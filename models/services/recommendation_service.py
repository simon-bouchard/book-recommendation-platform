# models/services/recommendation_service.py
"""
Recommendation service using singleton generators/filters/rankers.
"""

import logging
from typing import List

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from models.client.registry import get_metadata_client
from models.domain.candidate_generation import (
    create_joint_subject_generator,
    get_als_generator,
    get_popularity_generator,
)
from models.domain.config import RecommendationConfig
from models.domain.filters import get_read_books_filter
from models.domain.pipeline import RecommendationPipeline
from models.domain.rankers import get_noop_ranker
from models.domain.recommendation import Candidate, RecommendedBook
from models.domain.user import User

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


class RecommendationService:
    """
    Main recommendation service implementing business logic.

    Uses singleton generators/filters/rankers for optimal performance.
    No instance-level state — can be recreated cheaply.

    Trace structure per request:
        recommendation.service
        ├── pipeline.generate
        │   └── POST /als_recs  (httpx auto-instrumented)
        ├── pipeline.filter
        │   └── SELECT interactions  (SQLAlchemy auto-instrumented)
        └── recommendation.enrich
            ├── POST /enrich  (httpx auto-instrumented)
            └── metadata.build_index
    """

    def __init__(self):
        """Initialize recommendation service."""

    async def recommend(
        self, user: User, config: RecommendationConfig
    ) -> List[RecommendedBook]:
        """Generate personalized recommendations for a user."""
        with tracer.start_as_current_span("recommendation.service") as span:
            span.set_attribute("user.id", user.user_id)
            span.set_attribute("recommendation.mode", config.mode)
            span.set_attribute("recommendation.k", config.k)
            span.set_attribute("user.has_preferences", user.has_preferences)

            try:
                pipeline = await self._build_pipeline(user, config)

                logger.info(
                    "Recommendation started",
                    extra={
                        "user_id": user.user_id,
                        "mode": config.mode,
                        "has_preferences": user.has_preferences,
                        "k": config.k,
                    },
                )

                candidates = await pipeline.recommend(user, config.k)
                recommendations = await self._enrich(candidates)

                span.set_attribute("recommendation.result_count", len(recommendations))

                logger.info(
                    "Recommendation completed",
                    extra={
                        "user_id": user.user_id,
                        "mode": config.mode,
                        "count": len(recommendations),
                    },
                )

                return recommendations

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                logger.error(
                    "Recommendation failed",
                    extra={"user_id": user.user_id, "mode": config.mode, "error": str(exc)},
                    exc_info=True,
                )
                raise

    async def _build_pipeline(
        self, user: User, config: RecommendationConfig
    ) -> RecommendationPipeline:
        """
        Build recommendation pipeline using singleton components.

        is_warm() is only called for mode=auto, where the result determines
        which generator to use. For mode=behavioral and mode=subject the
        generator is fixed regardless of warmth, so the ALS server round-trip
        is skipped entirely (~4ms saving per request).

        Key insight: Generators/filters/rankers are singletons (one copy),
        but pipelines are recreated per request (they are lightweight wrappers).
        """
        if config.mode == "behavioral":
            primary_generator = get_als_generator()
            fallback_generator = get_popularity_generator()

        elif config.mode == "subject":
            primary_generator = create_joint_subject_generator(
                alpha=config.hybrid_config.subject_weight,
            )
            fallback_generator = get_popularity_generator()

        else:
            is_warm = await user.is_warm()
            if is_warm:
                primary_generator = get_als_generator()
                fallback_generator = get_popularity_generator()
            elif user.has_preferences:
                primary_generator = create_joint_subject_generator(
                    alpha=config.hybrid_config.subject_weight,
                )
                fallback_generator = get_popularity_generator()
            else:
                primary_generator = get_popularity_generator()
                fallback_generator = None

        return RecommendationPipeline(
            generator=primary_generator,
            fallback_generator=fallback_generator,
            filter=get_read_books_filter(),
            ranker=get_noop_ranker(),
        )

    async def _enrich(self, candidates: List[Candidate]) -> List[RecommendedBook]:
        """
        Enrich candidates with book metadata via the metadata model server.

        get_metadata_client().enrich() returns a dict[int, dict] keyed by
        item_idx. Fields are accessed directly from the raw dict, avoiding
        any Pydantic object construction on the hot path.

        The httpx auto-instrumentor attaches POST /enrich as a child of this
        span, making enrichment latency cleanly visible as a sibling of the
        pipeline span in the Jaeger waterfall.
        """
        with tracer.start_as_current_span("recommendation.enrich") as span:
            span.set_attribute("enrich.input_count", len(candidates))

            try:
                meta = await get_metadata_client().enrich([c.item_idx for c in candidates])

                result = [
                    RecommendedBook(
                        item_idx=c.item_idx,
                        title=book["title"],
                        score=c.score,
                        num_ratings=book["num_ratings"],
                        author=book["author"],
                        year=book["year"],
                        isbn=book["isbn"],
                        cover_id=book["cover_id"],
                        avg_rating=book["avg_rating"],
                    )
                    for c in candidates
                    if (book := meta.get(c.item_idx)) is not None
                ]

                span.set_attribute("enrich.output_count", len(result))
                span.set_attribute("enrich.missing_count", len(candidates) - len(result))
                return result

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
