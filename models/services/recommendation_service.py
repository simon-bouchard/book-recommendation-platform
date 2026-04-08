# models/services/recommendation_service.py
"""
Recommendation service using singleton generators and rankers.
"""

import asyncio
import logging
from typing import List

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from models.client.registry import get_metadata_client
from models.data.queries import get_read_books_for_candidates_async
from models.domain.candidate_generation import (
    create_joint_subject_generator,
    get_als_generator,
    get_popularity_generator,
)
from models.domain.config import RecommendationConfig
from models.domain.pipeline import RecommendationPipeline
from models.domain.rankers import get_noop_ranker
from models.domain.recommendation import Candidate, RecommendedBook
from models.domain.user import User

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


class RecommendationService:
    """
    Main recommendation service implementing business logic.

    Uses singleton generators and rankers for optimal performance.
    No instance-level state — can be recreated cheaply.

    Trace structure per request:
        recommendation.service
        ├── pipeline.generate
        │   └── POST /als_recs  (httpx auto-instrumented)
        └── recommendation.filter_and_enrich   ← filter + enrich run in parallel
            ├── interactions IN clause  (aiomysql)
            └── POST /enrich  (httpx auto-instrumented)
    """

    def __init__(self):
        """Initialize recommendation service."""

    async def recommend(self, user: User, config: RecommendationConfig) -> List[RecommendedBook]:
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

                if not candidates:
                    span.set_attribute("recommendation.result_count", 0)
                    return []

                recommendations = await self._filter_and_enrich(candidates, user, config.k)

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

    async def _filter_and_enrich(
        self,
        candidates: List[Candidate],
        user: User,
        k: int,
    ) -> List[RecommendedBook]:
        """
        Filter read books and enrich with metadata in parallel.

        Both operations are I/O-bound and independent: the DB query needs only
        the candidate item_idx list (available immediately), and the metadata
        HTTP call needs the same list. Running them with asyncio.gather overlaps
        the ~8ms DB round-trip with the ~10ms HTTP round-trip, reducing the
        combined cost to max(~8ms, ~10ms) instead of ~18ms sequential.

        After both complete, filtering and slicing are pure Python (no I/O).
        """
        with tracer.start_as_current_span("recommendation.filter_and_enrich") as span:
            item_ids = [c.item_idx for c in candidates]
            span.set_attribute("candidates.input_count", len(candidates))

            try:
                read_set, meta = await asyncio.gather(
                    get_read_books_for_candidates_async(user.user_id, item_ids),
                    get_metadata_client().enrich(item_ids),
                )

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
                    if c.item_idx not in read_set and (book := meta.get(c.item_idx)) is not None
                ][:k]

                span.set_attribute(
                    "candidates.filtered_count",
                    len(candidates) - len(read_set.intersection(set(item_ids))),
                )
                span.set_attribute("recommendation.result_count", len(result))
                return result

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def _build_pipeline(
        self, user: User, config: RecommendationConfig
    ) -> RecommendationPipeline:
        """
        Build recommendation pipeline using singleton components.

        is_warm() is only called for mode=auto, where the result determines
        which generator to use. For mode=behavioral and mode=subject the
        generator is fixed regardless of warmth, so the ALS server round-trip
        is skipped entirely.
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
            ranker=get_noop_ranker(),
        )

    async def _enrich(self, candidates: List[Candidate]) -> List[RecommendedBook]:
        """
        Enrich candidates with metadata. Used directly by tests and chatbot tools.

        For the recommendation hot path, use _filter_and_enrich instead so
        filtering and enrichment run concurrently.
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
