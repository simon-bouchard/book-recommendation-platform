# models/services/recommendation_service.py
"""
Recommendation service using singleton generators/filters/rankers.
"""

import logging
import time
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from models.domain.user import User
from models.domain.config import RecommendationConfig
from models.domain.recommendation import Candidate, RecommendedBook
from models.domain.pipeline import RecommendationPipeline
from models.domain.candidate_generation import (
    get_als_generator,
    get_popularity_generator,
    create_joint_subject_generator,
)
from models.domain.filters import get_read_books_filter
from models.domain.rankers import get_noop_ranker
from models.client.registry import get_metadata_client

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Main recommendation service implementing business logic.

    Uses singleton generators/filters/rankers for optimal performance.
    No instance-level state — can be recreated cheaply.
    """

    def __init__(self):
        """Initialize recommendation service."""

    async def recommend(
        self, user: User, config: RecommendationConfig, db: AsyncSession
    ) -> List[RecommendedBook]:
        """Generate personalized recommendations for a user."""
        start_time = time.time()

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

            candidates = await pipeline.recommend(user, config.k, db)
            recommendations = await self._enrich_candidates(candidates)

            latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Recommendation completed",
                extra={
                    "user_id": user.user_id,
                    "mode": config.mode,
                    "count": len(recommendations),
                    "latency_ms": latency_ms,
                },
            )

            return recommendations

        except Exception as e:
            logger.error(
                "Recommendation failed",
                extra={"user_id": user.user_id, "mode": config.mode, "error": str(e)},
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
            # Auto mode — only here do we need to know warmth.
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

    async def _enrich_candidates(self, candidates: List[Candidate]) -> List[RecommendedBook]:
        """Enrich candidates with book metadata via the metadata model server."""
        resp = await get_metadata_client().enrich([c.item_idx for c in candidates])
        meta = {b.item_idx: b for b in resp.books}

        return [
            RecommendedBook(
                item_idx=c.item_idx,
                title=meta[c.item_idx].title,
                score=c.score,
                num_ratings=meta[c.item_idx].num_ratings,
                author=meta[c.item_idx].author,
                year=meta[c.item_idx].year,
                isbn=meta[c.item_idx].isbn,
                cover_id=meta[c.item_idx].cover_id,
                avg_rating=meta[c.item_idx].avg_rating,
            )
            for c in candidates
            if c.item_idx in meta
        ]
