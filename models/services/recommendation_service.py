# models/services/recommendation_service.py
"""
Recommendation service using singleton generators/filters/rankers.
"""

import logging
import time
from typing import List
from sqlalchemy.orm import Session

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
from models.data.loaders import load_book_meta

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Main recommendation service implementing business logic.

    Uses singleton generators/filters/rankers for optimal performance.
    No instance-level state - can be recreated cheaply.
    """

    def __init__(self):
        """Initialize recommendation service."""
        self._book_meta = None

    def recommend(
        self, user: User, config: RecommendationConfig, db: Session
    ) -> List[RecommendedBook]:
        """Generate personalized recommendations for a user."""
        start_time = time.time()

        logger.info(
            "Recommendation started",
            extra={
                "user_id": user.user_id,
                "mode": config.mode,
                "is_warm": user.is_warm,
                "has_preferences": user.has_preferences,
                "k": config.k,
            },
        )

        try:
            pipeline = self._build_pipeline(user, config)
            candidates = pipeline.recommend(user, config.k, db)
            recommendations = self._enrich_candidates(candidates)

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

    def _build_pipeline(self, user: User, config: RecommendationConfig) -> RecommendationPipeline:
        """
        Build recommendation pipeline using singleton components.

        Key insight: Generators/filters/rankers are singletons (one copy),
        but pipelines are recreated per request (they're lightweight wrappers).
        """
        # Determine strategy based on mode
        if config.mode == "behavioral":
            primary_generator = get_als_generator()
            fallback_generator = get_popularity_generator()

        elif config.mode == "subject":
            primary_generator = create_joint_subject_generator(
                alpha=config.hybrid_config.subject_weight,
            )
            fallback_generator = get_popularity_generator()

        else:
            # Auto mode - decide based on user status
            if user.is_warm:
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

        # Pipeline is lightweight - recreate each time (just holds references)
        pipeline = RecommendationPipeline(
            generator=primary_generator,
            fallback_generator=fallback_generator,
            filter=get_read_books_filter(),
            ranker=get_noop_ranker(),
        )

        return pipeline

    def _enrich_candidates(self, candidates: List[Candidate]) -> List[RecommendedBook]:
        """Enrich candidates with book metadata."""
        if self._book_meta is None:
            self._book_meta = load_book_meta(use_cache=True)

        recommendations = []

        for candidate in candidates:
            if candidate.item_idx not in self._book_meta.index:
                continue

            row = self._book_meta.loc[candidate.item_idx]

            recommendations.append(
                RecommendedBook(
                    item_idx=candidate.item_idx,
                    title=str(row["title"]),
                    score=candidate.score,
                    num_ratings=int(row["book_num_ratings"]) if "book_num_ratings" in row else 0,
                    author=str(row["author"]) if "author" in row and row["author"] else None,
                    year=int(row["year"]) if "year" in row and row["year"] else None,
                    isbn=str(row["isbn"]) if "isbn" in row and row["isbn"] else None,
                    cover_id=str(row["cover_id"])
                    if "cover_id" in row and row["cover_id"]
                    else None,
                    avg_rating=float(row["book_avg_rating"])
                    if "book_avg_rating" in row and row["book_avg_rating"]
                    else None,
                )
            )

        return recommendations
