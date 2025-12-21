# models/services/recommendation_service.py
"""
Recommendation service implementing business rules for user-specific recommendation strategies.
Orchestrates pipeline selection, execution, and result enrichment.
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
    ALSBasedGenerator,
    SubjectBasedGenerator,
    BayesianPopularityGenerator,
    HybridGenerator,
)
from models.domain.filters import ReadBooksFilter, FilterChain
from models.domain.rankers import NoOpRanker
from models.data.loaders import load_book_meta

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Main recommendation service implementing business logic.

    Responsibilities:
    - Select appropriate recommendation strategy (warm/cold/fallback)
    - Execute recommendation pipeline
    - Enrich candidates with metadata
    - Provide structured logging for observability

    Business Rules:
    - Warm users (is_warm=True): Use ALS collaborative filtering
    - Cold users with preferences: Use hybrid (subject + popularity)
    - Cold users without preferences: Use popularity fallback

    Example:
        service = RecommendationService()
        config = RecommendationConfig(k=20, mode="auto")
        recommendations = service.recommend(user, config, db)
    """

    def __init__(self):
        """Initialize recommendation service."""
        self._book_meta = None

    def recommend(
        self, user: User, config: RecommendationConfig, db: Session
    ) -> List[RecommendedBook]:
        """
        Generate personalized recommendations for a user.

        Args:
            user: User to generate recommendations for
            config: Recommendation configuration
            db: Database session for filtering read books

        Returns:
            List of recommended books with metadata

        Raises:
            ValueError: If config is invalid
        """
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
            # Select and execute pipeline
            pipeline = self._build_pipeline(user, config)
            candidates = pipeline.recommend(user, config.k, db)

            # Enrich candidates with metadata
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
        Build recommendation pipeline based on user status and config.

        Args:
            user: User to build pipeline for
            config: Recommendation configuration

        Returns:
            Configured recommendation pipeline
        """
        # Determine strategy based on mode
        if config.mode == "behavioral":
            # Force behavioral (ALS) mode
            primary_generator = ALSBasedGenerator()
            fallback_generator = BayesianPopularityGenerator()

        elif config.mode == "subject":
            # Force subject-based mode
            primary_generator = self._build_hybrid_generator(config)
            fallback_generator = BayesianPopularityGenerator()

        else:
            # Auto mode - decide based on user status
            if user.is_warm:
                # Warm user: use ALS
                primary_generator = ALSBasedGenerator()
                fallback_generator = BayesianPopularityGenerator()

            elif user.has_preferences:
                # Cold user with preferences: use hybrid
                primary_generator = self._build_hybrid_generator(config)
                fallback_generator = BayesianPopularityGenerator()

            else:
                # Cold user without preferences: use popularity
                primary_generator = BayesianPopularityGenerator()
                fallback_generator = None

        # Build pipeline with read book filtering
        pipeline = RecommendationPipeline(
            generator=primary_generator,
            fallback_generator=fallback_generator,
            filter=ReadBooksFilter(),
            ranker=NoOpRanker(),
        )

        return pipeline

    def _build_hybrid_generator(self, config: RecommendationConfig) -> HybridGenerator:
        """
        Build hybrid generator combining subject similarity and popularity.

        Args:
            config: Configuration with hybrid_config settings

        Returns:
            Configured hybrid generator
        """
        subject_weight = config.hybrid_config.subject_weight
        popularity_weight = config.hybrid_config.popularity_weight

        generators = []

        if subject_weight > 0:
            generators.append((SubjectBasedGenerator(), subject_weight))

        if popularity_weight > 0:
            generators.append((BayesianPopularityGenerator(), popularity_weight))

        if not generators:
            raise ValueError("At least one generator weight must be > 0")

        return HybridGenerator(generators)

    def _enrich_candidates(self, candidates: List[Candidate]) -> List[RecommendedBook]:
        """
        Enrich candidates with book metadata.

        Args:
            candidates: Raw candidates from pipeline

        Returns:
            Enriched recommendations with full metadata
        """
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
