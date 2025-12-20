# models/domain/pipeline.py
"""
Recommendation pipeline orchestrating candidate generation, filtering, and ranking.
Provides composable recommendation flow: generate -> fallback -> filter -> rank -> top k.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from models.domain.recommendation import Candidate
from models.domain.user import User
from models.domain.candidate_generation import CandidateGenerator
from models.domain.filters import Filter, NoFilter
from models.domain.rankers import Ranker, NoOpRanker


class RecommendationPipeline:
    """
    Orchestrates the recommendation process.

    Pipeline stages:
    1. Generate candidates from primary generator
    2. If no candidates, use fallback generator
    3. Apply filters (e.g., remove read books)
    4. Rank remaining candidates
    5. Return top k

    Example:
        pipeline = RecommendationPipeline(
            generator=ALSBasedGenerator(),
            fallback_generator=BayesianPopularityGenerator(),
            filter=ReadBooksFilter(),
            ranker=NoOpRanker()
        )

        recommendations = pipeline.recommend(user, k=20, db=db_session)
    """

    def __init__(
        self,
        generator: CandidateGenerator,
        fallback_generator: Optional[CandidateGenerator] = None,
        filter: Filter = None,
        ranker: Ranker = None,
    ):
        """
        Initialize recommendation pipeline.

        Args:
            generator: Primary candidate generator
            fallback_generator: Generator to use if primary returns no candidates
            filter: Filter to apply to candidates (default: NoFilter)
            ranker: Ranker for final ordering (default: NoOpRanker)
        """
        self.generator = generator
        self.fallback_generator = fallback_generator
        self.filter = filter or NoFilter()
        self.ranker = ranker or NoOpRanker()

    def recommend(self, user: User, k: int, db: Session = None) -> List[Candidate]:
        """
        Generate recommendations for a user.

        Args:
            user: User to generate recommendations for
            k: Number of recommendations to return
            db: Database session (required if filter needs DB access)

        Returns:
            Top k candidates after generation, filtering, and ranking

        Process:
            1. Generate candidates (request more than k for filtering buffer)
            2. If empty and fallback exists, generate from fallback
            3. Apply filter
            4. Rank
            5. Return top k
        """
        if k <= 0:
            return []

        # Generate candidates (request 2x k to account for filtering)
        buffer_k = max(k * 2, 500)
        candidates = self.generator.generate(user, buffer_k)

        # Fallback if primary generator returned nothing
        if not candidates and self.fallback_generator is not None:
            candidates = self.fallback_generator.generate(user, buffer_k)

        # No candidates available
        if not candidates:
            return []

        # Filter candidates (e.g., remove read books)
        candidates = self.filter.apply(candidates, user, db)

        # No candidates left after filtering
        if not candidates:
            return []

        # Rank candidates
        candidates = self.ranker.rank(candidates, user)

        # Return top k
        return candidates[:k]
