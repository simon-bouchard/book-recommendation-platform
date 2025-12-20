# models/domain/rankers.py
"""
Rankers for sorting candidates by relevance.
Rankers determine final ordering of filtered candidates.
"""

from typing import Protocol, List

from models.domain.recommendation import Candidate
from models.domain.user import User


class Ranker(Protocol):
    """
    Protocol for candidate rankers.

    Rankers determine the final ordering of candidates after
    generation and filtering.
    """

    def rank(self, candidates: List[Candidate], user: User) -> List[Candidate]:
        """
        Rank candidates by relevance.

        Args:
            candidates: Candidates to rank
            user: User context for ranking decisions

        Returns:
            Candidates sorted by relevance (highest first)
        """
        ...


class NoOpRanker:
    """
    Pass-through ranker that preserves generator ordering.

    Returns candidates in their original order (assuming generators
    already sorted by score).
    """

    def rank(self, candidates: List[Candidate], user: User) -> List[Candidate]:
        """Return candidates in original order."""
        return candidates


class ScoreRanker:
    """
    Rank candidates by their score in descending order.

    Useful when combining candidates from multiple sources that may
    not be pre-sorted.
    """

    def rank(self, candidates: List[Candidate], user: User) -> List[Candidate]:
        """Sort candidates by score (highest first)."""
        return sorted(candidates, key=lambda c: c.score, reverse=True)
