# models/domain/rankers.py
"""
Rankers with module-level singleton instances.
"""

from typing import List, Protocol

from models.domain.recommendation import Candidate
from models.domain.user import User


class Ranker(Protocol):
    """Protocol for candidate rankers."""

    def rank(self, candidates: List[Candidate], user: User) -> List[Candidate]: ...


class NoOpRanker:
    """Pass-through ranker that preserves generator ordering."""

    def rank(self, candidates: List[Candidate], user: User) -> List[Candidate]:
        return candidates


class ScoreRanker:
    """Rank candidates by their score in descending order."""

    def rank(self, candidates: List[Candidate], user: User) -> List[Candidate]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)


# ============================================================================
# MODULE-LEVEL SINGLETONS
# ============================================================================

_noop_ranker = None
_score_ranker = None


def get_noop_ranker() -> NoOpRanker:
    """Get or create singleton NoOpRanker."""
    global _noop_ranker
    if _noop_ranker is None:
        _noop_ranker = NoOpRanker()
    return _noop_ranker


def get_score_ranker() -> ScoreRanker:
    """Get or create singleton ScoreRanker."""
    global _score_ranker
    if _score_ranker is None:
        _score_ranker = ScoreRanker()
    return _score_ranker
