# models/domain/candidate_generation.py
"""
Candidate generation strategies.

Each generator is a thin domain wrapper: it calls one infrastructure method,
converts the returned (item_ids, scores) arrays into Candidate objects, and
knows nothing about numpy, loaders, or scoring logic.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from models.domain.recommendation import Candidate
from models.domain.user import User
from models.infrastructure.subject_embedder import SubjectEmbedder
from models.infrastructure.subject_scorer import SubjectScorer
from models.infrastructure.als_model import ALSModel
from models.infrastructure.popularity_scorer import PopularityScorer


class CandidateGenerator(ABC):
    """Abstract base class for candidate generation strategies."""

    @abstractmethod
    def generate(self, user: User, k: int) -> List[Candidate]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class JointSubjectGenerator(CandidateGenerator):
    """
    Cold-start candidates via joint subject-popularity scoring.

    Embeds the user's subject preferences then delegates to SubjectScorer
    for a single-pass blended retrieval over all books.
    """

    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Subject similarity weight in [0, 1].
                   Popularity weight is (1 - alpha).
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha
        self._embedder = SubjectEmbedder()
        self._scorer = SubjectScorer()

    def generate(self, user: User, k: int) -> List[Candidate]:
        if not user.has_preferences or k <= 0:
            return []

        user_vec = self._embedder.embed(user.fav_subjects)
        item_ids, scores = self._scorer.score(user_vec, k=k, alpha=self.alpha)

        return [
            Candidate(item_idx=int(iid), score=float(s), source=self.name)
            for iid, s in zip(item_ids, scores)
        ]

    @property
    def name(self) -> str:
        return "joint_subject"


class ALSBasedGenerator(CandidateGenerator):
    """Warm-user candidates via ALS collaborative filtering."""

    def __init__(self, als_model: Optional[ALSModel] = None):
        self._als = als_model or ALSModel()

    def generate(self, user: User, k: int) -> List[Candidate]:
        if k <= 0:
            return []

        item_ids, scores = self._als.score(user.user_id, k=k)

        return [
            Candidate(item_idx=int(iid), score=float(s), source=self.name)
            for iid, s in zip(item_ids, scores)
        ]

    @property
    def name(self) -> str:
        return "als"


class PopularityBasedGenerator(CandidateGenerator):
    """Fallback candidates ranked by Bayesian popularity score."""

    def __init__(self, scorer: Optional[PopularityScorer] = None):
        self._scorer = scorer or PopularityScorer()

    def generate(self, user: User, k: int) -> List[Candidate]:
        if k <= 0:
            return []

        item_ids, scores = self._scorer.top_k(k=k)

        return [
            Candidate(item_idx=int(iid), score=float(s), source=self.name)
            for iid, s in zip(item_ids, scores)
        ]

    @property
    def name(self) -> str:
        return "popularity"


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

_als_generator: Optional[ALSBasedGenerator] = None
_popularity_generator: Optional[PopularityBasedGenerator] = None


def get_als_generator() -> ALSBasedGenerator:
    """Get or create singleton ALSBasedGenerator."""
    global _als_generator
    if _als_generator is None:
        _als_generator = ALSBasedGenerator()
    return _als_generator


def get_popularity_generator() -> PopularityBasedGenerator:
    """Get or create singleton PopularityBasedGenerator."""
    global _popularity_generator
    if _popularity_generator is None:
        _popularity_generator = PopularityBasedGenerator()
    return _popularity_generator


def create_joint_subject_generator(alpha: float = 0.6) -> JointSubjectGenerator:
    """
    Create a JointSubjectGenerator with the given blend weight.

    Not cached because alpha varies per request. The instance is cheap —
    it holds references to cached infrastructure singletons, not copies.

    Args:
        alpha: Subject similarity weight; popularity weight is (1 - alpha).
    """
    return JointSubjectGenerator(alpha=alpha)
