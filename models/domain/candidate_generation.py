# models/domain/candidate_generation.py
"""
Candidate generation strategies using shared FAISS indices for optimal performance.
All subject-based operations share a single singleton FAISS index.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from collections import defaultdict

import numpy as np

from models.domain.recommendation import Candidate
from models.domain.user import User
from models.infrastructure.subject_embedder import SubjectEmbedder
from models.infrastructure.als_model import ALSModel
from models.infrastructure.similarity_index import SimilarityIndex
from models.infrastructure.similarity_indices import get_subject_similarity_index
from models.data.loaders import (
    load_bayesian_scores,
    load_book_subject_embeddings,
)


class CandidateGenerator(ABC):
    """Abstract base class for candidate generation strategies."""

    @abstractmethod
    def generate(self, user: User, k: int) -> List[Candidate]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class SubjectBasedGenerator(CandidateGenerator):
    """
    Generate candidates based on subject similarity using FAISS.

    Uses shared singleton FAISS index for optimal performance and memory usage.
    """

    def __init__(
        self,
        embedder: Optional[SubjectEmbedder] = None,
        similarity_index: Optional[SimilarityIndex] = None,
    ):
        """
        Initialize subject-based generator.

        Args:
            embedder: Subject embedder instance. If None, uses singleton.
            similarity_index: Similarity index instance. If None, uses shared singleton.
                             Only provide for testing with mock data.
        """
        self.embedder = embedder or SubjectEmbedder()
        self.similarity_index = similarity_index or get_subject_similarity_index()

    def generate(self, user: User, k: int) -> List[Candidate]:
        if not user.has_preferences:
            return []

        if k <= 0:
            return []

        user_embedding = self.embedder.embed(user.fav_subjects)
        query = user_embedding.reshape(1, -1).astype(np.float32)

        search_k = min(k, self.similarity_index.index.ntotal)
        if search_k == 0:
            return []

        distances, indices = self.similarity_index.index.search(query, search_k)

        scores = distances[0]
        item_ids = self.similarity_index.ids_full[indices[0]]

        candidates = [
            Candidate(item_idx=int(item_id), score=float(score), source=self.name)
            for item_id, score in zip(item_ids, scores)
        ]

        return candidates

    @property
    def name(self) -> str:
        return "subject"


class ALSBasedGenerator(CandidateGenerator):
    """Generate candidates using ALS collaborative filtering."""

    def __init__(self, als_model: Optional[ALSModel] = None):
        """
        Initialize ALS-based generator.

        Args:
            als_model: ALS model instance. If None, uses singleton.
        """
        self.als_model = als_model or ALSModel()

    def generate(self, user: User, k: int) -> List[Candidate]:
        if not self.als_model.has_user(user.user_id):
            return []

        if k <= 0:
            return []

        user_factors = self.als_model.get_user_factors(user.user_id)
        if user_factors is None:
            return []

        scores = self.als_model.book_factors @ user_factors

        top_indices = np.argsort(-scores)[:k]
        top_scores = scores[top_indices]

        top_item_ids = [self.als_model.book_row_to_id[int(idx)] for idx in top_indices]

        candidates = [
            Candidate(item_idx=int(item_id), score=float(score), source=self.name)
            for item_id, score in zip(top_item_ids, top_scores)
        ]

        return candidates

    @property
    def name(self) -> str:
        return "als"


class BayesianPopularityGenerator(CandidateGenerator):
    """Generate candidates based on Bayesian popularity scores."""

    def __init__(self):
        self.bayesian_scores = load_bayesian_scores(use_cache=True)
        _, self.book_ids = load_book_subject_embeddings(use_cache=True)

    def generate(self, user: User, k: int) -> List[Candidate]:
        if k <= 0:
            return []

        top_indices = np.argsort(-self.bayesian_scores)[:k]
        top_scores = self.bayesian_scores[top_indices]
        top_item_ids = [self.book_ids[idx] for idx in top_indices]

        candidates = [
            Candidate(item_idx=int(item_id), score=float(score), source=self.name)
            for item_id, score in zip(top_item_ids, top_scores)
        ]

        return candidates

    @property
    def name(self) -> str:
        return "popularity"


class HybridGenerator(CandidateGenerator):
    """Generate candidates by blending multiple generators."""

    def __init__(self, generators: List[Tuple[CandidateGenerator, float]]):
        """
        Initialize hybrid generator.

        Args:
            generators: List of (generator, weight) tuples where weights are positive

        Raises:
            ValueError: If no generators provided or any weight is non-positive
        """
        if not generators:
            raise ValueError("HybridGenerator requires at least one generator")

        if any(weight <= 0 for _, weight in generators):
            raise ValueError("All generator weights must be positive")

        total_weight = sum(weight for _, weight in generators)
        self.generators = [(gen, weight / total_weight) for gen, weight in generators]

    def generate(self, user: User, k: int) -> List[Candidate]:
        if k <= 0:
            return []

        all_candidates_by_source: List[Tuple[List[Candidate], float]] = []

        for generator, weight in self.generators:
            candidates = generator.generate(user, k * 2)
            if candidates:
                all_candidates_by_source.append((candidates, weight))

        if not all_candidates_by_source:
            return []

        blended_scores = defaultdict(float)
        item_sources = defaultdict(list)

        for candidates, weight in all_candidates_by_source:
            scores = np.array([c.score for c in candidates])
            if scores.max() > scores.min():
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized_scores = np.ones_like(scores)

            for candidate, norm_score in zip(candidates, normalized_scores):
                blended_scores[candidate.item_idx] += weight * norm_score
                item_sources[candidate.item_idx].append(candidate.source)

        blended_candidates = [
            Candidate(
                item_idx=item_idx,
                score=score,
                source=f"hybrid({'+'.join(set(item_sources[item_idx]))})",
            )
            for item_idx, score in blended_scores.items()
        ]

        blended_candidates.sort(key=lambda c: c.score, reverse=True)
        return blended_candidates[:k]

    @property
    def name(self) -> str:
        source_names = [gen.name for gen, _ in self.generators]
        return f"hybrid({'+'.join(source_names)})"


class JointSubjectGenerator(CandidateGenerator):
    """
    Generate candidates via joint scoring over all books in a single pass.

    Blends cosine similarity between the user embedding and every book's
    subject embedding with pre-normalized Bayesian popularity scores:

        score = alpha * cosine_score + (1 - alpha) * bayesian_score

    Cosine scores are min-max normalized inline before blending so both
    components operate on the same [0, 1] scale. Bayesian scores are
    normalized once at load time and cached.

    This approach avoids the recall loss of the two-list merge pattern:
    a book with a moderate cosine score but high Bayesian score will be
    correctly ranked in the blended output because it is never dropped
    from the candidate pool during retrieval.
    """

    def __init__(self, alpha: float = 0.6):
        """
        Initialize the joint subject generator.

        Heavy data (embeddings, normalized Bayesian scores) is loaded from
        the module-level cache. Only the alpha scalar is instance-specific,
        so instances are cheap to create per request.

        Args:
            alpha: Weight for cosine similarity; (1 - alpha) is the
                   Bayesian popularity weight. Must be in [0, 1].
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha
        self._embeddings, self._book_ids = load_book_subject_embeddings(
            normalized=True, use_cache=True
        )
        self._bayesian_norm = load_bayesian_scores(normalized=True, use_cache=True)
        self._embedder = SubjectEmbedder()

    def generate(self, user: User, k: int) -> List[Candidate]:
        if not user.has_preferences:
            return []

        if k <= 0:
            return []

        user_vec = self._embedder.embed(user.fav_subjects).astype(np.float32)
        cosine_scores = self._embeddings @ user_vec

        lo, hi = cosine_scores.min(), cosine_scores.max()
        if hi > lo:
            cosine_norm = (cosine_scores - lo) / (hi - lo)
        else:
            cosine_norm = np.ones_like(cosine_scores)

        blended = self.alpha * cosine_norm + (1.0 - self.alpha) * self._bayesian_norm

        search_k = min(k, len(blended))
        top_rows = np.argpartition(-blended, search_k - 1)[:search_k]
        top_rows = top_rows[np.argsort(-blended[top_rows])]

        return [
            Candidate(
                item_idx=int(self._book_ids[row]),
                score=float(blended[row]),
                source=self.name,
            )
            for row in top_rows
        ]

    @property
    def name(self) -> str:
        return "joint_subject"


_subject_generator = None
_als_generator = None
_popularity_generator = None


def get_subject_generator() -> SubjectBasedGenerator:
    """Get or create singleton SubjectBasedGenerator."""
    global _subject_generator
    if _subject_generator is None:
        _subject_generator = SubjectBasedGenerator()
    return _subject_generator


def get_als_generator() -> ALSBasedGenerator:
    """Get or create singleton ALSBasedGenerator."""
    global _als_generator
    if _als_generator is None:
        _als_generator = ALSBasedGenerator()
    return _als_generator


def get_popularity_generator() -> BayesianPopularityGenerator:
    """Get or create singleton BayesianPopularityGenerator."""
    global _popularity_generator
    if _popularity_generator is None:
        _popularity_generator = BayesianPopularityGenerator()
    return _popularity_generator


def create_hybrid_generator(
    subject_weight: float = 0.6, popularity_weight: float = 0.4
) -> HybridGenerator:
    """
    Create HybridGenerator using singleton base generators.

    Note: HybridGenerator is NOT cached because weights can vary per request.
    But it reuses the cached base generators internally for efficiency.

    Args:
        subject_weight: Weight for subject-based scoring (0-1)
        popularity_weight: Weight for popularity-based scoring (0-1)

    Returns:
        HybridGenerator instance with specified weights

    Raises:
        ValueError: If both weights are zero
    """
    generators = []

    if subject_weight > 0:
        generators.append((get_subject_generator(), subject_weight))

    if popularity_weight > 0:
        generators.append((get_popularity_generator(), popularity_weight))

    if not generators:
        raise ValueError("At least one weight must be > 0")

    return HybridGenerator(generators)


def create_joint_subject_generator(alpha: float = 0.6) -> JointSubjectGenerator:
    """
    Create a JointSubjectGenerator with the given blend weight.

    Not cached because alpha varies per request. The generator is cheap
    to instantiate — it holds references to cached arrays, not copies.

    Args:
        alpha: Subject similarity weight; popularity weight is (1 - alpha).

    Returns:
        JointSubjectGenerator instance ready for candidate generation.
    """
    return JointSubjectGenerator(alpha=alpha)
