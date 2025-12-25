# models/domain/candidate_generation.py
"""
Candidate generation strategies with module-level singleton instances.
Import these singletons instead of creating new instances.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from collections import defaultdict

import numpy as np

from models.domain.recommendation import Candidate
from models.domain.user import User
from models.infrastructure.subject_embedder import SubjectEmbedder
from models.infrastructure.als_model import ALSModel
from models.infrastructure.similarity_index import SimilarityIndex
from models.data.loaders import (
    load_book_subject_embeddings,
    load_bayesian_scores,
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
    """Generate candidates based on subject similarity."""

    def __init__(self, embedder: SubjectEmbedder = None, similarity_index: SimilarityIndex = None):
        self.embedder = embedder or SubjectEmbedder()

        if similarity_index is None:
            embeddings, ids = load_book_subject_embeddings(normalized=True, use_cache=True)
            self.similarity_index = SimilarityIndex(
                embeddings=embeddings,
                ids=ids,
                normalize=False,
            )
        else:
            self.similarity_index = similarity_index

    def generate(self, user: User, k: int) -> List[Candidate]:
        if not user.has_preferences:
            return []

        if k <= 0:
            return []

        user_embedding = self.embedder.embed(user.fav_subjects)

        embeddings = self.similarity_index.embeddings_full
        scores = embeddings @ user_embedding

        top_indices = np.argsort(-scores)[:k]
        top_scores = scores[top_indices]
        top_item_ids = self.similarity_index.ids_full[top_indices]

        candidates = [
            Candidate(item_idx=int(item_id), score=float(score), source=self.name)
            for item_id, score in zip(top_item_ids, top_scores)
        ]

        return candidates

    @property
    def name(self) -> str:
        return "subject"


class ALSBasedGenerator(CandidateGenerator):
    """Generate candidates using ALS collaborative filtering."""

    def __init__(self, als_model: ALSModel = None):
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


# ============================================================================
# MODULE-LEVEL SINGLETONS - Import these instead of creating new instances
# ============================================================================

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
    But it reuses the cached base generators internally.
    """
    generators = []

    if subject_weight > 0:
        generators.append((get_subject_generator(), subject_weight))

    if popularity_weight > 0:
        generators.append((get_popularity_generator(), popularity_weight))

    if not generators:
        raise ValueError("At least one weight must be > 0")

    return HybridGenerator(generators)
