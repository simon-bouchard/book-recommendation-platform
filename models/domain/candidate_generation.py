# models/domain/candidate_generation.py
"""
Candidate generation strategies for recommendation system.
Each generator produces candidate books based on different relevance signals.
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
    """
    Abstract base class for candidate generation strategies.

    Each generator produces a ranked list of candidate books based on
    different relevance signals (subject similarity, collaborative filtering,
    popularity, or hybrid combinations).
    """

    @abstractmethod
    def generate(self, user: User, k: int) -> List[Candidate]:
        """
        Generate k candidate books for a user.

        Args:
            user: User to generate candidates for
            k: Number of candidates to generate

        Returns:
            List of Candidate objects, sorted by score (descending)
            Returns empty list if unable to generate candidates.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of this generator for tracking candidate sources.

        Returns:
            Human-readable generator name (e.g., "subject", "als", "popularity")
        """
        pass


class SubjectBasedGenerator(CandidateGenerator):
    """
    Generate candidates based on subject similarity.

    Computes user embedding from favorite subjects and finds books
    with similar subject embeddings using cosine similarity.
    """

    def __init__(self, embedder: SubjectEmbedder = None, similarity_index: SimilarityIndex = None):
        """
        Initialize subject-based generator.

        Args:
            embedder: SubjectEmbedder instance (or None to use singleton)
            similarity_index: SimilarityIndex instance (or None to build from loaded embeddings)
        """
        self.embedder = embedder or SubjectEmbedder()

        if similarity_index is None:
            embeddings, ids = load_book_subject_embeddings(normalized=True)
            self.similarity_index = SimilarityIndex(
                embeddings=embeddings,
                ids=ids,
                normalize=False,  # Already normalized
            )
        else:
            self.similarity_index = similarity_index

    def generate(self, user: User, k: int) -> List[Candidate]:
        """
        Generate subject-based candidates.

        Returns empty list if user has no valid subject preferences.
        """
        if not user.has_preferences:
            return []

        if k <= 0:
            return []

        # Compute user embedding from favorite subjects
        user_embedding = self.embedder.embed(user.fav_subjects)

        # Find similar books
        # Search in embedding space (not by item_id, so we need to use the index differently)
        # Actually, we need to compute similarities manually since we have a query embedding
        # not a query item_id

        # Compute similarities: embeddings @ user_embedding
        embeddings = self.similarity_index.embeddings_full
        cosine_scores = embeddings @ user_embedding

        # Transform cosine similarity [-1, 1] to non-negative scores [0, 1]
        # This is required because Candidate enforces non-negative scores
        scores = (cosine_scores + 1) / 2

        # Get top k
        top_indices = np.argsort(-scores)[:k]
        top_scores = scores[top_indices]
        top_item_ids = self.similarity_index.ids_full[top_indices]

        # Create candidates
        candidates = [
            Candidate(item_idx=int(item_id), score=float(score), source=self.name)
            for item_id, score in zip(top_item_ids, top_scores)
        ]

        return candidates

    @property
    def name(self) -> str:
        return "subject"


class ALSBasedGenerator(CandidateGenerator):
    """
    Generate candidates using ALS collaborative filtering.

    Uses matrix factorization latent factors to predict user preferences
    based on past interaction patterns.
    """

    def __init__(self, als_model: ALSModel = None):
        """
        Initialize ALS-based generator.

        Args:
            als_model: ALSModel instance (or None to use singleton)
        """
        self.als_model = als_model or ALSModel()

    def generate(self, user: User, k: int) -> List[Candidate]:
        """
        Generate ALS-based candidates.

        Returns empty list if user not in ALS model (cold user).
        """
        if not self.als_model.has_user(user.user_id):
            return []

        if k <= 0:
            return []

        # Get user factors
        user_factors = self.als_model.get_user_factors(user.user_id)
        if user_factors is None:
            return []

        # Compute scores for all books: book_factors @ user_factors
        scores = self.als_model.book_factors @ user_factors

        # Get top k
        top_indices = np.argsort(-scores)[:k]
        top_scores = scores[top_indices]

        # Map row indices to item_idx
        top_item_ids = [self.als_model.book_row_to_id[int(idx)] for idx in top_indices]

        # Create candidates
        candidates = [
            Candidate(item_idx=int(item_id), score=float(score), source=self.name)
            for item_id, score in zip(top_item_ids, top_scores)
        ]

        return candidates

    @property
    def name(self) -> str:
        return "als"


class BayesianPopularityGenerator(CandidateGenerator):
    """
    Generate candidates based on Bayesian popularity scores.

    Returns books ranked by smoothed average ratings, providing a
    quality-weighted popularity signal. Always succeeds (fallback generator).
    """

    def __init__(self):
        """Initialize popularity generator with precomputed Bayesian scores."""
        self.bayesian_scores = load_bayesian_scores(use_cache=True)
        _, self.book_ids = load_book_subject_embeddings(use_cache=True)

    def generate(self, user: User, k: int) -> List[Candidate]:
        """
        Generate popularity-based candidates.

        Always succeeds - does not depend on user preferences or history.
        """
        if k <= 0:
            return []

        # Get top k by Bayesian score
        top_indices = np.argsort(-self.bayesian_scores)[:k]
        top_scores = self.bayesian_scores[top_indices]
        top_item_ids = [self.book_ids[idx] for idx in top_indices]

        # Create candidates
        candidates = [
            Candidate(item_idx=int(item_id), score=float(score), source=self.name)
            for item_id, score in zip(top_item_ids, top_scores)
        ]

        return candidates

    @property
    def name(self) -> str:
        return "popularity"


class HybridGenerator(CandidateGenerator):
    """
    Generate candidates by blending multiple generators.

    Combines scores from multiple generators using weighted averaging,
    deduplicating candidates and keeping the highest blended score.
    """

    def __init__(self, generators: List[Tuple[CandidateGenerator, float]]):
        """
        Initialize hybrid generator.

        Args:
            generators: List of (generator, weight) tuples.
                       Weights should sum to 1.0 but will be normalized if not.

        Raises:
            ValueError: If generators list is empty or weights are invalid
        """
        if not generators:
            raise ValueError("HybridGenerator requires at least one generator")

        if any(weight <= 0 for _, weight in generators):
            raise ValueError("All generator weights must be positive")

        # Normalize weights to sum to 1.0
        total_weight = sum(weight for _, weight in generators)
        self.generators = [(gen, weight / total_weight) for gen, weight in generators]

    def generate(self, user: User, k: int) -> List[Candidate]:
        """
        Generate hybrid candidates by blending multiple sources.

        Process:
        1. Generate candidates from each generator (requesting k from each)
        2. Normalize scores within each source to [0, 1]
        3. Blend scores using weighted average
        4. Deduplicate (keep highest blended score)
        5. Return top k by blended score
        """
        if k <= 0:
            return []

        # Collect candidates from all generators
        all_candidates_by_source: List[Tuple[List[Candidate], float]] = []

        for generator, weight in self.generators:
            candidates = generator.generate(user, k * 2)  # Request more for diversity
            if candidates:
                all_candidates_by_source.append((candidates, weight))

        if not all_candidates_by_source:
            return []

        # Normalize scores within each source and blend
        blended_scores = defaultdict(float)
        item_sources = defaultdict(list)  # Track which sources contributed

        for candidates, weight in all_candidates_by_source:
            # Normalize scores to [0, 1] within this source
            scores = np.array([c.score for c in candidates])
            if scores.max() > scores.min():
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized_scores = np.ones_like(scores)

            # Add weighted normalized scores
            for candidate, norm_score in zip(candidates, normalized_scores):
                blended_scores[candidate.item_idx] += weight * norm_score
                item_sources[candidate.item_idx].append(candidate.source)

        # Create blended candidates
        blended_candidates = [
            Candidate(
                item_idx=item_idx,
                score=score,
                source=f"hybrid({'+'.join(set(item_sources[item_idx]))})",
            )
            for item_idx, score in blended_scores.items()
        ]

        # Sort by blended score and return top k
        blended_candidates.sort(key=lambda c: c.score, reverse=True)
        return blended_candidates[:k]

    @property
    def name(self) -> str:
        source_names = [gen.name for gen, _ in self.generators]
        return f"hybrid({'+'.join(source_names)})"
