# models/infrastructure/popularity_scorer.py
"""
Popularity scorer for fallback and cold-start candidate generation.

Owns the raw Bayesian scores array and exposes a simple top-k retrieval.
"""

from typing import List, Optional, Tuple

import numpy as np


class PopularityScorer:
    """
    Returns the top-k books by precomputed Bayesian popularity score.

    Uses singleton pattern. Supports injection for testing.

    Example:
        scorer = PopularityScorer()
        item_ids, scores = scorer.top_k(k=100)
    """

    _instance: Optional["PopularityScorer"] = None

    def __new__(
        cls,
        bayesian_scores: Optional[np.ndarray] = None,
        book_ids: Optional[List[int]] = None,
    ):
        if any(x is not None for x in [bayesian_scores, book_ids]):
            instance = super().__new__(cls)
            instance._initialized = False
            return instance

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(
        self,
        bayesian_scores: Optional[np.ndarray] = None,
        book_ids: Optional[List[int]] = None,
    ):
        if self._initialized:
            return

        self._initialized = True

        if bayesian_scores is not None:
            self._scores = bayesian_scores.astype(np.float32, copy=False)
            self._book_ids = np.array(book_ids, dtype=np.int64)
        else:
            from models.data.loaders import load_bayesian_scores, load_book_subject_embeddings

            self._scores = load_bayesian_scores(use_cache=True).astype(np.float32)
            _, ids = load_book_subject_embeddings(use_cache=True)
            self._book_ids = np.array(ids, dtype=np.int64)

    def top_k(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the top-k books by Bayesian score.

        Args:
            k: Number of results to return.

        Returns:
            Tuple of (item_ids, scores) as numpy arrays of length <= k,
            ordered by descending score.
        """
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        search_k = min(k, len(self._scores))
        top_rows = np.argpartition(-self._scores, search_k - 1)[:search_k]
        top_rows = top_rows[np.argsort(-self._scores[top_rows])]

        return self._book_ids[top_rows], self._scores[top_rows]

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance for model reloading."""
        cls._instance = None
