# models/infrastructure/subject_scorer.py
"""
Joint subject-popularity scorer for cold-start user recommendations.

Owns the normalized subject embedding matrix and normalized Bayesian scores.
Performs the single-pass joint matmul that produces blended recommendation
scores without an intermediate retrieval step.
"""

from typing import List, Optional, Tuple

import numpy as np


class SubjectScorer:
    """
    Scores all books jointly against a user embedding and Bayesian popularity.

    Blends cosine similarity and popularity in a single pass over all books:

        score = alpha * normalize(cosine_scores) + (1 - alpha) * normalize(bayesian_scores)

    Both components are independently min-max normalized before blending so
    neither dominates by scale regardless of the raw value ranges.

    Bayesian scores are normalized once at load time and cached. Cosine scores
    are normalized inline per request since they depend on the query vector.

    Uses singleton pattern for performance. Supports injection for testing.

    Example:
        scorer = SubjectScorer()
        item_ids, scores = scorer.score(user_vector=emb, k=200, alpha=0.6)
    """

    _instance: Optional["SubjectScorer"] = None

    def __new__(
        cls,
        embeddings: Optional[np.ndarray] = None,
        book_ids: Optional[List[int]] = None,
        bayesian_scores_norm: Optional[np.ndarray] = None,
    ):
        """
        Singleton instantiation with optional injection.

        Args:
            embeddings: Injected normalized subject embeddings for testing.
            book_ids: Injected book ID list for testing.
            bayesian_scores_norm: Injected normalized Bayesian scores for testing.
        """
        if any(x is not None for x in [embeddings, book_ids, bayesian_scores_norm]):
            instance = super().__new__(cls)
            instance._initialized = False
            return instance

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        book_ids: Optional[List[int]] = None,
        bayesian_scores_norm: Optional[np.ndarray] = None,
    ):
        if self._initialized:
            return

        self._initialized = True

        if embeddings is not None:
            self._embeddings = embeddings.astype(np.float32, copy=False)
            self._book_ids = np.array(book_ids, dtype=np.int64)
            self._bayesian_norm = bayesian_scores_norm.astype(np.float32, copy=False)
        else:
            from models.data.loaders import load_bayesian_scores, load_book_subject_embeddings

            embs, ids = load_book_subject_embeddings(normalized=True, use_cache=True)
            self._embeddings = embs.astype(np.float32, copy=False)
            self._book_ids = np.array(ids, dtype=np.int64)
            self._bayesian_norm = load_bayesian_scores(normalized=True, use_cache=True)

    def score(
        self,
        user_vector: np.ndarray,
        k: int,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute blended scores for all books and return the top-k.

        Args:
            user_vector: L2-normalized user embedding of shape (D,).
            k: Number of results to return.
            alpha: Subject similarity weight in [0, 1].
                   Popularity weight is (1 - alpha).

        Returns:
            Tuple of (item_ids, scores) as numpy arrays of length <= k,
            ordered by descending score.
        """
        cosine_scores = self._embeddings @ user_vector.astype(np.float32)

        lo, hi = cosine_scores.min(), cosine_scores.max()
        cosine_norm = (cosine_scores - lo) / (hi - lo) if hi > lo else np.ones_like(cosine_scores)

        blended = alpha * cosine_norm + (1.0 - alpha) * self._bayesian_norm

        search_k = min(k, len(blended))
        top_rows = np.argpartition(-blended, search_k - 1)[:search_k]
        top_rows = top_rows[np.argsort(-blended[top_rows])]

        return self._book_ids[top_rows], blended[top_rows]

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance for model reloading."""
        cls._instance = None
