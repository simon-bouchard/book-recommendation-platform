# models/infrastructure/hybrid_scorer.py
"""
Joint subject-ALS hybrid scorer for item-to-item similarity.

Owns the subject/ALS alignment map and performs the single-pass joint matmul
that produces hybrid similarity scores across all books simultaneously.
"""

from typing import Optional, Tuple

import numpy as np


class HybridScorer:
    """
    Scores all books against a query item using blended subject and ALS similarity.

    Performs a single-pass joint matmul over both embedding matrices:

        score = (1 - alpha) * subject_cosine + alpha * als_cosine

    Books without ALS factors receive a zero ALS contribution — they are not
    excluded and can still appear in results if their subject score is strong.

    The subject/ALS alignment map is built once on initialization and reused
    across all requests. Uses singleton pattern. Supports injection for testing.

    Example:
        scorer = HybridScorer()
        item_ids, scores = scorer.score(item_idx=42, k=200, alpha=0.6)
    """

    _instance: Optional["HybridScorer"] = None

    HYBRID_MIN_RATINGS: int = 5

    def __new__(
        cls,
        subject_embeddings: Optional[np.ndarray] = None,
        subject_ids: Optional[list] = None,
        als_factors: Optional[np.ndarray] = None,
        als_row_for_subject: Optional[np.ndarray] = None,
        rating_counts: Optional[np.ndarray] = None,
    ):
        """
        Singleton instantiation with optional injection.

        All five parameters must be provided together for injection mode,
        or all must be None for singleton mode.
        """
        injected = [
            subject_embeddings,
            subject_ids,
            als_factors,
            als_row_for_subject,
            rating_counts,
        ]
        if any(x is not None for x in injected):
            instance = super().__new__(cls)
            instance._initialized = False
            return instance

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(
        self,
        subject_embeddings: Optional[np.ndarray] = None,
        subject_ids: Optional[list] = None,
        als_factors: Optional[np.ndarray] = None,
        als_row_for_subject: Optional[np.ndarray] = None,
        rating_counts: Optional[np.ndarray] = None,
    ):
        if self._initialized:
            return

        self._initialized = True

        if subject_embeddings is not None:
            self._subject_embeddings = subject_embeddings.astype(np.float32, copy=False)
            self._subject_ids = np.array(subject_ids, dtype=np.int64)
            self._item_to_subject_row = {int(idx): i for i, idx in enumerate(subject_ids)}
            self._als_factors = als_factors.astype(np.float32, copy=False)
            self._als_row_for_subject = als_row_for_subject
            self._rating_counts = rating_counts
        else:
            from models.data.loaders import (
                load_als_factors,
                load_book_meta,
                load_book_subject_embeddings,
            )

            subject_embs, subject_ids_list = load_book_subject_embeddings(
                normalized=True, use_cache=True
            )
            _, als_factors_loaded, _, als_row_map = load_als_factors(
                normalized=True, use_cache=True
            )

            item_to_als_row = {int(item_idx): row for row, item_idx in als_row_map.items()}

            n_subject = len(subject_ids_list)
            als_row_for_subj = np.full(n_subject, -1, dtype=np.int32)
            for subj_row, item_idx in enumerate(subject_ids_list):
                als_row_for_subj[subj_row] = item_to_als_row.get(int(item_idx), -1)

            metadata = load_book_meta(use_cache=True)
            counts = (
                metadata.reindex(subject_ids_list)["book_num_ratings"]
                .fillna(0)
                .astype(np.int32)
                .to_numpy()
            )

            self._subject_embeddings = subject_embs.astype(np.float32, copy=False)
            self._subject_ids = np.array(subject_ids_list, dtype=np.int64)
            self._item_to_subject_row = {int(idx): i for i, idx in enumerate(subject_ids_list)}
            self._als_factors = als_factors_loaded.astype(np.float32, copy=False)
            self._als_row_for_subject = als_row_for_subj
            self._rating_counts = counts

    def has_item(self, item_idx: int) -> bool:
        """Return True if item_idx has a subject embedding and can be queried."""
        return int(item_idx) in self._item_to_subject_row

    def score(
        self,
        item_idx: int,
        k: int,
        alpha: float,
        filter_candidates: bool = True,
        min_rating_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute hybrid similarity scores for all books against a query item.

        Args:
            item_idx: Query book item index.
            k: Number of results to return.
            alpha: ALS weight in [0, 1]; subject weight is (1 - alpha).
            filter_candidates: If True, restrict results to books meeting
                               the rating threshold.
            min_rating_count: Minimum rating count threshold. If None,
                              uses HYBRID_MIN_RATINGS.

        Returns:
            Tuple of (item_ids, scores) as numpy arrays of length <= k,
            ordered by descending score. Both arrays are empty if item_idx
            has no subject embedding.
        """
        if not self.has_item(item_idx):
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        subj_row = self._item_to_subject_row[item_idx]
        query_subject = self._subject_embeddings[subj_row]
        subject_scores = self._subject_embeddings @ query_subject

        als_row = int(self._als_row_for_subject[subj_row])
        if als_row >= 0:
            query_als = self._als_factors[als_row]
            als_scores_all = self._als_factors @ query_als
            als_scores_aligned = np.zeros(len(subject_scores), dtype=np.float32)
            valid = self._als_row_for_subject >= 0
            als_scores_aligned[valid] = als_scores_all[self._als_row_for_subject[valid]]
        else:
            als_scores_aligned = np.zeros(len(subject_scores), dtype=np.float32)

        final_scores = (1.0 - alpha) * subject_scores + alpha * als_scores_aligned

        if filter_candidates:
            threshold = (
                min_rating_count if min_rating_count is not None else self.HYBRID_MIN_RATINGS
            )
            candidate_indices = np.flatnonzero(self._rating_counts >= threshold)
        else:
            candidate_indices = np.arange(len(final_scores), dtype=np.int64)

        if candidate_indices.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        search_k = min(k + 1, candidate_indices.size)
        candidate_scores = final_scores[candidate_indices]
        top_k_in_candidates = np.argpartition(-candidate_scores, search_k - 1)[:search_k]
        top_k_in_candidates = top_k_in_candidates[
            np.argsort(-candidate_scores[top_k_in_candidates])
        ]

        top_indices = candidate_indices[top_k_in_candidates]
        top_item_ids = self._subject_ids[top_indices]
        top_scores = final_scores[top_indices]

        mask = top_item_ids != item_idx
        top_item_ids = top_item_ids[mask][:k]
        top_scores = top_scores[mask][:k]

        return top_item_ids, top_scores

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance for model reloading."""
        cls._instance = None
