# models/infrastructure/hybrid_scorer.py
"""
Joint subject-ALS hybrid scorer for item-to-item similarity.

At initialisation a single joint matrix J of shape (N, D_s + D_a) is built
once, with subject embeddings in the left columns and ALS factors (aligned to
subject rows, zeros where a book has no ALS factor) in the right columns.

At query time the joint query vector is scaled by alpha and (1 - alpha) so that
a single BLAS matmul produces the blended scores:

    J @ q  =  (1 - alpha) * (S @ q_s)  +  alpha * (A_aligned @ q_a)

This is mathematically identical to scoring the two spaces separately but
requires one memory pass instead of two, eliminates the per-request alignment
scatter, and reduces temporary allocations from O(4 N) to O(N + D).
"""

from typing import Optional, Tuple

import numpy as np


class HybridScorer:
    """
    Scores all books against a query item using blended subject and ALS similarity.

    Computes:
        score = (1 - alpha) * subject_cosine + alpha * als_cosine

    via a single matmul over a pre-built joint embedding matrix.  Books without
    ALS factors have zero in their ALS columns and receive zero ALS contribution;
    they remain eligible to appear in results if their subject score is strong.

    The joint matrix is built once at initialisation and shared across all
    requests.  Uses the singleton + injectable pattern: call with no arguments
    for the production singleton, or pass all five parameters for an injected
    instance (typically in tests).

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
        or all must be None for singleton mode (production use).
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
            self._build(
                subject_embeddings=subject_embeddings.astype(np.float32, copy=False),
                subject_ids=subject_ids,
                als_factors=als_factors.astype(np.float32, copy=False),
                als_row_for_subject=als_row_for_subject,
                rating_counts=rating_counts,
            )
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

            self._build(
                subject_embeddings=subject_embs.astype(np.float32, copy=False),
                subject_ids=subject_ids_list,
                als_factors=als_factors_loaded.astype(np.float32, copy=False),
                als_row_for_subject=als_row_for_subj,
                rating_counts=counts,
            )

    def _build(
        self,
        subject_embeddings: np.ndarray,
        subject_ids: list,
        als_factors: np.ndarray,
        als_row_for_subject: np.ndarray,
        rating_counts: np.ndarray,
    ) -> None:
        """
        Build the joint matrix and supporting lookup structures.

        Joint matrix layout (N rows, D_s + D_a columns):
            columns  0 : D_s          subject embeddings (all N books)
            columns  D_s : D_s + D_a  ALS factors aligned to subject rows;
                                      zero-filled for books with no ALS factor

        Args:
            subject_embeddings: (N, D_s) float32, L2-normalised.
            subject_ids: List of N item_idx values in subject row order.
            als_factors: (M, D_a) float32, L2-normalised, M <= N.
            als_row_for_subject: (N,) int32; als_row_for_subject[i] is the row
                in als_factors for subject row i, or -1 if the book has no ALS.
            rating_counts: (N,) int32 rating counts aligned to subject rows.
        """
        n, d_s = subject_embeddings.shape
        d_a = als_factors.shape[1]

        joint = np.zeros((n, d_s + d_a), dtype=np.float32)
        joint[:, :d_s] = subject_embeddings

        has_als = als_row_for_subject >= 0
        joint[has_als, d_s:] = als_factors[als_row_for_subject[has_als]]

        self._joint_matrix: np.ndarray = joint
        self._d_subject: int = d_s
        self._subject_ids: np.ndarray = np.array(subject_ids, dtype=np.int64)
        self._item_to_subject_row: dict = {int(idx): i for i, idx in enumerate(subject_ids)}
        self._rating_counts: np.ndarray = rating_counts

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

        Builds a scaled joint query vector from the query book's row in the
        joint matrix and performs a single matmul to produce blended scores.
        Alpha is applied at query time so it can vary freely per call without
        any index rebuild.

        Args:
            item_idx: Query book item index.
            k: Number of results to return.
            alpha: ALS weight in [0, 1]; subject weight is (1 - alpha).
            filter_candidates: If True, restrict results to books meeting
                the rating threshold.
            min_rating_count: Minimum rating count threshold.  If None,
                uses HYBRID_MIN_RATINGS.

        Returns:
            Tuple of (item_ids, scores) as numpy arrays of length <= k,
            ordered by descending score.  Both arrays are empty if item_idx
            has no subject embedding.
        """
        if not self.has_item(item_idx):
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        subj_row = self._item_to_subject_row[item_idx]
        d_s = self._d_subject

        query_row = self._joint_matrix[subj_row]
        query_vec = np.empty_like(query_row)
        query_vec[:d_s] = (1.0 - alpha) * query_row[:d_s]
        query_vec[d_s:] = alpha * query_row[d_s:]

        final_scores: np.ndarray = self._joint_matrix @ query_vec

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
        return top_item_ids[mask][:k], top_scores[mask][:k]

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance for model reloading."""
        cls._instance = None
