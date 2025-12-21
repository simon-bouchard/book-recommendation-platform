# models/services/similarity_service.py
"""
Book similarity service providing subject-based, ALS-based, and hybrid similarity search.
Implements mode-specific filtering and two-pool architecture for quality control.
"""

import logging
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from models.infrastructure.similarity_index import SimilarityIndex
from models.data.loaders import (
    load_book_subject_embeddings,
    load_als_factors,
    load_book_meta,
)

logger = logging.getLogger(__name__)


class SimilarityService:
    """
    Book similarity service with mode-specific filtering.

    Provides three similarity modes:
    - Subject: Semantic similarity based on book subjects (no filtering)
    - ALS: Collaborative filtering based on user behavior (10+ ratings filter)
    - Hybrid: Blended subject + ALS scores (5+ ratings filter, configurable)

    Two-pool architecture:
    - Query pool: Any book can be queried (even low-rated)
    - Candidate pool: Only high-quality books in results

    Example:
        service = SimilarityService()

        # Subject similarity (no filtering)
        results = service.get_similar(item_idx=123, mode="subject", k=20)

        # ALS similarity (10+ ratings)
        results = service.get_similar(item_idx=123, mode="als", k=20)

        # Hybrid with custom blend
        results = service.get_similar(
            item_idx=123, mode="hybrid", k=20, alpha=0.7
        )
    """

    ALS_MIN_RATINGS = 10
    HYBRID_MIN_RATINGS = 5

    def __init__(self):
        """Initialize similarity service with lazy-loaded indices."""
        self._subject_index = None
        self._als_index = None
        self._book_meta = None

        # Hybrid mode cached data
        self._hybrid_initialized = False
        self._hybrid_subject_embs = None
        self._hybrid_subject_ids = None
        self._hybrid_als_factors = None
        self._hybrid_als_row_map = None
        self._hybrid_als_row_for_subject = None
        self._hybrid_counts = None
        self._hybrid_item_to_subject_row = None

    def get_similar(
        self,
        item_idx: int,
        mode: str = "subject",
        k: int = 200,
        alpha: float = 0.6,
        min_rating_count: Optional[int] = None,
        filter_candidates: bool = True,
    ) -> List[Dict]:
        """
        Find similar books with mode-specific filtering.

        Args:
            item_idx: Book to find similar items for
            mode: "subject" | "als" | "hybrid"
            k: Number of results to return
            alpha: Blend weight for hybrid (0.0=subject, 1.0=ALS)
            min_rating_count: Override default rating threshold
            filter_candidates: Enable/disable candidate filtering

        Returns:
            List of similar books with metadata

        Raises:
            ValueError: If mode is invalid
            HTTPException: If book not found or has no data for requested mode
        """
        start_time = time.time()

        logger.info(
            "Similarity search started",
            extra={"item_idx": item_idx, "mode": mode, "k": k},
        )

        try:
            if mode == "subject":
                results = self._get_similar_subject(item_idx, k)
            elif mode == "als":
                results = self._get_similar_als(item_idx, k, min_rating_count, filter_candidates)
            elif mode == "hybrid":
                results = self._get_similar_hybrid(
                    item_idx, k, alpha, min_rating_count, filter_candidates
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Similarity search completed",
                extra={
                    "item_idx": item_idx,
                    "mode": mode,
                    "count": len(results),
                    "latency_ms": latency_ms,
                },
            )

            return results

        except Exception as e:
            logger.error(
                "Similarity search failed",
                extra={"item_idx": item_idx, "mode": mode, "error": str(e)},
                exc_info=True,
            )
            raise

    def _get_similar_subject(self, item_idx: int, k: int) -> List[Dict]:
        """
        Get subject-based similarity with no filtering.

        Args:
            item_idx: Query book
            k: Number of results

        Returns:
            Similar books ranked by subject similarity
        """
        if self._subject_index is None:
            embeddings, ids = load_book_subject_embeddings(normalized=True, use_cache=True)
            self._subject_index = SimilarityIndex(embeddings=embeddings, ids=ids, normalize=False)

        scores, item_ids = self._subject_index.search(
            query_item_id=item_idx, k=k, exclude_query=True
        )

        return self._format_results(item_ids, scores)

    def _get_similar_als(
        self,
        item_idx: int,
        k: int,
        min_rating_count: Optional[int],
        filter_candidates: bool,
    ) -> List[Dict]:
        """
        Get ALS-based similarity with rating count filtering.

        Args:
            item_idx: Query book
            k: Number of results
            min_rating_count: Override default threshold (10+)
            filter_candidates: Enable/disable filtering

        Returns:
            Similar books ranked by collaborative filtering
        """
        if self._als_index is None:
            _, book_factors, _, book_row_map = load_als_factors(normalized=True, use_cache=True)
            book_ids = [book_row_map[i] for i in range(book_factors.shape[0])]

            if filter_candidates:
                threshold = (
                    min_rating_count if min_rating_count is not None else self.ALS_MIN_RATINGS
                )
                metadata = self._get_book_meta()

                self._als_index = SimilarityIndex.create_filtered_index(
                    embeddings=book_factors,
                    ids=book_ids,
                    metadata=metadata,
                    min_rating_count=threshold,
                    normalize=False,
                )
            else:
                self._als_index = SimilarityIndex(
                    embeddings=book_factors, ids=book_ids, normalize=False
                )

        scores, item_ids = self._als_index.search(query_item_id=item_idx, k=k, exclude_query=True)

        return self._format_results(item_ids, scores)

    def _get_similar_hybrid(
        self,
        item_idx: int,
        k: int,
        alpha: float,
        min_rating_count: Optional[int],
        filter_candidates: bool,
    ) -> List[Dict]:
        """
        Get hybrid similarity by blending subject and ALS scores.

        Process:
        1. Compute subject scores for all books
        2. Compute ALS scores for all books
        3. Align ALS scores to subject space (handle missing items)
        4. Blend: final = (1-alpha)*subject + alpha*ALS
        5. Filter by rating count (default 5+)
        6. Return top k

        Args:
            item_idx: Query book
            k: Number of results
            alpha: Blend weight (0.0=subject, 1.0=ALS)
            min_rating_count: Override default threshold (5+)
            filter_candidates: Enable/disable filtering

        Returns:
            Similar books ranked by blended scores
        """
        # Initialize hybrid data structures (one-time cost)
        if not self._hybrid_initialized:
            self._initialize_hybrid()

        # Check if query item exists
        if item_idx not in self._hybrid_item_to_subject_row:
            return []

        # Get query row in subject space
        subject_row = self._hybrid_item_to_subject_row[item_idx]
        query_subject = self._hybrid_subject_embs[subject_row].astype(np.float32)

        # Compute subject scores (all books)
        subject_scores = self._hybrid_subject_embs @ query_subject

        # Compute ALS scores (if query has ALS factors)
        als_row = self._hybrid_als_row_for_subject[subject_row]

        if als_row >= 0:
            query_als = self._hybrid_als_factors[als_row].astype(np.float32)
            als_scores_all = self._hybrid_als_factors @ query_als

            # Align ALS scores to subject space
            als_scores_aligned = np.zeros_like(subject_scores, dtype=np.float32)
            valid_mask = self._hybrid_als_row_for_subject >= 0
            als_scores_aligned[valid_mask] = als_scores_all[
                self._hybrid_als_row_for_subject[valid_mask]
            ]
        else:
            als_scores_aligned = np.zeros_like(subject_scores, dtype=np.float32)

        # Blend scores
        final_scores = (1.0 - alpha) * subject_scores + alpha * als_scores_aligned

        # Apply rating count filter
        if filter_candidates:
            threshold = (
                min_rating_count if min_rating_count is not None else self.HYBRID_MIN_RATINGS
            )
            candidate_mask = self._hybrid_counts >= threshold
            candidate_indices = np.flatnonzero(candidate_mask)
        else:
            candidate_indices = np.arange(len(final_scores), dtype=np.int64)

        if candidate_indices.size == 0:
            return []

        # Get top k from candidates (excluding query)
        search_k = min(k + 1, candidate_indices.size)
        candidate_scores = final_scores[candidate_indices]

        # Partial sort to get top k
        top_k_indices_in_candidates = np.argpartition(-candidate_scores, search_k - 1)[:search_k]
        top_k_indices_in_candidates = top_k_indices_in_candidates[
            np.argsort(-candidate_scores[top_k_indices_in_candidates])
        ]

        top_indices = candidate_indices[top_k_indices_in_candidates]
        top_scores = final_scores[top_indices]
        top_item_ids = self._hybrid_subject_ids[top_indices]

        # Remove query item
        mask = top_item_ids != item_idx
        top_item_ids = top_item_ids[mask][:k]
        top_scores = top_scores[mask][:k]

        return self._format_results(top_item_ids, top_scores)

    def _initialize_hybrid(self):
        """
        Initialize hybrid similarity data structures.

        Builds alignment mapping between subject and ALS spaces.
        This is a one-time cost on first hybrid query.
        """
        # Load embeddings
        self._hybrid_subject_embs, subject_ids_list = load_book_subject_embeddings(
            normalized=True, use_cache=True
        )
        self._hybrid_subject_ids = np.array(subject_ids_list, dtype=np.int64)

        _, self._hybrid_als_factors, _, als_row_map = load_als_factors(
            normalized=True, use_cache=True
        )

        # Build mapping: item_idx -> subject row
        self._hybrid_item_to_subject_row = {
            item_idx: i for i, item_idx in enumerate(subject_ids_list)
        }

        # Build mapping: ALS row -> item_idx
        als_row_to_item = {row: item_idx for row, item_idx in als_row_map.items()}

        # Build alignment: subject row -> ALS row
        n_subject = len(subject_ids_list)
        self._hybrid_als_row_for_subject = np.full(n_subject, -1, dtype=np.int32)

        # Create reverse lookup: item_idx -> ALS row
        item_to_als_row = {item_idx: row for row, item_idx in als_row_to_item.items()}

        for subject_row, item_idx in enumerate(subject_ids_list):
            als_row = item_to_als_row.get(int(item_idx), -1)
            self._hybrid_als_row_for_subject[subject_row] = als_row

        # Load rating counts aligned to subject order
        metadata = self._get_book_meta()
        self._hybrid_counts = (
            metadata.reindex(self._hybrid_subject_ids)["book_num_ratings"]
            .fillna(0)
            .astype(np.int32)
            .to_numpy()
        )

        self._hybrid_initialized = True

    def _get_book_meta(self) -> pd.DataFrame:
        """Get cached book metadata."""
        if self._book_meta is None:
            self._book_meta = load_book_meta(use_cache=True)
        return self._book_meta

    def _format_results(self, item_ids: np.ndarray, scores: np.ndarray) -> List[Dict]:
        """
        Format results with book metadata.

        Args:
            item_ids: Array of item indices
            scores: Array of similarity scores

        Returns:
            List of dicts with book metadata and scores
        """
        metadata = self._get_book_meta()
        results = []

        for item_id, score in zip(item_ids, scores):
            item_id = int(item_id)

            if item_id not in metadata.index:
                continue

            row = metadata.loc[item_id]

            results.append(
                {
                    "item_idx": item_id,
                    "title": str(row["title"]),
                    "author": str(row["author"]) if pd.notnull(row.get("author")) else None,
                    "year": int(row["year"]) if pd.notnull(row.get("year")) else None,
                    "isbn": str(row["isbn"]) if pd.notnull(row.get("isbn")) else None,
                    "cover_id": str(row["cover_id"]) if pd.notnull(row.get("cover_id")) else None,
                    "score": float(score),
                }
            )

        return results
