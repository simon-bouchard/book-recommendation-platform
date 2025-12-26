# models/services/similarity_service.py
"""
Book similarity service using shared FAISS indices for optimal performance.
Hybrid mode uses FAISS for initial retrieval then blends scores for accuracy.
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


_subject_index = None
_als_index = None
_hybrid_data = None


def _get_subject_index() -> SimilarityIndex:
    """Get or create singleton subject similarity index."""
    global _subject_index
    if _subject_index is None:
        embeddings, ids = load_book_subject_embeddings(normalized=True, use_cache=True)
        _subject_index = SimilarityIndex(embeddings=embeddings, ids=ids, normalize=False)
    return _subject_index


def _get_als_index() -> SimilarityIndex:
    """Get or create singleton ALS similarity index with filtering."""
    global _als_index
    if _als_index is None:
        _, book_factors, _, book_row_map = load_als_factors(normalized=True, use_cache=True)
        book_ids = [book_row_map[i] for i in range(book_factors.shape[0])]
        metadata = load_book_meta(use_cache=True)

        _als_index = SimilarityIndex.create_filtered_index(
            embeddings=book_factors,
            ids=book_ids,
            metadata=metadata,
            min_rating_count=10,
            normalize=False,
        )
    return _als_index


def _get_hybrid_data() -> dict:
    """
    Get or create singleton hybrid similarity data structures.

    Pre-computes alignment between subject and ALS embeddings for fast hybrid similarity.
    This approach uses full matmuls (like original code) which is faster than FAISS retrieval.
    """
    global _hybrid_data
    if _hybrid_data is None:
        subject_embeddings, subject_ids_list = load_book_subject_embeddings(
            normalized=True, use_cache=True
        )
        _, als_factors, _, als_row_map = load_als_factors(normalized=True, use_cache=True)

        als_row_to_item = {row: item_idx for row, item_idx in als_row_map.items()}
        item_to_als_row = {item_idx: row for row, item_idx in als_row_to_item.items()}

        n_subject = len(subject_ids_list)
        als_row_for_subject = np.full(n_subject, -1, dtype=np.int32)

        for subject_row, item_idx in enumerate(subject_ids_list):
            als_row = item_to_als_row.get(int(item_idx), -1)
            als_row_for_subject[subject_row] = als_row

        metadata = load_book_meta(use_cache=True)
        counts = (
            metadata.reindex(subject_ids_list)["book_num_ratings"]
            .fillna(0)
            .astype(np.int32)
            .to_numpy()
        )

        item_to_subject_row = {item_idx: i for i, item_idx in enumerate(subject_ids_list)}

        _hybrid_data = {
            "subject_embeddings": subject_embeddings,
            "subject_ids": np.array(subject_ids_list, dtype=np.int64),
            "als_factors": als_factors,
            "als_row_for_subject": als_row_for_subject,
            "counts": counts,
            "item_to_subject_row": item_to_subject_row,
        }

    return _hybrid_data


class SimilarityService:
    """
    Book similarity service using singleton FAISS indices.

    All subject operations share the same FAISS index for memory efficiency.
    Hybrid mode uses FAISS for fast initial retrieval then exact scoring for accuracy.
    """

    ALS_MIN_RATINGS = 10
    HYBRID_MIN_RATINGS = 5
    HYBRID_RETRIEVAL_MULTIPLIER = 3

    def __init__(self):
        """Initialize similarity service."""
        self._book_meta = None

    def get_similar(
        self,
        item_idx: int,
        mode: str = "subject",
        k: int = 200,
        alpha: float = 0.6,
        min_rating_count: Optional[int] = None,
        filter_candidates: bool = True,
    ) -> List[Dict]:
        """Find similar books with mode-specific filtering."""
        start_time = time.time()

        logger.info("Similarity search started", extra={"item_idx": item_idx, "mode": mode, "k": k})

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
        """Get subject-based similarity using shared FAISS index."""
        index = _get_subject_index()
        scores, item_ids = index.search(query_item_id=item_idx, k=k, exclude_query=True)
        return self._format_results(item_ids, scores)

    def _get_similar_als(
        self,
        item_idx: int,
        k: int,
        min_rating_count: Optional[int],
        filter_candidates: bool,
    ) -> List[Dict]:
        """Get ALS-based similarity using filtered FAISS index."""
        index = _get_als_index()
        scores, item_ids = index.search(query_item_id=item_idx, k=k, exclude_query=True)
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
        Get hybrid similarity using original full matmul approach.

        Uses pre-computed alignment between subject and ALS embeddings.
        This approach (from original code) is faster than FAISS retrieval for hybrid mode.
        """
        data = _get_hybrid_data()

        if item_idx not in data["item_to_subject_row"]:
            return []

        subj_row = data["item_to_subject_row"][item_idx]
        query_subject = data["subject_embeddings"][subj_row].astype(np.float32, copy=False)

        subject_scores = data["subject_embeddings"] @ query_subject

        als_row = data["als_row_for_subject"][subj_row]
        if als_row >= 0:
            query_als = data["als_factors"][als_row].astype(np.float32, copy=False)
            als_scores_all = data["als_factors"] @ query_als

            als_scores_aligned = np.zeros_like(subject_scores, dtype=np.float32)
            valid_mask = data["als_row_for_subject"] >= 0
            als_scores_aligned[valid_mask] = als_scores_all[data["als_row_for_subject"][valid_mask]]
        else:
            als_scores_aligned = np.zeros_like(subject_scores, dtype=np.float32)

        final_scores = (1.0 - alpha) * subject_scores + alpha * als_scores_aligned

        if filter_candidates:
            threshold = (
                min_rating_count if min_rating_count is not None else self.HYBRID_MIN_RATINGS
            )
            candidate_indices = np.flatnonzero(data["counts"] >= threshold)
        else:
            candidate_indices = np.arange(len(final_scores), dtype=np.int64)

        if candidate_indices.size == 0:
            return []

        search_k = min(k + 1, candidate_indices.size)
        candidate_scores = final_scores[candidate_indices]

        top_k_in_candidates = np.argpartition(-candidate_scores, search_k - 1)[:search_k]
        top_k_in_candidates = top_k_in_candidates[
            np.argsort(-candidate_scores[top_k_in_candidates])
        ]

        top_indices = candidate_indices[top_k_in_candidates]
        top_scores = final_scores[top_indices]
        top_item_ids = data["subject_ids"][top_indices]

        mask = top_item_ids != item_idx
        top_item_ids = top_item_ids[mask][:k]
        top_scores = top_scores[mask][:k]

        return self._format_results(top_item_ids, top_scores)

    def _get_book_meta(self) -> pd.DataFrame:
        """Get cached book metadata."""
        if self._book_meta is None:
            self._book_meta = load_book_meta(use_cache=True)
        return self._book_meta

    def _format_results(self, item_ids: np.ndarray, scores: np.ndarray) -> List[Dict]:
        """Format results with book metadata."""
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
