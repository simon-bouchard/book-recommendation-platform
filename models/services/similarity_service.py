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
        Get hybrid similarity using FAISS for initial retrieval then exact scoring.

        Strategy:
        1. Use FAISS to get top-N candidates from subject similarity (N = 3*k)
        2. Use FAISS to get top-N candidates from ALS similarity
        3. Take union of candidates
        4. For each candidate, compute exact subject + ALS scores
        5. Blend and rank
        6. Return top-k

        This avoids computing scores for ALL books while maintaining accuracy.
        """
        subject_index = _get_subject_index()
        als_index = _get_als_index()

        if not subject_index.has_item(item_idx):
            return []

        retrieval_k = k * self.HYBRID_RETRIEVAL_MULTIPLIER

        subject_scores_initial, subject_ids_initial = subject_index.search(
            query_item_id=item_idx, k=retrieval_k, exclude_query=True
        )

        candidate_set = set(int(iid) for iid in subject_ids_initial)

        if als_index.has_item(item_idx):
            als_scores_initial, als_ids_initial = als_index.search(
                query_item_id=item_idx, k=retrieval_k, exclude_query=True
            )
            candidate_set.update(int(iid) for iid in als_ids_initial)

        if not candidate_set:
            return []

        subject_embeddings, subject_ids_list = load_book_subject_embeddings(
            normalized=True, use_cache=True
        )
        subject_id_to_row = {iid: i for i, iid in enumerate(subject_ids_list)}

        query_row = subject_id_to_row.get(item_idx)
        if query_row is None:
            return []

        query_vec = subject_embeddings[query_row]

        candidate_list = sorted(candidate_set)
        candidate_rows = [
            subject_id_to_row[iid] for iid in candidate_list if iid in subject_id_to_row
        ]

        if not candidate_rows:
            return []

        candidate_embeddings = subject_embeddings[candidate_rows]
        subject_scores = candidate_embeddings @ query_vec

        if als_index.has_item(item_idx):
            _, als_factors, _, als_row_map = load_als_factors(normalized=True, use_cache=True)
            als_id_to_row = {item_idx: row for row, item_idx in als_row_map.items()}

            query_als_row = als_id_to_row.get(item_idx)
            if query_als_row is not None:
                query_als_vec = als_factors[query_als_row]

                candidate_item_ids = [subject_ids_list[row] for row in candidate_rows]
                candidate_als_rows = np.array(
                    [als_id_to_row.get(iid, -1) for iid in candidate_item_ids]
                )

                valid_als_mask = candidate_als_rows >= 0
                als_scores = np.zeros(len(candidate_rows), dtype=np.float32)

                if valid_als_mask.any():
                    valid_als_rows = candidate_als_rows[valid_als_mask]
                    valid_als_factors = als_factors[valid_als_rows]
                    als_scores[valid_als_mask] = valid_als_factors @ query_als_vec
            else:
                als_scores = np.zeros(len(candidate_rows), dtype=np.float32)
        else:
            als_scores = np.zeros(len(candidate_rows), dtype=np.float32)

        blended_scores = (1.0 - alpha) * subject_scores + alpha * als_scores

        if filter_candidates:
            metadata = self._get_book_meta()
            threshold = (
                min_rating_count if min_rating_count is not None else self.HYBRID_MIN_RATINGS
            )

            valid_mask = np.array(
                [
                    metadata.loc[subject_ids_list[row], "book_num_ratings"] >= threshold
                    if subject_ids_list[row] in metadata.index
                    else False
                    for row in candidate_rows
                ]
            )

            if not valid_mask.any():
                return []

            blended_scores = blended_scores[valid_mask]
            candidate_rows = [
                candidate_rows[i] for i in range(len(candidate_rows)) if valid_mask[i]
            ]

        top_k_indices = np.argsort(-blended_scores)[:k]
        top_scores = blended_scores[top_k_indices]
        top_item_ids = np.array([subject_ids_list[candidate_rows[i]] for i in top_k_indices])

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
