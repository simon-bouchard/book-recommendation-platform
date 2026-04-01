# models/infrastructure/similarity_index.py
"""
FAISS-based similarity index with two-pool filtering system.

Supports querying any item while only returning high-quality candidates.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import faiss


class SimilarityIndex:
    """
    FAISS similarity index with optional candidate filtering.

    Implements two-pool system:
    - Full embeddings: ANY item can be queried (even low-quality)
    - Filtered candidates: Only high-quality items in results

    This allows users to ask "what's similar to this obscure book?"
    while ensuring results are only from trusted, high-quality books.

    Example:
        # Create index with rating count filter
        index = SimilarityIndex.create_filtered_index(
            embeddings=book_embeddings,
            ids=book_ids,
            metadata=book_meta,
            min_rating_count=10,
            normalize=True
        )

        # Query any book (even if it has <10 ratings)
        scores, item_ids = index.search(query_item_id=12345, k=20)
        # Results will only be books with 10+ ratings
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        ids: list[int],
        normalize: bool = True,
        candidate_mask: Optional[np.ndarray] = None,
        use_hnsw: bool = False,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 200,
    ):
        """
        Initialize similarity index with optional candidate filtering.

        Args:
            embeddings: NxD array of ALL embeddings
            ids: List of ALL item IDs (length N)
            normalize: If True, L2 normalize embeddings for cosine similarity
            candidate_mask: Boolean array (length N) - True = valid candidate.
                           If None, all items are candidates.
            use_hnsw: If True, build an IndexHNSWFlat instead of IndexFlatIP.
                      HNSW is approximate but much faster for large catalogs.
                      Recommended only for normalized embeddings where inner
                      product equals cosine similarity.
            hnsw_m: HNSW graph connectivity (edges per node). Higher = better
                    recall and more memory. Typical range: 16-64.
            hnsw_ef_construction: Build-time search depth. Higher = better
                    graph quality but slower startup. Typical range: 100-400.
            hnsw_ef_search: Query-time search depth. Higher = better recall
                    but slower search. Should be >= k. Typical range: 100-500.

        Raises:
            ValueError: If embeddings and ids have mismatched lengths
        """
        if len(embeddings) != len(ids):
            raise ValueError(
                f"Embeddings and IDs length mismatch: "
                f"{len(embeddings)} embeddings but {len(ids)} IDs"
            )

        # Store full data for queries
        self.ids_full = np.array(ids, dtype=np.int64)
        self.id_to_row_full = {item_id: row for row, item_id in enumerate(ids)}

        # Prepare embeddings (normalize if requested)
        embs = embeddings.astype(np.float32, copy=False)
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embs = embs / norms

        self.embeddings_full = embs  # Keep full matrix for queries

        # Build candidate pool
        if candidate_mask is not None:
            # Filtered candidates only
            self.candidate_rows = np.flatnonzero(candidate_mask)
            self.candidate_ids = self.ids_full[candidate_mask]
            candidate_embs = embs[candidate_mask]
        else:
            # No filtering - all items are candidates
            self.candidate_rows = np.arange(len(ids), dtype=np.int64)
            self.candidate_ids = self.ids_full
            candidate_embs = embs

        # Build FAISS index over CANDIDATES only
        dim = embs.shape[1]
        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = hnsw_ef_construction
            self.index.hnsw.efSearch = hnsw_ef_search
        else:
            self.index = faiss.IndexFlatIP(dim)
        if candidate_embs.shape[0] > 0:
            self.index.add(candidate_embs)

    def search(
        self, query_item_id: int, k: int, exclude_query: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar items to query_item_id.

        Query can be ANY item (even if filtered from candidates).
        Results are only from the candidate pool.

        Args:
            query_item_id: Item to find similar items for
            k: Number of neighbors to return
            exclude_query: If True, remove query from results (if present)

        Returns:
            Tuple of (scores, candidate_item_ids)
            - scores: Array of similarity scores (length <= k)
            - candidate_item_ids: Array of item IDs (length <= k)

            Returns empty arrays if query_item_id not found or no candidates.
        """
        # Check if query exists in full matrix
        if query_item_id not in self.id_to_row_full:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        # Check if candidate pool is empty
        if self.index.ntotal == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        # Get query vector from FULL matrix (allows querying any item)
        query_row = self.id_to_row_full[query_item_id]
        query_vec = self.embeddings_full[query_row].reshape(1, -1)

        # Search in candidate pool only
        search_k = k + 1 if exclude_query else k
        search_k = min(search_k, max(1, self.index.ntotal))

        distances, cand_indices = self.index.search(query_vec, search_k)

        # Map candidate indices back to item IDs
        item_ids = self.candidate_ids[cand_indices[0]]
        scores = distances[0]

        # Filter out query if requested
        if exclude_query:
            mask = item_ids != query_item_id
            item_ids = item_ids[mask][:k]
            scores = scores[mask][:k]

        return scores, item_ids

    def has_item(self, item_id: int) -> bool:
        """
        Check if item exists in the full index (can be queried).

        Args:
            item_id: Item to check

        Returns:
            True if item can be queried
        """
        return int(item_id) in self.id_to_row_full

    def is_candidate(self, item_id: int) -> bool:
        """
        Check if item is in the candidate pool (can appear in results).

        Args:
            item_id: Item to check

        Returns:
            True if item can appear in search results
        """
        return int(item_id) in set(self.candidate_ids)

    @property
    def num_total(self) -> int:
        """Total number of items that can be queried."""
        return len(self.ids_full)

    @property
    def num_candidates(self) -> int:
        """Number of items that can appear in results."""
        return len(self.candidate_ids)

    @staticmethod
    def create_filtered_index(
        embeddings: np.ndarray,
        ids: list[int],
        metadata: pd.DataFrame,
        min_rating_count: int = 0,
        normalize: bool = True,
    ) -> "SimilarityIndex":
        """
        Create similarity index with rating count filtering.

        Convenience factory for creating indices with quality thresholds.

        Args:
            embeddings: Book embeddings (NxD array)
            ids: Book IDs (length N)
            metadata: DataFrame indexed by item_idx with "book_num_ratings" column
            min_rating_count: Minimum ratings to be a candidate (0 = no filter)
            normalize: L2 normalize embeddings

        Returns:
            SimilarityIndex with filtered candidates

        Example:
            # ALS similarity with 10+ rating threshold
            als_index = SimilarityIndex.create_filtered_index(
                embeddings=als_factors,
                ids=als_ids,
                metadata=book_meta,
                min_rating_count=10
            )
        """
        if min_rating_count <= 0:
            # No filtering - all items are candidates
            return SimilarityIndex(embeddings, ids, normalize=normalize)

        # Build filter mask based on rating counts
        # Use reindex to handle missing items gracefully (fills with 0)
        rating_counts = metadata.reindex(ids)["book_num_ratings"].fillna(0).astype(int)
        candidate_mask = rating_counts.values >= min_rating_count

        return SimilarityIndex(embeddings, ids, normalize=normalize, candidate_mask=candidate_mask)
