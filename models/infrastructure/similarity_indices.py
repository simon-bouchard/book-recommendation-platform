# models/infrastructure/similarity_indices.py
"""
Shared FAISS index registry for optimal memory usage and performance.
All similarity operations use the same underlying indices to avoid duplicate index building.
"""

from typing import Optional

from models.infrastructure.similarity_index import SimilarityIndex
from models.data.loaders import (
    load_book_subject_embeddings,
    load_als_factors,
    load_book_meta,
)


_subject_index: Optional[SimilarityIndex] = None
_als_index: Optional[SimilarityIndex] = None


def get_subject_similarity_index() -> SimilarityIndex:
    """
    Get or create the shared subject-based similarity index.

    Used by both SubjectBasedGenerator and SimilarityService.
    Built once, reused everywhere for memory efficiency.

    Returns:
        Shared SimilarityIndex for subject-based similarity
    """
    global _subject_index

    if _subject_index is None:
        embeddings, ids = load_book_subject_embeddings(normalized=True, use_cache=True)
        _subject_index = SimilarityIndex(
            embeddings=embeddings,
            ids=ids,
            normalize=False,
        )

    return _subject_index


def get_als_similarity_index() -> SimilarityIndex:
    """
    Get or create the shared ALS-based similarity index with filtering.

    Filters out books with fewer than 10 ratings for quality control.

    Returns:
        Shared SimilarityIndex for ALS-based similarity with rating filter
    """
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


def reset_indices():
    """
    Clear all cached indices.

    Used for testing or when reloading models after training.
    """
    global _subject_index, _als_index
    _subject_index = None
    _als_index = None


def preload_indices():
    """
    Preload all FAISS indices at application startup.

    Call this from FastAPI startup event to eliminate cold start latency
    on first request. Builds indices synchronously during startup.
    """
    get_subject_similarity_index()
    get_als_similarity_index()
