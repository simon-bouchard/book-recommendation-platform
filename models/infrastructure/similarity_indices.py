# models/infrastructure/similarity_indices.py
"""
Shared FAISS index registry for optimal memory usage and performance.
All similarity operations use the same underlying indices to avoid duplicate index building.
"""

from typing import Optional

from models.core.paths import PATHS
from models.infrastructure.similarity_index import SimilarityIndex


_subject_index: Optional[SimilarityIndex] = None
_als_index: Optional[SimilarityIndex] = None


def get_subject_similarity_index() -> SimilarityIndex:
    """
    Get or lazily load the shared subject-based similarity index.

    The index is pre-built by build_similarity_indices.py during training
    and stored as a versioned artifact. First call loads from disk; subsequent
    calls return the cached instance.

    Returns:
        Shared SimilarityIndex for subject-based similarity
    """
    global _subject_index

    if _subject_index is None:
        _subject_index = SimilarityIndex.load(PATHS.subject_similarity_index_dir)

    return _subject_index


def get_als_similarity_index() -> SimilarityIndex:
    """
    Get or lazily load the shared ALS-based similarity index.

    The index is pre-built by build_similarity_indices.py during training
    with a min_rating_count=10 candidate filter, and stored as a versioned
    artifact. First call loads from disk; subsequent calls return the cached
    instance.

    Returns:
        Shared SimilarityIndex for ALS-based similarity with rating filter
    """
    global _als_index

    if _als_index is None:
        _als_index = SimilarityIndex.load(PATHS.als_similarity_index_dir)

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
