# models/cache/__init__.py
"""
ML-specific caching layer for recommendation and similarity endpoints.
Provides decorators and cache key builders optimized for ML model results.
"""

from models.cache.decorators import cached_recommendations, cached_similarity
from models.cache.invalidation import clear_ml_cache, clear_recommendation_cache
from models.cache.keys import hash_subjects, recommendation_key, similarity_key

__all__ = [
    "cached_similarity",
    "cached_recommendations",
    "clear_ml_cache",
    "clear_recommendation_cache",
    "similarity_key",
    "recommendation_key",
    "hash_subjects",
]
