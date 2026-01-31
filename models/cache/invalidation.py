# models/cache/invalidation.py
"""
Cache invalidation utilities for ML model results.
Provides functions to clear caches on model reload or other trigger events.
"""

import logging
from typing import Optional

from app.cache import get_redis_client

logger = logging.getLogger(__name__)


def clear_ml_cache() -> int:
    """
    Clear all ML-related cache entries.

    This should be called when models are reloaded, as all cached predictions
    become stale with new model artifacts.

    Returns:
        Number of keys deleted
    """
    client = get_redis_client()

    if not client.available:
        logger.warning("Cache unavailable, skipping ML cache clear")
        return 0

    deleted = client.delete_pattern("ml:*")
    logger.info(f"Cleared {deleted} ML cache entries")
    return deleted


def clear_similarity_cache(mode: Optional[str] = None) -> int:
    """
    Clear similarity cache entries.

    Args:
        mode: If provided, only clear this mode ("subject", "als", "hybrid")
              If None, clear all similarity cache

    Returns:
        Number of keys deleted
    """
    client = get_redis_client()

    if not client.available:
        logger.warning("Cache unavailable, skipping similarity cache clear")
        return 0

    if mode:
        pattern = f"ml:sim:{mode}:*"
    else:
        pattern = "ml:sim:*"

    deleted = client.delete_pattern(pattern)
    logger.info(f"Cleared {deleted} similarity cache entries (pattern: {pattern})")
    return deleted


def clear_recommendation_cache(mode: Optional[str] = None, user_id: Optional[int] = None) -> int:
    """
    Clear recommendation cache entries.

    Args:
        mode: If provided, only clear this mode ("behavioral", "subject", "auto")
              If None, clear all recommendation cache
        user_id: If provided, only clear cache for this user (behavioral and auto modes)
                 Ignored for subject mode

    Returns:
        Number of keys deleted
    """
    client = get_redis_client()

    if not client.available:
        logger.warning("Cache unavailable, skipping recommendation cache clear")
        return 0

    if mode and user_id is not None:
        if mode in ("behavioral", "auto"):
            pattern = f"ml:rec:{mode}:{user_id}:*"
        else:
            pattern = f"ml:rec:{mode}:*"
    elif mode:
        pattern = f"ml:rec:{mode}:*"
    else:
        pattern = "ml:rec:*"

    deleted = client.delete_pattern(pattern)
    logger.info(f"Cleared {deleted} recommendation cache entries (pattern: {pattern})")
    return deleted
