# models/cache/invalidation.py
"""
Cache invalidation utilities for ML model results.
Provides functions to clear caches on model reload or other trigger events.
"""

import logging
from typing import Optional

from app.cache import get_redis_client

logger = logging.getLogger(__name__)


async def clear_als_cache() -> int:
    """
    Clear all cache entries that depend on ALS factors.

    Should be called after every ALS training run. Subject-only entries
    (ml:sim:subject:* and ml:rec:subject:*) are intentionally preserved
    because subject embeddings are not updated during ALS training.

    Patterns cleared:
      - ml:sim:als:*      pure ALS similarity
      - ml:sim:hybrid:*   hybrid similarity (blends ALS + subject)
      - ml:rec:behavioral:*  ALS-driven recommendations
      - ml:rec:auto:*        auto-routed (warm users go through ALS)

    Returns:
        Total number of keys deleted across all patterns.
    """
    client = await get_redis_client()

    if not client.available:
        logger.warning("Cache unavailable, skipping ALS cache clear")
        return 0

    patterns = [
        "ml:sim:als:*",
        "ml:sim:hybrid:*",
        "ml:rec:behavioral:*",
        "ml:rec:auto:*",
    ]

    total = 0
    for pattern in patterns:
        deleted = await client.delete_pattern(pattern)
        logger.info("Cleared %d entries for pattern '%s'", deleted, pattern)
        total += deleted

    logger.info("ALS cache flush complete: %d total entries cleared", total)
    return total


async def clear_ml_cache() -> int:
    """
    Clear all ML-related cache entries.

    Nuclear option — clears both ALS and subject caches. Use clear_als_cache()
    when only ALS factors have been updated (the common training case).

    Returns:
        Number of keys deleted.
    """
    client = await get_redis_client()

    if not client.available:
        logger.warning("Cache unavailable, skipping ML cache clear")
        return 0

    deleted = await client.delete_pattern("ml:*")
    logger.info("Cleared %d ML cache entries", deleted)
    return deleted


async def clear_similarity_cache(mode: Optional[str] = None) -> int:
    """
    Clear similarity cache entries.

    Args:
        mode: If provided, only clear this mode ("subject", "als", "hybrid").
              If None, clear all similarity cache.

    Returns:
        Number of keys deleted.
    """
    client = await get_redis_client()

    if not client.available:
        logger.warning("Cache unavailable, skipping similarity cache clear")
        return 0

    pattern = f"ml:sim:{mode}:*" if mode else "ml:sim:*"
    deleted = await client.delete_pattern(pattern)
    logger.info("Cleared %d similarity cache entries (pattern: %s)", deleted, pattern)
    return deleted


async def clear_recommendation_cache(
    mode: Optional[str] = None,
    user_id: Optional[int] = None,
) -> int:
    """
    Clear recommendation cache entries.

    Args:
        mode: If provided, only clear this mode ("behavioral", "subject", "auto").
              If None, clear all recommendation cache.
        user_id: If provided, only clear cache for this specific user.
                 Only meaningful for "behavioral" and "auto" modes; ignored for "subject".

    Returns:
        Number of keys deleted.
    """
    client = await get_redis_client()

    if not client.available:
        logger.warning("Cache unavailable, skipping recommendation cache clear")
        return 0

    if mode and user_id is not None and mode in ("behavioral", "auto"):
        pattern = f"ml:rec:{mode}:{user_id}:*"
    elif mode:
        pattern = f"ml:rec:{mode}:*"
    else:
        pattern = "ml:rec:*"

    deleted = await client.delete_pattern(pattern)
    logger.info("Cleared %d recommendation cache entries (pattern: %s)", deleted, pattern)
    return deleted
