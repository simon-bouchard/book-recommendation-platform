# app/cache/decorators.py
"""
Generic caching decorator with automatic serialization and graceful degradation.
Provides @cached decorator that works with any function returning serializable data.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Optional

from app.cache.client import get_redis_client
from app.cache.serializers import deserialize, serialize

logger = logging.getLogger(__name__)


def cached(
    key_func: Callable[..., str],
    ttl: Optional[int] = None,
    log_hits: bool = True,
):
    """
    Generic caching decorator with automatic serialization.

    Caches function results in Redis with automatic JSON serialization.
    Falls through to the actual function if the cache is unavailable or on error.
    Supports both sync and async wrapped functions; sync functions are wrapped
    in an async wrapper since the underlying cache client is async.

    Args:
        key_func: Function that takes the same args as the wrapped function
                  and returns a string cache key.
        ttl: Time to live in seconds (None = no expiration).
        log_hits: Whether to log cache hits and misses.

    Example:
        @cached(key_func=lambda x, y: f"sum:{x}:{y}", ttl=3600)
        async def add(x: int, y: int) -> int:
            return x + y
    """

    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            client = await get_redis_client()

            try:
                cache_key = key_func(*args, **kwargs)
            except Exception as exc:
                logger.warning("Cache key generation failed for %s: %s", func.__name__, exc)
                return await func(*args, **kwargs) if is_async else func(*args, **kwargs)

            if client.available:
                cached_value = await client.get(cache_key)

                if cached_value is not None:
                    result = deserialize(cached_value)
                    if result is not None:
                        if log_hits:
                            logger.info("Cache HIT: %s", cache_key)
                        return result

            if log_hits:
                logger.info("Cache MISS: %s", cache_key)

            start_time = time.time()
            result = await func(*args, **kwargs) if is_async else func(*args, **kwargs)
            compute_ms = int((time.time() - start_time) * 1000)

            if client.available:
                serialized = serialize(result)
                if serialized is not None:

                    async def _write_cache():
                        success = await client.set(cache_key, serialized, ttl)
                        if success and log_hits:
                            logger.info(
                                "Cache SET: %s (computed in %dms, ttl=%ss)",
                                cache_key,
                                compute_ms,
                                ttl,
                            )

                    asyncio.create_task(_write_cache())

            return result

        return wrapper

    return decorator
