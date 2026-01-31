# app/cache/decorators.py
"""
Generic caching decorator with automatic serialization and graceful degradation.
Provides @cached decorator that works with any function returning serializable data.
"""

from functools import wraps
from typing import Callable, Optional, Any
import logging
import time

from app.cache.client import get_redis_client
from app.cache.serializers import serialize, deserialize

logger = logging.getLogger(__name__)


def cached(
    key_func: Callable[..., str],
    ttl: Optional[int] = None,
    log_hits: bool = True,
):
    """
    Generic caching decorator with automatic serialization.

    Caches function results in Redis with automatic JSON serialization.
    Falls through to actual function if cache unavailable or on error.

    Args:
        key_func: Function that takes same args as wrapped function and returns cache key
        ttl: Time to live in seconds (None = no expiration)
        log_hits: Whether to log cache hits/misses

    Example:
        @cached(key_func=lambda x, y: f"sum:{x}:{y}", ttl=3600)
        def add(x: int, y: int) -> int:
            return x + y
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            client = get_redis_client()

            try:
                cache_key = key_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Cache key generation failed for {func.__name__}: {e}")
                return func(*args, **kwargs)

            if client.available:
                cached_value = client.get(cache_key)

                if cached_value is not None:
                    result = deserialize(cached_value)
                    if result is not None:
                        if log_hits:
                            logger.info(f"Cache HIT: {cache_key}")
                        return result

            if log_hits:
                logger.info(f"Cache MISS: {cache_key}")

            start_time = time.time()
            result = func(*args, **kwargs)
            compute_ms = int((time.time() - start_time) * 1000)

            if client.available:
                serialized = serialize(result)
                if serialized is not None:
                    success = client.set(cache_key, serialized, ttl)
                    if success and log_hits:
                        logger.info(
                            f"Cache SET: {cache_key} (computed in {compute_ms}ms, ttl={ttl}s)"
                        )

            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            client = get_redis_client()

            try:
                cache_key = key_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Cache key generation failed for {func.__name__}: {e}")
                return await func(*args, **kwargs)

            if client.available:
                cached_value = client.get(cache_key)

                if cached_value is not None:
                    result = deserialize(cached_value)
                    if result is not None:
                        if log_hits:
                            logger.info(f"Cache HIT: {cache_key}")
                        return result

            if log_hits:
                logger.info(f"Cache MISS: {cache_key}")

            start_time = time.time()
            result = await func(*args, **kwargs)
            compute_ms = int((time.time() - start_time) * 1000)

            if client.available:
                serialized = serialize(result)
                if serialized is not None:
                    success = client.set(cache_key, serialized, ttl)
                    if success and log_hits:
                        logger.info(
                            f"Cache SET: {cache_key} (computed in {compute_ms}ms, ttl={ttl}s)"
                        )

            return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
