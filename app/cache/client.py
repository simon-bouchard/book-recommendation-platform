# app/cache/client.py
"""
Redis connection management with connection pooling and error handling.
Provides singleton access to Redis client with graceful degradation.
"""

import redis
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class CacheClient:
    """
    Redis cache client with connection pooling and graceful degradation.

    Provides simple interface for cache operations with automatic error handling.
    Falls back gracefully if Redis is unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        max_connections: int = 50,
    ):
        """
        Initialize Redis cache client with connection pool.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password if required
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            max_connections: Maximum connections in pool
        """
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            max_connections=max_connections,
            decode_responses=True,
        )

        self.client = redis.Redis(connection_pool=self.pool)
        self._available = self._check_connection()

        if self._available:
            logger.info(f"Redis cache connected: {host}:{port}/{db}")
        else:
            logger.warning(f"Redis cache unavailable: {host}:{port}/{db} - running without cache")

    def _check_connection(self) -> bool:
        """
        Check if Redis connection is available.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis connection check failed: {e}")
            return False

    @property
    def available(self) -> bool:
        """Check if cache is currently available."""
        return self._available

    def get(self, key: str) -> Optional[str]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or error
        """
        if not self._available:
            return None

        try:
            return self.client.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self._available:
            return False

        try:
            if ttl:
                return self.client.setex(key, ttl, value)
            else:
                return self.client.set(key, value)
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if successful, False otherwise
        """
        if not self._available:
            return False

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "ml:*")

        Returns:
            Number of keys deleted, or 0 on error
        """
        if not self._available:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache delete_pattern failed for pattern {pattern}: {e}")
            return 0

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self._available:
            return False

        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.warning(f"Cache exists check failed for key {key}: {e}")
            return False

    def close(self):
        """Close Redis connection pool."""
        try:
            self.pool.disconnect()
            logger.info("Redis connection pool closed")
        except Exception as e:
            logger.warning(f"Error closing Redis connection pool: {e}")


_cache_client: Optional[CacheClient] = None


def get_redis_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None,
    password: Optional[str] = None,
) -> CacheClient:
    """
    Get or create singleton Redis cache client.

    Args:
        host: Redis host (default: from env or localhost)
        port: Redis port (default: from env or 6379)
        db: Redis database number (default: from env or 0)
        password: Redis password (default: from env or None)

    Returns:
        CacheClient instance
    """
    global _cache_client

    if _cache_client is None:
        host = host or os.getenv("REDIS_HOST", "localhost")
        port = port or int(os.getenv("REDIS_PORT", "6379"))
        db = db or int(os.getenv("REDIS_DB", "0"))
        password = password or os.getenv("REDIS_PASSWORD")

        _cache_client = CacheClient(
            host=host,
            port=port,
            db=db,
            password=password,
        )

    return _cache_client


def reset_cache_client():
    """
    Reset singleton cache client.

    Used for testing or when reconnection is needed.
    """
    global _cache_client

    if _cache_client:
        _cache_client.close()
        _cache_client = None
