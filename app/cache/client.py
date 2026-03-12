# app/cache/client.py
"""
Redis connection management with connection pooling and graceful degradation.
Provides singleton access to a Redis client, reading connection details from
REDIS_URL (preferred) or the individual REDIS_HOST / REDIS_PORT / REDIS_DB /
REDIS_PASSWORD env vars.
"""

import logging
import os
from typing import Optional
from urllib.parse import urlparse

import redis

logger = logging.getLogger(__name__)


def _parse_redis_url(url: str) -> dict:
    """
    Parse a redis:// URL into connection kwargs.

    Returns a dict with host, port, db, and password keys, using sensible
    defaults for any component that is absent from the URL.
    """
    parsed = urlparse(url)
    db_str = parsed.path.lstrip("/")
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 6379,
        "db": int(db_str) if db_str.isdigit() else 0,
        "password": parsed.password or None,
    }


class CacheClient:
    """
    Redis cache client with connection pooling and graceful degradation.

    Falls back gracefully when Redis is unreachable — all operations become
    no-ops and available returns False, so callers can skip cache logic
    without additional error handling.
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
            logger.info("Redis cache connected: %s:%s/%s", host, port, db)
        else:
            logger.warning(
                "Redis cache unavailable: %s:%s/%s - running without cache", host, port, db
            )

    def _check_connection(self) -> bool:
        """Ping Redis to confirm the connection is live."""
        try:
            self.client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning("Redis connection check failed: %s", exc)
            return False

    @property
    def available(self) -> bool:
        """True if Redis is reachable and operations will be attempted."""
        return self._available

    def get(self, key: str) -> Optional[str]:
        """Return the cached value for key, or None on miss or error."""
        if not self._available:
            return None
        try:
            return self.client.get(key)
        except Exception as exc:
            logger.warning("Cache get failed for key %s: %s", key, exc)
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Store value under key with an optional TTL in seconds."""
        if not self._available:
            return False
        try:
            if ttl:
                return bool(self.client.setex(key, ttl, value))
            return bool(self.client.set(key, value))
        except Exception as exc:
            logger.warning("Cache set failed for key %s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """Delete a single key. Returns True on success."""
        if not self._available:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception as exc:
            logger.warning("Cache delete failed for key %s: %s", key, exc)
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern. Returns the count deleted."""
        if not self._available:
            return 0
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as exc:
            logger.warning("Cache delete_pattern failed for pattern %s: %s", pattern, exc)
            return 0

    def exists(self, key: str) -> bool:
        """Return True if key exists in the cache."""
        if not self._available:
            return False
        try:
            return bool(self.client.exists(key))
        except Exception as exc:
            logger.warning("Cache exists check failed for key %s: %s", key, exc)
            return False

    def close(self) -> None:
        """Disconnect the connection pool."""
        try:
            self.pool.disconnect()
            logger.info("Redis connection pool closed")
        except Exception as exc:
            logger.warning("Error closing Redis connection pool: %s", exc)


_cache_client: Optional[CacheClient] = None


def get_redis_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None,
    password: Optional[str] = None,
) -> CacheClient:
    """
    Return the singleton CacheClient, creating it on first call.

    Connection resolution order:
      1. REDIS_URL env var  (e.g. redis://localhost:6379/0)
      2. REDIS_HOST / REDIS_PORT / REDIS_DB / REDIS_PASSWORD env vars
      3. Arguments passed directly to this function
      4. Hardcoded defaults (localhost:6379/0)

    To disable caching without code changes, point REDIS_URL at a port
    where nothing is listening:
        REDIS_URL=redis://localhost:9999 uvicorn ...
    """
    global _cache_client

    if _cache_client is None:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            conn = _parse_redis_url(redis_url)
        else:
            conn = {
                "host": host or os.getenv("REDIS_HOST", "localhost"),
                "port": port or int(os.getenv("REDIS_PORT", "6379")),
                "db": db or int(os.getenv("REDIS_DB", "0")),
                "password": password or os.getenv("REDIS_PASSWORD"),
            }

        _cache_client = CacheClient(**conn)

    return _cache_client


def reset_cache_client() -> None:
    """
    Tear down the singleton and release its connection pool.

    Intended for testing and for cases where the Redis configuration
    needs to change at runtime without a process restart.
    """
    global _cache_client

    if _cache_client:
        _cache_client.close()
        _cache_client = None
