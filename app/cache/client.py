# app/cache/client.py
"""
Redis connection management with async connection pooling and graceful degradation.
Provides singleton access to an async Redis client, reading connection details from
REDIS_URL (preferred) or the individual REDIS_HOST / REDIS_PORT / REDIS_DB /
REDIS_PASSWORD env vars.
"""

import asyncio
import logging
import os
from typing import Optional
from urllib.parse import urlparse

import redis.asyncio as redis

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
    Async Redis cache client with connection pooling and graceful degradation.

    Must be instantiated via the async factory method CacheClient.create() rather
    than the constructor directly, as the startup connectivity check is async.

    Falls back gracefully when Redis is unreachable — all operations become
    no-ops and available returns False, so callers can skip cache logic
    without additional error handling.
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        password: Optional[str],
        socket_timeout: int,
        socket_connect_timeout: int,
        max_connections: int,
    ):
        self._host = host
        self._port = port
        self._db = db

        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            max_connections=max_connections,
            decode_responses=False,
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self._available = False

    @classmethod
    async def create(
        cls,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        max_connections: int = 50,
    ) -> "CacheClient":
        """
        Async factory that constructs a CacheClient and confirms connectivity.

        Use this instead of instantiating CacheClient directly. The startup
        ping is awaited here so availability is known before the first request.
        """
        instance = cls(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            max_connections=max_connections,
        )
        instance._available = await instance._check_connection()

        if instance._available:
            logger.info("Redis cache connected: %s:%s/%s", host, port, db)
        else:
            logger.warning(
                "Redis cache unavailable: %s:%s/%s - running without cache", host, port, db
            )

        return instance

    async def _check_connection(self) -> bool:
        """Ping Redis to confirm the connection is live."""
        try:
            await self.client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning("Redis connection check failed: %s", exc)
            return False

    @property
    def available(self) -> bool:
        """True if Redis is reachable and operations will be attempted."""
        return self._available

    async def get(self, key: str) -> Optional[str]:
        """Return the cached value for key, or None on miss or error."""
        if not self._available:
            return None
        try:
            return await self.client.get(key)
        except Exception as exc:
            logger.warning("Cache get failed for key %s: %s", key, exc)
            return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Store value under key with an optional TTL in seconds."""
        if not self._available:
            return False
        try:
            if ttl:
                return bool(await self.client.setex(key, ttl, value))
            return bool(await self.client.set(key, value))
        except Exception as exc:
            logger.warning("Cache set failed for key %s: %s", key, exc)
            return False

    async def delete(self, key: str) -> bool:
        """Delete a single key. Returns True on success."""
        if not self._available:
            return False
        try:
            await self.client.delete(key)
            return True
        except Exception as exc:
            logger.warning("Cache delete failed for key %s: %s", key, exc)
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern. Returns the count deleted.

        Uses SCAN internally to avoid blocking the server on large keyspaces.
        """
        if not self._available:
            return 0
        try:
            keys = [key async for key in self.client.scan_iter(pattern)]
            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as exc:
            logger.warning("Cache delete_pattern failed for pattern %s: %s", pattern, exc)
            return 0

    async def exists(self, key: str) -> bool:
        """Return True if key exists in the cache."""
        if not self._available:
            return False
        try:
            return bool(await self.client.exists(key))
        except Exception as exc:
            logger.warning("Cache exists check failed for key %s: %s", key, exc)
            return False

    async def aclose(self) -> None:
        """Disconnect the connection pool."""
        try:
            await self.client.aclose()
            logger.info("Redis connection pool closed")
        except Exception as exc:
            logger.warning("Error closing Redis connection pool: %s", exc)


_cache_client: Optional[CacheClient] = None
_cache_client_lock: asyncio.Lock = asyncio.Lock()


async def get_redis_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None,
    password: Optional[str] = None,
) -> CacheClient:
    """
    Return the singleton CacheClient, creating it on first call.

    Thread-safe via asyncio.Lock — concurrent coroutines racing on first call
    will wait rather than creating multiple clients.

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

    if _cache_client is not None:
        return _cache_client

    async with _cache_client_lock:
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

            _cache_client = await CacheClient.create(**conn)

    return _cache_client


async def reset_cache_client() -> None:
    """
    Tear down the singleton and release its connection pool.

    Intended for testing and for cases where the Redis configuration
    needs to change at runtime without a process restart.
    """
    global _cache_client

    if _cache_client:
        await _cache_client.aclose()
        _cache_client = None
