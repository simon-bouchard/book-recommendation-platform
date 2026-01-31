# app/cache/__init__.py
"""
Generic caching infrastructure for the application.
Provides Redis connection management and caching decorators.
"""

from app.cache.client import get_redis_client, CacheClient
from app.cache.decorators import cached

__all__ = [
    "get_redis_client",
    "CacheClient",
    "cached",
]
