# models/cache/decorators.py
"""
ML-specific caching decorators for similarity and recommendation endpoints.
Wraps generic @cached decorator with ML-optimized key functions and TTLs.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable

from app.cache import cached, get_redis_client
from app.cache.serializers import deserialize, serialize
from models.cache.keys import recommendation_key, similarity_key

logger = logging.getLogger(__name__)


SIMILARITY_TTL = 86400
RECOMMENDATION_TTL = 3600


def cached_similarity(func: Callable) -> Callable:
    """
    Cache decorator for book similarity endpoints.

    Caches results for 24 hours. Similarity results are static between model reloads.

    Expected function signature:
        get_similar(item_idx: int, mode: str, alpha: float, top_k: int, **kwargs)

    Cache key includes: item_idx, mode, top_k, alpha (for hybrid mode only).
    """

    def key_func(item_idx: int, mode: str, alpha: float, top_k: int, **kwargs) -> str:
        return similarity_key(item_idx, mode, top_k, alpha)

    return cached(key_func=key_func, ttl=SIMILARITY_TTL, log_hits=True)(func)


async def _get_subject_idxs_for_user(user: str, _id: bool) -> list[int]:
    """
    Fetch favorite subject indices for a user via raw aiomysql.

    Used by cached_recommendations to build a content-addressable cache key
    for subject-mode requests so that two users with identical subject
    preferences share the same cached result.
    """
    from app.database import get_aiomysql_pool

    pool = get_aiomysql_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            if _id:
                await cur.execute(
                    "SELECT subject_idx FROM user_fav_subjects WHERE user_id = %s",
                    (int(user),),
                )
            else:
                await cur.execute(
                    "SELECT subject_idx FROM user_fav_subjects"
                    " WHERE user_id = (SELECT user_id FROM users WHERE username = %s)",
                    (user,),
                )
            rows = await cur.fetchall()

    return [row[0] for row in rows]


def cached_recommendations(func: Callable) -> Callable:
    """
    Cache decorator for recommendation endpoints.

    Caches results for 1 hour. Key strategy varies by mode:
    - behavioral: user_id based (user-specific)
    - subject: subject_hash based (shareable across users with identical subjects)
    - auto: user_id based (warm/cold decision is user-specific)

    Expected function signature:
        recommend(user: str, _id: bool, top_n: int, mode: str, w: float, db: AsyncSession)
    """

    @wraps(func)
    async def wrapper(
        user: str, _id: bool, top_n: int, mode: str, w: float, **kwargs
    ):
        client = await get_redis_client()

        if not client.available:
            logger.info("Cache unavailable, falling through to computation")
            return await func(user, _id, top_n, mode, w, **kwargs)

        user_id = int(user) if _id else None
        subject_idxs = None

        if mode == "subject":
            subject_idxs = await _get_subject_idxs_for_user(user, _id)

        try:
            cache_key = recommendation_key(
                mode=mode,
                user_id=user_id,
                subject_idxs=subject_idxs,
                top_n=top_n,
                w=w,
            )
        except ValueError as exc:
            logger.warning("Cache key generation failed: %s", exc)
            return await func(user, _id, top_n, mode, w, **kwargs)

        cached_value = await client.get(cache_key)

        if cached_value is not None:
            result = deserialize(cached_value)
            if result is not None:
                logger.info("Cache HIT: %s", cache_key)
                return result

        logger.info("Cache MISS: %s", cache_key)

        start_time = time.time()
        result = await func(user, _id, top_n, mode, w, **kwargs)
        compute_ms = int((time.time() - start_time) * 1000)

        serialized = serialize(result)
        if serialized is not None:
            async def _write_cache():
                success = await client.set(cache_key, serialized, RECOMMENDATION_TTL)
                if success:
                    logger.info(
                        "Cache SET: %s (computed in %dms, ttl=%ss)",
                        cache_key,
                        compute_ms,
                        RECOMMENDATION_TTL,
                    )

            asyncio.create_task(_write_cache())

        return result

    return wrapper
