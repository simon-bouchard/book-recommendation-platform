# models/cache/decorators.py
"""
ML-specific caching decorators for similarity and recommendation endpoints.
Wraps generic @cached decorator with ML-optimized key functions and TTLs.
"""

from functools import wraps
from typing import Callable
import logging
import time

from app.cache import get_redis_client, cached
from app.cache.serializers import serialize, deserialize
from app.table_models import User as ORMUser, UserFavSubject
from models.cache.keys import similarity_key, recommendation_key

logger = logging.getLogger(__name__)


SIMILARITY_TTL = 86400
RECOMMENDATION_TTL = 3600


def cached_similarity(func: Callable) -> Callable:
    """
    Cache decorator for book similarity endpoints.

    Caches results for 24 hours. Similarity results are static between model reloads.

    Expected function signature:
        get_similar(item_idx: int, mode: str, k: int, alpha: float = 0.6, **kwargs)

    Cache key includes: item_idx, mode, k, alpha (for hybrid mode only)

    Example:
        @cached_similarity
        def get_similar_books(item_idx: int, mode: str, k: int, alpha: float):
            # ... computation
            return results
    """

    def key_func(item_idx: int, mode: str, alpha: float, top_k: int, **kwargs) -> str:
        return similarity_key(item_idx, mode, top_k, alpha)

    return cached(key_func=key_func, ttl=SIMILARITY_TTL, log_hits=True)(func)


def cached_recommendations(func: Callable) -> Callable:
    """
    Cache decorator for recommendation endpoints.

    Caches results for 1 hour. Key strategy varies by mode:
    - behavioral: user_id based (user-specific)
    - subject: subject_hash based (shareable across users)
    - auto: user_id based (decision is user-specific)

    Expected function signature:
        recommend(user: str, _id: bool, top_n: int, mode: str, w: float, db: Session)

    Cache key depends on mode - see recommendation_key() for details.

    Example:
        @cached_recommendations
        async def recommend_for_user(user: str, _id: bool, top_n: int, mode: str, w: float, db):
            # ... computation
            return results
    """

    @wraps(func)
    async def wrapper(user: str, _id: bool, top_n: int, mode: str, w: float, db, **kwargs):
        client = get_redis_client()

        if not client.available:
            logger.info("Cache unavailable, falling through to computation")
            return await func(user, _id, top_n, mode, w, db, **kwargs)

        user_id = int(user) if _id else None
        subject_idxs = None

        if mode == "subject":
            if _id:
                subject_idxs = [
                    row[0]
                    for row in db.query(UserFavSubject.subject_idx)
                    .filter(UserFavSubject.user_id == int(user))
                    .all()
                ]
            else:
                subquery = (
                    db.query(ORMUser.user_id).filter(ORMUser.username == user).scalar_subquery()
                )
                subject_idxs = [
                    row[0]
                    for row in db.query(UserFavSubject.subject_idx)
                    .filter(UserFavSubject.user_id == subquery)
                    .all()
                ]

        try:
            cache_key = recommendation_key(
                mode=mode,
                user_id=user_id,
                subject_idxs=subject_idxs,
                top_n=top_n,
                w=w,
            )
        except ValueError as e:
            logger.warning(f"Cache key generation failed: {e}")
            return await func(user, _id, top_n, mode, w, db, **kwargs)

        cached_value = client.get(cache_key)

        if cached_value is not None:
            result = deserialize(cached_value)
            if result is not None:
                logger.info(f"Cache HIT: {cache_key}")
                return result

        logger.info(f"Cache MISS: {cache_key}")

        start_time = time.time()
        result = await func(user, _id, top_n, mode, w, db, **kwargs)
        compute_ms = int((time.time() - start_time) * 1000)

        serialized = serialize(result)
        if serialized is not None:
            success = client.set(cache_key, serialized, RECOMMENDATION_TTL)
            if success:
                logger.info(
                    f"Cache SET: {cache_key} (computed in {compute_ms}ms, ttl={RECOMMENDATION_TTL}s)"
                )

        return result

    return wrapper
