# ops/training/flush_cache.py
"""
Flushes stale Redis cache entries after a successful model promotion.

Intended to be called as a subprocess step in the automated training pipeline,
immediately after model server containers have been signalled to reload.
Exits with code 1 if the cache client is unavailable, so the pipeline can
decide whether to treat that as a hard failure.
"""

import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.cache import get_redis_client
from models.cache.invalidation import clear_als_cache, clear_popularity_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Flush all stale cache entries and report the result."""
    logger.info("Starting cache flush...")

    client = await get_redis_client()
    if not client.available:
        logger.error("Redis is unavailable — cannot flush cache.")
        sys.exit(1)

    deleted = await clear_als_cache()
    deleted += await clear_popularity_cache()

    if deleted == 0:
        logger.warning("No cache entries were deleted — cache may already be empty.")
    else:
        logger.info("Cache flush complete: %d entries removed.", deleted)


if __name__ == "__main__":
    asyncio.run(main())
