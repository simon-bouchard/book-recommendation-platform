# models/data/queries.py
"""
Database queries for the recommendation pipeline.

get_read_books_for_candidates_async is the hot-path variant used by
ReadBooksFilter: it runs natively on the event loop via aiomysql, eliminating
the asyncio.to_thread dispatch that was the dominant latency contributor in
the recommendation endpoint.
"""

from typing import List, Set

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from app.table_models import Interaction
except ImportError:
    Interaction = None


def get_read_books(user_id: int, db: Session) -> Set[int]:
    """
    Get the full set of item_idx that a user has already interacted with.

    Args:
        user_id: User ID to query
        db: SQLAlchemy database session

    Returns:
        Set of item indices the user has read or rated
    """
    if Interaction is None:
        return set()

    return {
        row.item_idx
        for row in db.query(Interaction.item_idx).filter(Interaction.user_id == user_id).all()
    }


def get_read_books_for_candidates(user_id: int, candidate_ids: List[int], db: Session) -> Set[int]:
    """
    Synchronous variant — retained for non-async callers and tests.

    Get the subset of candidate_ids that the user has already read or rated.
    Scoped to the candidate list via an IN clause. With the composite index on
    (user_id, item_idx), MySQL performs a targeted seek per candidate ID rather
    than scanning the full interaction history. The index is covering, so no
    table row access is required.

    Prefer get_read_books_for_candidates_async in async request paths.

    Args:
        user_id: User ID to query
        candidate_ids: Candidate item indices to check membership for
        db: Synchronous SQLAlchemy session

    Returns:
        Subset of candidate_ids the user has already interacted with
    """
    if Interaction is None or not candidate_ids:
        return set()

    return {
        row.item_idx
        for row in db.query(Interaction.item_idx)
        .filter(Interaction.user_id == user_id, Interaction.item_idx.in_(candidate_ids))
        .all()
    }


async def get_read_books_for_candidates_async(
    user_id: int, candidate_ids: List[int]
) -> Set[int]:
    """
    Async variant used by ReadBooksFilter in the recommendation hot path.

    Uses a raw aiomysql connection from the module-level pool, bypassing
    SQLAlchemy's ORM and expanding-bind-parameter overhead. The composite
    (user_id, item_idx) covering index is used by MySQL for this query.

    Args:
        user_id: User ID to query
        candidate_ids: Candidate item indices to check membership for

    Returns:
        Subset of candidate_ids the user has already interacted with
    """
    if not candidate_ids:
        return set()

    from app.database import get_aiomysql_pool

    placeholders = ",".join(["%s"] * len(candidate_ids))
    sql = (
        f"SELECT item_idx FROM interactions "
        f"WHERE user_id = %s AND item_idx IN ({placeholders})"
    )

    pool = get_aiomysql_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, [user_id] + list(candidate_ids))
            rows = await cur.fetchall()

    return {row[0] for row in rows}


def get_user_num_ratings(user_id: int) -> int:
    """
    Get number of ratings for a user from cached metadata.

    Args:
        user_id: User ID to query

    Returns:
        Number of ratings, or 0 if user not found
    """
    try:
        from models.data.loaders import load_user_meta

        meta = load_user_meta()

        if user_id in meta.index:
            val = meta.loc[user_id].get("user_num_ratings")
            return int(val) if val is not None else 0
    except Exception:
        pass

    return 0
