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
from sqlalchemy.ext.asyncio import AsyncSession
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
    user_id: int, candidate_ids: List[int], db: AsyncSession
) -> Set[int]:
    """
    Async variant used by ReadBooksFilter in the recommendation hot path.

    Identical query semantics to get_read_books_for_candidates — scoped IN
    clause over the composite (user_id, item_idx) covering index — but runs
    natively on the event loop via aiomysql. No thread pool dispatch, no
    synchronous socket I/O blocking the event loop.

    Args:
        user_id: User ID to query
        candidate_ids: Candidate item indices to check membership for
        db: Async SQLAlchemy session

    Returns:
        Subset of candidate_ids the user has already interacted with
    """
    if Interaction is None or not candidate_ids:
        return set()

    stmt = select(Interaction.item_idx).where(
        Interaction.user_id == user_id,
        Interaction.item_idx.in_(candidate_ids),
    )
    result = await db.execute(stmt)
    return {row.item_idx for row in result}


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
