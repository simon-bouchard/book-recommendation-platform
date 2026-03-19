# models/domain/filters.py
"""
Filters with async apply() and module-level singleton instances.

ReadBooksFilter.apply() accepts an AsyncSession and calls
get_read_books_for_candidates_async directly on the event loop — no thread
pool dispatch, no synchronous socket I/O. This is the primary latency fix for
the recommendation endpoint: the previous asyncio.to_thread approach added
90-110ms of sync pymysql overhead on every request.

The IN-clause scoped query remains the same: with the composite index on
(user_id, item_idx), MySQL performs one targeted seek per candidate ID,
making the query cost bounded by the candidate count rather than the user's
full interaction history.
"""

from typing import List, Optional, Protocol, Set

from sqlalchemy.ext.asyncio import AsyncSession

from models.domain.recommendation import Candidate
from models.domain.user import User
from models.data.queries import get_read_books_for_candidates_async


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Filter(Protocol):
    """Protocol for candidate filters."""

    async def apply(
        self,
        candidates: List[Candidate],
        user: User,
        db: Optional[AsyncSession] = None,
    ) -> List[Candidate]: ...


# ---------------------------------------------------------------------------
# Concrete filters
# ---------------------------------------------------------------------------


class ReadBooksFilter:
    """
    Filter out books the user has already read or rated.

    Calls get_read_books_for_candidates_async directly — the query runs
    natively on the event loop via aiomysql with no thread pool involvement.
    The IN clause is scoped to the candidate set, making this a covering index
    query bounded by the number of candidates regardless of how many total
    interactions the user has.
    """

    async def apply(
        self,
        candidates: List[Candidate],
        user: User,
        db: Optional[AsyncSession] = None,
    ) -> List[Candidate]:
        if db is None:
            raise ValueError("ReadBooksFilter requires a database session")

        candidate_ids = [c.item_idx for c in candidates]
        read_item_ids: Set[int] = await get_read_books_for_candidates_async(
            user.user_id, candidate_ids, db
        )

        return [c for c in candidates if c.item_idx not in read_item_ids]


class MinRatingCountFilter:
    """Filter out books with insufficient ratings."""

    def __init__(self, min_count: int = 10):
        if min_count < 0:
            raise ValueError(f"min_count must be non-negative, got {min_count}")

        self.min_count = min_count

    async def apply(
        self,
        candidates: List[Candidate],
        user: User,
        db: Optional[AsyncSession] = None,
    ) -> List[Candidate]:
        if self.min_count == 0:
            return candidates

        from models.client.registry import get_metadata_client

        candidate_ids = [c.item_idx for c in candidates]
        meta = await get_metadata_client().enrich(candidate_ids)

        return [
            c for c in candidates
            if meta.get(c.item_idx, {}).get("num_ratings", 0) >= self.min_count
        ]


class FilterChain:
    """Compose multiple filters into a single filter."""

    def __init__(self, filters: List[Filter]):
        self.filters = filters

    async def apply(
        self,
        candidates: List[Candidate],
        user: User,
        db: Optional[AsyncSession] = None,
    ) -> List[Candidate]:
        result = candidates
        for f in self.filters:
            result = await f.apply(result, user, db)
        return result


class NoFilter:
    """Pass-through filter that doesn't remove any candidates."""

    async def apply(
        self,
        candidates: List[Candidate],
        user: User,
        db: Optional[AsyncSession] = None,
    ) -> List[Candidate]:
        return candidates


# ============================================================================
# MODULE-LEVEL SINGLETONS
# ============================================================================

_read_books_filter = None
_no_filter = None


def get_read_books_filter() -> ReadBooksFilter:
    """Get or create singleton ReadBooksFilter."""
    global _read_books_filter
    if _read_books_filter is None:
        _read_books_filter = ReadBooksFilter()
    return _read_books_filter


def get_no_filter() -> NoFilter:
    """Get or create singleton NoFilter."""
    global _no_filter
    if _no_filter is None:
        _no_filter = NoFilter()
    return _no_filter
