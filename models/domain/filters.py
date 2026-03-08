# models/domain/filters.py
"""
Filters with async apply() and module-level singleton instances.

ReadBooksFilter.apply() is async: it wraps the synchronous SQLAlchemy query in
asyncio.to_thread so the event loop is never blocked while the DB call runs.

The IN-clause scoped query (get_read_books_for_candidates) is used rather than
the full history query: with the composite index on (user_id, item_idx), MySQL
performs one targeted seek per candidate ID, which is faster than scanning the
user's full interaction history for warm users with many interactions.
"""

import asyncio
from typing import List, Optional, Protocol, Set

from sqlalchemy.orm import Session

from models.domain.recommendation import Candidate
from models.domain.user import User
from models.data.queries import get_read_books_for_candidates
from models.data.loaders import load_book_meta


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Filter(Protocol):
    """Protocol for candidate filters."""

    async def apply(
        self,
        candidates: List[Candidate],
        user: User,
        db: Optional[Session] = None,
    ) -> List[Candidate]: ...


# ---------------------------------------------------------------------------
# Concrete filters
# ---------------------------------------------------------------------------


class ReadBooksFilter:
    """
    Filter out books the user has already read or rated.

    Uses get_read_books_for_candidates to scope the DB query to the candidate
    set via an IN clause.  With the composite index on (user_id, item_idx)
    this is a covering index query bounded by the number of candidates, making
    it fast regardless of how many total interactions the user has.

    The synchronous SQLAlchemy call is dispatched to asyncio.to_thread so
    it never blocks the event loop.
    """

    async def apply(
        self,
        candidates: List[Candidate],
        user: User,
        db: Optional[Session] = None,
    ) -> List[Candidate]:
        if db is None:
            raise ValueError("ReadBooksFilter requires a database session")

        candidate_ids = [c.item_idx for c in candidates]
        read_item_ids: Set[int] = await asyncio.to_thread(
            get_read_books_for_candidates, user.user_id, candidate_ids, db
        )

        return [c for c in candidates if c.item_idx not in read_item_ids]


class MinRatingCountFilter:
    """Filter out books with insufficient ratings."""

    def __init__(self, min_count: int = 10):
        if min_count < 0:
            raise ValueError(f"min_count must be non-negative, got {min_count}")

        self.min_count = min_count
        self._book_meta = None

    async def apply(
        self,
        candidates: List[Candidate],
        user: User,
        db: Optional[Session] = None,
    ) -> List[Candidate]:
        if self.min_count == 0:
            return candidates

        if self._book_meta is None:
            self._book_meta = load_book_meta(use_cache=True)

        filtered = []
        for candidate in candidates:
            if candidate.item_idx not in self._book_meta.index:
                continue

            rating_count = self._book_meta.loc[candidate.item_idx, "book_num_ratings"]
            if rating_count >= self.min_count:
                filtered.append(candidate)

        return filtered


class FilterChain:
    """Compose multiple filters into a single filter."""

    def __init__(self, filters: List[Filter]):
        self.filters = filters

    async def apply(
        self,
        candidates: List[Candidate],
        user: User,
        db: Optional[Session] = None,
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
        db: Optional[Session] = None,
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
