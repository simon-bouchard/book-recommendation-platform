# models/domain/filters.py
"""
Filters with module-level singleton instances.
"""

from typing import Protocol, List
from sqlalchemy.orm import Session

from models.domain.recommendation import Candidate
from models.domain.user import User
from models.data.queries import get_read_books
from models.data.loaders import load_book_meta


class Filter(Protocol):
    """Protocol for candidate filters."""

    def apply(
        self, candidates: List[Candidate], user: User, db: Session = None
    ) -> List[Candidate]: ...


class ReadBooksFilter:
    """Filter out books the user has already read or rated."""

    def apply(self, candidates: List[Candidate], user: User, db: Session = None) -> List[Candidate]:
        if db is None:
            raise ValueError("ReadBooksFilter requires database session")

        read_item_ids = get_read_books(user.user_id, db)
        return [candidate for candidate in candidates if candidate.item_idx not in read_item_ids]


class MinRatingCountFilter:
    """Filter out books with insufficient ratings."""

    def __init__(self, min_count: int = 10):
        if min_count < 0:
            raise ValueError(f"min_count must be non-negative, got {min_count}")

        self.min_count = min_count
        self._book_meta = None

    def apply(self, candidates: List[Candidate], user: User, db: Session = None) -> List[Candidate]:
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

    def apply(self, candidates: List[Candidate], user: User, db: Session = None) -> List[Candidate]:
        result = candidates
        for filter_instance in self.filters:
            result = filter_instance.apply(result, user, db)
        return result


class NoFilter:
    """Pass-through filter that doesn't remove any candidates."""

    def apply(self, candidates: List[Candidate], user: User, db: Session = None) -> List[Candidate]:
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
