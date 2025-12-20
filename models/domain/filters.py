# models/domain/filters.py
"""
Filters for removing unwanted candidates from recommendation results.
Filters apply quality control and user-specific exclusions.
"""

from typing import Protocol, List
from sqlalchemy.orm import Session

from models.domain.recommendation import Candidate
from models.domain.user import User
from models.data.queries import get_read_books
from models.data.loaders import load_book_meta


class Filter(Protocol):
    """
    Protocol for candidate filters.

    Filters remove candidates based on quality thresholds,
    user preferences, or business rules.
    """

    def apply(self, candidates: List[Candidate], user: User, db: Session = None) -> List[Candidate]:
        """
        Filter candidate list.

        Args:
            candidates: List of candidates to filter
            user: User context for filtering decisions
            db: Database session (optional, for filters that need DB access)

        Returns:
            Filtered list of candidates
        """
        ...


class ReadBooksFilter:
    """
    Filter out books the user has already read or rated.

    Prevents recommending books the user has already interacted with.
    """

    def apply(self, candidates: List[Candidate], user: User, db: Session = None) -> List[Candidate]:
        """
        Remove books user has already read.

        Args:
            candidates: Candidates to filter
            user: User to check interactions for
            db: Database session (required)

        Returns:
            Candidates excluding user's read books

        Raises:
            ValueError: If db is None
        """
        if db is None:
            raise ValueError("ReadBooksFilter requires database session")

        read_item_ids = get_read_books(user.user_id, db)

        return [candidate for candidate in candidates if candidate.item_idx not in read_item_ids]


class MinRatingCountFilter:
    """
    Filter out books with insufficient ratings.

    Ensures recommended books have enough user feedback to be trustworthy.
    """

    def __init__(self, min_count: int = 10):
        """
        Initialize rating count filter.

        Args:
            min_count: Minimum number of ratings required
        """
        if min_count < 0:
            raise ValueError(f"min_count must be non-negative, got {min_count}")

        self.min_count = min_count
        self._book_meta = None

    def apply(self, candidates: List[Candidate], user: User, db: Session = None) -> List[Candidate]:
        """
        Remove books with too few ratings.

        Args:
            candidates: Candidates to filter
            user: User context (unused)
            db: Database session (unused)

        Returns:
            Candidates with sufficient rating counts
        """
        if self.min_count == 0:
            return candidates

        # Lazy load book metadata
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
    """
    Compose multiple filters into a single filter.

    Applies filters in sequence, passing output of one filter
    as input to the next.
    """

    def __init__(self, filters: List[Filter]):
        """
        Initialize filter chain.

        Args:
            filters: List of filters to apply in order
        """
        self.filters = filters

    def apply(self, candidates: List[Candidate], user: User, db: Session = None) -> List[Candidate]:
        """
        Apply all filters in sequence.

        Args:
            candidates: Initial candidates
            user: User context
            db: Database session

        Returns:
            Candidates after all filters applied
        """
        result = candidates
        for filter_instance in self.filters:
            result = filter_instance.apply(result, user, db)

        return result


class NoFilter:
    """
    Pass-through filter that doesn't remove any candidates.

    Useful for testing or when no filtering is desired.
    """

    def apply(self, candidates: List[Candidate], user: User, db: Session = None) -> List[Candidate]:
        """Return candidates unchanged."""
        return candidates
