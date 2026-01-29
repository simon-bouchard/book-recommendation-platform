# tests/unit/models/domain/test_filters.py
"""
Unit tests for recommendation filters.
Tests filtering logic with mocked database and metadata.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.domain.filters import (
    Filter,
    ReadBooksFilter,
    MinRatingCountFilter,
    FilterChain,
    NoFilter,
)
from models.domain.recommendation import Candidate
from models.domain.user import User


@pytest.fixture
def mock_user():
    """Create a mock user."""
    return User(
        user_id=123,
        fav_subjects=[5, 12, 23],
        country="US",
        age=25,
    )


@pytest.fixture
def sample_candidates():
    """Create sample candidates for filtering."""
    return [
        Candidate(item_idx=100, score=0.9, source="test"),
        Candidate(item_idx=101, score=0.8, source="test"),
        Candidate(item_idx=102, score=0.7, source="test"),
        Candidate(item_idx=103, score=0.6, source="test"),
        Candidate(item_idx=104, score=0.5, source="test"),
    ]


@pytest.fixture
def mock_db():
    """Create mock database session."""
    db = MagicMock()
    return db


@pytest.fixture
def mock_book_meta():
    """Create mock book metadata DataFrame."""
    return pd.DataFrame(
        {
            "item_idx": [100, 101, 102, 103, 104],
            "book_num_ratings": [50, 5, 100, 8, 200],
            "title": ["Book A", "Book B", "Book C", "Book D", "Book E"],
        }
    ).set_index("item_idx")


class TestReadBooksFilter:
    """Test ReadBooksFilter with mocked database."""

    def test_implements_filter_protocol(self):
        """Should implement Filter protocol."""
        filter_obj = ReadBooksFilter()

        assert hasattr(filter_obj, "apply")
        assert callable(filter_obj.apply)

    def test_removes_books_user_has_read(self, sample_candidates, mock_user, mock_db):
        """Should filter out books the user has already read."""
        # Mock query to return items 100 and 102 as read
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [
            Mock(item_idx=100),
            Mock(item_idx=102),
        ]
        mock_db.query.return_value = mock_query

        filter_obj = ReadBooksFilter()
        filtered = filter_obj.apply(sample_candidates, mock_user, mock_db)

        # Should remove items 100 and 102
        remaining_ids = [c.item_idx for c in filtered]
        assert 100 not in remaining_ids
        assert 102 not in remaining_ids
        assert 101 in remaining_ids
        assert 103 in remaining_ids
        assert 104 in remaining_ids

    def test_preserves_order_of_candidates(self, sample_candidates, mock_user, mock_db):
        """Should maintain original candidate order."""
        # Mock no read books
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_db.query.return_value = mock_query

        filter_obj = ReadBooksFilter()
        filtered = filter_obj.apply(sample_candidates, mock_user, mock_db)

        # Order should be preserved
        assert [c.item_idx for c in filtered] == [c.item_idx for c in sample_candidates]

    def test_returns_empty_if_all_read(self, sample_candidates, mock_user, mock_db):
        """Should return empty list if user has read all candidates."""
        # Mock all books as read
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [Mock(item_idx=c.item_idx) for c in sample_candidates]
        mock_db.query.return_value = mock_query

        filter_obj = ReadBooksFilter()
        filtered = filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert filtered == []

    def test_handles_empty_candidate_list(self, mock_user, mock_db):
        """Should handle empty candidate list gracefully."""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_db.query.return_value = mock_query

        filter_obj = ReadBooksFilter()
        filtered = filter_obj.apply([], mock_user, mock_db)

        assert filtered == []

    def test_queries_database_with_correct_user_id(self, sample_candidates, mock_user, mock_db):
        """Should query interactions for the correct user."""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_db.query.return_value = mock_query

        filter_obj = ReadBooksFilter()
        filter_obj.apply(sample_candidates, mock_user, mock_db)

        # Should have filtered by user_id (checking that filter was called)
        assert mock_query.filter.called


class TestMinRatingCountFilter:
    """Test MinRatingCountFilter with mocked metadata."""

    def test_implements_filter_protocol(self):
        """Should implement Filter protocol."""
        filter_obj = MinRatingCountFilter(min_count=10)

        assert hasattr(filter_obj, "apply")
        assert callable(filter_obj.apply)

    def test_removes_books_with_low_rating_count(
        self, sample_candidates, mock_user, mock_db, mock_book_meta, monkeypatch
    ):
        """Should filter out books below rating threshold."""

        # Mock load_book_meta to return our test data
        def mock_load_book_meta(use_cache=True):
            return mock_book_meta

        monkeypatch.setattr(
            "models.domain.filters.load_book_meta",
            mock_load_book_meta,
        )

        # Threshold of 10 should keep 100 (50), 102 (100), 104 (200)
        # and remove 101 (5), 103 (8)
        filter_obj = MinRatingCountFilter(min_count=10)
        filtered = filter_obj.apply(sample_candidates, mock_user, mock_db)

        remaining_ids = [c.item_idx for c in filtered]
        assert 100 in remaining_ids  # 50 ratings >= 10
        assert 102 in remaining_ids  # 100 ratings >= 10
        assert 104 in remaining_ids  # 200 ratings >= 10
        assert 101 not in remaining_ids  # 5 ratings < 10
        assert 103 not in remaining_ids  # 8 ratings < 10

    def test_preserves_order_of_candidates(
        self, sample_candidates, mock_user, mock_db, mock_book_meta, monkeypatch
    ):
        """Should maintain original candidate order."""

        def mock_load_book_meta(use_cache=True):
            return mock_book_meta

        monkeypatch.setattr(
            "models.domain.filters.load_book_meta",
            mock_load_book_meta,
        )

        filter_obj = MinRatingCountFilter(min_count=10)
        filtered = filter_obj.apply(sample_candidates, mock_user, mock_db)

        # Should be in order: 100, 102, 104 (original order preserved)
        assert [c.item_idx for c in filtered] == [100, 102, 104]

    def test_returns_empty_if_all_below_threshold(
        self, sample_candidates, mock_user, mock_db, monkeypatch
    ):
        """Should return empty list if all candidates below threshold."""
        # All books have 0 ratings
        low_rating_meta = pd.DataFrame(
            {
                "item_idx": [100, 101, 102, 103, 104],
                "book_num_ratings": [0, 0, 0, 0, 0],
            }
        ).set_index("item_idx")

        def mock_load_book_meta(use_cache=True):
            return low_rating_meta

        monkeypatch.setattr(
            "models.domain.filters.load_book_meta",
            mock_load_book_meta,
        )

        filter_obj = MinRatingCountFilter(min_count=10)
        filtered = filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert filtered == []

    def test_handles_missing_books_in_metadata(
        self, mock_user, mock_db, mock_book_meta, monkeypatch
    ):
        """Should handle books not in metadata (treats as 0 ratings)."""

        def mock_load_book_meta(use_cache=True):
            return mock_book_meta

        monkeypatch.setattr(
            "models.domain.filters.load_book_meta",
            mock_load_book_meta,
        )

        # Candidate for book not in metadata
        candidates = [Candidate(item_idx=99999, score=0.9, source="test")]

        filter_obj = MinRatingCountFilter(min_count=10)
        filtered = filter_obj.apply(candidates, mock_user, mock_db)

        # Should be filtered out (missing -> 0 ratings)
        assert filtered == []

    def test_min_count_zero_keeps_all(
        self, sample_candidates, mock_user, mock_db, mock_book_meta, monkeypatch
    ):
        """min_count=0 should keep all candidates."""

        def mock_load_book_meta(use_cache=True):
            return mock_book_meta

        monkeypatch.setattr(
            "models.domain.filters.load_book_meta",
            mock_load_book_meta,
        )

        filter_obj = MinRatingCountFilter(min_count=0)
        filtered = filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert len(filtered) == len(sample_candidates)


class TestFilterChain:
    """Test FilterChain for composing multiple filters."""

    def test_applies_filters_in_sequence(self, sample_candidates, mock_user, mock_db):
        """Should apply filters one after another."""
        # Create two mock filters
        mock_filter1 = Mock(spec=Filter)
        mock_filter1.apply.return_value = sample_candidates[:3]  # Keep first 3

        mock_filter2 = Mock(spec=Filter)
        mock_filter2.apply.return_value = sample_candidates[:2]  # Keep first 2

        chain = FilterChain([mock_filter1, mock_filter2])
        result = chain.apply(sample_candidates, mock_user, mock_db)

        # Both filters should have been called
        mock_filter1.apply.assert_called_once()
        mock_filter2.apply.assert_called_once()

        # Second filter should receive output of first filter
        second_call_input = mock_filter2.apply.call_args[0][0]
        assert len(second_call_input) == 3

    def test_empty_chain_returns_all_candidates(self, sample_candidates, mock_user, mock_db):
        """Empty filter chain should act as no-op."""
        chain = FilterChain([])
        result = chain.apply(sample_candidates, mock_user, mock_db)

        assert result == sample_candidates

    def test_single_filter_chain_works(self, sample_candidates, mock_user, mock_db):
        """Single filter in chain should work correctly."""
        mock_filter = Mock(spec=Filter)
        mock_filter.apply.return_value = sample_candidates[:2]

        chain = FilterChain([mock_filter])
        result = chain.apply(sample_candidates, mock_user, mock_db)

        assert len(result) == 2
        mock_filter.apply.assert_called_once()

    def test_chain_handles_filter_returning_empty(self, sample_candidates, mock_user, mock_db):
        """Should handle filter that returns empty list."""
        mock_filter1 = Mock(spec=Filter)
        mock_filter1.apply.return_value = []  # First filter removes all

        mock_filter2 = Mock(spec=Filter)
        mock_filter2.apply.return_value = []

        chain = FilterChain([mock_filter1, mock_filter2])
        result = chain.apply(sample_candidates, mock_user, mock_db)

        assert result == []

        # Second filter should still be called (with empty list)
        mock_filter2.apply.assert_called_once()
        assert mock_filter2.apply.call_args[0][0] == []

    def test_chain_with_real_filters(
        self, sample_candidates, mock_user, mock_db, mock_book_meta, monkeypatch
    ):
        """Test chain with actual filter implementations."""
        # Setup mocks
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [Mock(item_idx=100)]  # User read book 100
        mock_db.query.return_value = mock_query

        def mock_load_book_meta(use_cache=True):
            return mock_book_meta

        monkeypatch.setattr(
            "models.domain.filters.load_book_meta",
            mock_load_book_meta,
        )

        # Create chain: remove read books, then filter by rating count
        read_filter = ReadBooksFilter()
        rating_filter = MinRatingCountFilter(min_count=10)
        chain = FilterChain([read_filter, rating_filter])

        result = chain.apply(sample_candidates, mock_user, mock_db)

        # Should remove:
        # - 100 (read by user)
        # - 101 (only 5 ratings < 10)
        # - 103 (only 8 ratings < 10)
        # Should keep: 102, 104
        remaining_ids = [c.item_idx for c in result]
        assert remaining_ids == [102, 104]


class TestNoFilter:
    """Test NoFilter pass-through implementation."""

    def test_implements_filter_protocol(self):
        """Should implement Filter protocol."""
        filter_obj = NoFilter()

        assert hasattr(filter_obj, "apply")
        assert callable(filter_obj.apply)

    def test_returns_all_candidates_unchanged(self, sample_candidates, mock_user, mock_db):
        """Should return all candidates without modification."""
        filter_obj = NoFilter()
        result = filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert result == sample_candidates
        assert result is sample_candidates  # Same object

    def test_handles_empty_candidates(self, mock_user, mock_db):
        """Should handle empty candidate list."""
        filter_obj = NoFilter()
        result = filter_obj.apply([], mock_user, mock_db)

        assert result == []


class TestEdgeCases:
    """Test edge cases across all filters."""

    def test_read_filter_handles_none_db(self, sample_candidates, mock_user):
        """ReadBooksFilter should raise ValueError when db is None."""
        filter_obj = ReadBooksFilter()

        # Should raise ValueError when db is None
        with pytest.raises(ValueError, match="ReadBooksFilter requires database session"):
            filter_obj.apply(sample_candidates, mock_user, None)

    def test_rating_filter_handles_none_db(self, sample_candidates, mock_user, monkeypatch):
        """MinRatingCountFilter should work without database (uses loader)."""
        mock_meta = pd.DataFrame(
            {
                "item_idx": [100, 101, 102, 103, 104],
                "book_num_ratings": [50, 5, 100, 8, 200],
            }
        ).set_index("item_idx")

        def mock_load_book_meta(use_cache=True):
            return mock_meta

        monkeypatch.setattr(
            "models.domain.filters.load_book_meta",
            mock_load_book_meta,
        )

        filter_obj = MinRatingCountFilter(min_count=10)

        # Should work fine with None db (doesn't use it)
        result = filter_obj.apply(sample_candidates, mock_user, None)

        assert len(result) == 3  # 100, 102, 104

    def test_filters_preserve_candidate_attributes(self, sample_candidates, mock_user, mock_db):
        """Filters should not modify candidate attributes."""
        original_scores = [c.score for c in sample_candidates]
        original_sources = [c.source for c in sample_candidates]

        filter_obj = NoFilter()
        filtered = filter_obj.apply(sample_candidates, mock_user, mock_db)

        # Attributes should be unchanged
        assert [c.score for c in filtered] == original_scores
        assert [c.source for c in filtered] == original_sources

    def test_min_rating_filter_handles_negative_threshold(
        self, sample_candidates, mock_user, mock_db
    ):
        """MinRatingCountFilter should raise ValueError for negative threshold."""
        # Should raise ValueError during construction for negative min_count
        with pytest.raises(ValueError, match="min_count must be non-negative"):
            MinRatingCountFilter(min_count=-10)
