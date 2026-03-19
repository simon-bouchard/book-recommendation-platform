# tests/unit/models/domain/test_filters.py
"""
Unit tests for recommendation filters.
Tests filtering logic with mocked database and metadata.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, MagicMock, patch

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
    """Create mock async database session."""
    return MagicMock()


def _meta_dict(ratings_by_id: dict) -> dict:
    """Build a dict[int, dict] as returned by MetadataClient.enrich()."""
    return {idx: {"item_idx": idx, "num_ratings": n} for idx, n in ratings_by_id.items()}


def _patch_meta_client(enrich_return_value: dict):
    """Return a context manager that patches get_metadata_client with the given enrich result."""
    client = AsyncMock()
    client.enrich.return_value = enrich_return_value
    return patch("models.client.registry.get_metadata_client", return_value=client)


# Default ratings used by most MinRatingCountFilter tests
_DEFAULT_RATINGS = {100: 50, 101: 5, 102: 100, 103: 8, 104: 200}


class TestReadBooksFilter:
    """Test ReadBooksFilter with mocked async database query."""

    def test_implements_filter_protocol(self):
        """Should implement Filter protocol."""
        filter_obj = ReadBooksFilter()

        assert hasattr(filter_obj, "apply")
        assert callable(filter_obj.apply)

    @pytest.mark.asyncio
    async def test_removes_books_user_has_read(
        self, sample_candidates, mock_user, mock_db, monkeypatch
    ):
        """Should filter out books the user has already read."""
        monkeypatch.setattr(
            "models.domain.filters.get_read_books_for_candidates_async",
            AsyncMock(return_value={100, 102}),
        )

        filter_obj = ReadBooksFilter()
        filtered = await filter_obj.apply(sample_candidates, mock_user, mock_db)

        remaining_ids = [c.item_idx for c in filtered]
        assert 100 not in remaining_ids
        assert 102 not in remaining_ids
        assert 101 in remaining_ids
        assert 103 in remaining_ids
        assert 104 in remaining_ids

    @pytest.mark.asyncio
    async def test_preserves_order_of_candidates(
        self, sample_candidates, mock_user, mock_db, monkeypatch
    ):
        """Should maintain original candidate order."""
        monkeypatch.setattr(
            "models.domain.filters.get_read_books_for_candidates_async",
            AsyncMock(return_value=set()),
        )

        filter_obj = ReadBooksFilter()
        filtered = await filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert [c.item_idx for c in filtered] == [c.item_idx for c in sample_candidates]

    @pytest.mark.asyncio
    async def test_returns_empty_if_all_read(
        self, sample_candidates, mock_user, mock_db, monkeypatch
    ):
        """Should return empty list if user has read all candidates."""
        monkeypatch.setattr(
            "models.domain.filters.get_read_books_for_candidates_async",
            AsyncMock(return_value={c.item_idx for c in sample_candidates}),
        )

        filter_obj = ReadBooksFilter()
        filtered = await filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert filtered == []

    @pytest.mark.asyncio
    async def test_handles_empty_candidate_list(self, mock_user, mock_db, monkeypatch):
        """Should handle empty candidate list gracefully."""
        mock_query = AsyncMock(return_value=set())
        monkeypatch.setattr(
            "models.domain.filters.get_read_books_for_candidates_async",
            mock_query,
        )

        filter_obj = ReadBooksFilter()
        filtered = await filter_obj.apply([], mock_user, mock_db)

        assert filtered == []

    @pytest.mark.asyncio
    async def test_queries_database_with_correct_user_id(
        self, sample_candidates, mock_user, mock_db, monkeypatch
    ):
        """Should call the async query helper with the correct user_id."""
        mock_query = AsyncMock(return_value=set())
        monkeypatch.setattr(
            "models.domain.filters.get_read_books_for_candidates_async",
            mock_query,
        )

        filter_obj = ReadBooksFilter()
        await filter_obj.apply(sample_candidates, mock_user, mock_db)

        mock_query.assert_called_once()
        call_args = mock_query.call_args
        assert call_args[0][0] == mock_user.user_id


class TestMinRatingCountFilter:
    """Test MinRatingCountFilter with mocked metadata."""

    def test_implements_filter_protocol(self):
        """Should implement Filter protocol."""
        filter_obj = MinRatingCountFilter(min_count=10)

        assert hasattr(filter_obj, "apply")
        assert callable(filter_obj.apply)

    @pytest.mark.asyncio
    async def test_removes_books_with_low_rating_count(
        self, sample_candidates, mock_user, mock_db
    ):
        """Should filter out books below rating threshold."""
        with _patch_meta_client(_meta_dict(_DEFAULT_RATINGS)):
            filter_obj = MinRatingCountFilter(min_count=10)
            filtered = await filter_obj.apply(sample_candidates, mock_user, mock_db)

        remaining_ids = [c.item_idx for c in filtered]
        assert 100 in remaining_ids  # 50 ratings >= 10
        assert 102 in remaining_ids  # 100 ratings >= 10
        assert 104 in remaining_ids  # 200 ratings >= 10
        assert 101 not in remaining_ids  # 5 ratings < 10
        assert 103 not in remaining_ids  # 8 ratings < 10

    @pytest.mark.asyncio
    async def test_preserves_order_of_candidates(self, sample_candidates, mock_user, mock_db):
        """Should maintain original candidate order."""
        with _patch_meta_client(_meta_dict(_DEFAULT_RATINGS)):
            filter_obj = MinRatingCountFilter(min_count=10)
            filtered = await filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert [c.item_idx for c in filtered] == [100, 102, 104]

    @pytest.mark.asyncio
    async def test_returns_empty_if_all_below_threshold(
        self, sample_candidates, mock_user, mock_db
    ):
        """Should return empty list if all candidates below threshold."""
        with _patch_meta_client(_meta_dict({100: 0, 101: 0, 102: 0, 103: 0, 104: 0})):
            filter_obj = MinRatingCountFilter(min_count=10)
            filtered = await filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert filtered == []

    @pytest.mark.asyncio
    async def test_handles_missing_books_in_metadata(self, mock_user, mock_db):
        """Should handle books not in metadata (treats as 0 ratings)."""
        with _patch_meta_client({}):  # enrich returns nothing for unknown book
            candidates = [Candidate(item_idx=99999, score=0.9, source="test")]
            filter_obj = MinRatingCountFilter(min_count=10)
            filtered = await filter_obj.apply(candidates, mock_user, mock_db)

        assert filtered == []

    @pytest.mark.asyncio
    async def test_min_count_zero_keeps_all(self, sample_candidates, mock_user, mock_db):
        """min_count=0 should keep all candidates without calling enrich."""
        filter_obj = MinRatingCountFilter(min_count=0)
        filtered = await filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert len(filtered) == len(sample_candidates)


class TestFilterChain:
    """Test FilterChain for composing multiple filters."""

    @pytest.mark.asyncio
    async def test_applies_filters_in_sequence(self, sample_candidates, mock_user, mock_db):
        """Should apply filters one after another."""
        mock_filter1 = Mock(spec=Filter)
        mock_filter1.apply = AsyncMock(return_value=sample_candidates[:3])

        mock_filter2 = Mock(spec=Filter)
        mock_filter2.apply = AsyncMock(return_value=sample_candidates[:2])

        chain = FilterChain([mock_filter1, mock_filter2])
        result = await chain.apply(sample_candidates, mock_user, mock_db)

        mock_filter1.apply.assert_called_once()
        mock_filter2.apply.assert_called_once()

        second_call_input = mock_filter2.apply.call_args[0][0]
        assert len(second_call_input) == 3

    @pytest.mark.asyncio
    async def test_empty_chain_returns_all_candidates(self, sample_candidates, mock_user, mock_db):
        """Empty filter chain should act as no-op."""
        chain = FilterChain([])
        result = await chain.apply(sample_candidates, mock_user, mock_db)

        assert result == sample_candidates

    @pytest.mark.asyncio
    async def test_single_filter_chain_works(self, sample_candidates, mock_user, mock_db):
        """Single filter in chain should work correctly."""
        mock_filter = Mock(spec=Filter)
        mock_filter.apply = AsyncMock(return_value=sample_candidates[:2])

        chain = FilterChain([mock_filter])
        result = await chain.apply(sample_candidates, mock_user, mock_db)

        assert len(result) == 2
        mock_filter.apply.assert_called_once()

    @pytest.mark.asyncio
    async def test_chain_handles_filter_returning_empty(
        self, sample_candidates, mock_user, mock_db
    ):
        """Should handle filter that returns empty list."""
        mock_filter1 = Mock(spec=Filter)
        mock_filter1.apply = AsyncMock(return_value=[])

        mock_filter2 = Mock(spec=Filter)
        mock_filter2.apply = AsyncMock(return_value=[])

        chain = FilterChain([mock_filter1, mock_filter2])
        result = await chain.apply(sample_candidates, mock_user, mock_db)

        assert result == []
        mock_filter2.apply.assert_called_once()
        assert mock_filter2.apply.call_args[0][0] == []

    @pytest.mark.asyncio
    async def test_chain_with_real_filters(
        self, sample_candidates, mock_user, mock_db, monkeypatch
    ):
        """Test chain with actual filter implementations."""
        monkeypatch.setattr(
            "models.domain.filters.get_read_books_for_candidates_async",
            AsyncMock(return_value={100}),
        )

        with _patch_meta_client(_meta_dict(_DEFAULT_RATINGS)):
            read_filter = ReadBooksFilter()
            rating_filter = MinRatingCountFilter(min_count=10)
            chain = FilterChain([read_filter, rating_filter])
            result = await chain.apply(sample_candidates, mock_user, mock_db)

        remaining_ids = [c.item_idx for c in result]
        assert remaining_ids == [102, 104]


class TestNoFilter:
    """Test NoFilter pass-through implementation."""

    def test_implements_filter_protocol(self):
        """Should implement Filter protocol."""
        filter_obj = NoFilter()

        assert hasattr(filter_obj, "apply")
        assert callable(filter_obj.apply)

    @pytest.mark.asyncio
    async def test_returns_all_candidates_unchanged(self, sample_candidates, mock_user, mock_db):
        """Should return all candidates without modification."""
        filter_obj = NoFilter()
        result = await filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert result == sample_candidates
        assert result is sample_candidates

    @pytest.mark.asyncio
    async def test_handles_empty_candidates(self, mock_user, mock_db):
        """Should handle empty candidate list."""
        filter_obj = NoFilter()
        result = await filter_obj.apply([], mock_user, mock_db)

        assert result == []


class TestEdgeCases:
    """Test edge cases across all filters."""

    @pytest.mark.asyncio
    async def test_read_filter_handles_none_db(self, sample_candidates, mock_user):
        """ReadBooksFilter should raise ValueError when db is None."""
        filter_obj = ReadBooksFilter()

        with pytest.raises(ValueError, match="ReadBooksFilter requires a database session"):
            await filter_obj.apply(sample_candidates, mock_user, None)

    @pytest.mark.asyncio
    async def test_rating_filter_handles_none_db(self, sample_candidates, mock_user):
        """MinRatingCountFilter should work without database (uses metadata client)."""
        with _patch_meta_client(_meta_dict(_DEFAULT_RATINGS)):
            filter_obj = MinRatingCountFilter(min_count=10)
            result = await filter_obj.apply(sample_candidates, mock_user, None)

        assert len(result) == 3  # 100, 102, 104

    @pytest.mark.asyncio
    async def test_filters_preserve_candidate_attributes(
        self, sample_candidates, mock_user, mock_db
    ):
        """Filters should not modify candidate attributes."""
        original_scores = [c.score for c in sample_candidates]
        original_sources = [c.source for c in sample_candidates]

        filter_obj = NoFilter()
        filtered = await filter_obj.apply(sample_candidates, mock_user, mock_db)

        assert [c.score for c in filtered] == original_scores
        assert [c.source for c in filtered] == original_sources

    def test_min_rating_filter_handles_negative_threshold(
        self, sample_candidates, mock_user, mock_db
    ):
        """MinRatingCountFilter should raise ValueError for negative threshold."""
        with pytest.raises(ValueError, match="min_count must be non-negative"):
            MinRatingCountFilter(min_count=-10)
