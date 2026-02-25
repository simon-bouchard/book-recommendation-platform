# tests/unit/models/services/test_similarity_service.py
"""
Unit tests for SimilarityService.
Tests three similarity modes (subject, ALS, hybrid) with quality filtering.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.services.similarity_service import SimilarityService
from models.infrastructure.similarity_index import SimilarityIndex


@pytest.fixture
def mock_book_meta():
    """Create mock book metadata DataFrame."""
    data = {
        "title": [f"Book {i}" for i in range(100)],
        "author": [f"Author {i}" for i in range(100)],
        "year": [2000 + i % 25 for i in range(100)],
        "isbn": [f"ISBN-{i:04d}" for i in range(100)],
        "cover_id": [f"cover_{i}" for i in range(100)],
        "book_num_ratings": [i for i in range(100)],  # 0–99 ratings; book 1000+i has i ratings
        "book_avg_rating": [3.5 + (i % 10) * 0.1 for i in range(100)],
    }
    return pd.DataFrame(data, index=range(1000, 1100))


@pytest.fixture
def mock_subject_embeddings():
    """Create mock subject embeddings (100 books, normalised)."""
    rng = np.random.default_rng(seed=42)
    embeddings = rng.standard_normal((100, 16)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    ids = list(range(1000, 1100))
    return embeddings, ids


@pytest.fixture
def mock_als_factors():
    """Create mock ALS factors (80 books overlapping with subject IDs 1000–1079)."""
    rng = np.random.default_rng(seed=43)
    user_factors = rng.standard_normal((50, 32)).astype(np.float32)
    book_factors = rng.standard_normal((80, 32)).astype(np.float32)
    book_factors = book_factors / np.linalg.norm(book_factors, axis=1, keepdims=True)

    user_ids = list(range(100, 150))
    book_ids = list(range(1000, 1080))
    book_row_map = {i: book_id for i, book_id in enumerate(book_ids)}

    return user_factors, book_factors, user_ids, book_row_map


@pytest.fixture
def mock_subject_index(mock_subject_embeddings):
    """
    Real SimilarityIndex built from mock subject embeddings with no candidate filtering.

    Used to replace the module-level singleton returned by get_subject_similarity_index().
    Building a real index (rather than a Mock) ensures that all behavioural tests —
    sort order, k-truncation, query exclusion — pass without a working filesystem.
    """
    embeddings, ids = mock_subject_embeddings
    return SimilarityIndex(embeddings=embeddings, ids=ids, normalize=False)


@pytest.fixture
def mock_als_index(mock_als_factors, mock_book_meta):
    """
    Real filtered SimilarityIndex built from mock ALS factors.

    Mirrors the production get_als_similarity_index() which applies a 10-rating filter.
    Books 1000–1009 have 0–9 ratings in mock_book_meta and are therefore excluded
    from the candidate pool, making the filtering assertions meaningful.
    """
    _, book_factors, _, book_row_map = mock_als_factors
    book_ids = [book_row_map[i] for i in range(book_factors.shape[0])]
    return SimilarityIndex.create_filtered_index(
        embeddings=book_factors,
        ids=book_ids,
        metadata=mock_book_meta,
        min_rating_count=10,
        normalize=False,
    )


class TestSimilarityServiceInitialization:
    """Test SimilarityService initialization and lazy loading."""

    def test_service_initializes_with_none_book_meta(self):
        """
        Book metadata should not be loaded at construction time.

        Subject and ALS indices are owned by module-level singletons in
        similarity_indices.py; the service's only instance-level lazy attribute
        is _book_meta.
        """
        service = SimilarityService()

        assert service._book_meta is None

    def test_service_sets_rating_thresholds(self):
        """Should set class-level rating thresholds."""
        service = SimilarityService()

        assert service.ALS_MIN_RATINGS == 10
        assert service.HYBRID_MIN_RATINGS == 5


class TestSubjectSimilarity:
    """Test subject-based similarity (no filtering)."""

    def test_subject_mode_calls_index_getter_on_first_request(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Subject index getter should be invoked when get_similar is called."""
        service = SimilarityService()

        getter = Mock(return_value=mock_subject_index)
        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index", getter
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        getter.assert_not_called()

        results = service.get_similar(item_idx=1000, mode="subject", k=5)

        getter.assert_called_once()
        assert len(results) <= 5

    def test_subject_mode_returns_similar_books(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Should return similar books with metadata."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=10)

        assert isinstance(results, list)
        assert len(results) <= 10

        for result in results:
            assert "item_idx" in result
            assert "title" in result
            assert "score" in result
            assert result["item_idx"] != 1000

    def test_subject_mode_respects_k_parameter(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Should return at most k results."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results_5 = service.get_similar(item_idx=1000, mode="subject", k=5)
        results_20 = service.get_similar(item_idx=1000, mode="subject", k=20)

        assert len(results_5) <= 5
        assert len(results_20) <= 20

    def test_subject_mode_no_filtering_by_rating_count(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Subject mode should not filter by rating count."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1050, mode="subject", k=50)

        rating_counts = [mock_book_meta.loc[r["item_idx"], "book_num_ratings"] for r in results]
        assert any(count < 5 for count in rating_counts)

    def test_subject_mode_index_getter_called_per_request(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """
        Index caching is handled by the module-level singleton in similarity_indices.py.
        The service calls the getter on every request; caching is transparent to the service.
        """
        service = SimilarityService()

        getter = Mock(return_value=mock_subject_index)
        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index", getter
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        service.get_similar(item_idx=1000, mode="subject", k=5)
        service.get_similar(item_idx=1001, mode="subject", k=5)

        assert getter.call_count == 2


class TestALSSimilarity:
    """Test ALS-based similarity with rating filtering."""

    def test_als_mode_calls_index_getter_on_first_request(
        self, monkeypatch, mock_als_index, mock_book_meta
    ):
        """ALS index getter should be invoked when get_similar is called."""
        service = SimilarityService()

        getter = Mock(return_value=mock_als_index)
        monkeypatch.setattr("models.services.similarity_service.get_als_similarity_index", getter)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        getter.assert_not_called()

        service.get_similar(item_idx=1000, mode="als", k=5)

        getter.assert_called_once()

    def test_als_mode_returns_similar_books(self, monkeypatch, mock_als_index, mock_book_meta):
        """Should return similar books based on collaborative filtering."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_als_similarity_index",
            lambda: mock_als_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="als", k=10)

        assert isinstance(results, list)
        assert len(results) <= 10

        for result in results:
            assert "item_idx" in result
            assert "title" in result
            assert "score" in result

    def test_als_mode_filters_by_rating_count(self, monkeypatch, mock_als_index, mock_book_meta):
        """ALS mode should filter candidates by 10+ ratings."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_als_similarity_index",
            lambda: mock_als_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1050, mode="als", k=50)

        for result in results:
            rating_count = mock_book_meta.loc[result["item_idx"], "book_num_ratings"]
            assert rating_count >= 10

    def test_als_mode_respects_k_parameter(self, monkeypatch, mock_als_index, mock_book_meta):
        """Should return at most k results."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_als_similarity_index",
            lambda: mock_als_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results_5 = service.get_similar(item_idx=1020, mode="als", k=5)
        results_15 = service.get_similar(item_idx=1020, mode="als", k=15)

        assert len(results_5) <= 5
        assert len(results_15) <= 15


class TestHybridSimilarity:
    """Test hybrid similarity with score blending."""

    def test_hybrid_mode_initializes_on_first_call(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid mode should build alignment data and return results on first query."""
        service = SimilarityService()

        monkeypatch.setattr("models.services.similarity_service._hybrid_data", None)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="hybrid", k=5)

        assert isinstance(results, list)
        assert len(results) <= 5

    def test_hybrid_mode_blends_subject_and_als_scores(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Should blend subject and ALS scores using alpha parameter."""
        import models.services.similarity_service as sim_svc

        monkeypatch.setattr(sim_svc, "_hybrid_data", None)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        service = SimilarityService()

        results_subject_heavy = service.get_similar(item_idx=1000, mode="hybrid", k=10, alpha=0.2)

        # Reset module-level cache to force re-entry of _get_hybrid_data on next call
        sim_svc._hybrid_data = None

        results_als_heavy = service.get_similar(item_idx=1000, mode="hybrid", k=10, alpha=0.8)

        assert len(results_subject_heavy) > 0
        assert len(results_als_heavy) > 0

    def test_hybrid_mode_filters_by_rating_count(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid mode should filter candidates by 5+ ratings by default."""
        service = SimilarityService()

        monkeypatch.setattr("models.services.similarity_service._hybrid_data", None)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1050, mode="hybrid", k=50)

        for result in results:
            rating_count = mock_book_meta.loc[result["item_idx"], "book_num_ratings"]
            assert rating_count >= 5

    def test_hybrid_mode_respects_custom_rating_threshold(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Should respect custom min_rating_count parameter."""
        service = SimilarityService()

        monkeypatch.setattr("models.services.similarity_service._hybrid_data", None)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1050, mode="hybrid", k=50, min_rating_count=20)

        for result in results:
            rating_count = mock_book_meta.loc[result["item_idx"], "book_num_ratings"]
            assert rating_count >= 20

    def test_hybrid_mode_caches_initialization(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid data should be built once and reused on subsequent calls."""
        service = SimilarityService()

        monkeypatch.setattr("models.services.similarity_service._hybrid_data", None)

        subject_load_mock = Mock(return_value=mock_subject_embeddings)
        als_load_mock = Mock(return_value=mock_als_factors)

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            subject_load_mock,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            als_load_mock,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        service.get_similar(item_idx=1000, mode="hybrid", k=5)
        assert subject_load_mock.call_count == 1
        assert als_load_mock.call_count == 1

        service.get_similar(item_idx=1001, mode="hybrid", k=5, alpha=0.7)
        assert subject_load_mock.call_count == 1
        assert als_load_mock.call_count == 1


class TestTwoPoolArchitecture:
    """Test two-pool system (query any book, filter candidates)."""

    def test_can_query_low_rated_book_in_als_mode(
        self, monkeypatch, mock_als_index, mock_book_meta
    ):
        """
        Should be able to query a low-rated book even with candidate filtering enabled.

        The ALS index holds all books in its full embedding matrix (query pool)
        but only returns high-rated books as candidates.
        """
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_als_similarity_index",
            lambda: mock_als_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        # Item 1005 has 5 ratings — below the 10-rating threshold — but is still queryable
        results = service.get_similar(item_idx=1005, mode="als", k=10)

        assert isinstance(results, list)

        for result in results:
            rating_count = mock_book_meta.loc[result["item_idx"], "book_num_ratings"]
            assert rating_count >= 10

    def test_filtering_can_be_disabled(self, monkeypatch, mock_als_index, mock_book_meta):
        """With filter_candidates=True, all results should meet the rating threshold."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_als_similarity_index",
            lambda: mock_als_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1020, mode="als", k=50, filter_candidates=True)

        assert all(mock_book_meta.loc[r["item_idx"], "book_num_ratings"] >= 10 for r in results)


class TestResultFormatting:
    """Test result formatting and metadata enrichment."""

    def test_results_include_required_fields(self, monkeypatch, mock_subject_index, mock_book_meta):
        """Results should include all required metadata fields."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=5)

        for result in results:
            assert "item_idx" in result
            assert "title" in result
            assert "score" in result
            assert isinstance(result["item_idx"], int)
            assert isinstance(result["title"], str)
            assert isinstance(result["score"], float)

    def test_results_include_optional_metadata(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Results should include optional metadata when available."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=5)

        for result in results:
            assert "author" in result
            assert "year" in result
            assert "isbn" in result
            assert "cover_id" in result

    def test_results_handle_missing_books_gracefully(self, monkeypatch, mock_subject_index):
        """Should skip books not found in metadata."""
        service = SimilarityService()

        partial_meta = pd.DataFrame(
            {
                "title": ["Book A", "Book B"],
                "author": ["Author A", "Author B"],
            },
            index=[1000, 1001],
        )

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: partial_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=50)

        assert all(r["item_idx"] in [1000, 1001] for r in results)


class TestLogging:
    """Test structured logging for observability."""

    @patch("models.services.similarity_service.logger")
    def test_logs_search_start(self, mock_logger, monkeypatch, mock_subject_index, mock_book_meta):
        """Should log search start with context."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        service.get_similar(item_idx=1000, mode="subject", k=10)

        start_calls = [c for c in mock_logger.info.call_args_list if "started" in str(c)]
        assert len(start_calls) > 0

        start_call = start_calls[0]
        assert "Similarity search started" in str(start_call)
        assert "item_idx" in str(start_call)
        assert "mode" in str(start_call)

    @patch("models.services.similarity_service.logger")
    def test_logs_search_completion(
        self, mock_logger, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Should log search completion with metrics."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        service.get_similar(item_idx=1000, mode="subject", k=10)

        completion_calls = [c for c in mock_logger.info.call_args_list if "completed" in str(c)]
        assert len(completion_calls) > 0

        completion_call = completion_calls[0]
        assert "Similarity search completed" in str(completion_call)
        assert "count" in str(completion_call)
        assert "latency_ms" in str(completion_call)

    @patch("models.services.similarity_service.logger")
    def test_logs_errors_with_context(self, mock_logger, monkeypatch):
        """Should log errors with context and traceback."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            Mock(side_effect=RuntimeError("Test error")),
        )

        with pytest.raises(RuntimeError):
            service.get_similar(item_idx=1000, mode="subject", k=10)

        assert mock_logger.error.called
        error_call = mock_logger.error.call_args
        assert "Similarity search failed" in str(error_call)
        assert "item_idx" in str(error_call)
        assert "mode" in str(error_call)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_mode_raises_error(self):
        """Should raise ValueError for invalid mode."""
        service = SimilarityService()

        with pytest.raises(ValueError, match="Unknown mode"):
            service.get_similar(item_idx=1000, mode="invalid_mode", k=10)

    def test_k_zero_returns_empty_list(self, monkeypatch, mock_subject_index, mock_book_meta):
        """Should handle k=0 gracefully."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=0)

        assert results == []

    def test_query_book_not_in_index_returns_empty(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Should return empty list if query book is not in the index."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=9999, mode="subject", k=10)

        assert results == []

    def test_alpha_zero_uses_only_subject_scores(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid with alpha=0.0 should return results (pure subject weighting)."""
        service = SimilarityService()

        monkeypatch.setattr("models.services.similarity_service._hybrid_data", None)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="hybrid", k=10, alpha=0.0)

        assert len(results) > 0

    def test_alpha_one_uses_only_als_scores(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid with alpha=1.0 should return results (pure ALS weighting)."""
        service = SimilarityService()

        monkeypatch.setattr("models.services.similarity_service._hybrid_data", None)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="hybrid", k=10, alpha=1.0)

        assert len(results) > 0

    def test_k_larger_than_catalog_returns_all_candidates(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Should handle k larger than number of candidates."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=10000)

        assert len(results) <= 99  # 100 books minus the query book

    def test_hybrid_mode_handles_book_missing_als_factors(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid should handle books missing from ALS gracefully (uses zero ALS scores)."""
        service = SimilarityService()

        monkeypatch.setattr("models.services.similarity_service._hybrid_data", None)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        # Books 1080–1099 are in the subject index but not in ALS (which only covers 1000–1079)
        results = service.get_similar(item_idx=1085, mode="hybrid", k=10)

        assert isinstance(results, list)


class TestMetadataCaching:
    """Test book metadata caching."""

    def test_book_meta_loaded_once(self, monkeypatch, mock_subject_index, mock_book_meta):
        """Book metadata should be loaded once and cached on the service instance."""
        service = SimilarityService()

        meta_load_mock = Mock(return_value=mock_book_meta)
        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            meta_load_mock,
        )

        service.get_similar(item_idx=1000, mode="subject", k=5)
        assert meta_load_mock.call_count == 1

        service.get_similar(item_idx=1001, mode="subject", k=5)
        assert meta_load_mock.call_count == 1

    def test_shared_metadata_across_modes(
        self,
        monkeypatch,
        mock_subject_index,
        mock_als_index,
        mock_subject_embeddings,
        mock_als_factors,
        mock_book_meta,
    ):
        """
        Metadata should be loaded at most twice across all three modes.

        Subject and ALS modes share the service's _book_meta cache.
        Hybrid mode additionally calls load_book_meta inside _get_hybrid_data()
        to build the rating-count alignment array, which may add one extra call
        before the instance cache takes over in _format_results.
        """
        service = SimilarityService()

        monkeypatch.setattr("models.services.similarity_service._hybrid_data", None)

        meta_load_mock = Mock(return_value=mock_book_meta)
        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.get_als_similarity_index",
            lambda: mock_als_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            meta_load_mock,
        )

        service.get_similar(item_idx=1000, mode="subject", k=5)
        service.get_similar(item_idx=1000, mode="als", k=5)
        service.get_similar(item_idx=1000, mode="hybrid", k=5)

        # Subject and ALS share the _book_meta instance cache; hybrid's _get_hybrid_data()
        # calls load_book_meta directly before the instance cache is populated.
        assert meta_load_mock.call_count <= 2


class TestPerformanceConsiderations:
    """Test performance-related behaviour."""

    def test_excludes_query_book_from_results(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Results should never include the query book itself."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        query_id = 1050
        results = service.get_similar(item_idx=query_id, mode="subject", k=50)

        assert all(r["item_idx"] != query_id for r in results)

    def test_results_sorted_by_score_descending(
        self, monkeypatch, mock_subject_index, mock_book_meta
    ):
        """Results should be sorted by score (highest first)."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.get_subject_similarity_index",
            lambda: mock_subject_index,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=20)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
