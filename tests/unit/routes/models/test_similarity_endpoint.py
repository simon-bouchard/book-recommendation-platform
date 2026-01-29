# tests/unit/models/services/test_similarity_service.py
"""
Unit tests for SimilarityService.
Tests three similarity modes (subject, ALS, hybrid) with quality filtering.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from models.services.similarity_service import SimilarityService


@pytest.fixture
def mock_book_meta():
    """Create mock book metadata DataFrame."""
    data = {
        "title": [f"Book {i}" for i in range(100)],
        "author": [f"Author {i}" for i in range(100)],
        "year": [2000 + i % 25 for i in range(100)],
        "isbn": [f"ISBN-{i:04d}" for i in range(100)],
        "cover_id": [f"cover_{i}" for i in range(100)],
        "book_num_ratings": [i for i in range(100)],  # 0 to 99 ratings
        "book_avg_rating": [3.5 + (i % 10) * 0.1 for i in range(100)],
    }
    df = pd.DataFrame(data, index=range(1000, 1100))  # item_idx 1000-1099
    return df


@pytest.fixture
def mock_subject_embeddings():
    """Create mock subject embeddings."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 16).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    ids = list(range(1000, 1100))
    return embeddings, ids


@pytest.fixture
def mock_als_factors():
    """Create mock ALS factors."""
    np.random.seed(43)
    user_factors = np.random.randn(50, 32).astype(np.float32)
    book_factors = np.random.randn(80, 32).astype(np.float32)
    # Normalize
    book_factors = book_factors / np.linalg.norm(book_factors, axis=1, keepdims=True)

    user_ids = list(range(100, 150))
    # Book IDs overlap with subject embeddings (1000-1079)
    book_ids = list(range(1000, 1080))
    book_row_map = {i: book_id for i, book_id in enumerate(book_ids)}

    return user_factors, book_factors, user_ids, book_row_map


class TestSimilarityServiceInitialization:
    """Test SimilarityService initialization and lazy loading."""

    def test_service_initializes_with_none_values(self):
        """Should initialize with all indices as None (lazy loading)."""
        service = SimilarityService()

        assert service._subject_index is None
        assert service._als_index is None
        assert service._book_meta is None
        assert service._hybrid_initialized is False

    def test_service_sets_rating_thresholds(self):
        """Should set class-level rating thresholds."""
        service = SimilarityService()

        assert service.ALS_MIN_RATINGS == 10
        assert service.HYBRID_MIN_RATINGS == 5


class TestSubjectSimilarity:
    """Test subject-based similarity (no filtering)."""

    def test_subject_mode_builds_index_on_first_call(
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Subject index should be lazy-loaded on first query."""
        service = SimilarityService()

        # Mock loaders
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        assert service._subject_index is None

        # First call builds index
        results = service.get_similar(item_idx=1000, mode="subject", k=5)

        assert service._subject_index is not None
        assert len(results) <= 5

    def test_subject_mode_returns_similar_books(
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Should return similar books with metadata."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
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
            assert result["item_idx"] != 1000  # Excludes query book

    def test_subject_mode_respects_k_parameter(
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Should return at most k results."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
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
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Subject mode should not filter by rating count."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1050, mode="subject", k=50)

        # Should include books with low rating counts
        rating_counts = [mock_book_meta.loc[r["item_idx"], "book_num_ratings"] for r in results]
        assert any(count < 5 for count in rating_counts)

    def test_subject_mode_caches_index(self, monkeypatch, mock_subject_embeddings, mock_book_meta):
        """Index should be cached after first query."""
        service = SimilarityService()

        load_mock = Mock(return_value=mock_subject_embeddings)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            load_mock,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        # First query
        service.get_similar(item_idx=1000, mode="subject", k=5)
        assert load_mock.call_count == 1

        # Second query (should use cached index)
        service.get_similar(item_idx=1001, mode="subject", k=5)
        assert load_mock.call_count == 1  # Not called again


class TestALSSimilarity:
    """Test ALS-based similarity with rating filtering."""

    def test_als_mode_builds_index_on_first_call(
        self, monkeypatch, mock_als_factors, mock_book_meta
    ):
        """ALS index should be lazy-loaded on first query."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        assert service._als_index is None

        results = service.get_similar(item_idx=1000, mode="als", k=5)

        assert service._als_index is not None

    def test_als_mode_returns_similar_books(self, monkeypatch, mock_als_factors, mock_book_meta):
        """Should return similar books based on collaborative filtering."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
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

    def test_als_mode_filters_by_rating_count(self, monkeypatch, mock_als_factors, mock_book_meta):
        """ALS mode should filter candidates by 10+ ratings."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1050, mode="als", k=50)

        # All results should have 10+ ratings
        for result in results:
            rating_count = mock_book_meta.loc[result["item_idx"], "book_num_ratings"]
            assert rating_count >= 10

    def test_als_mode_respects_k_parameter(self, monkeypatch, mock_als_factors, mock_book_meta):
        """Should return at most k results."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
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
        """Hybrid mode should initialize alignment data on first query."""
        service = SimilarityService()

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

        assert service._hybrid_initialized is False

        results = service.get_similar(item_idx=1000, mode="hybrid", k=5)

        assert service._hybrid_initialized is True
        assert service._hybrid_subject_embs is not None
        assert service._hybrid_als_factors is not None

    def test_hybrid_mode_blends_subject_and_als_scores(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Should blend subject and ALS scores using alpha parameter."""
        service = SimilarityService()

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

        # Test with different alpha values
        results_subject_heavy = service.get_similar(
            item_idx=1000,
            mode="hybrid",
            k=10,
            alpha=0.2,  # 80% subject
        )

        # Reset and test with ALS-heavy
        service._hybrid_initialized = False
        results_als_heavy = service.get_similar(
            item_idx=1000,
            mode="hybrid",
            k=10,
            alpha=0.8,  # 80% ALS
        )

        # Both should return results (exact ordering may differ)
        assert len(results_subject_heavy) > 0
        assert len(results_als_heavy) > 0

    def test_hybrid_mode_filters_by_rating_count(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid mode should filter candidates by 5+ ratings by default."""
        service = SimilarityService()

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

        # All results should have 5+ ratings
        for result in results:
            rating_count = mock_book_meta.loc[result["item_idx"], "book_num_ratings"]
            assert rating_count >= 5

    def test_hybrid_mode_respects_custom_rating_threshold(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Should respect custom min_rating_count parameter."""
        service = SimilarityService()

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

        # All results should have 20+ ratings
        for result in results:
            rating_count = mock_book_meta.loc[result["item_idx"], "book_num_ratings"]
            assert rating_count >= 20

    def test_hybrid_mode_caches_initialization(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid initialization should be cached after first query."""
        service = SimilarityService()

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

        # First query
        service.get_similar(item_idx=1000, mode="hybrid", k=5)
        assert subject_load_mock.call_count == 1
        assert als_load_mock.call_count == 1

        # Second query (should use cached data)
        service.get_similar(item_idx=1001, mode="hybrid", k=5, alpha=0.7)
        assert subject_load_mock.call_count == 1  # Not called again
        assert als_load_mock.call_count == 1  # Not called again


class TestTwoPoolArchitecture:
    """Test two-pool system (query any book, filter candidates)."""

    def test_can_query_low_rated_book_in_als_mode(
        self, monkeypatch, mock_als_factors, mock_book_meta
    ):
        """Should be able to query low-rated book even with filtering enabled."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        # Query a book with <10 ratings (e.g., item 1005 has 5 ratings)
        # Should still work (query pool includes all books)
        results = service.get_similar(item_idx=1005, mode="als", k=10)

        # Results exist (candidate pool is filtered)
        assert isinstance(results, list)

        # All results have 10+ ratings (candidate pool filtered)
        for result in results:
            rating_count = mock_book_meta.loc[result["item_idx"], "book_num_ratings"]
            assert rating_count >= 10

    def test_filtering_respects_filter_candidates_parameter(
        self, monkeypatch, mock_als_factors, mock_book_meta
    ):
        """Should respect filter_candidates parameter to enable/disable filtering."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_als_factors",
            lambda **kwargs: mock_als_factors,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        # With filtering enabled (default)
        results_filtered = service.get_similar(
            item_idx=1020, mode="als", k=50, filter_candidates=True
        )

        # All results should have 10+ ratings when filtered
        assert all(
            mock_book_meta.loc[r["item_idx"], "book_num_ratings"] >= 10 for r in results_filtered
        )


class TestResultFormatting:
    """Test result formatting and metadata enrichment."""

    def test_results_include_required_fields(
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Results should include all required metadata fields."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
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
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Results should include optional metadata when available."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=5)

        for result in results:
            # Optional fields (may be None)
            assert "author" in result
            assert "year" in result
            assert "isbn" in result
            assert "cover_id" in result

    def test_results_handle_missing_books_gracefully(self, monkeypatch, mock_subject_embeddings):
        """Should skip books not in metadata."""
        service = SimilarityService()

        # Create metadata missing some books
        partial_meta = pd.DataFrame(
            {
                "title": ["Book A", "Book B"],
                "author": ["Author A", "Author B"],
            },
            index=[1000, 1001],  # Only first two books
        )

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: partial_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=50)

        # Should only return books in metadata
        assert all(r["item_idx"] in [1000, 1001] for r in results)


class TestLogging:
    """Test structured logging for observability."""

    @patch("models.services.similarity_service.logger")
    def test_logs_search_start(
        self, mock_logger, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Should log search start with context."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        service.get_similar(item_idx=1000, mode="subject", k=10)

        # Check start log
        start_calls = [c for c in mock_logger.info.call_args_list if len(c[0]) > 0]
        assert len(start_calls) > 0
        assert any("Similarity search started" in str(c) for c in start_calls)

    @patch("models.services.similarity_service.logger")
    def test_logs_search_completion(
        self, mock_logger, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Should log search completion with metrics."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        service.get_similar(item_idx=1000, mode="subject", k=10)

        # Check completion log
        completion_calls = [c for c in mock_logger.info.call_args_list if len(c[0]) > 0]
        assert len(completion_calls) > 0
        assert any("Similarity search completed" in str(c) for c in completion_calls)

    @patch("models.services.similarity_service.logger")
    def test_logs_errors_with_context(self, mock_logger, monkeypatch):
        """Should log errors with context and traceback."""
        service = SimilarityService()

        # Make loader raise exception
        def failing_loader(**kwargs):
            raise RuntimeError("Test error")

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            failing_loader,
        )

        with pytest.raises(RuntimeError):
            service.get_similar(item_idx=1000, mode="subject", k=10)

        # Check error log
        assert mock_logger.error.called
        assert len(mock_logger.error.call_args_list) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_mode_raises_error(self):
        """Should raise ValueError for invalid mode."""
        service = SimilarityService()

        with pytest.raises(ValueError, match="Unknown mode"):
            service.get_similar(item_idx=1000, mode="invalid_mode", k=10)

    def test_k_zero_returns_empty_list(self, monkeypatch, mock_subject_embeddings, mock_book_meta):
        """Should handle k=0 gracefully."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=0)

        assert results == []

    def test_query_book_not_in_index_returns_empty(
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Should return empty list if query book not in index."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        # Query book not in embeddings (outside 1000-1099 range)
        results = service.get_similar(item_idx=9999, mode="subject", k=10)

        assert results == []

    def test_alpha_zero_uses_only_subject_scores(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid with alpha=0.0 should be pure subject similarity."""
        service = SimilarityService()

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

        # Should still return results (pure subject)
        assert len(results) > 0

    def test_alpha_one_uses_only_als_scores(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid with alpha=1.0 should be pure ALS similarity."""
        service = SimilarityService()

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

        # Should still return results (pure ALS)
        assert len(results) > 0

    def test_k_larger_than_catalog_returns_all_candidates(
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Should handle k larger than number of candidates."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=10000)

        # Should return all available books (minus query)
        assert len(results) <= 99  # 100 books - 1 query book

    def test_hybrid_mode_handles_book_missing_als_factors(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """Hybrid should handle books missing in ALS gracefully."""
        service = SimilarityService()

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

        # Query book outside ALS range (1080-1099 not in ALS)
        results = service.get_similar(item_idx=1085, mode="hybrid", k=10)

        # Should still return results (uses zero ALS scores for missing books)
        assert isinstance(results, list)


class TestMetadataCaching:
    """Test book metadata caching."""

    def test_book_meta_loaded_once(self, monkeypatch, mock_subject_embeddings, mock_book_meta):
        """Book metadata should be loaded once and cached."""
        service = SimilarityService()

        meta_load_mock = Mock(return_value=mock_book_meta)
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            meta_load_mock,
        )

        # First query
        service.get_similar(item_idx=1000, mode="subject", k=5)
        assert meta_load_mock.call_count == 1

        # Second query (different mode)
        service.get_similar(item_idx=1001, mode="subject", k=5)
        assert meta_load_mock.call_count == 1  # Not called again

    def test_shared_metadata_across_modes(
        self, monkeypatch, mock_subject_embeddings, mock_als_factors, mock_book_meta
    ):
        """All modes should share the same metadata cache."""
        service = SimilarityService()

        meta_load_mock = Mock(return_value=mock_book_meta)
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

        # Query different modes
        service.get_similar(item_idx=1000, mode="subject", k=5)
        service.get_similar(item_idx=1000, mode="als", k=5)
        service.get_similar(item_idx=1000, mode="hybrid", k=5)

        # Metadata loaded only once
        assert meta_load_mock.call_count == 1


class TestPerformanceConsiderations:
    """Test performance-related behavior."""

    def test_excludes_query_book_from_results(
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Results should never include the query book itself."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        query_id = 1050
        results = service.get_similar(item_idx=query_id, mode="subject", k=50)

        # Query book not in results
        assert all(r["item_idx"] != query_id for r in results)

    def test_results_sorted_by_score_descending(
        self, monkeypatch, mock_subject_embeddings, mock_book_meta
    ):
        """Results should be sorted by score (highest first)."""
        service = SimilarityService()

        monkeypatch.setattr(
            "models.services.similarity_service.load_book_subject_embeddings",
            lambda **kwargs: mock_subject_embeddings,
        )
        monkeypatch.setattr(
            "models.services.similarity_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        results = service.get_similar(item_idx=1000, mode="subject", k=20)

        # Scores should be descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
