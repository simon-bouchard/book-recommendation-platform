# tests/unit/models/services/test_recommendation_service.py
"""
Unit tests for RecommendationService.
Tests strategy selection, pipeline building, candidate enrichment, and logging.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.services.recommendation_service import RecommendationService
from models.domain.user import User
from models.domain.config import RecommendationConfig, HybridConfig
from models.domain.recommendation import Candidate, RecommendedBook
from models.core.constants import PAD_IDX


@pytest.fixture
def mock_book_meta():
    """Create mock book metadata DataFrame."""
    data = {
        "title": ["Book A", "Book B", "Book C"],
        "author": ["Author 1", "Author 2", "Author 3"],
        "year": [2020, 2021, 2022],
        "isbn": ["111", "222", "333"],
        "cover_id": ["aaa", "bbb", "ccc"],
        "book_num_ratings": [100, 200, 50],
        "book_avg_rating": [4.5, 4.2, 3.9],
    }
    df = pd.DataFrame(data, index=[1000, 1001, 1002])
    df.index.name = "item_idx"
    return df


@pytest.fixture
def mock_candidates():
    """Create mock candidates for testing."""
    return [
        Candidate(item_idx=1000, score=0.9, source="test"),
        Candidate(item_idx=1001, score=0.8, source="test"),
        Candidate(item_idx=1002, score=0.7, source="test"),
    ]


@pytest.fixture
def warm_user():
    """Create warm user (has ALS factors)."""
    user = User(user_id=123, fav_subjects=[5, 12, 23])
    return user


@pytest.fixture
def cold_user_with_prefs():
    """Create cold user with preferences."""
    user = User(user_id=456, fav_subjects=[5, 12, 23])
    return user


@pytest.fixture
def cold_user_no_prefs():
    """Create cold user without preferences."""
    user = User(user_id=789, fav_subjects=[PAD_IDX])
    return user


@pytest.fixture
def mock_db():
    """Create mock database session."""
    return Mock()


class TestRecommendationServiceInitialization:
    """Test RecommendationService initialization."""

    def test_service_initializes_successfully(self):
        """Should initialize without errors."""
        service = RecommendationService()
        assert service is not None

    def test_book_meta_starts_as_none(self):
        """Book metadata should be lazily loaded."""
        service = RecommendationService()
        assert service._book_meta is None


class TestStrategySelection:
    """Test automatic strategy selection based on user type."""

    def test_warm_user_gets_als_pipeline(self, warm_user, mock_db, monkeypatch):
        """Warm user should use ALS-based pipeline."""
        # Mock ALSModel to return True for has_user
        mock_als_model = Mock()
        mock_als_model.has_user.return_value = True

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch("models.services.recommendation_service.load_book_meta") as mock_load_meta:
                mock_load_meta.return_value = pd.DataFrame(
                    {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
                )

                # Mock pipeline to return empty (we're just testing pipeline selection)
                with patch(
                    "models.services.recommendation_service.RecommendationPipeline"
                ) as mock_pipeline_class:
                    mock_pipeline = Mock()
                    mock_pipeline.recommend.return_value = []
                    mock_pipeline_class.return_value = mock_pipeline

                    service = RecommendationService()
                    config = RecommendationConfig.default()

                    service.recommend(warm_user, config, mock_db)

                    # Verify ALSBasedGenerator was used
                    call_args = mock_pipeline_class.call_args
                    generator = call_args[1]["generator"]
                    assert generator.__class__.__name__ == "ALSBasedGenerator"

    def test_cold_user_with_prefs_gets_hybrid_pipeline(
        self, cold_user_with_prefs, mock_db, monkeypatch
    ):
        """Cold user with preferences should use hybrid pipeline."""
        # Mock ALSModel to return False for has_user
        mock_als_model = Mock()
        mock_als_model.has_user.return_value = False

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch("models.services.recommendation_service.load_book_meta") as mock_load_meta:
                mock_load_meta.return_value = pd.DataFrame(
                    {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
                )

                with patch(
                    "models.services.recommendation_service.RecommendationPipeline"
                ) as mock_pipeline_class:
                    mock_pipeline = Mock()
                    mock_pipeline.recommend.return_value = []
                    mock_pipeline_class.return_value = mock_pipeline

                    service = RecommendationService()
                    config = RecommendationConfig.default()

                    service.recommend(cold_user_with_prefs, config, mock_db)

                    # Verify HybridGenerator was used
                    call_args = mock_pipeline_class.call_args
                    generator = call_args[1]["generator"]
                    assert generator.__class__.__name__ == "HybridGenerator"

    def test_cold_user_no_prefs_gets_popularity_pipeline(
        self, cold_user_no_prefs, mock_db, monkeypatch
    ):
        """Cold user without preferences should use popularity fallback."""
        # Mock ALSModel to return False
        mock_als_model = Mock()
        mock_als_model.has_user.return_value = False

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch("models.services.recommendation_service.load_book_meta") as mock_load_meta:
                mock_load_meta.return_value = pd.DataFrame(
                    {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
                )

                with patch(
                    "models.services.recommendation_service.RecommendationPipeline"
                ) as mock_pipeline_class:
                    mock_pipeline = Mock()
                    mock_pipeline.recommend.return_value = []
                    mock_pipeline_class.return_value = mock_pipeline

                    service = RecommendationService()
                    config = RecommendationConfig.default()

                    service.recommend(cold_user_no_prefs, config, mock_db)

                    # Verify BayesianPopularityGenerator was used
                    call_args = mock_pipeline_class.call_args
                    generator = call_args[1]["generator"]
                    assert generator.__class__.__name__ == "BayesianPopularityGenerator"

    def test_behavioral_mode_forces_als(self, cold_user_with_prefs, mock_db):
        """Behavioral mode should force ALS even for cold users."""
        mock_als_model = Mock()
        mock_als_model.has_user.return_value = False

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch("models.services.recommendation_service.load_book_meta") as mock_load_meta:
                mock_load_meta.return_value = pd.DataFrame(
                    {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
                )

                with patch(
                    "models.services.recommendation_service.RecommendationPipeline"
                ) as mock_pipeline_class:
                    mock_pipeline = Mock()
                    mock_pipeline.recommend.return_value = []
                    mock_pipeline_class.return_value = mock_pipeline

                    service = RecommendationService()
                    config = RecommendationConfig(k=20, mode="behavioral")

                    service.recommend(cold_user_with_prefs, config, mock_db)

                    # Should use ALS despite being cold user
                    call_args = mock_pipeline_class.call_args
                    generator = call_args[1]["generator"]
                    assert generator.__class__.__name__ == "ALSBasedGenerator"

    def test_subject_mode_forces_hybrid(self, warm_user, mock_db):
        """Subject mode should force hybrid even for warm users."""
        mock_als_model = Mock()
        mock_als_model.has_user.return_value = True

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch("models.services.recommendation_service.load_book_meta") as mock_load_meta:
                mock_load_meta.return_value = pd.DataFrame(
                    {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
                )

                with patch(
                    "models.services.recommendation_service.RecommendationPipeline"
                ) as mock_pipeline_class:
                    mock_pipeline = Mock()
                    mock_pipeline.recommend.return_value = []
                    mock_pipeline_class.return_value = mock_pipeline

                    service = RecommendationService()
                    config = RecommendationConfig(k=20, mode="subject")

                    service.recommend(warm_user, config, mock_db)

                    # Should use hybrid despite being warm user
                    call_args = mock_pipeline_class.call_args
                    generator = call_args[1]["generator"]
                    assert generator.__class__.__name__ == "HybridGenerator"


class TestCandidateEnrichment:
    """Test candidate enrichment with book metadata."""

    def test_enriches_candidates_with_metadata(self, mock_candidates, mock_book_meta, monkeypatch):
        """Should add metadata from book_meta to candidates."""
        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        service = RecommendationService()
        recommendations = service._enrich_candidates(mock_candidates)

        assert len(recommendations) == 3
        assert all(isinstance(r, RecommendedBook) for r in recommendations)

        # Check first recommendation
        assert recommendations[0].item_idx == 1000
        assert recommendations[0].title == "Book A"
        assert recommendations[0].author == "Author 1"
        assert recommendations[0].year == 2020
        assert recommendations[0].isbn == "111"
        assert recommendations[0].cover_id == "aaa"
        assert recommendations[0].score == 0.9
        assert recommendations[0].num_ratings == 100
        assert recommendations[0].avg_rating == 4.5

    def test_skips_candidates_not_in_metadata(self, mock_book_meta, monkeypatch):
        """Should skip candidates missing from metadata."""
        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        candidates = [
            Candidate(item_idx=1000, score=0.9, source="test"),
            Candidate(item_idx=9999, score=0.8, source="test"),  # Not in metadata
            Candidate(item_idx=1001, score=0.7, source="test"),
        ]

        service = RecommendationService()
        recommendations = service._enrich_candidates(candidates)

        # Should skip 9999
        assert len(recommendations) == 2
        assert recommendations[0].item_idx == 1000
        assert recommendations[1].item_idx == 1001

    def test_handles_missing_optional_fields(self, monkeypatch):
        """Should handle books with missing optional metadata."""
        incomplete_meta = pd.DataFrame(
            {
                "title": ["Book A"],
                "book_num_ratings": [10],
                # Missing: author, year, isbn, cover_id, book_avg_rating
            },
            index=[1000],
        )

        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: incomplete_meta,
        )

        candidates = [Candidate(item_idx=1000, score=0.9, source="test")]

        service = RecommendationService()
        recommendations = service._enrich_candidates(candidates)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.title == "Book A"
        assert rec.author is None
        assert rec.year is None
        assert rec.isbn is None
        assert rec.cover_id is None
        assert rec.avg_rating is None

    def test_handles_empty_candidate_list(self, mock_book_meta, monkeypatch):
        """Should handle empty candidate list."""
        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: mock_book_meta,
        )

        service = RecommendationService()
        recommendations = service._enrich_candidates([])

        assert recommendations == []

    def test_caches_book_metadata(self, mock_book_meta, monkeypatch):
        """Should cache book metadata after first load."""
        load_count = [0]

        def mock_load(**kwargs):
            load_count[0] += 1
            return mock_book_meta

        monkeypatch.setattr("models.services.recommendation_service.load_book_meta", mock_load)

        service = RecommendationService()
        candidates = [Candidate(item_idx=1000, score=0.9, source="test")]

        # First enrichment
        service._enrich_candidates(candidates)
        assert load_count[0] == 1

        # Second enrichment should use cache
        service._enrich_candidates(candidates)
        assert load_count[0] == 1  # Still 1, not 2


class TestLogging:
    """Test structured logging for observability."""

    def test_logs_recommendation_start(self, warm_user, mock_db, monkeypatch):
        """Should log when recommendation starts."""
        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: pd.DataFrame(
                {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
            ),
        )

        mock_als_model = Mock()
        mock_als_model.has_user.return_value = True

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch(
                "models.services.recommendation_service.RecommendationPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline.recommend.return_value = []
                mock_pipeline_class.return_value = mock_pipeline

                with patch("models.services.recommendation_service.logger") as mock_logger:
                    service = RecommendationService()
                    config = RecommendationConfig.default()

                    service.recommend(warm_user, config, mock_db)

                    # Check start log
                    mock_logger.info.assert_any_call(
                        "Recommendation started",
                        extra={
                            "user_id": 123,
                            "mode": "auto",
                            "is_warm": True,
                            "has_preferences": True,
                            "k": 200,
                        },
                    )

    def test_logs_recommendation_completion(self, warm_user, mock_db, monkeypatch):
        """Should log when recommendation completes successfully."""
        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: pd.DataFrame(
                {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
            ),
        )

        mock_als_model = Mock()
        mock_als_model.has_user.return_value = True

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch(
                "models.services.recommendation_service.RecommendationPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline.recommend.return_value = []
                mock_pipeline_class.return_value = mock_pipeline

                with patch("models.services.recommendation_service.logger") as mock_logger:
                    service = RecommendationService()
                    config = RecommendationConfig.default()

                    service.recommend(warm_user, config, mock_db)

                    # Check completion log
                    completion_calls = [
                        call
                        for call in mock_logger.info.call_args_list
                        if "Recommendation completed" in str(call)
                    ]
                    assert len(completion_calls) == 1

                    # Verify it includes latency_ms
                    extra = completion_calls[0][1]["extra"]
                    assert "latency_ms" in extra
                    assert "count" in extra
                    assert extra["user_id"] == 123

    def test_logs_errors_with_context(self, warm_user, mock_db, monkeypatch):
        """Should log errors with context and traceback."""
        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: pd.DataFrame(
                {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
            ),
        )

        mock_als_model = Mock()
        mock_als_model.has_user.return_value = True

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch(
                "models.services.recommendation_service.RecommendationPipeline"
            ) as mock_pipeline_class:
                # Make pipeline raise an exception
                mock_pipeline = Mock()
                mock_pipeline.recommend.side_effect = ValueError("Test error")
                mock_pipeline_class.return_value = mock_pipeline

                with patch("models.services.recommendation_service.logger") as mock_logger:
                    service = RecommendationService()
                    config = RecommendationConfig.default()

                    with pytest.raises(ValueError):
                        service.recommend(warm_user, config, mock_db)

                    # Check error log
                    mock_logger.error.assert_called_once()
                    call_args = mock_logger.error.call_args

                    assert "Recommendation failed" in call_args[0][0]
                    assert call_args[1]["extra"]["user_id"] == 123
                    assert "Test error" in call_args[1]["extra"]["error"]
                    assert call_args[1]["exc_info"] is True


class TestEndToEndFlow:
    """Test complete recommendation flow."""

    def test_complete_recommendation_flow(self, warm_user, mock_db, monkeypatch):
        """Test complete flow from request to enriched results."""
        # Mock book metadata
        mock_meta = pd.DataFrame(
            {
                "title": ["Book A", "Book B"],
                "author": ["Author 1", "Author 2"],
                "year": [2020, 2021],
                "isbn": ["111", "222"],
                "cover_id": ["aaa", "bbb"],
                "book_num_ratings": [100, 200],
                "book_avg_rating": [4.5, 4.2],
            },
            index=[1000, 1001],
        )

        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: mock_meta,
        )

        # Mock ALSModel
        mock_als_model = Mock()
        mock_als_model.has_user.return_value = True

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            # Mock pipeline to return candidates
            with patch(
                "models.services.recommendation_service.RecommendationPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline.recommend.return_value = [
                    Candidate(item_idx=1000, score=0.9, source="als"),
                    Candidate(item_idx=1001, score=0.8, source="als"),
                ]
                mock_pipeline_class.return_value = mock_pipeline

                service = RecommendationService()
                config = RecommendationConfig(k=10, mode="auto")

                results = service.recommend(warm_user, config, mock_db)

                # Verify results
                assert len(results) == 2
                assert all(isinstance(r, RecommendedBook) for r in results)

                # Check first result
                assert results[0].item_idx == 1000
                assert results[0].title == "Book A"
                assert results[0].score == 0.9


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_invalid_config(self, warm_user, mock_db):
        """Should raise error for invalid config."""
        with pytest.raises(ValueError):
            config = RecommendationConfig(k=0, mode="auto")  # k must be >= 1

    def test_handles_invalid_mode(self, warm_user, mock_db):
        """Should raise error for invalid mode."""
        with pytest.raises(ValueError):
            config = RecommendationConfig(k=10, mode="invalid")

    def test_handles_empty_pipeline_results(self, warm_user, mock_db, monkeypatch):
        """Should handle pipeline returning no candidates."""
        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: pd.DataFrame(
                {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
            ),
        )

        mock_als_model = Mock()
        mock_als_model.has_user.return_value = True

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch(
                "models.services.recommendation_service.RecommendationPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline.recommend.return_value = []  # Empty results
                mock_pipeline_class.return_value = mock_pipeline

                service = RecommendationService()
                config = RecommendationConfig.default()

                results = service.recommend(warm_user, config, mock_db)

                assert results == []

    def test_respects_k_parameter(self, warm_user, mock_db, monkeypatch):
        """Should respect k parameter when requesting recommendations."""
        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: pd.DataFrame(
                {"title": [f"Book {i}"], "book_num_ratings": [10]} for i in range(100)
            ),
        )

        mock_als_model = Mock()
        mock_als_model.has_user.return_value = True

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch(
                "models.services.recommendation_service.RecommendationPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline.recommend.return_value = []  # Add return value
                mock_pipeline_class.return_value = mock_pipeline  # Add this line!

                service = RecommendationService()
                config = RecommendationConfig(k=20, mode="auto")

                service.recommend(warm_user, config, mock_db)

                # Verify pipeline was called with correct k
                call_args = mock_pipeline.recommend.call_args
                assert call_args[0][1] == 20  # k parameter

    def test_passes_db_session_to_pipeline(self, warm_user, mock_db, monkeypatch):
        """Should pass database session to pipeline for filtering."""
        monkeypatch.setattr(
            "models.services.recommendation_service.load_book_meta",
            lambda **kwargs: pd.DataFrame(
                {"title": ["Test"], "book_num_ratings": [10]}, index=[1000]
            ),
        )

        mock_als_model = Mock()
        mock_als_model.has_user.return_value = True

        with patch("models.infrastructure.als_model.ALSModel", return_value=mock_als_model):
            with patch(
                "models.services.recommendation_service.RecommendationPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline.recommend.return_value = []
                mock_pipeline_class.return_value = mock_pipeline

                service = RecommendationService()
                config = RecommendationConfig.default()

                service.recommend(warm_user, config, mock_db)

                # Verify db was passed to pipeline
                call_args = mock_pipeline.recommend.call_args
                assert call_args[0][2] == mock_db  # db parameter
