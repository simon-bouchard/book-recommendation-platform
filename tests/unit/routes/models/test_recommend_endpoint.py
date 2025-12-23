# tests/unit/routes/models/test_recommend_endpoint.py
"""
Unit tests for GET /profile/recommend endpoint.
Tests request validation, user lookup, service integration, and response formatting.
"""

import pytest
from unittest.mock import Mock

from models.domain.user import User
from models.domain.config import RecommendationConfig
from models.core.constants import PAD_IDX


class TestRecommendEndpointBasics:
    """Test basic endpoint functionality and happy paths."""

    def test_endpoint_returns_200_with_valid_user_id(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should return 200 and recommendations for valid user ID."""
        # Mock database query chain
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        # Mock service
        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        # Mock get_db dependency
        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=123&top_n=10")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3  # From sample_recommendations fixture

    def test_endpoint_calls_service_with_correct_config(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should call RecommendationService with properly configured parameters."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        test_client.get("/profile/recommend?user=123&top_n=50&mode=subject&w=0.7")

        # Assert
        assert mock_recommendation_service.recommend.called
        call_args = mock_recommendation_service.recommend.call_args

        # Check User object
        user_arg = call_args[0][0]
        assert isinstance(user_arg, User)
        assert user_arg.user_id == 123
        assert user_arg.fav_subjects == [5, 12, 23]

        # Check Config object
        config_arg = call_args[0][1]
        assert isinstance(config_arg, RecommendationConfig)
        assert config_arg.k == 50
        assert config_arg.mode == "subject"
        assert config_arg.hybrid_config.subject_weight == 0.7

        # Check db session
        db_arg = call_args[0][2]
        assert db_arg is mock_db


class TestRecommendEndpointUserLookup:
    """Test user lookup logic (by ID vs username)."""

    def test_lookup_by_user_id_when_id_true(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should query by user_id when _id=true."""
        query_mock = Mock()
        filter_mock = Mock()
        filter_mock.first.return_value = mock_orm_user
        query_mock.filter.return_value = filter_mock
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        test_client.get("/profile/recommend?user=123&_id=true")

        # Assert - check filter was called with user_id
        from app.table_models import User as ORMUser

        filter_call = query_mock.filter.call_args[0][0]
        # The filter expression compares ORMUser.user_id == 123
        assert "user_id" in str(filter_call)

    def test_lookup_by_username_when_id_false(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should query by username when _id=false."""
        query_mock = Mock()
        filter_mock = Mock()
        filter_mock.first.return_value = mock_orm_user
        query_mock.filter.return_value = filter_mock
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        test_client.get("/profile/recommend?user=testuser&_id=false")

        # Assert - check filter was called with username
        filter_call = query_mock.filter.call_args[0][0]
        assert "username" in str(filter_call)

    def test_returns_404_when_user_not_found(self, test_client, mock_db, monkeypatch):
        """Should return 404 if user doesn't exist."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = None  # User not found
        mock_db.query.return_value.options.return_value = query_mock

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=99999")

        # Assert
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestRecommendEndpointUserConversion:
    """Test ORM User to domain User conversion."""

    def test_converts_orm_user_with_preferences(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should convert ORM user with favorite subjects to domain User."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        test_client.get("/profile/recommend?user=123")

        # Assert
        user_arg = mock_recommendation_service.recommend.call_args[0][0]
        assert user_arg.user_id == 123
        assert user_arg.fav_subjects == [5, 12, 23]
        assert user_arg.country == "US"
        assert user_arg.age == 30

    def test_converts_orm_user_without_preferences_to_pad_idx(
        self,
        test_client,
        mock_db,
        mock_orm_user_no_preferences,
        mock_recommendation_service,
        monkeypatch,
    ):
        """Should use PAD_IDX when user has no favorite subjects."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user_no_preferences
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        test_client.get("/profile/recommend?user=456")

        # Assert
        user_arg = mock_recommendation_service.recommend.call_args[0][0]
        assert user_arg.fav_subjects == [PAD_IDX]


class TestRecommendEndpointResponseFormat:
    """Test response formatting for backward compatibility."""

    def test_response_is_flat_list_not_nested(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should return flat list, not nested under 'recommendations' key."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=123")
        data = response.json()

        # Assert - should be a list at root level
        assert isinstance(data, list)
        assert "recommendations" not in data  # Not nested

    def test_response_uses_old_field_names(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should use book_avg_rating and book_num_ratings (old field names)."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=123")
        data = response.json()

        # Assert - check field names match old API
        first_book = data[0]
        assert "book_avg_rating" in first_book  # Old name
        assert "book_num_ratings" in first_book  # Old name
        assert "avg_rating" not in first_book  # New name should NOT be present
        assert "num_ratings" not in first_book  # New name should NOT be present

    def test_response_includes_all_required_fields(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should include all fields from old API format."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=123")
        data = response.json()
        first_book = data[0]

        # Assert - all required fields present
        assert "item_idx" in first_book
        assert "title" in first_book
        assert "score" in first_book
        assert "book_avg_rating" in first_book
        assert "book_num_ratings" in first_book
        assert "cover_id" in first_book
        assert "author" in first_book
        assert "year" in first_book
        assert "isbn" in first_book


class TestRecommendEndpointParameterValidation:
    """Test query parameter validation and defaults."""

    def test_default_parameters_when_not_provided(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should use default values when parameters omitted."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act - minimal parameters
        test_client.get("/profile/recommend?user=123")

        # Assert - check defaults
        config_arg = mock_recommendation_service.recommend.call_args[0][1]
        assert config_arg.k == 200  # Default top_n
        assert config_arg.mode == "auto"  # Default mode
        assert config_arg.hybrid_config.subject_weight == 0.6  # Default w

    def test_rejects_invalid_mode(self, test_client, mock_db, mock_orm_user, monkeypatch):
        """Should return 422 for invalid mode."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=123&mode=invalid")

        # Assert
        assert response.status_code == 422

    def test_rejects_top_n_out_of_range(self, test_client, mock_db, mock_orm_user):
        """Should return 422 for top_n out of valid range."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act - top_n too high
        response = test_client.get("/profile/recommend?user=123&top_n=1000")

        # Assert
        assert response.status_code == 422

    def test_rejects_weight_out_of_range(self, test_client, mock_db, mock_orm_user):
        """Should return 422 for w outside [0, 1]."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=123&w=1.5")

        # Assert
        assert response.status_code == 422


class TestRecommendEndpointModes:
    """Test different recommendation modes."""

    def test_auto_mode_passes_through(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should pass mode='auto' to service."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        test_client.get("/profile/recommend?user=123&mode=auto")

        # Assert
        config = mock_recommendation_service.recommend.call_args[0][1]
        assert config.mode == "auto"

    def test_subject_mode_passes_through(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should pass mode='subject' to service."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        test_client.get("/profile/recommend?user=123&mode=subject")

        # Assert
        config = mock_recommendation_service.recommend.call_args[0][1]
        assert config.mode == "subject"

    def test_behavioral_mode_passes_through(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should pass mode='behavioral' to service."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        test_client.get("/profile/recommend?user=123&mode=behavioral")

        # Assert
        config = mock_recommendation_service.recommend.call_args[0][1]
        assert config.mode == "behavioral"


class TestRecommendEndpointErrorHandling:
    """Test error handling and edge cases."""

    def test_returns_500_when_service_raises_exception(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should return 500 if service raises unexpected exception."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        # Make service raise exception
        mock_recommendation_service.recommend.side_effect = RuntimeError("Model crashed")

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=123")

        # Assert
        assert response.status_code == 500

    def test_returns_422_when_service_raises_value_error(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should return 422 if service raises ValueError (validation error)."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        # Make service raise ValueError
        mock_recommendation_service.recommend.side_effect = ValueError("Invalid config")

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=123")

        # Assert
        assert response.status_code == 422
        assert "Invalid config" in response.json()["detail"]

    def test_handles_empty_recommendations_gracefully(
        self, test_client, mock_db, mock_orm_user, mock_recommendation_service, monkeypatch
    ):
        """Should return empty list if service returns no recommendations."""
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_orm_user
        mock_db.query.return_value.options.return_value = query_mock

        # Service returns empty
        mock_recommendation_service.recommend.return_value = []

        monkeypatch.setattr(
            "routes.models.RecommendationService", lambda: mock_recommendation_service
        )

        def override_get_db():
            yield mock_db

        from routes import models as routes_models

        test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

        # Act
        response = test_client.get("/profile/recommend?user=123")

        # Assert
        assert response.status_code == 200
        assert response.json() == []
