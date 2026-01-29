# tests/unit/models/infrastructure/test_als_model.py
"""
Unit tests for ALSModel infrastructure component.
Tests singleton pattern, injection, recommendations, and factor retrieval.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.infrastructure.als_model import ALSModel


@pytest.fixture
def mock_als_data():
    """Create mock ALS factors for testing."""
    np.random.seed(42)

    n_users = 10
    n_books = 50
    n_factors = 16

    user_factors = np.random.randn(n_users, n_factors).astype(np.float32)
    book_factors = np.random.randn(n_books, n_factors).astype(np.float32)
    user_ids = list(range(100, 100 + n_users))
    book_ids = list(range(1000, 1000 + n_books))

    return {
        "user_factors": user_factors,
        "book_factors": book_factors,
        "user_ids": user_ids,
        "book_ids": book_ids,
    }


@pytest.fixture
def als_model(mock_als_data):
    """Create ALSModel with injected mock data."""
    model = ALSModel(
        user_factors=mock_als_data["user_factors"],
        book_factors=mock_als_data["book_factors"],
        user_ids=mock_als_data["user_ids"],
        book_ids=mock_als_data["book_ids"],
    )
    yield model
    ALSModel.reset()


class TestALSModelInitialization:
    """Test ALSModel initialization and singleton pattern."""

    def test_injection_bypasses_singleton(self, mock_als_data):
        """Injecting data should create non-singleton instance."""
        ALSModel.reset()

        model1 = ALSModel(
            user_factors=mock_als_data["user_factors"],
            book_factors=mock_als_data["book_factors"],
            user_ids=mock_als_data["user_ids"],
            book_ids=mock_als_data["book_ids"],
        )

        model2 = ALSModel(
            user_factors=mock_als_data["user_factors"],
            book_factors=mock_als_data["book_factors"],
            user_ids=mock_als_data["user_ids"],
            book_ids=mock_als_data["book_ids"],
        )

        assert model1 is not model2

        ALSModel.reset()

    def test_stores_injected_data_correctly(self, als_model, mock_als_data):
        """Model should store injected factors and IDs."""
        assert als_model.user_factors.shape == mock_als_data["user_factors"].shape
        assert als_model.book_factors.shape == mock_als_data["book_factors"].shape
        assert als_model.user_ids == mock_als_data["user_ids"]
        assert als_model.book_ids == mock_als_data["book_ids"]

    def test_creates_correct_mappings(self, als_model, mock_als_data):
        """Model should create bidirectional mappings."""
        for i, user_id in enumerate(mock_als_data["user_ids"]):
            assert als_model.user_id_to_row[user_id] == i

        for i, book_id in enumerate(mock_als_data["book_ids"]):
            assert als_model.book_row_to_id[i] == book_id
            assert als_model.book_id_to_row[book_id] == i


class TestHasUserHasBook:
    """Test has_user and has_book membership checks."""

    def test_has_user_returns_true_for_existing_user(self, als_model, mock_als_data):
        """Should return True for users in the model."""
        user_id = mock_als_data["user_ids"][0]
        assert als_model.has_user(user_id) is True

    def test_has_user_returns_false_for_missing_user(self, als_model):
        """Should return False for users not in the model."""
        assert als_model.has_user(-99999) is False

    def test_has_book_returns_true_for_existing_book(self, als_model, mock_als_data):
        """Should return True for books in the model."""
        book_id = mock_als_data["book_ids"][0]
        assert als_model.has_book(book_id) is True

    def test_has_book_returns_false_for_missing_book(self, als_model):
        """Should return False for books not in the model."""
        assert als_model.has_book(-99999) is False

    def test_has_user_handles_type_coercion(self, als_model, mock_als_data):
        """Should handle int-like types correctly."""
        user_id = mock_als_data["user_ids"][0]
        assert als_model.has_user(str(user_id)) is True
        assert als_model.has_user(float(user_id)) is True


class TestRecommend:
    """Test recommendation generation."""

    def test_recommend_returns_list_of_book_ids(self, als_model, mock_als_data):
        """Should return list of book IDs."""
        user_id = mock_als_data["user_ids"][0]
        recommendations = als_model.recommend(user_id, k=10)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10
        assert all(isinstance(book_id, int) for book_id in recommendations)

    def test_recommend_returns_books_in_model(self, als_model, mock_als_data):
        """Recommended books should be from the model's book set."""
        user_id = mock_als_data["user_ids"][0]
        recommendations = als_model.recommend(user_id, k=10)

        book_id_set = set(mock_als_data["book_ids"])
        assert all(book_id in book_id_set for book_id in recommendations)

    def test_recommend_respects_k_parameter(self, als_model, mock_als_data):
        """Should return at most k recommendations."""
        user_id = mock_als_data["user_ids"][0]

        recs_5 = als_model.recommend(user_id, k=5)
        recs_20 = als_model.recommend(user_id, k=20)

        assert len(recs_5) <= 5
        assert len(recs_20) <= 20

    def test_recommend_returns_empty_for_missing_user(self, als_model):
        """Should return empty list for users not in model."""
        recommendations = als_model.recommend(user_id=-99999, k=10)

        assert recommendations == []

    def test_recommend_returns_different_results_for_different_users(
        self, als_model, mock_als_data
    ):
        """Different users should get different recommendations."""
        user1 = mock_als_data["user_ids"][0]
        user2 = mock_als_data["user_ids"][1]

        recs1 = als_model.recommend(user1, k=10)
        recs2 = als_model.recommend(user2, k=10)

        assert recs1 != recs2

    def test_recommend_is_deterministic(self, als_model, mock_als_data):
        """Same user should get same recommendations."""
        user_id = mock_als_data["user_ids"][0]

        recs1 = als_model.recommend(user_id, k=10)
        recs2 = als_model.recommend(user_id, k=10)

        assert recs1 == recs2

    def test_recommend_handles_k_larger_than_catalog(self, als_model, mock_als_data):
        """Should handle k larger than number of books."""
        user_id = mock_als_data["user_ids"][0]
        n_books = len(mock_als_data["book_ids"])

        recs = als_model.recommend(user_id, k=n_books + 100)

        assert len(recs) == n_books


class TestGetFactors:
    """Test factor retrieval methods."""

    def test_get_user_factors_returns_array(self, als_model, mock_als_data):
        """Should return numpy array of user factors."""
        user_id = mock_als_data["user_ids"][0]
        factors = als_model.get_user_factors(user_id)

        assert isinstance(factors, np.ndarray)
        assert factors.ndim == 1
        assert factors.shape[0] == mock_als_data["user_factors"].shape[1]

    def test_get_user_factors_returns_correct_factors(self, als_model, mock_als_data):
        """Should return correct factors for user."""
        user_id = mock_als_data["user_ids"][0]
        expected = mock_als_data["user_factors"][0]

        factors = als_model.get_user_factors(user_id)

        np.testing.assert_array_almost_equal(factors, expected)

    def test_get_user_factors_returns_none_for_missing_user(self, als_model):
        """Should return None for users not in model."""
        factors = als_model.get_user_factors(-99999)

        assert factors is None

    def test_get_book_factors_returns_array(self, als_model, mock_als_data):
        """Should return numpy array of book factors."""
        book_id = mock_als_data["book_ids"][0]
        factors = als_model.get_book_factors(book_id)

        assert isinstance(factors, np.ndarray)
        assert factors.ndim == 1
        assert factors.shape[0] == mock_als_data["book_factors"].shape[1]

    def test_get_book_factors_returns_correct_factors(self, als_model, mock_als_data):
        """Should return correct factors for book."""
        book_id = mock_als_data["book_ids"][0]
        expected = mock_als_data["book_factors"][0]

        factors = als_model.get_book_factors(book_id)

        np.testing.assert_array_almost_equal(factors, expected)

    def test_get_book_factors_returns_none_for_missing_book(self, als_model):
        """Should return None for books not in model."""
        factors = als_model.get_book_factors(-99999)

        assert factors is None


class TestProperties:
    """Test model property accessors."""

    def test_num_users_returns_correct_count(self, als_model, mock_als_data):
        """Should return number of users in model."""
        assert als_model.num_users == len(mock_als_data["user_ids"])

    def test_num_books_returns_correct_count(self, als_model, mock_als_data):
        """Should return number of books in model."""
        assert als_model.num_books == len(mock_als_data["book_ids"])

    def test_num_factors_returns_correct_dimension(self, als_model, mock_als_data):
        """Should return dimensionality of latent factors."""
        expected_dim = mock_als_data["user_factors"].shape[1]
        assert als_model.num_factors == expected_dim


class TestResetSingleton:
    """Test singleton reset functionality."""

    def test_reset_clears_singleton_instance(self):
        """Reset should clear the singleton instance."""
        ALSModel.reset()

        assert ALSModel._instance is None

    def test_can_create_new_instance_after_reset(self, mock_als_data):
        """Should be able to create new instance after reset."""
        model1 = ALSModel(
            user_factors=mock_als_data["user_factors"],
            book_factors=mock_als_data["book_factors"],
            user_ids=mock_als_data["user_ids"],
            book_ids=mock_als_data["book_ids"],
        )

        ALSModel.reset()

        model2 = ALSModel(
            user_factors=mock_als_data["user_factors"],
            book_factors=mock_als_data["book_factors"],
            user_ids=mock_als_data["user_ids"],
            book_ids=mock_als_data["book_ids"],
        )

        assert model1 is not model2

        ALSModel.reset()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_model_has_no_users_or_books(self):
        """Model with empty data should handle gracefully."""
        model = ALSModel(
            user_factors=np.zeros((0, 16)),
            book_factors=np.zeros((0, 16)),
            user_ids=[],
            book_ids=[],
        )

        assert model.num_users == 0
        assert model.num_books == 0
        assert not model.has_user(123)
        assert not model.has_book(456)
        assert model.recommend(123, k=10) == []

        ALSModel.reset()

    def test_single_user_single_book_model(self):
        """Model with minimal data should work correctly."""
        model = ALSModel(
            user_factors=np.array([[1.0, 2.0, 3.0]]),
            book_factors=np.array([[4.0, 5.0, 6.0]]),
            user_ids=[100],
            book_ids=[200],
        )

        assert model.num_users == 1
        assert model.num_books == 1
        assert model.has_user(100)
        assert model.has_book(200)

        recs = model.recommend(100, k=5)
        assert recs == [200]

        ALSModel.reset()

    def test_recommend_with_k_zero(self, als_model, mock_als_data):
        """Should handle k=0 gracefully."""
        user_id = mock_als_data["user_ids"][0]
        recs = als_model.recommend(user_id, k=0)

        assert recs == []

    def test_recommend_with_negative_k(self, als_model, mock_als_data):
        """Should handle negative k gracefully."""
        user_id = mock_als_data["user_ids"][0]
        recs = als_model.recommend(user_id, k=-5)

        assert recs == []
