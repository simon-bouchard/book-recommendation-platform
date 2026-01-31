# tests/unit/chatbot/tools/recsys/conftest.py
"""
Recsys-specific fixtures for tool testing.
Provides mocked recommendation components and services.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


@pytest.fixture
def mock_semantic_searcher(sample_semantic_search_raw):
    """
    Mock SemanticSearcher with realistic results.

    Returns a searcher that produces results with nested meta fields,
    matching the actual SemanticSearcher interface.
    """
    searcher = Mock()
    searcher.search.return_value = sample_semantic_search_raw
    return searcher


@pytest.fixture
def mock_recommendation_service():
    """
    Mock RecommendationService for ALS, subject hybrid, and popular books tests.

    Returns a service that produces RecommendedBook-like objects with proper attributes.
    """

    def create_mock_recommended_book(item_idx, title, author, year, score, num_ratings):
        """Helper to create a mock RecommendedBook with proper attributes."""
        book = Mock()
        book.item_idx = item_idx
        book.title = title
        book.author = author
        book.year = year
        book.score = score
        book.num_ratings = num_ratings
        return book

    service = Mock()

    # Default return value - list of mock RecommendedBook objects
    # IMPORTANT: Return as a real list (not Mock) so iteration works properly
    service.recommend.return_value = [
        create_mock_recommended_book(1, "The Great Gatsby", "F. Scott Fitzgerald", 1925, 0.95, 100),
        create_mock_recommended_book(2, "1984", "George Orwell", 1949, 0.92, 200),
    ]

    return service


@pytest.fixture
def mock_load_book_meta(mock_book_meta):
    """
    Mock the load_book_meta function.

    Patches models.data.loaders.load_book_meta to return test data.
    """
    with patch("app.agents.tools.recsys.native_tools.load_book_meta") as mock:
        mock.return_value = mock_book_meta
        yield mock


@pytest.fixture
def mock_load_bayesian_scores():
    """
    Mock load_bayesian_scores for popular_books tool.

    Returns precomputed Bayesian average scores as a dict.
    """
    with patch("app.agents.tools.recsys.native_tools.load_bayesian_scores") as mock:
        # Return as dict mapping item_idx -> score
        mock.return_value = {
            1: 4.5,
            2: 4.3,
            3: 4.1,
        }
        yield mock


@pytest.fixture
def mock_settings_embedder():
    """
    Mock the settings.embedder for semantic search.

    Prevents actual embedding model loading during tests.
    """
    with patch("app.agents.tools.recsys.native_tools.settings") as mock_settings:
        mock_embedder = Mock()
        mock_settings.embedder = mock_embedder
        yield mock_settings


@pytest.fixture
def mock_semantic_searcher_class(mock_semantic_searcher):
    """
    Mock the SemanticSearcher class constructor.

    Prevents file system access and FAISS index loading.
    """
    with patch("app.agents.tools.recsys.native_tools.SemanticSearcher") as mock_class:
        mock_class.return_value = mock_semantic_searcher
        yield mock_class


@pytest.fixture
def mock_get_all_subject_counts():
    """
    Mock get_all_subject_counts for subject_id_search.

    Returns sample subject data for TF-IDF index building.
    This function is imported from app.models in the subject_search module.
    """
    with patch("app.models.get_all_subject_counts") as mock:
        mock.return_value = [
            {"subject_idx": 1, "subject": "Science Fiction", "count": 1000},
            {"subject_idx": 2, "subject": "Fantasy", "count": 800},
            {"subject_idx": 3, "subject": "Mystery", "count": 600},
            {"subject_idx": 4, "subject": "Romance", "count": 900},
            {"subject_idx": 5, "subject": "Historical Fiction", "count": 500},
        ]
        yield mock


@pytest.fixture
def internal_tools_factory(mock_user, mock_db_session):
    """
    Factory fixture for creating InternalTools instances with mocked dependencies.

    Usage:
        tools = internal_tools_factory(allow_profile=True)
        tools = internal_tools_factory(current_user=None)  # Explicitly no user
    """

    def _create(
        current_user="USE_DEFAULT",
        db="USE_DEFAULT",
        user_num_ratings=0,
        allow_profile=False,
    ):
        from app.agents.tools.recsys.native_tools import InternalTools

        # Use sentinel values to distinguish between "not provided" and "explicitly None"
        actual_user = mock_user if current_user == "USE_DEFAULT" else current_user
        actual_db = mock_db_session if db == "USE_DEFAULT" else db

        return InternalTools(
            current_user=actual_user,
            db=actual_db,
            user_num_ratings=user_num_ratings,
            allow_profile=allow_profile,
        )

    return _create
