# tests/unit/routes/models/conftest.py
"""
Shared fixtures for the /profile/recommend endpoint tests.

Key design decisions
--------------------
- mock_fetch_user is an autouse AsyncMock that patches _fetch_user_and_subjects,
  the raw-aiomysql helper the route calls to load user + subjects. Tests that
  need a different result (e.g. user not found) override its return_value.

- test_client is session-scoped; the mock_fetch_user patch is function-scoped
  so each test gets a fresh mock without cross-test pollution.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

import os

from models.domain.recommendation import RecommendedBook
from models.core.constants import PAD_IDX


os.environ["SECURE_MODE"] = "false"


# ---------------------------------------------------------------------------
# Application client
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def disable_cache():
    """
    Disable Redis caching for all route tests.

    Both @cached_recommendations and @cached_similarity check client.available
    and fall through to the real function when False. Patching both import
    sites prevents any test from polluting the cache seen by a later test.
    """
    unavailable = MagicMock()
    unavailable.available = False

    with (
        patch("models.cache.decorators.get_redis_client", return_value=unavailable),
        patch("app.cache.decorators.get_redis_client", return_value=unavailable),
    ):
        yield


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """
    TestClient wrapping the main FastAPI application.

    Session-scoped so the app is only instantiated once.
    """
    from main import app

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# User fetch mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_fetch_user() -> AsyncMock:
    """
    Patch _fetch_user_and_subjects for every test.

    Returns the default user tuple (user_id=123, country="US", age=30,
    filled_age=None, fav_subjects=[5, 12, 23]). Tests that need a different
    result override mock_fetch_user.return_value before making the request.
    """
    default = (123, "US", 30, None, [5, 12, 23])
    with patch(
        "routes.models._fetch_user_and_subjects",
        new_callable=AsyncMock,
        return_value=default,
    ) as m:
        yield m


# ---------------------------------------------------------------------------
# User data variants
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user_no_preferences(mock_fetch_user: AsyncMock) -> AsyncMock:
    """Override mock_fetch_user to return a user with no subject preferences."""
    mock_fetch_user.return_value = (456, "US", 25, None, [])
    return mock_fetch_user


# ---------------------------------------------------------------------------
# Sample recommendations
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_recommendations() -> list[RecommendedBook]:
    """
    Three fully-populated RecommendedBook instances.

    Used as the default return value of mock_recommendation_service so that
    response-format and field-name assertions have concrete data to inspect.
    All optional fields are populated so `assert field in first_book` tests pass.
    """
    return [
        RecommendedBook(
            item_idx=1000,
            title="Book One",
            score=0.95,
            num_ratings=500,
            author="Author One",
            year=2019,
            isbn="978-0-00-000001-0",
            cover_id="OL001M",
            avg_rating=4.2,
        ),
        RecommendedBook(
            item_idx=1001,
            title="Book Two",
            score=0.88,
            num_ratings=300,
            author="Author Two",
            year=2020,
            isbn="978-0-00-000002-0",
            cover_id="OL002M",
            avg_rating=3.9,
        ),
        RecommendedBook(
            item_idx=1002,
            title="Book Three",
            score=0.75,
            num_ratings=150,
            author="Author Three",
            year=2021,
            isbn="978-0-00-000003-0",
            cover_id="OL003M",
            avg_rating=4.5,
        ),
    ]


# ---------------------------------------------------------------------------
# Similarity fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_similar_books() -> list[dict]:
    """
    Two similar-book dicts matching the format returned by SimilarityService.get_similar().

    All fields the response-format tests assert on are populated.
    """
    return [
        {
            "item_idx": 2000,
            "title": "Similar Book One",
            "score": 0.92,
            "author": "Author A",
            "year": 2018,
            "isbn": "978-0-00-002000-0",
            "cover_id": "OL2000M",
        },
        {
            "item_idx": 2001,
            "title": "Similar Book Two",
            "score": 0.81,
            "author": "Author B",
            "year": 2017,
            "isbn": "978-0-00-002001-0",
            "cover_id": "OL2001M",
        },
    ]


@pytest.fixture
def mock_similarity_service(sample_similar_books: list[dict]) -> MagicMock:
    """
    Mock SimilarityService with an async get_similar method.

    get_similar must be AsyncMock because the endpoint handler awaits it.
    Returns sample_similar_books by default; individual tests can override
    return_value or set side_effect as needed.
    """
    service = MagicMock()
    service.get_similar = AsyncMock(return_value=sample_similar_books)
    return service


# ---------------------------------------------------------------------------
# Recommendation service mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_recommendation_service(sample_recommendations: list[RecommendedBook]) -> MagicMock:
    """
    Mock RecommendationService with an async recommend method.

    recommend must be AsyncMock because the endpoint handler awaits it.
    Returns sample_recommendations by default; individual tests can override
    return_value or set side_effect as needed.
    """
    service = MagicMock()
    service.recommend = AsyncMock(return_value=sample_recommendations)
    return service
