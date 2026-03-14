# tests/unit/routes/models/conftest.py
"""
Shared fixtures for the /profile/recommend endpoint tests.

Key design decisions
--------------------
- mock_db is an AsyncMock whose execute() returns a chain matching the
  SQLAlchemy Core pattern used in the route:
    result = await db.execute(stmt)
    user_obj = result.unique().scalar_one_or_none()

- mock_orm_user exposes a `favorite_subjects` list of mocks with a
  `subject_idx` attribute, matching the relationship the route iterates:
    fav_subjects = [s.subject_idx for s in user_obj.favorite_subjects]

- The `override_db_dependency` autouse fixture injects mock_db as the
  get_async_read_only_db dependency for every test and cleans up afterward,
  so individual tests do not need to repeat the dependency_overrides pattern.

- test_client is session-scoped; dependency_overrides is a plain dict that
  can be mutated per test without recreating the app.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

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

    Session-scoped so the app is only instantiated once. Tests override
    dependencies via app.dependency_overrides, which the autouse fixture
    manages per test.
    """
    from main import app

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Database session mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db(mock_orm_user: MagicMock) -> AsyncMock:
    """
    AsyncMock standing in for an async SQLAlchemy Session.

    Pre-wired to return mock_orm_user through the execute/unique/
    scalar_one_or_none chain the route uses. Tests that need a different
    result (e.g. user not found) can override:
        mock_db.execute.return_value.unique.return_value.scalar_one_or_none.return_value = None
    """
    db = AsyncMock()
    result = Mock()
    result.unique.return_value.scalar_one_or_none.return_value = mock_orm_user
    db.execute.return_value = result
    return db


# ---------------------------------------------------------------------------
# Autouse dependency override
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def override_db_dependency(test_client: TestClient, mock_db: AsyncMock):
    """
    Replace get_async_read_only_db with mock_db for every test.

    Using autouse eliminates the repeated dependency_overrides boilerplate
    from every test body. Cleanup removes the override after each test so
    session-scoped test_client is not polluted between test classes.
    """
    from app.database import get_async_read_only_db

    async def _override():
        yield mock_db

    test_client.app.dependency_overrides[get_async_read_only_db] = _override
    yield
    test_client.app.dependency_overrides.pop(get_async_read_only_db, None)


# ---------------------------------------------------------------------------
# ORM user mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_orm_user() -> MagicMock:
    """
    ORM User mock with fully populated profile.

    favorite_subjects is a list of mocks with subject_idx attributes, matching
    the relationship the route iterates:
        fav_subjects = [s.subject_idx for s in user_obj.favorite_subjects]

    country, age, and filled_age match the assertions in
    TestRecommendEndpointUserConversion.
    """
    user = MagicMock()
    user.user_id = 123
    user.username = "testuser"
    user.favorite_subjects = [
        Mock(subject_idx=5),
        Mock(subject_idx=12),
        Mock(subject_idx=23),
    ]
    user.country = "US"
    user.age = 30
    user.filled_age = None
    return user


@pytest.fixture
def mock_orm_user_no_preferences() -> MagicMock:
    """
    ORM User mock with no subject preferences.

    favorite_subjects is empty, causing the route to fall back to [PAD_IDX]
    and exercising the cold-start path.
    """
    user = MagicMock()
    user.user_id = 456
    user.username = "colduser"
    user.favorite_subjects = []
    user.country = "US"
    user.age = 25
    user.filled_age = None
    return user


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
