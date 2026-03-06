# tests/unit/routes/models/conftest.py
"""
Shared fixtures for API layer tests.
Provides mocks for services, database, and test client setup.
"""

import sys
import pytest
from unittest.mock import AsyncMock, Mock
from sqlalchemy.orm import Session
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.domain.user import User
from models.domain.recommendation import RecommendedBook


@pytest.fixture
def mock_db():
    """Mock database session."""
    db = Mock(spec=Session)
    db.query = Mock()
    db.close = Mock()
    return db


@pytest.fixture
def mock_orm_user():
    """Mock SQLAlchemy User ORM object."""
    user = Mock()
    user.user_id = 123
    user.username = "testuser"
    user.country = "US"
    user.age = 30
    user.filled_age = "30-35"

    subject_1 = Mock()
    subject_1.subject_idx = 5
    subject_2 = Mock()
    subject_2.subject_idx = 12
    subject_3 = Mock()
    subject_3.subject_idx = 23

    user.favorite_subjects = [subject_1, subject_2, subject_3]

    return user


@pytest.fixture
def mock_orm_user_no_preferences():
    """Mock ORM user with no favorite subjects."""
    user = Mock()
    user.user_id = 456
    user.username = "colduser"
    user.country = "CA"
    user.age = 25
    user.filled_age = "25-30"
    user.favorite_subjects = []
    return user


@pytest.fixture
def sample_domain_user():
    """Sample domain User for testing."""
    return User(user_id=123, fav_subjects=[5, 12, 23], country="US", age=30, filled_age="30-35")


@pytest.fixture
def sample_recommendations():
    """Sample RecommendedBook objects."""
    return [
        RecommendedBook(
            item_idx=1001,
            title="The Great Gatsby",
            score=0.92,
            num_ratings=1500,
            author="F. Scott Fitzgerald",
            year=1925,
            isbn="978-0-123456-78-9",
            cover_id="abc123",
            avg_rating=4.5,
        ),
        RecommendedBook(
            item_idx=1002,
            title="1984",
            score=0.88,
            num_ratings=2000,
            author="George Orwell",
            year=1949,
            isbn="978-0-987654-32-1",
            cover_id="def456",
            avg_rating=4.6,
        ),
        RecommendedBook(
            item_idx=1003,
            title="To Kill a Mockingbird",
            score=0.85,
            num_ratings=1800,
            author="Harper Lee",
            year=1960,
            isbn="978-0-111111-11-1",
            cover_id="ghi789",
            avg_rating=4.7,
        ),
    ]


@pytest.fixture
def sample_similar_books():
    """Sample similarity results."""
    return [
        {
            "item_idx": 2001,
            "title": "Tender Is the Night",
            "score": 0.89,
            "author": "F. Scott Fitzgerald",
            "year": 1934,
            "isbn": "978-0-222222-22-2",
            "cover_id": "jkl012",
        },
        {
            "item_idx": 2002,
            "title": "This Side of Paradise",
            "score": 0.84,
            "author": "F. Scott Fitzgerald",
            "year": 1920,
            "isbn": "978-0-333333-33-3",
            "cover_id": "mno345",
        },
    ]


@pytest.fixture
def mock_recommendation_service(sample_recommendations):
    """Mock RecommendationService with async recommend method."""
    service = Mock()
    service.recommend = AsyncMock(return_value=sample_recommendations)
    return service


@pytest.fixture
def mock_similarity_service(sample_similar_books):
    """Mock SimilarityService with async get_similar method."""
    service = Mock()
    service.get_similar = AsyncMock(return_value=sample_similar_books)
    return service


@pytest.fixture
def test_app(monkeypatch):
    """
    Create a test FastAPI app with response caching neutralised.

    The @cached_recommendations and @cached_similarity decorators are applied
    to the route handlers at module import time.  If routes.models is already
    in sys.modules the cached wrappers are reused across tests, meaning the
    real handler is bypassed after the first request with a given parameter
    combination and mock call assertions fail.

    Fix: patch both decorators to identity functions in the cache module, then
    remove routes.models from sys.modules so the next import re-executes the
    module body with the inert decorators.  monkeypatch restores both the
    decorator patch and the sys.modules entry on teardown, leaving no state
    between tests.
    """
    import models.cache as cache_module

    monkeypatch.setattr(cache_module, "cached_recommendations", lambda f: f)
    monkeypatch.setattr(cache_module, "cached_similarity", lambda f: f)

    # Force re-execution of routes.models with the patched decorators.
    monkeypatch.delitem(sys.modules, "routes.models", raising=False)

    from fastapi import FastAPI
    import routes.models as routes_module

    app = FastAPI()
    app.include_router(routes_module.router)

    return app


@pytest.fixture
def test_client(test_app):
    """FastAPI test client."""
    from fastapi.testclient import TestClient

    return TestClient(test_app)
