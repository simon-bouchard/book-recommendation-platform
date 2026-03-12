# tests/integration/chatbot/conftest.py
"""
Generic fixtures for ALL chatbot integration tests.
Agent-specific fixtures are in their subdirectories or recsys_fixtures.py.
"""

import pytest
from sqlalchemy.orm import Session

from app.database import SessionLocal


@pytest.fixture(scope="session")
def db_session() -> Session:
    """
    Provide a database session for integration tests.

    Uses actual database connection (requires DATABASE_URL env var).
    Tests should not modify production data.

    Scope is 'session' to reuse connection across all tests in the test session.
    """
    if SessionLocal is None:
        pytest.skip("Database not configured. Set DATABASE_URL environment variable.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
async def reset_model_clients():
    """
    Reset model server client singletons before and after each async test.

    httpx.AsyncClient binds its internal connection pool to the event loop
    that is running when the first request is made. pytest-asyncio runs each
    test function on its own function-scoped event loop, so clients created
    (or first used) during one test hold references to a loop that is closed
    before the next test starts. Reusing those stale clients causes
    'Event loop is closed' errors.

    Resetting the registry to None before each test ensures every test
    creates fresh clients bound to its own live loop. The teardown awaits
    aclose() so connection pool resources are released cleanly rather than
    leaking into the next test's loop.
    """
    import models.client.registry as registry

    registry._embedder = None
    registry._similarity = None
    registry._als = None
    registry._metadata = None

    yield

    await registry.close_all()
