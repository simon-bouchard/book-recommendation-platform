# tests/integration/chatbot/tools/conftest.py
"""
Pytest configuration for chatbot tool integration tests.

Provides a session-scoped event loop so that httpx AsyncClient singletons
in the model server client registry are not bound to a loop that closes
between tests. Without this, async tests after the first would fail with
'Event loop is closed' because the registry clients hold a reference to
the loop from the first test.
"""

import asyncio
import pytest
from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def event_loop():
    """
    Session-scoped event loop shared across all async tests in this module.

    pytest-asyncio creates a new loop per test by default, which causes
    httpx AsyncClient singletons to bind to a loop that is then closed.
    A single session-scoped loop keeps those clients alive and reusable
    for the duration of the test session.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def db_session() -> Session:
    """
    Session-scoped database connection shared across all tests.

    Session scope avoids repeated connection setup costs and keeps the
    same connection alive for the full duration of the test run.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
