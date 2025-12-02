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
