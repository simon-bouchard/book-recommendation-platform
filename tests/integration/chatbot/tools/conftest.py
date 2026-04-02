# tests/integration/chatbot/tools/conftest.py
"""
Pytest configuration for chatbot tool integration tests.

db_session (sync, session-scoped): for direct ORM queries in test setup,
e.g. db_session.query(User).filter(...).first(). Also used as the db
argument for sync tools (popular_books, semantic_search, subject_id_search).

Async tools (subject_hybrid_pool, als_recs) no longer require an AsyncSession
— they use the global aiomysql pool initialized by the app lifespan, so no
separate async_db_session fixture is needed.
"""

import pytest
from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def db_session() -> Session:
    """
    Session-scoped synchronous database session for direct ORM queries.

    Used in test setup to look up test users and as the db argument for sync
    tools (popular_books, semantic_search, subject_id_search).
    """
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
