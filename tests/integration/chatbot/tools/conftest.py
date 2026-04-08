# tests/integration/chatbot/tools/conftest.py
"""
Pytest configuration for chatbot tool integration tests.

db_session (sync, session-scoped): for direct ORM queries in test setup,
e.g. db_session.query(User).filter(...).first(). Also used as the db
argument for sync tools (popular_books, semantic_search, subject_id_search).

Async tools (subject_hybrid_pool, als_recs) use the global aiomysql pool
directly. aiomysql_pool initializes it once for the test session and tears
it down afterwards.
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


@pytest.fixture(autouse=True)
async def aiomysql_pool():
    """
    Initialize the global aiomysql pool for each async test.

    Async tools (subject_hybrid_pool, als_recs) call get_aiomysql_pool()
    directly instead of using a passed-in session. The pool is created and
    destroyed per test so it is always bound to the test's own event loop
    (pytest-asyncio creates a new loop per test function by default).
    """
    from app.database import close_aiomysql_pool, init_aiomysql_pool

    await init_aiomysql_pool()
    yield
    await close_aiomysql_pool()
