# tests/integration/chatbot/tools/conftest.py
"""
Pytest configuration for chatbot tool integration tests.

Two database session fixtures are provided:

db_session (sync, session-scoped): for direct ORM queries in test setup,
e.g. db_session.query(User).filter(...).first(). Also used as the db
argument for sync tools (popular_books, semantic_search, subject_id_search).

async_db_session (async, function-scoped): for async tools only
(subject_hybrid_pool, als_recs). These tools call service.recommend(), which
routes through ReadBooksFilter -> get_read_books_for_candidates_async() ->
await db.execute(stmt), which requires an AsyncSession.

Event loop and greenlet interaction
------------------------------------
The parent conftest gives each test its own function-scoped event loop.
aiomysql calls asyncio.get_event_loop() (not get_running_loop()) during
connection establishment inside SQLAlchemy's greenlet bridge. A greenlet
runs synchronously on the thread — it does not execute as a coroutine frame
— so asyncio does not automatically supply the running loop. Instead,
get_event_loop() returns whatever loop is registered as the thread's current
loop via set_event_loop(). Between tests, pytest-asyncio closes the old loop
and creates a new one. If set_event_loop() is not called before the new loop
starts running (which varies by pytest-asyncio version and fixture ordering),
get_event_loop() returns the old closed loop. aiomysql then stores that stale
loop as self._loop, and any subsequent Future created on it conflicts with
the running loop, producing 'Future attached to a different loop'.

The fix is asyncio.set_event_loop(asyncio.get_running_loop()) at the top of
the async fixture. This synchronises the thread-level current loop with the
loop that is actually running the test, so get_event_loop() inside any
greenlet spawned during this fixture's lifetime returns the correct loop.

NullPool is used so connections are never pooled across tests. A pooled
connection carries a StreamReader whose _loop was bound to a previous test's
loop; NullPool guarantees every checkout creates a fresh connection.
"""

import asyncio
import pytest
from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def db_session() -> Session:
    """
    Session-scoped synchronous database session for direct ORM queries.

    Used in test setup to look up test users (e.g. db_session.query(User)...)
    and as the db argument for sync tools (popular_books, semantic_search,
    subject_id_search). Must not be passed to async tools — those require an
    AsyncSession. Use async_db_session for those.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
async def async_db_session():
    """
    Function-scoped async database session for async tools.

    Pass this as the db argument to InternalTools when testing
    subject_hybrid_pool or als_recs. These tools call await db.execute()
    deep in the filter chain, which requires an AsyncSession.

    Pins the thread-level current event loop to the running loop before any
    I/O so that aiomysql's get_event_loop() calls inside SQLAlchemy greenlets
    always see the correct loop. Uses NullPool so connections are never
    recycled across tests.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.pool import NullPool
    from app.database import ASYNC_DATABASE_URL

    asyncio.set_event_loop(asyncio.get_running_loop())

    engine = create_async_engine(ASYNC_DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(
            engine.execution_options(isolation_level="AUTOCOMMIT"),
            expire_on_commit=False,
        ) as session:
            yield session
    finally:
        await engine.dispose()
