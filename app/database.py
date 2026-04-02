# app/database.py
"""
SQLAlchemy engine, session factories, and FastAPI dependency providers.

Three session factories are provided:

- SessionLocal: full read-write sessions using the sync pymysql engine.
  Use for any route that writes to the database.

- ReadOnlySessionLocal: sync sessions bound to an AUTOCOMMIT sub-engine.
  MySQL never opens an explicit transaction, so close() issues no ROLLBACK
  or COMMIT round-trip. Retained for sync read-only routes.

- AsyncReadOnlySessionLocal: async sessions using aiomysql. Use for async
  route handlers that need native non-blocking DB access. Currently used
  exclusively by the recommendation endpoint to eliminate the asyncio.to_thread
  dispatch that was the dominant latency contributor.

The sync and async engines are independent objects with separate connection
pools. The async pool is intentionally smaller (pool_size=5) because async
connections are multiplexed across concurrent coroutines — fewer connections
serve the same concurrency as a larger sync pool.

pool_pre_ping is intentionally absent on both engines. pool_recycle=3600 is
sufficient given MySQL wait_timeout=28800 — connections are replaced well
before the server would drop them, without the cost of a SELECT 1 on every
checkout.
"""

import os
from typing import Optional
from urllib.parse import urlparse

import aiomysql
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
ASYNC_DATABASE_URL = (
    DATABASE_URL.replace("mysql+pymysql://", "mysql+aiomysql://") if DATABASE_URL else None
)

# ---------------------------------------------------------------------------
# Raw aiomysql pool — used by hot-path queries that bypass SQLAlchemy overhead
# ---------------------------------------------------------------------------

_aiomysql_pool: Optional[aiomysql.Pool] = None


def _parse_mysql_url(url: str) -> dict:
    """Extract aiomysql connect kwargs from a SQLAlchemy MySQL URL."""
    stripped = url.replace("mysql+pymysql://", "mysql://").replace("mysql+aiomysql://", "mysql://")
    parsed = urlparse(stripped)
    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "user": parsed.username,
        "password": parsed.password,
        "db": parsed.path.lstrip("/"),
        "autocommit": True,
        "charset": "utf8mb4",
    }


async def init_aiomysql_pool() -> None:
    """Create the raw aiomysql connection pool. Call once at application startup."""
    global _aiomysql_pool
    if DATABASE_URL and _aiomysql_pool is None:
        _aiomysql_pool = await aiomysql.create_pool(
            minsize=2,
            maxsize=5,
            **_parse_mysql_url(DATABASE_URL),
        )


async def close_aiomysql_pool() -> None:
    """Drain and close the pool. Call at application shutdown."""
    global _aiomysql_pool
    if _aiomysql_pool is not None:
        _aiomysql_pool.close()
        await _aiomysql_pool.wait_closed()
        _aiomysql_pool = None


def get_aiomysql_pool() -> aiomysql.Pool:
    """Return the active pool. Raises if called before init_aiomysql_pool()."""
    if _aiomysql_pool is None:
        raise RuntimeError("aiomysql pool not initialized — call init_aiomysql_pool() at startup")
    return _aiomysql_pool

Base = declarative_base()

if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_recycle=3600,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        connect_args={"connect_timeout": 10},
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    ReadOnlySessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine.execution_options(isolation_level="AUTOCOMMIT"),
    )
else:
    engine = None
    SessionLocal = None
    ReadOnlySessionLocal = None

if ASYNC_DATABASE_URL:
    async_engine = create_async_engine(
        ASYNC_DATABASE_URL,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        connect_args={"connect_timeout": 10},
    )
    AsyncReadOnlySessionLocal = async_sessionmaker(
        bind=async_engine.execution_options(isolation_level="AUTOCOMMIT"),
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
else:
    async_engine = None
    AsyncReadOnlySessionLocal = None


def get_db() -> Session:
    """Read-write session dependency for routes that modify the database."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_read_only_db() -> Session:
    """
    Read-only session dependency for sync routes that never write.

    The underlying connection operates in AUTOCOMMIT mode — MySQL runs each
    statement without an explicit transaction. close() returns the connection
    to the pool with no ROLLBACK or COMMIT issued to the server.
    """
    db = ReadOnlySessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_read_only_db() -> AsyncSession:
    """
    Read-only async session dependency for async routes that never write.

    Uses aiomysql under the hood — the DB call runs natively on the event
    loop with no thread pool dispatch. AUTOCOMMIT mode means no transaction
    overhead on session close.

    Usage:
        db: AsyncSession = Depends(get_async_read_only_db)
    """
    async with AsyncReadOnlySessionLocal() as session:
        yield session
