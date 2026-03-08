# app/database.py
"""
SQLAlchemy engine, session factories, and FastAPI dependency providers.

Two session factories share a single connection pool:
- SessionLocal: full read-write sessions with standard transaction management.
  Use for any route that writes to the database.
- ReadOnlySessionLocal: sessions bound to an AUTOCOMMIT sub-engine. MySQL
  never opens an explicit transaction, so close() issues no ROLLBACK or COMMIT
  round-trip. Use for read-only routes (e.g. recommendations).

The read-only sub-engine is created via engine.execution_options(), which
shares the same underlying connection pool as the main engine. Connections
are reset to the DBAPI default isolation level when returned to the pool, so
the two factories do not interfere.

pool_pre_ping is intentionally absent. pool_recycle=3600 is sufficient
given MySQL wait_timeout=28800 — connections are replaced well before the
server would drop them, without the cost of a SELECT 1 on every checkout.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

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


def get_db() -> Session:
    """Read-write session dependency for routes that modify the database."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_read_only_db() -> Session:
    """
    Read-only session dependency for routes that never write.

    The underlying connection operates in AUTOCOMMIT mode — MySQL runs each
    statement without an explicit transaction. close() returns the connection
    to the pool with no ROLLBACK or COMMIT issued to the server.

    All other behaviour is identical to get_db(): same connection pool, same
    Session API, same ORM query support including joinedload.
    """
    db = ReadOnlySessionLocal()
    try:
        yield db
    finally:
        db.close()
