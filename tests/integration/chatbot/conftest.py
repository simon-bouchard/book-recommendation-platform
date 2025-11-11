# tests/integration/chatbot/conftest.py
"""
Shared fixtures for Conductor integration tests.
Provides test users with different states (warm/cold, with/without profiles).
"""
import pytest
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.table_models import User, Interaction, Subject


@pytest.fixture(scope="session")
def db_session() -> Session:
    """
    Provide a database session for integration tests.
    
    Uses actual database connection (requires DATABASE_URL env var).
    Tests should not modify production data.
    """
    if SessionLocal is None:
        pytest.skip("Database not configured. Set DATABASE_URL environment variable.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def test_user_warm(db_session) -> User:
    """
    Get or create a warm user (≥10 ratings).
    
    Returns user ID 278859 from test data.
    Falls back to any user with ≥10 ratings if not found.
    """
    # Try specific test user first
    user = db_session.query(User).filter(User.user_id == 278859).first()
    
    if user:
        # Verify rating count
        rating_count = db_session.query(Interaction).filter(
            Interaction.user_id == user.user_id
        ).count()
        
        if rating_count >= 10:
            return user
    
    # Fallback: find any warm user
    from sqlalchemy import func
    warm_user = (
        db_session.query(User)
        .join(Interaction, User.user_id == Interaction.user_id)
        .group_by(User.user_id)
        .having(func.count(Interaction.id) >= 10)
        .first()
    )
    
    if not warm_user:
        pytest.skip("No warm users (≥10 ratings) available in test database")
    
    return warm_user


@pytest.fixture
def test_user_cold(db_session) -> User:
    """
    Get or create a cold user (<10 ratings).
    
    Returns user ID 278857 from test data.
    Falls back to any user with <10 ratings if not found.
    """
    # Try specific test user first
    user = db_session.query(User).filter(User.user_id == 278857).first()
    
    if user:
        # Verify rating count
        rating_count = db_session.query(Interaction).filter(
            Interaction.user_id == user.user_id
        ).count()
        
        if rating_count < 10:
            return user
    
    # Fallback: find any cold user
    from sqlalchemy import func
    cold_user = (
        db_session.query(User)
        .outerjoin(Interaction, User.user_id == Interaction.user_id)
        .group_by(User.user_id)
        .having(func.count(Interaction.id) < 10)
        .first()
    )
    
    if not cold_user:
        pytest.skip("No cold users (<10 ratings) available in test database")
    
    return cold_user


@pytest.fixture
def test_user_new(db_session) -> User:
    """
    Get a new user (0 ratings, no profile).
    
    Returns user ID 278867 from test data.
    Falls back to any user with 0 ratings if not found.
    """
    # Try specific test user first
    user = db_session.query(User).filter(User.user_id == 278867).first()
    
    if user:
        rating_count = db_session.query(Interaction).filter(
            Interaction.user_id == user.user_id
        ).count()
        
        if rating_count == 0:
            return user
    
    # Fallback: find any user with 0 ratings
    from sqlalchemy import func
    new_user = (
        db_session.query(User)
        .outerjoin(Interaction, User.user_id == Interaction.user_id)
        .group_by(User.user_id)
        .having(func.count(Interaction.id) == 0)
        .first()
    )
    
    if not new_user:
        pytest.skip("No new users (0 ratings) available in test database")
    
    return new_user


@pytest.fixture
def test_user_with_profile(db_session) -> User:
    """
    Get a user with profile data (favorite subjects).
    
    Prefers warm user with profile for realistic testing.
    """
    from sqlalchemy import func, exists, select
    from app.table_models import UserFavSubject
    
    # Find warm user with favorite_subjects (using exists subquery)
    subquery = exists(select(UserFavSubject.id).where(UserFavSubject.user_id == User.user_id))
    
    user = (
        db_session.query(User)
        .join(Interaction, User.user_id == Interaction.user_id)
        .filter(subquery)
        .group_by(User.user_id)
        .having(func.count(Interaction.id) >= 10)
        .first()
    )
    
    if not user:
        # Fallback: any user with profile
        user = db_session.query(User).filter(subquery).first()
    
    if not user:
        pytest.skip("No users with profile data available in test database")
    
    return user


def get_user_rating_count(db_session: Session, user_id: int) -> int:
    """Helper to get rating count for a user."""
    return db_session.query(Interaction).filter(
        Interaction.user_id == user_id
    ).count()
