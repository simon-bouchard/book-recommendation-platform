# tests/integration/chatbot/recsys_fixtures.py
"""
Reusable fixtures for recommendation system tests.
Provides test users with different rating states (warm/cold/new) and profile configurations.
"""

import pytest
from sqlalchemy.orm import Session
from sqlalchemy import func, exists, select

from app.table_models import User, Interaction, UserFavSubject


@pytest.fixture
def test_user_warm(db_session) -> User:
    """
    Get or create a warm user (≥10 ratings).

    Warm users have sufficient rating history to use collaborative filtering
    (ALS-based recommendations).

    Returns user ID 278859 from test data.
    Falls back to any user with ≥10 ratings if not found.
    """
    # Try specific test user first
    user = db_session.query(User).filter(User.user_id == 278859).first()

    if user:
        # Verify rating count
        rating_count = (
            db_session.query(Interaction).filter(Interaction.user_id == user.user_id).count()
        )

        if rating_count >= 10:
            return user

    # Fallback: find any warm user
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

    Cold users don't have sufficient history for collaborative filtering
    and must use content-based or subject-based recommendations.

    Returns user ID 278857 from test data.
    Falls back to any user with <10 ratings if not found.
    """
    # Try specific test user first
    user = db_session.query(User).filter(User.user_id == 278857).first()

    if user:
        # Verify rating count
        rating_count = (
            db_session.query(Interaction).filter(Interaction.user_id == user.user_id).count()
        )

        if rating_count < 10:
            return user

    # Fallback: find any cold user
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

    New users have no interaction history and require subject-based or
    popular recommendations.

    Returns user ID 278867 from test data.
    Falls back to any user with 0 ratings if not found.
    """
    # Try specific test user first
    user = db_session.query(User).filter(User.user_id == 278867).first()

    if user:
        rating_count = (
            db_session.query(Interaction).filter(Interaction.user_id == user.user_id).count()
        )

        if rating_count == 0:
            return user

    # Fallback: find any user with 0 ratings
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

    Users with profiles allow testing profile-aware recommendation strategies.
    Prefers warm user with profile for realistic testing.
    """
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
    """
    Get the number of ratings for a user.

    Args:
        db_session: Database session
        user_id: User ID to check

    Returns:
        Number of ratings the user has made
    """
    return db_session.query(Interaction).filter(Interaction.user_id == user_id).count()
