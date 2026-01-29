# models/domain/user.py
"""
User domain model representing a user with their preferences and status.
"""

from dataclasses import dataclass
from typing import Optional

from models.core.constants import PAD_IDX


@dataclass
class User:
    """
    User domain model with book preferences and computed status.

    Attributes:
        user_id: Unique user identifier
        fav_subjects: List of favorite subject indices
        country: User's country code (optional)
        age: User's age (optional)
        filled_age: Age category if actual age not available (optional)

    Properties:
        has_preferences: True if user has valid non-PAD favorite subjects
        is_warm: True if user has sufficient interaction history for collaborative filtering
    """

    user_id: int
    fav_subjects: list[int]
    country: Optional[str] = None
    age: Optional[int] = None
    filled_age: Optional[str] = None

    @property
    def has_preferences(self) -> bool:
        """
        Check if user has valid subject preferences.

        Returns False if fav_subjects is empty or contains only PAD_IDX.
        """
        if not self.fav_subjects:
            return False
        return not all(s == PAD_IDX for s in self.fav_subjects)

    @property
    def is_warm(self) -> bool:
        """
        Check if user is warm (has sufficient interaction history for ALS).

        Directly checks if user exists in the trained ALS model.
        This is faster and more robust than counting ratings.
        Uses lazy import to avoid circular dependencies.
        """
        try:
            from models.infrastructure.als_model import ALSModel

            return ALSModel().has_user(self.user_id)
        except Exception:
            return False

    @classmethod
    def from_orm(cls, orm_user) -> "User":
        """
        Create User domain model from SQLAlchemy ORM user.

        Args:
            orm_user: User instance from app.table_models.User

        Returns:
            User domain model
        """
        fav_subjects = getattr(orm_user, "fav_subjects_idxs", [])
        if not fav_subjects:
            fav_subjects = [PAD_IDX]

        return cls(
            user_id=orm_user.user_id,
            fav_subjects=fav_subjects,
            country=getattr(orm_user, "country", None),
            age=getattr(orm_user, "age", None),
            filled_age=getattr(orm_user, "filled_age", None),
        )
