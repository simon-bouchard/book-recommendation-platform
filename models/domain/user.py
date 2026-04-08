# models/domain/user.py
"""
User domain model representing a user with their preferences and status.
"""

from dataclasses import dataclass
from typing import Optional

from models.client.registry import get_als_client
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

    Async methods:
        is_warm: True if the ALS model server holds factors for this user.
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

    async def is_warm(self) -> bool:
        """
        Check if user has ALS factors on the model server (warm/cold gate).

        Delegates to the ALS client so the check uses the same HTTP path as
        all other model server interactions, rather than importing infrastructure
        directly into the domain layer.

        Returns False on any network or server error to fail safe toward the
        cold-start path.
        """
        try:
            resp = await get_als_client().has_als_user(self.user_id)
            return resp.is_warm
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
