# models/domain/__init__.py
"""
Domain models layer - pure business objects representing core entities.
"""

from models.domain.user import User
from models.domain.recommendation import Candidate, RecommendedBook
from models.domain.config import RecommendationConfig, HybridConfig

__all__ = [
    "User",
    "Candidate",
    "RecommendedBook",
    "RecommendationConfig",
    "HybridConfig",
]
