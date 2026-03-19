# tests/unit/models/data/test_queries.py
"""
Unit tests for models.data.queries module.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.data.queries import get_user_num_ratings


class TestGetUserNumRatings:
    """Test getting user rating count from cached metadata."""

    def test_returns_integer(self):
        """Should return integer."""
        from models.data.loaders import load_user_meta

        user_meta = load_user_meta()

        if len(user_meta) > 0:
            user_id = user_meta.index[0]
            count = get_user_num_ratings(user_id)

            assert isinstance(count, int)

    def test_returns_zero_for_nonexistent_user(self):
        """Should return 0 for user not in metadata."""
        count = get_user_num_ratings(-99999)

        assert count == 0

    def test_returns_correct_count_for_known_user(self):
        """Should return actual rating count from metadata."""
        from models.data.loaders import load_user_meta

        user_meta = load_user_meta()

        if len(user_meta) > 0:
            user_id = user_meta.index[0]
            expected = user_meta.loc[user_id].get("user_num_ratings")

            if expected is not None:
                count = get_user_num_ratings(user_id)
                assert count == int(expected)
