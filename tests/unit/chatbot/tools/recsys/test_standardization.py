# tests/unit/chatbot/tools/recsys/test_standardization.py
"""
Unit tests for InternalTools._standardize_tool_output().

Tests the normalization logic that all retrieval tools pass results through.
The only external dependency is load_book_meta(), which is mocked throughout.
"""

import pytest
from unittest.mock import patch
from app.agents.tools.recsys.native_tools import InternalTools

_PATCH_TARGET = "app.agents.tools.recsys.native_tools.load_book_meta"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_CORE_FIELDS = {"item_idx", "title", "author", "year", "num_ratings", "cover_id", "score"}


def _make_tools(mock_book_meta, db=None):
    """Return an InternalTools instance with load_book_meta patched."""
    return InternalTools(current_user=None, db=db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStandardizeToolOutput:
    """Tests for _standardize_tool_output() normalization logic."""

    def test_empty_input_returns_empty_list(self, mock_book_meta):
        """Empty input should return empty list without touching book_meta."""
        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output([])

        assert result == []

    def test_error_dict_passes_through_unchanged(self, mock_book_meta):
        """Dicts with 'error' key are included in output without modification."""
        error_entry = {"error": "Something went wrong"}

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output([error_entry])

        assert len(result) == 1
        assert result[0] == error_entry

    def test_book_without_item_idx_is_skipped(self, mock_book_meta):
        """Books missing item_idx are silently dropped."""
        raw = [
            {"title": "No ID Book", "author": "Unknown", "score": 0.5},
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "score": 0.9,
            },
        ]

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output(raw)

        assert len(result) == 1
        assert result[0]["item_idx"] == 1

    def test_num_ratings_populated_from_book_meta(self, mock_book_meta):
        """num_ratings should come from book_meta.book_num_ratings for known books."""
        raw = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "score": 0.9,
            },
            {"item_idx": 2, "title": "1984", "author": "George Orwell", "score": 0.8},
        ]

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output(raw)

        assert result[0]["num_ratings"] == 100
        assert result[1]["num_ratings"] == 200

    def test_num_ratings_defaults_to_zero_for_unknown_book(self, mock_book_meta):
        """Books not in book_meta index get num_ratings=0."""
        raw = [{"item_idx": 999, "title": "Unknown Book", "author": "Ghost", "score": 0.5}]

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output(raw)

        assert len(result) == 1
        assert result[0]["num_ratings"] == 0

    def test_cover_id_populated_from_book_meta(self, mock_book_meta):
        """cover_id should be read from book_meta when present and truthy."""
        raw = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "score": 0.9,
            }
        ]

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output(raw)

        assert result[0]["cover_id"] == "cov_1"

    def test_cover_id_is_none_when_falsy_in_book_meta(self, mock_book_meta):
        """cover_id should be None when book_meta has None or empty value."""
        # item_idx 3 has cover_id=None in mock_book_meta
        raw = [
            {"item_idx": 3, "title": "To Kill a Mockingbird", "author": "Harper Lee", "score": 0.7}
        ]

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output(raw)

        assert result[0]["cover_id"] is None

    def test_cover_id_is_none_for_unknown_book(self, mock_book_meta):
        """Books not in book_meta index get cover_id=None."""
        raw = [{"item_idx": 999, "title": "Unknown", "author": "Ghost", "score": 0.5}]

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output(raw)

        assert result[0]["cover_id"] is None

    def test_output_contains_exactly_core_fields(self, mock_book_meta):
        """Output dicts should contain exactly the expected core fields — no extras."""
        raw = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "year": 1925,
                "score": 0.9,
                "subjects": ["Jazz Age"],
                "vibe": "Melancholic",
            }
        ]

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output(raw)

        assert set(result[0].keys()) == EXPECTED_CORE_FIELDS

    def test_score_preserved_from_input(self, mock_book_meta):
        """score field should be carried through unchanged."""
        raw = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "score": 0.87654,
            }
        ]

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output(raw)

        assert result[0]["score"] == 0.87654

    def test_mixed_valid_error_and_missing_id(self, mock_book_meta):
        """Valid books, error dicts, and missing-id books are all handled correctly together."""
        raw = [
            {"error": "fetch failed"},
            {"title": "No ID", "score": 0.3},
            {"item_idx": 2, "title": "1984", "author": "George Orwell", "score": 0.92},
        ]

        with patch(_PATCH_TARGET, return_value=mock_book_meta):
            tools = InternalTools(current_user=None, db=None)
            result = tools._standardize_tool_output(raw)

        assert len(result) == 2
        assert result[0] == {"error": "fetch failed"}
        assert result[1]["item_idx"] == 2
