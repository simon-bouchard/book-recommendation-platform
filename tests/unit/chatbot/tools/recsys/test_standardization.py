# tests/unit/chatbot/tools/recsys/test_standardization.py
"""
Unit tests for InternalTools._standardize_tool_output().

Tests the normalization logic that all retrieval tools pass results through.
num_ratings and cover_id are read directly from the input dict — no external
dependency on load_book_meta.
"""

import pytest
from app.agents.tools.recsys.native_tools import InternalTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_CORE_FIELDS = {"item_idx", "title", "author", "year", "num_ratings", "cover_id", "score"}


def _tools():
    return InternalTools(current_user=None, db=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStandardizeToolOutput:
    """Tests for _standardize_tool_output() normalization logic."""

    def test_empty_input_returns_empty_list(self):
        """Empty input should return empty list."""
        assert _tools()._standardize_tool_output([]) == []

    def test_error_dict_passes_through_unchanged(self):
        """Dicts with 'error' key are included in output without modification."""
        error_entry = {"error": "Something went wrong"}
        result = _tools()._standardize_tool_output([error_entry])

        assert len(result) == 1
        assert result[0] == error_entry

    def test_book_without_item_idx_is_skipped(self):
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
        result = _tools()._standardize_tool_output(raw)

        assert len(result) == 1
        assert result[0]["item_idx"] == 1

    def test_num_ratings_read_from_input(self):
        """num_ratings should be taken from the input dict."""
        raw = [
            {"item_idx": 1, "title": "Book A", "author": "Author A", "score": 0.9, "num_ratings": 100},
            {"item_idx": 2, "title": "Book B", "author": "Author B", "score": 0.8, "num_ratings": 200},
        ]
        result = _tools()._standardize_tool_output(raw)

        assert result[0]["num_ratings"] == 100
        assert result[1]["num_ratings"] == 200

    def test_num_ratings_defaults_to_zero_when_absent(self):
        """Books without num_ratings in input get num_ratings=0."""
        raw = [{"item_idx": 999, "title": "Unknown Book", "author": "Ghost", "score": 0.5}]
        result = _tools()._standardize_tool_output(raw)

        assert len(result) == 1
        assert result[0]["num_ratings"] == 0

    def test_cover_id_read_from_input(self):
        """cover_id should be read from the input dict when present and truthy."""
        raw = [
            {"item_idx": 1, "title": "Book A", "author": "Author A", "score": 0.9, "cover_id": "cov_1"}
        ]
        result = _tools()._standardize_tool_output(raw)

        assert result[0]["cover_id"] == "cov_1"

    def test_cover_id_is_none_when_falsy_in_input(self):
        """cover_id should be None when input has None or empty value."""
        raw = [
            {"item_idx": 1, "title": "Book A", "author": "Author A", "score": 0.9, "cover_id": None},
            {"item_idx": 2, "title": "Book B", "author": "Author B", "score": 0.8, "cover_id": ""},
        ]
        result = _tools()._standardize_tool_output(raw)

        assert result[0]["cover_id"] is None
        assert result[1]["cover_id"] is None

    def test_cover_id_is_none_when_absent(self):
        """Books with no cover_id in input get cover_id=None."""
        raw = [{"item_idx": 999, "title": "Unknown", "author": "Ghost", "score": 0.5}]
        result = _tools()._standardize_tool_output(raw)

        assert result[0]["cover_id"] is None

    def test_output_contains_exactly_core_fields(self):
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
        result = _tools()._standardize_tool_output(raw)

        assert set(result[0].keys()) == EXPECTED_CORE_FIELDS

    def test_score_preserved_from_input(self):
        """score field should be carried through unchanged."""
        raw = [
            {"item_idx": 1, "title": "Book", "author": "Author", "score": 0.87654}
        ]
        result = _tools()._standardize_tool_output(raw)

        assert result[0]["score"] == 0.87654

    def test_mixed_valid_error_and_missing_id(self):
        """Valid books, error dicts, and missing-id books are all handled correctly together."""
        raw = [
            {"error": "fetch failed"},
            {"title": "No ID", "score": 0.3},
            {"item_idx": 2, "title": "1984", "author": "George Orwell", "score": 0.92, "num_ratings": 200},
        ]
        result = _tools()._standardize_tool_output(raw)

        assert len(result) == 2
        assert result[0] == {"error": "fetch failed"}
        assert result[1]["item_idx"] == 2
