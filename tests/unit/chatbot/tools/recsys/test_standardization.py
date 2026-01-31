# tests/unit/chatbot/tools/recsys/test_standardization.py
"""
Tests for the _standardize_tool_output() method in InternalTools.
This method is responsible for normalizing all retrieval tool outputs to a consistent schema.
"""

import pytest
from unittest.mock import patch
from app.agents.tools.recsys.native_tools import InternalTools


class TestStandardizeToolOutput:
    """Test the _standardize_tool_output method in isolation."""

    def test_adds_num_ratings_from_book_meta(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should add num_ratings from book metadata."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "score": 0.95,
            }
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert len(standardized) == 1
        assert standardized[0]["num_ratings"] == 100

    def test_resolves_tone_ids_to_names(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
        mock_tone_map,
    ):
        """Should convert tone_ids to tone names using database mapping."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "tone_ids": [2, 4],
                "score": 0.95,
            }
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert "tones" in standardized[0]
        assert standardized[0]["tones"] == ["Dark", "Reflective"]

    def test_excludes_empty_enrichment_fields(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should not include enrichment fields when they're empty."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "subjects": [],
                "tone_ids": [],
                "genre": "",
                "vibe": "",
                "score": 0.95,
            }
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert "subjects" not in standardized[0]
        assert "tones" not in standardized[0]
        assert "genre" not in standardized[0]
        assert "vibe" not in standardized[0]

    def test_includes_enrichment_when_present(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should include enrichment fields when they have content."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "subjects": ["American Dream", "Jazz Age"],
                "genre": "Literary Fiction",
                "vibe": "Melancholic exploration of wealth",
                "score": 0.95,
            }
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert standardized[0]["subjects"] == ["American Dream", "Jazz Age"]
        assert standardized[0]["genre"] == "Literary Fiction"
        assert standardized[0]["vibe"] == "Melancholic exploration of wealth"

    def test_preserves_score_field(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should keep score field for debugging purposes."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "score": 0.95,
            }
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert standardized[0]["score"] == 0.95

    def test_handles_missing_item_idx(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should skip books without item_idx."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {"title": "Book Without ID", "author": "Unknown"},
            {"item_idx": 1, "title": "Valid Book", "author": "Author"},
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert len(standardized) == 1
        assert standardized[0]["item_idx"] == 1

    def test_handles_books_not_in_book_meta(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should default num_ratings to 0 for unknown books."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {
                "item_idx": 999,
                "title": "Unknown Book",
                "author": "Unknown Author",
                "score": 0.5,
            }
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert standardized[0]["num_ratings"] == 0

    def test_handles_missing_tone_ids(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should not include tones field when tone_ids is empty or missing."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {
                "item_idx": 1,
                "title": "Book Without Tones",
                "author": "Author",
                "score": 0.8,
            }
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert "tones" not in standardized[0]

    def test_filters_invalid_tone_ids(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
        mock_tone_map,
    ):
        """Should only include tones that exist in the tone map."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {
                "item_idx": 1,
                "title": "Book",
                "author": "Author",
                "tone_ids": [1, 999, 2],
                "score": 0.8,
            }
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert standardized[0]["tones"] == ["Fast-paced", "Dark"]

    def test_passes_through_error_dicts(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should not modify dicts with 'error' key."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {"error": "Something went wrong"},
            {"item_idx": 1, "title": "Valid Book", "author": "Author"},
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert len(standardized) == 2
        assert standardized[0] == {"error": "Something went wrong"}

    def test_handles_empty_results(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should return empty list for empty input."""
        tools = internal_tools_factory(db=mock_db_session)

        standardized = tools._standardize_tool_output([])

        assert standardized == []

    def test_preserves_core_metadata_fields(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should always include core fields: item_idx, title, author, year, num_ratings."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {
                "item_idx": 1,
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "year": 1925,
                "score": 0.95,
            }
        ]

        standardized = tools._standardize_tool_output(raw_results)

        assert "item_idx" in standardized[0]
        assert "title" in standardized[0]
        assert "author" in standardized[0]
        assert "year" in standardized[0]
        assert "num_ratings" in standardized[0]
        assert "score" in standardized[0]

    def test_tone_map_caching(
        self,
        internal_tools_factory,
        mock_db_session,
        mock_load_book_meta,
    ):
        """Should cache tone map after first database query."""
        tools = internal_tools_factory(db=mock_db_session)

        raw_results = [
            {"item_idx": 1, "title": "Book 1", "author": "Author", "tone_ids": [1, 2]},
            {"item_idx": 2, "title": "Book 2", "author": "Author", "tone_ids": [3, 4]},
        ]

        tools._standardize_tool_output(raw_results)
        tools._standardize_tool_output(raw_results)

        # Database query should only be called once
        assert mock_db_session.query.call_count == 1
