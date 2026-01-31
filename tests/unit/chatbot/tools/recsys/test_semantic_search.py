# tests/unit/chatbot/tools/recsys/test_semantic_search.py
"""
Tests for the semantic_search tool.
Validates tool interface, meta field unpacking, and standardization integration.
"""

import pytest
from unittest.mock import patch, Mock


class TestSemanticSearchTool:
    """Test semantic_search tool behavior."""

    def test_returns_correct_schema(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should return standardized list with expected fields."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")
        results = semantic_tool.execute(query="dark fantasy", top_k=10)

        assert isinstance(results, list)
        assert len(results) > 0

        book = results[0]
        assert "item_idx" in book
        assert "title" in book
        assert "author" in book
        assert "year" in book
        assert "num_ratings" in book

    def test_unpacks_meta_field(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should unpack nested meta field to top level."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")
        results = semantic_tool.execute(query="dystopian novels", top_k=5)

        # Check that enrichment fields are at top level (not nested in meta)
        book = results[0]
        assert "subjects" in book
        assert "tones" in book
        assert "genre" in book
        assert "vibe" in book

        # Meta field should not exist anymore
        assert "meta" not in book

    def test_resolves_tone_names(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
        mock_tone_map,
    ):
        """Should convert tone_ids to tone names."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")
        results = semantic_tool.execute(query="suspenseful thriller", top_k=3)

        book = results[0]
        assert "tones" in book
        assert isinstance(book["tones"], list)
        assert all(isinstance(tone, str) for tone in book["tones"])

        # Check actual tone names from mock
        expected_tones = ["Dark", "Reflective"]
        assert book["tones"] == expected_tones

    def test_adds_num_ratings(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should add num_ratings from book metadata."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")
        results = semantic_tool.execute(query="literary fiction", top_k=5)

        assert results[0]["num_ratings"] == 100
        assert results[1]["num_ratings"] == 200

    def test_handles_top_k_limits(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should enforce top_k bounds (1 to 500)."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")

        # Test minimum
        results = semantic_tool.execute(query="test", top_k=0)
        mock_semantic_searcher_class.return_value.search.assert_called_with(
            query="test",
            top_k=1,
        )

        # Test maximum
        results = semantic_tool.execute(query="test", top_k=1000)
        mock_semantic_searcher_class.return_value.search.assert_called_with(
            query="test",
            top_k=500,
        )

    def test_handles_empty_results(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should handle empty search results gracefully."""
        # Mock searcher to return empty results
        mock_semantic_searcher_class.return_value.search.return_value = []

        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")
        results = semantic_tool.execute(query="nonexistent query", top_k=10)

        assert results == []

    def test_returns_error_on_exception(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should return error dict when search fails."""
        # Mock searcher to raise exception
        mock_semantic_searcher_class.return_value.search.side_effect = Exception("Search failed")

        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")
        results = semantic_tool.execute(query="test", top_k=10)

        assert len(results) == 1
        assert "error" in results[0]
        assert "Search failed" in results[0]["error"]

    def test_works_without_authentication(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should work for anonymous users."""
        tools = internal_tools_factory(current_user=None, db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")
        results = semantic_tool.execute(query="science fiction", top_k=5)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_lazy_loads_semantic_searcher(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should only create SemanticSearcher when first needed."""
        tools = internal_tools_factory(db=mock_db_session)

        # Searcher not created yet
        assert mock_semantic_searcher_class.call_count == 0

        # First search creates searcher
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)
        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")

        with patch("app.agents.tools.recsys.native_tools.load_book_meta") as mock_meta:
            mock_meta.return_value = Mock()
            semantic_tool.execute(query="test", top_k=5)

        assert mock_semantic_searcher_class.call_count == 1

        # Second search reuses searcher
        with patch("app.agents.tools.recsys.native_tools.load_book_meta") as mock_meta:
            mock_meta.return_value = Mock()
            semantic_tool.execute(query="another test", top_k=5)

        # Still only one searcher instance
        assert mock_semantic_searcher_class.call_count == 1

    def test_excludes_empty_enrichment_fields(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should not include enrichment fields when empty."""
        # Mock searcher with minimal metadata
        mock_searcher = Mock()
        mock_searcher.search.return_value = [
            {
                "item_idx": 1,
                "score": 0.8,
                "meta": {
                    "title": "Minimal Book",
                    "author": "Author",
                    "subjects": [],
                    "tone_ids": [],
                },
            }
        ]
        mock_semantic_searcher_class.return_value = mock_searcher

        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")
        results = semantic_tool.execute(query="test", top_k=5)

        book = results[0]
        assert "subjects" not in book
        assert "tones" not in book
        assert "genre" not in book
        assert "vibe" not in book

    def test_filters_books_without_item_idx(
        self,
        internal_tools_factory,
        mock_semantic_searcher_class,
        mock_load_book_meta,
        mock_settings_embedder,
        mock_db_session,
    ):
        """Should skip results without item_idx."""
        # Mock searcher with one invalid result
        mock_searcher = Mock()
        mock_searcher.search.return_value = [
            {"score": 0.9, "meta": {"title": "No ID"}},
            {
                "item_idx": 1,
                "score": 0.8,
                "meta": {"title": "Has ID", "author": "Author"},
            },
        ]
        mock_semantic_searcher_class.return_value = mock_searcher

        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        semantic_tool = next(t for t in retrieval_tools if t.name == "book_semantic_search")
        results = semantic_tool.execute(query="test", top_k=10)

        assert len(results) == 1
        assert results[0]["item_idx"] == 1
