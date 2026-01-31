# tests/unit/chatbot/tools/recsys/test_popular_books.py
"""
Tests for the popular_books tool (popularity-based recommendations).
Validates that the tool returns popular books using RecommendationService.
"""

import pytest
from unittest.mock import patch, Mock


class TestPopularBooksTool:
    """Test popular_books tool behavior."""

    def test_returns_correct_schema(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should return standardized list with basic metadata."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            results = popular_tool.execute(top_k=10)

            assert isinstance(results, list)
            assert len(results) > 0

            book = results[0]
            assert "item_idx" in book
            assert "title" in book
            assert "author" in book
            assert "year" in book
            assert "num_ratings" in book
            assert "score" in book

    def test_does_not_include_enrichment_metadata(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Popular books returns basic metadata only."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            results = popular_tool.execute(top_k=10)

            book = results[0]
            assert "subjects" not in book
            assert "tones" not in book
            assert "genre" not in book
            assert "vibe" not in book

    def test_works_without_authentication(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should work for anonymous users when database is present."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(current_user=None, db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            results = popular_tool.execute(top_k=5)

            assert isinstance(results, list)
            assert len(results) > 0

    def test_returns_error_when_no_database(
        self,
        internal_tools_factory,
    ):
        """Popular books should return error when no database available."""
        tools = internal_tools_factory(db=None)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        # Tool is still in list (pattern: tool checks internally)
        popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
        results = popular_tool.execute(top_k=10)

        # Should return error when no database
        assert len(results) == 1
        assert "error" in results[0]
        assert "database" in results[0]["error"].lower()

    def test_handles_top_k_limits(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should enforce top_k bounds (1 to 500)."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.recommend.return_value = []
            mock_service_class.return_value = mock_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")

            # Test minimum
            popular_tool.execute(top_k=0)
            call_args = mock_service.recommend.call_args
            assert call_args[0][1].k == 1

            # Test maximum
            popular_tool.execute(top_k=1000)
            call_args = mock_service.recommend.call_args
            assert call_args[0][1].k == 500

    def test_calls_recommendation_service_correctly(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should call RecommendationService with auto mode."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            popular_tool.execute(top_k=50)

            # Verify service was called
            mock_recommendation_service.recommend.assert_called_once()

            # Check arguments
            call_args = mock_recommendation_service.recommend.call_args
            config = call_args[0][1]

            # Uses "auto" mode which falls back to popularity
            assert config.mode == "auto"
            assert config.k == 50

    def test_adds_num_ratings_from_book_meta(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should add num_ratings field from book metadata."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            results = popular_tool.execute(top_k=10)

            # num_ratings should come from mock_book_meta
            assert results[0]["num_ratings"] == 100
            assert results[1]["num_ratings"] == 200

    def test_returns_error_on_exception(
        self,
        internal_tools_factory,
        mock_db_session,
        mock_load_book_meta,
    ):
        """Should return error dict when service fails."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.recommend.side_effect = Exception("Popularity model error")
            mock_service_class.return_value = mock_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            results = popular_tool.execute(top_k=10)

            assert len(results) == 1
            assert "error" in results[0]
            assert "Popularity model error" in results[0]["error"]

    def test_tool_always_in_list(
        self,
        internal_tools_factory,
        mock_db_session,
    ):
        """Popular books tool is always in list (errors are returned at execution time)."""
        # Anonymous user with DB
        tools1 = internal_tools_factory(current_user=None, db=mock_db_session)
        tools1_list = tools1.get_retrieval_tools(is_warm=False)
        assert "popular_books" in [t.name for t in tools1_list]

        # Authenticated user with DB
        tools2 = internal_tools_factory(current_user=Mock(), db=mock_db_session)
        tools2_list = tools2.get_retrieval_tools(is_warm=False)
        assert "popular_books" in [t.name for t in tools2_list]

        # Without DB - still in list, but returns error when executed
        tools3 = internal_tools_factory(current_user=Mock(), db=None)
        tools3_list = tools3.get_retrieval_tools(is_warm=True)
        assert "popular_books" in [t.name for t in tools3_list]

    def test_converts_recommended_book_objects_to_dicts(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should convert RecommendedBook objects to standardized dicts."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            results = popular_tool.execute(top_k=10)

            # All results should be dicts, not objects
            assert all(isinstance(book, dict) for book in results)

    def test_creates_anonymous_user_when_not_authenticated(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should create domain user with PAD_IDX for anonymous users."""
        with (
            patch(
                "app.agents.tools.recsys.native_tools.RecommendationService"
            ) as mock_service_class,
            patch("app.agents.tools.recsys.native_tools.User") as mock_user_class,
        ):
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(current_user=None, db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            popular_tool.execute(top_k=10)

            # Should have created User with user_id=-1
            mock_user_class.assert_called_once()
            call_args = mock_user_class.call_args
            assert call_args[1]["user_id"] == -1

    def test_preserves_score_from_service(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should preserve popularity score from service."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            results = popular_tool.execute(top_k=10)

            # Scores from mock_recommendation_service
            assert results[0]["score"] == 0.95
            assert results[1]["score"] == 0.92

    def test_handles_empty_results(
        self,
        internal_tools_factory,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should handle empty recommendations gracefully."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.recommend.return_value = []
            mock_service_class.return_value = mock_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            results = popular_tool.execute(top_k=10)

            assert results == []

    def test_standardization_integration(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should call standardization on raw results."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)

            # Spy on _standardize_tool_output
            original_standardize = tools._standardize_tool_output
            standardize_called = []

            def spy_standardize(results):
                standardize_called.append(True)
                return original_standardize(results)

            tools._standardize_tool_output = spy_standardize

            retrieval_tools = tools.get_retrieval_tools(is_warm=False)
            popular_tool = next(t for t in retrieval_tools if t.name == "popular_books")
            popular_tool.execute(top_k=10)

            assert len(standardize_called) > 0
