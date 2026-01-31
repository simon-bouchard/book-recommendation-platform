# tests/unit/chatbot/tools/recsys/test_als_recs.py
"""
Tests for the als_recs tool (collaborative filtering recommendations).
Validates authentication requirements, service integration, and basic metadata handling.
"""

import pytest
from unittest.mock import patch, Mock


class TestALSRecsTool:
    """Test als_recs collaborative filtering tool."""

    def test_returns_correct_schema(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_user,
        mock_db_session,
    ):
        """Should return standardized list with basic metadata."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                user_num_ratings=25,
            )
            retrieval_tools = tools.get_retrieval_tools(is_warm=True)

            als_tool = next(t for t in retrieval_tools if t.name == "als_recs")
            results = als_tool.execute(top_k=10)

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
        mock_user,
        mock_db_session,
    ):
        """ALS returns basic metadata only, no enrichment fields."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                user_num_ratings=25,
            )
            retrieval_tools = tools.get_retrieval_tools(is_warm=True)

            als_tool = next(t for t in retrieval_tools if t.name == "als_recs")
            results = als_tool.execute(top_k=10)

            book = results[0]
            assert "subjects" not in book
            assert "tones" not in book
            assert "genre" not in book
            assert "vibe" not in book

    def test_requires_authenticated_user(
        self,
        internal_tools_factory,
        mock_db_session,
    ):
        """ALS tool should not be available for unauthenticated users."""
        tools = internal_tools_factory(
            current_user=None,
            db=mock_db_session,
            user_num_ratings=0,
        )
        # ALS requires warm user, so pass is_warm=False to test it's not available
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        tool_names = [t.name for t in retrieval_tools]
        assert "als_recs" not in tool_names

    def test_requires_database_connection(
        self,
        internal_tools_factory,
        mock_user,
    ):
        """ALS tool should not be available without database connection."""
        tools = internal_tools_factory(
            current_user=mock_user,
            db=None,
            user_num_ratings=25,
        )
        retrieval_tools = tools.get_retrieval_tools(is_warm=True)

        tool_names = [t.name for t in retrieval_tools]
        assert "als_recs" not in tool_names

    def test_handles_top_k_limits(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_user,
        mock_db_session,
    ):
        """Should enforce top_k bounds (1 to 500)."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.recommend.return_value = []
            mock_service_class.return_value = mock_service

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                user_num_ratings=25,
            )
            retrieval_tools = tools.get_retrieval_tools(is_warm=True)
            als_tool = next(t for t in retrieval_tools if t.name == "als_recs")

            # Test minimum
            als_tool.execute(top_k=0)
            call_args = mock_service.recommend.call_args
            assert call_args[0][1].k == 1

            # Test maximum
            als_tool.execute(top_k=1000)
            call_args = mock_service.recommend.call_args
            assert call_args[0][1].k == 500

    def test_calls_recommendation_service_correctly(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_user,
        mock_db_session,
    ):
        """Should call RecommendationService with ALS mode."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                user_num_ratings=25,
            )
            retrieval_tools = tools.get_retrieval_tools(is_warm=True)

            als_tool = next(t for t in retrieval_tools if t.name == "als_recs")
            als_tool.execute(top_k=50)

            # Verify service was called
            mock_recommendation_service.recommend.assert_called_once()

            # Check arguments
            call_args = mock_recommendation_service.recommend.call_args
            domain_user = call_args[0][0]
            config = call_args[0][1]
            db_session = call_args[0][2]

            assert config.mode == "als"
            assert config.k == 50
            assert db_session == mock_db_session

    def test_converts_recommended_book_objects_to_dicts(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_user,
        mock_db_session,
    ):
        """Should convert RecommendedBook objects to standardized dicts."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                user_num_ratings=25,
            )
            retrieval_tools = tools.get_retrieval_tools(is_warm=True)

            als_tool = next(t for t in retrieval_tools if t.name == "als_recs")
            results = als_tool.execute(top_k=10)

            # All results should be dicts, not objects
            assert all(isinstance(book, dict) for book in results)

    def test_only_available_for_warm_users(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """ALS tool should only be available when is_warm=True."""
        tools = internal_tools_factory(
            current_user=mock_user,
            db=mock_db_session,
            user_num_ratings=25,
        )

        # Cold user - no ALS tool
        cold_tools = tools.get_retrieval_tools(is_warm=False)
        cold_tool_names = [t.name for t in cold_tools]
        assert "als_recs" not in cold_tool_names

        # Warm user - has ALS tool
        warm_tools = tools.get_retrieval_tools(is_warm=True)
        warm_tool_names = [t.name for t in warm_tools]
        assert "als_recs" in warm_tool_names

    def test_returns_error_on_exception(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
        mock_load_book_meta,
    ):
        """Should return error dict when recommendation service fails."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.recommend.side_effect = Exception("Model error")
            mock_service_class.return_value = mock_service

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                user_num_ratings=25,
            )
            retrieval_tools = tools.get_retrieval_tools(is_warm=True)

            als_tool = next(t for t in retrieval_tools if t.name == "als_recs")
            results = als_tool.execute(top_k=10)

            assert len(results) == 1
            assert "error" in results[0]
            assert "Model error" in results[0]["error"]

    def test_adds_num_ratings_from_book_meta(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_user,
        mock_db_session,
    ):
        """Should add num_ratings field from book metadata."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                user_num_ratings=25,
            )
            retrieval_tools = tools.get_retrieval_tools(is_warm=True)

            als_tool = next(t for t in retrieval_tools if t.name == "als_recs")
            results = als_tool.execute(top_k=10)

            # num_ratings should come from mock_book_meta
            assert results[0]["num_ratings"] == 100
            assert results[1]["num_ratings"] == 200

    def test_preserves_score_from_service(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_user,
        mock_db_session,
    ):
        """Should preserve ALS similarity score from service."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                user_num_ratings=25,
            )
            retrieval_tools = tools.get_retrieval_tools(is_warm=True)

            als_tool = next(t for t in retrieval_tools if t.name == "als_recs")
            results = als_tool.execute(top_k=10)

            # Scores from mock_recommendation_service
            assert results[0]["score"] == 0.95
            assert results[1]["score"] == 0.92
