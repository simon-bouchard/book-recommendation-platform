# tests/unit/chatbot/tools/recsys/test_subject_hybrid.py
"""
Tests for the subject_hybrid_pool tool (subject-based recommendations with popularity blending).
Validates subject-based retrieval, parameter handling, and database requirements.
"""

import pytest
from unittest.mock import patch, Mock


class TestSubjectHybridTool:
    """Test subject_hybrid_pool tool behavior."""

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

            subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")
            results = subject_tool.execute(fav_subjects_idxs=["1", "2"], top_k=10)

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
        """Subject hybrid returns basic metadata only."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")
            results = subject_tool.execute(fav_subjects_idxs=["1", "2"], top_k=10)

            book = results[0]
            assert "subjects" not in book
            assert "tones" not in book
            assert "genre" not in book
            assert "vibe" not in book

    def test_requires_database_connection(
        self,
        internal_tools_factory,
    ):
        """Subject hybrid tool should not be available without database."""
        tools = internal_tools_factory(db=None)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        # Tool should not be in the list when no database
        tool_names = [t.name for t in retrieval_tools]
        assert "subject_hybrid_pool" not in tool_names

    def test_requires_subject_indices(
        self,
        internal_tools_factory,
        mock_db_session,
    ):
        """Should return error when no subjects provided."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")

        # Empty list
        results = subject_tool.execute(fav_subjects_idxs=[], top_k=10)
        assert len(results) == 1
        assert "error" in results[0]
        assert "at least one subject" in results[0]["error"].lower()

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
            subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")

            # Test minimum
            subject_tool.execute(fav_subjects_idxs=["1"], top_k=0)
            call_args = mock_service.recommend.call_args
            assert call_args[0][1].k == 1

            # Test maximum
            subject_tool.execute(fav_subjects_idxs=["1"], top_k=1000)
            call_args = mock_service.recommend.call_args
            assert call_args[0][1].k == 500

    def test_validates_subject_weight(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should clamp subject_weight to 0-1 range."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.recommend.return_value = []
            mock_service_class.return_value = mock_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)
            subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")

            # Test below minimum
            subject_tool.execute(fav_subjects_idxs=["1"], subject_weight=-0.5)
            call_args = mock_service.recommend.call_args
            assert call_args[0][1].hybrid_config.subject_weight == 0.0

            # Test above maximum
            subject_tool.execute(fav_subjects_idxs=["1"], subject_weight=1.5)
            call_args = mock_service.recommend.call_args
            assert call_args[0][1].hybrid_config.subject_weight == 1.0

    def test_calls_recommendation_service_correctly(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should call RecommendationService with subject mode."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")
            subject_tool.execute(fav_subjects_idxs=["1", "2", "3"], top_k=50, subject_weight=0.7)

            # Verify service was called
            mock_recommendation_service.recommend.assert_called_once()

            # Check arguments
            call_args = mock_recommendation_service.recommend.call_args
            domain_user = call_args[0][0]
            config = call_args[0][1]
            db_session = call_args[0][2]

            assert config.mode == "subject"
            assert config.k == 50
            assert config.hybrid_config.subject_weight == 0.7
            assert db_session == mock_db_session

    def test_converts_subject_indices_list(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should pass subject indices to domain user correctly."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")
            subject_tool.execute(fav_subjects_idxs=["1", "2", "3"], top_k=10)

            # Check domain user has correct subject indices
            call_args = mock_recommendation_service.recommend.call_args
            domain_user = call_args[0][0]

            # Subject indices should be converted to list and passed to user
            assert hasattr(domain_user, "fav_subjects_idxs") or hasattr(
                domain_user, "subject_indices"
            )

    def test_works_without_authentication(
        self,
        internal_tools_factory,
        mock_recommendation_service,
        mock_load_book_meta,
        mock_db_session,
    ):
        """Should work for anonymous users with subject preferences."""
        with patch(
            "app.agents.tools.recsys.native_tools.RecommendationService"
        ) as mock_service_class:
            mock_service_class.return_value = mock_recommendation_service

            tools = internal_tools_factory(current_user=None, db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")
            results = subject_tool.execute(fav_subjects_idxs=["1", "2"], top_k=10)

            assert isinstance(results, list)
            assert len(results) > 0

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
            mock_service.recommend.side_effect = Exception("Subject model error")
            mock_service_class.return_value = mock_service

            tools = internal_tools_factory(db=mock_db_session)
            retrieval_tools = tools.get_retrieval_tools(is_warm=False)

            subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")
            results = subject_tool.execute(fav_subjects_idxs=["1", "2"], top_k=10)

            assert len(results) == 1
            assert "error" in results[0]
            assert "Subject model error" in results[0]["error"]

    def test_available_when_db_present(
        self,
        internal_tools_factory,
        mock_db_session,
    ):
        """Subject hybrid should only be available with database connection."""
        # With database - should be present
        tools_with_db = internal_tools_factory(db=mock_db_session)
        tools_list = tools_with_db.get_retrieval_tools(is_warm=False)
        tool_names = [t.name for t in tools_list]
        assert "subject_hybrid_pool" in tool_names

        # Without database - should NOT be present
        tools_without_db = internal_tools_factory(db=None)
        tools_list = tools_without_db.get_retrieval_tools(is_warm=False)
        tool_names = [t.name for t in tools_list]
        assert "subject_hybrid_pool" not in tool_names

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

            subject_tool = next(t for t in retrieval_tools if t.name == "subject_hybrid_pool")
            results = subject_tool.execute(fav_subjects_idxs=["1", "2"], top_k=10)

            # num_ratings should come from mock_book_meta
            assert results[0]["num_ratings"] == 100
            assert results[1]["num_ratings"] == 200
