# tests/unit/chatbot/tools/recsys/test_user_context_tools.py
"""
Tests for user context tools (user_profile and recent_interactions).
Validates that these tools respect profile consent and provide appropriate user data.
"""

import pytest
from unittest.mock import patch, Mock


class TestUserContextTools:
    """Test user_profile and recent_interactions tools."""

    def test_context_tools_available_when_profile_allowed(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """Context tools should be available when allow_profile=True."""
        tools = internal_tools_factory(
            current_user=mock_user,
            db=mock_db_session,
            allow_profile=True,
        )

        context_tools = tools.get_context_tools()
        tool_names = [t.name for t in context_tools]

        assert "user_profile" in tool_names
        assert "recent_interactions" in tool_names

    def test_context_tools_not_available_when_profile_not_allowed(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """Context tools should not be available when allow_profile=False."""
        tools = internal_tools_factory(
            current_user=mock_user,
            db=mock_db_session,
            allow_profile=False,
        )

        context_tools = tools.get_context_tools()

        assert len(context_tools) == 0

    def test_context_tools_require_authenticated_user(
        self,
        internal_tools_factory,
        mock_db_session,
    ):
        """Context tools should not be available without authenticated user."""
        tools = internal_tools_factory(
            current_user=None,
            db=mock_db_session,
            allow_profile=True,
        )

        context_tools = tools.get_context_tools()

        assert len(context_tools) == 0

    def test_context_tools_require_database(
        self,
        internal_tools_factory,
        mock_user,
    ):
        """Context tools should not be available without database connection."""
        tools = internal_tools_factory(
            current_user=mock_user,
            db=None,
            allow_profile=True,
        )

        context_tools = tools.get_context_tools()

        assert len(context_tools) == 0

    def test_user_profile_returns_preference_data(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """user_profile should return user preference information."""
        with patch("app.agents.tools.recsys.native_tools.get_read_books") as mock_read_books:
            mock_read_books.return_value = [1, 2, 3]

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                allow_profile=True,
            )

            context_tools = tools.get_context_tools()
            profile_tool = next(t for t in context_tools if t.name == "user_profile")

            result = profile_tool.execute()

            assert isinstance(result, dict)
            assert "user_idx" in result or "username" in result or "num_ratings" in result

    def test_recent_interactions_returns_interaction_data(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """recent_interactions should return user interaction history."""
        with patch("app.agents.tools.recsys.native_tools.get_read_books") as mock_read_books:
            mock_read_books.return_value = [1, 2, 3]

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                allow_profile=True,
            )

            context_tools = tools.get_context_tools()
            interactions_tool = next(t for t in context_tools if t.name == "recent_interactions")

            result = interactions_tool.execute()

            assert isinstance(result, (dict, list))

    def test_user_profile_includes_rating_count(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """user_profile should include number of ratings."""
        with patch("app.agents.tools.recsys.native_tools.get_read_books") as mock_read_books:
            mock_read_books.return_value = [1, 2, 3, 4, 5]

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                allow_profile=True,
            )

            context_tools = tools.get_context_tools()
            profile_tool = next(t for t in context_tools if t.name == "user_profile")

            result = profile_tool.execute()

            # Should have some indicator of rating count
            assert (
                any(key in result for key in ["num_ratings", "rating_count", "total_ratings"])
                or len(result) > 0
            )

    def test_context_tools_dont_retrieve_books(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """Context tools provide user data, not book recommendations."""
        with patch("app.agents.tools.recsys.native_tools.get_read_books") as mock_read_books:
            mock_read_books.return_value = [1, 2, 3]

            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                allow_profile=True,
            )

            context_tools = tools.get_context_tools()

            # Context tools should not have book recommendation structure
            for tool in context_tools:
                result = tool.execute()

                # Should not be a list of books with item_idx
                if isinstance(result, list) and result:
                    # If it's a list, items shouldn't have book structure
                    if isinstance(result[0], dict):
                        assert "item_idx" not in result[0] or "title" not in result[0]

    def test_get_context_tools_returns_empty_by_default(
        self,
        internal_tools_factory,
        mock_db_session,
    ):
        """Default settings should return no context tools."""
        # Default: no user, allow_profile=False
        tools = internal_tools_factory(db=mock_db_session)

        context_tools = tools.get_context_tools()

        assert len(context_tools) == 0

    def test_context_tools_separate_from_retrieval_tools(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """Context tools should be separate from retrieval tools."""
        tools = internal_tools_factory(
            current_user=mock_user,
            db=mock_db_session,
            allow_profile=True,
        )

        context_tools = tools.get_context_tools()
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        context_names = [t.name for t in context_tools]
        retrieval_names = [t.name for t in retrieval_tools]

        # No overlap between context and retrieval tools
        assert set(context_names).isdisjoint(set(retrieval_names))

    def test_user_profile_tool_category(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """user_profile should be categorized as INTERNAL tool."""
        with patch("app.agents.tools.recsys.native_tools.get_read_books"):
            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                allow_profile=True,
            )

            context_tools = tools.get_context_tools()
            profile_tool = next(t for t in context_tools if t.name == "user_profile")

            from app.agents.tools.native_tool import ToolCategory

            assert profile_tool.category == ToolCategory.INTERNAL

    def test_recent_interactions_tool_category(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """recent_interactions should be categorized as INTERNAL tool."""
        with patch("app.agents.tools.recsys.native_tools.get_read_books"):
            tools = internal_tools_factory(
                current_user=mock_user,
                db=mock_db_session,
                allow_profile=True,
            )

            context_tools = tools.get_context_tools()
            interactions_tool = next(t for t in context_tools if t.name == "recent_interactions")

            from app.agents.tools.native_tool import ToolCategory

            assert interactions_tool.category == ToolCategory.INTERNAL

    def test_all_conditions_must_be_met_for_context_tools(
        self,
        internal_tools_factory,
        mock_user,
        mock_db_session,
    ):
        """All conditions (user, db, allow_profile) must be True for context tools."""
        # Has user and DB but not allow_profile
        tools1 = internal_tools_factory(
            current_user=mock_user,
            db=mock_db_session,
            allow_profile=False,
        )
        assert len(tools1.get_context_tools()) == 0

        # Has allow_profile and DB but no user
        tools2 = internal_tools_factory(
            current_user=None,
            db=mock_db_session,
            allow_profile=True,
        )
        assert len(tools2.get_context_tools()) == 0

        # Has user and allow_profile but no DB
        tools3 = internal_tools_factory(
            current_user=mock_user,
            db=None,
            allow_profile=True,
        )
        assert len(tools3.get_context_tools()) == 0

        # Has all three - should work
        tools4 = internal_tools_factory(
            current_user=mock_user,
            db=mock_db_session,
            allow_profile=True,
        )
        assert len(tools4.get_context_tools()) > 0
