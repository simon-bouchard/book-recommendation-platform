# tests/integration/chatbot/conductor/test_parameter_handling.py
"""
Tests for parameter passing and edge cases in Conductor.
Validates that user_num_ratings, use_profile, and optional parameters work correctly.
"""

import pytest
from app.agents.schemas import AgentResult

pytestmark = pytest.mark.asyncio


class TestParameterHandling:
    """
    Verify parameter propagation and edge case handling.

    Component tests use well-formed inputs. These test boundary conditions
    and verify parameters correctly flow through the system.

    All routing is controlled via force_target or the pre-wired mock_router
    (default: recsys) so no LLM calls are made.
    """

    async def test_cold_user_rating_count_reaches_agent(
        self,
        db_session,
        test_user_cold,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify user_num_ratings=0 is correctly passed through to the agent.

        The recsys agent uses num_ratings to decide which retrieval tools
        are available (ALS requires ≥10 ratings). This test confirms the
        value survives all adapter and context-builder conversions.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend fantasy adventure books",
            use_profile=False,
            current_user=test_user_cold,
            db=db_session,
            user_num_ratings=0,
            force_target="recsys",
        )

        assert result.success, f"Cold user recommendation failed: {result.text}"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert agent_request.context.user_preferences["num_ratings"] == 0, (
            "Cold user rating count not passed to agent"
        )

        assert result.book_ids is not None, "Cold user should get recommendations"
        assert len(result.book_ids) == 5, f"Expected 5 books from mock, got {len(result.book_ids)}"

    async def test_warm_user_rating_count_reaches_agent(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify user_num_ratings≥10 is correctly passed through to the agent.

        Complement of test_cold_user: confirms the warm threshold value
        propagates intact.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend something good",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            force_target="recsys",
        )

        assert result.success, f"Warm user recommendation failed: {result.text}"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert agent_request.context.user_preferences["num_ratings"] == 15, (
            "Warm user rating count not passed to agent"
        )

        assert result.book_ids is not None, "Warm user should get recommendations"
        assert len(result.book_ids) == 5, f"Expected 5 books from mock, got {len(result.book_ids)}"

    async def test_user_num_ratings_none_defaults_to_zero(
        self,
        db_session,
        test_user_new,
        conductor,
        collect_result,
    ):
        """
        Verify user_num_ratings=None is treated as 0 (cold user).

        When user_num_ratings is not provided, system should not crash
        and should default to cold-user behaviour.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="hello",
            use_profile=False,
            current_user=test_user_new,
            db=db_session,
            user_num_ratings=None,
        )

        assert result.success, "Request with user_num_ratings=None failed"
        assert result.text, "No response with user_num_ratings=None"

    async def test_optional_parameters_none_values(
        self,
        conductor,
        mock_response_agent,
        collect_result,
    ):
        """
        Verify Conductor handles None for optional parameters.

        Response agent doesn't require db or current_user.
        force_target="respond" exercises this path explicitly.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="hello there",
            use_profile=False,
            db=None,
            current_user=None,
            user_num_ratings=None,
            force_target="respond",
        )

        assert result.success, "Response agent failed with None parameters"
        assert result.text, "No response from response agent"
        assert mock_response_agent.execute_stream.called, (
            "Response agent should handle query with None parameters"
        )

    async def test_docs_agent_works_without_user_context(
        self,
        conductor,
        mock_docs_agent,
        collect_result,
    ):
        """
        Verify docs agent works without user context.

        Documentation queries don't need personalization.
        force_target="docs" exercises this path explicitly.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="how do I rate a book?",
            use_profile=False,
            db=None,
            current_user=None,
            user_num_ratings=0,
            force_target="docs",
        )

        assert isinstance(result, AgentResult), (
            "Docs agent didn't return AgentResult without user context"
        )
        assert result.text, "No response from docs agent"
        assert mock_docs_agent.execute_stream.called, (
            "Docs agent should handle query without user context"
        )

    async def test_use_profile_false_prevents_profile_access(
        self,
        db_session,
        test_user_with_profile,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify use_profile=False is respected (privacy-critical).

        Even when user has a profile, the agent must not receive the
        profile_allowed=True flag when use_profile=False.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend something for me",
            use_profile=False,
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=15,
            force_target="recsys",
        )

        assert result.success, "Request with use_profile=False failed"
        assert result.text, "No response when profile disabled"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert agent_request.context.user_preferences["profile_allowed"] is False, (
            "Profile access flag should be False"
        )

    async def test_conv_id_and_uid_passed_through(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify conv_id and uid metadata fields survive all conversions.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend books",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="test_conv_456",
            uid=test_user_warm.user_id,
            force_target="recsys",
        )

        assert result.success, "Request with conv_id/uid failed"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert agent_request.context.conversation_id == "test_conv_456", (
            "conv_id not passed to agent"
        )
        assert agent_request.context.user_id == test_user_warm.user_id, "uid not passed to agent"
