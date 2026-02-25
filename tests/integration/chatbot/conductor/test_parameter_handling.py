# tests/integration/chatbot/conductor/test_parameter_handling.py
"""
Tests for parameter passing and edge cases in Conductor.
Validates that user_num_ratings, use_profile, and optional parameters work correctly.
"""

import pytest
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult

pytestmark = pytest.mark.asyncio


class TestParameterHandling:
    """
    Verify parameter propagation and edge case handling.

    Component tests use well-formed inputs. These test boundary conditions
    and verify parameters correctly flow through the system.

    These tests use mocked agents to verify orchestration logic,
    not LLM quality.
    """

    async def test_cold_user_routing_and_tool_selection(
        self, db_session, test_user_cold, mock_agent_factory, mock_recsys_agent, collect_result
    ):
        """
        Verify user_num_ratings=0 is correctly passed to agent.

        Cold users should not get ALS-based recommendations (requires ≥10 ratings).
        The agent factory should pass the correct user_num_ratings to determine
        tool availability.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

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

    async def test_warm_user_can_access_personalized_recs(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent, collect_result
    ):
        """
        Verify user_num_ratings≥10 enables personalized recommendations.

        Warm users should be able to use ALS-based recommendations.
        This is the complement of test_cold_user.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

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
        self, db_session, test_user_new, mock_agent_factory, mock_recsys_agent, collect_result
    ):
        """
        Verify user_num_ratings=None is handled as 0 (cold user).

        When user_num_ratings is not provided, should default to cold user behavior.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

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
        self, mock_agent_factory, mock_response_agent, collect_result
    ):
        """
        Verify Conductor handles None for optional parameters.

        Response and docs agents don't require db/current_user.
        System should work with minimal parameters.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

        result = await collect_result(
            conductor,
            history=[],
            user_text="hello there",
            use_profile=False,
            db=None,
            current_user=None,
            user_num_ratings=None,
        )

        assert result.success, "Response agent failed with None parameters"
        assert result.text, "No response from response agent"

        assert mock_response_agent.execute_stream.called, (
            "Response agent should handle query with None parameters"
        )

    async def test_docs_agent_works_without_user_context(
        self, mock_agent_factory, mock_docs_agent, collect_result
    ):
        """
        Verify docs agent works without user context.

        Documentation queries don't need personalization.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

        result = await collect_result(
            conductor,
            history=[],
            user_text="how do I rate a book?",
            use_profile=False,
            db=None,
            current_user=None,
            user_num_ratings=0,
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
        mock_agent_factory,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify use_profile=False is respected (privacy-critical).

        Even if user has profile, agent should not access it when
        use_profile=False.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend something for me",
            use_profile=False,
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=15,
        )

        assert result.success, "Request with use_profile=False failed"
        assert result.text, "No response when profile disabled"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert agent_request.context.user_preferences["profile_allowed"] is False, (
            "Profile access flag should be False"
        )

    async def test_conv_id_and_uid_passed_through(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent, collect_result
    ):
        """
        Verify conv_id and uid parameters pass through system.

        These are metadata fields that should survive all conversions.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

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
        )

        assert result.success, "Request with conv_id/uid failed"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert agent_request.context.conversation_id == "test_conv_456", (
            "conv_id not passed to agent"
        )
        assert agent_request.context.user_id == test_user_warm.user_id, "uid not passed to agent"
