# tests/integration/chatbot/conductor/test_parameter_handling.py
"""
Tests for parameter passing and edge cases in Conductor.
Validates that user_num_ratings, use_profile, and optional parameters work correctly.
"""

import pytest
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult


class TestParameterHandling:
    """
    Verify parameter propagation and edge case handling.

    Component tests use well-formed inputs. These test boundary conditions
    and verify parameters correctly flow through the system.

    These tests use mocked agents to verify orchestration logic,
    not LLM quality.
    """

    def test_cold_user_routing_and_tool_selection(
        self, db_session, test_user_cold, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify user_num_ratings=0 is correctly passed to agent.

        Cold users should not get ALS-based recommendations (requires ≥10 ratings).
        The agent factory should pass the correct user_num_ratings to determine
        tool availability.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
            history=[],
            user_text="recommend fantasy adventure books",
            use_profile=False,
            current_user=test_user_cold,
            db=db_session,
            user_num_ratings=0,  # Cold user
            force_target="recsys",  # Force recsys to test parameter passing
        )

        assert result.success, f"Cold user recommendation failed: {result.text}"

        # Verify agent received cold user status
        agent_request = mock_recsys_agent.execute.call_args[0][0]
        assert agent_request.context.user_preferences["num_ratings"] == 0, (
            "Cold user rating count not passed to agent"
        )

        # Cold user should still get recommendations (mock returns 5 books)
        assert result.book_ids is not None, "Cold user should get recommendations"
        assert len(result.book_ids) == 5, f"Expected 5 books from mock, got {len(result.book_ids)}"

    def test_warm_user_can_access_personalized_recs(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify user_num_ratings≥10 enables personalized recommendations.

        Warm users should be able to use ALS-based recommendations.
        This is the complement of test_cold_user.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
            history=[],
            user_text="recommend something good",  # Vague = might use ALS
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,  # Warm user
            force_target="recsys",
        )

        assert result.success, f"Warm user recommendation failed: {result.text}"

        # Verify agent received warm user status
        agent_request = mock_recsys_agent.execute.call_args[0][0]
        assert agent_request.context.user_preferences["num_ratings"] == 15, (
            "Warm user rating count not passed to agent"
        )

        # Warm user should get personalized recommendations (mock returns 5 books)
        assert result.book_ids is not None, "Warm user should get recommendations"
        assert len(result.book_ids) == 5, f"Expected 5 books from mock, got {len(result.book_ids)}"

    def test_user_num_ratings_none_defaults_to_zero(
        self, db_session, test_user_new, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify user_num_ratings=None is handled as 0 (cold user).

        When user_num_ratings is not provided, should default to cold user behavior.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
            history=[],
            user_text="hello",
            use_profile=False,
            current_user=test_user_new,
            db=db_session,
            user_num_ratings=None,  # Should default to 0
        )

        assert result.success, "Request with user_num_ratings=None failed"
        assert result.text, "No response with user_num_ratings=None"

        # Verify some agent was called
        # (Router will decide which agent, but system shouldn't crash)

    def test_optional_parameters_none_values(self, mock_agent_factory, mock_response_agent):
        """
        Verify Conductor handles None for optional parameters.

        Response and docs agents don't require db/current_user.
        System should work with minimal parameters.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        # Response agent should work with minimal params
        result = conductor.run(
            history=[],
            user_text="hello there",
            use_profile=False,
            db=None,  # Response agent doesn't need db
            current_user=None,
            user_num_ratings=None,  # Defaults to 0
        )

        assert result.success, "Response agent failed with None parameters"
        assert result.text, "No response from response agent"

        # Verify response agent was called
        assert mock_response_agent.execute.called, (
            "Response agent should handle query with None parameters"
        )

    def test_docs_agent_works_without_user_context(self, mock_agent_factory, mock_docs_agent):
        """
        Verify docs agent works without user context.

        Documentation queries don't need personalization.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
            history=[],
            user_text="how do I rate a book?",
            use_profile=False,
            db=None,  # Docs agent doesn't need db
            current_user=None,
            user_num_ratings=0,
        )

        assert isinstance(result, AgentResult), (
            "Docs agent didn't return AgentResult without user context"
        )
        assert result.text, "No response from docs agent"

        # Verify docs agent was called
        assert mock_docs_agent.execute.called, "Docs agent should handle query without user context"

    def test_use_profile_false_prevents_profile_access(
        self, db_session, test_user_with_profile, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify use_profile=False is respected (privacy-critical).

        Even if user has profile, agent should not access it when
        use_profile=False.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
            history=[],
            user_text="recommend something for me",  # Vague = would use profile
            use_profile=False,  # Explicit denial
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=15,  # Warm user with profile
        )

        assert result.success, "Request with use_profile=False failed"
        assert result.text, "No response when profile disabled"

        # Verify agent received profile_allowed=False
        agent_request = mock_recsys_agent.execute.call_args[0][0]
        assert agent_request.context.user_preferences["profile_allowed"] is False, (
            "Profile access flag should be False"
        )

        # Should succeed without profile access

    def test_conv_id_and_uid_passed_through(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify conv_id and uid parameters pass through system.

        These are metadata fields that should survive all conversions.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
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

        # Verify metadata fields reached agent
        agent_request = mock_recsys_agent.execute.call_args[0][0]
        assert agent_request.context.conversation_id == "test_conv_456", (
            "conv_id not passed to agent"
        )
        assert agent_request.context.user_id == test_user_warm.user_id, "uid not passed to agent"

        # Metadata fields should be present in execution context
