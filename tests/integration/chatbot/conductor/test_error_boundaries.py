# tests/integration/chatbot/conductor/test_error_boundaries.py
"""
Tests for error handling and resilience in Conductor.
Validates that failures are caught and handled gracefully at orchestration level.
"""

import pytest
from unittest.mock import Mock
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult, RoutePlan
from app.agents.domain.entities import AgentResponse


class TestErrorBoundaries:
    """
    Verify Conductor handles failures gracefully.

    Component tests assume happy path. These test failure modes
    and verify system doesn't crash or leak exceptions.

    These tests use mocked agents to inject failures and verify
    error handling, not LLM quality.
    """

    def test_agent_execution_failure_bubbles_up(self, db_session, mock_agent_factory):
        """
        Verify agent execution failures bubble up as exceptions.

        Current behavior: Conductor does not catch exceptions from agents.
        This test verifies that behavior is consistent.
        """
        conductor = Conductor()

        # Create a factory that returns a failing agent
        failing_agent = Mock()
        failing_agent.execute.side_effect = RuntimeError("Simulated agent execution failure")

        failing_factory = Mock()
        failing_factory.create_agent = Mock(return_value=failing_agent)

        conductor.factory = failing_factory

        # Exception should bubble up (not caught by Conductor)
        with pytest.raises(RuntimeError, match="Simulated agent execution failure"):
            conductor.run(
                history=[],
                user_text="recommend books",
                use_profile=False,
                db=db_session,
                user_num_ratings=10,
            )

    def test_router_classification_failure_bubbles_up(self, db_session, mock_agent_factory):
        """
        Verify router classification failures bubble up as exceptions.

        Current behavior: Conductor does not catch router exceptions.
        This test verifies that behavior is consistent.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        # Mock router to raise exception
        failing_router = Mock()
        failing_router.classify.side_effect = RuntimeError("Router classification failed")

        conductor.router = failing_router

        # Exception should bubble up (not caught by Conductor)
        with pytest.raises(RuntimeError, match="Router classification failed"):
            conductor.run(
                history=[],
                user_text="test query",
                use_profile=False,
                db=db_session,
                user_num_ratings=0,
            )

    def test_empty_query_handled_gracefully(
        self, db_session, mock_agent_factory, mock_response_agent
    ):
        """
        Verify empty user_text doesn't break system.

        Edge case: user submits empty string.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
            history=[],
            user_text="",  # Empty query
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
        )

        # Should handle gracefully (route to response agent)
        assert isinstance(result, AgentResult), "Should return AgentResult for empty query"
        assert result.text, "Should have response even for empty query"

        # Verify some agent was called (likely response agent)
        assert mock_response_agent.execute.called, "Response agent should handle empty query"

    def test_whitespace_only_query_handled(
        self, db_session, mock_agent_factory, mock_response_agent
    ):
        """
        Verify whitespace-only query is handled.

        Edge case: user submits only spaces/newlines.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
            history=[],
            user_text="   \n\t  ",  # Whitespace only
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
        )

        assert isinstance(result, AgentResult), "Should return AgentResult for whitespace query"
        assert result.text, "Should have response for whitespace query"

    def test_very_long_query_handled(self, db_session, mock_agent_factory, mock_recsys_agent):
        """
        Verify extremely long queries don't break system.

        Edge case: user submits massive query (potential token overflow).
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        # Generate very long query (1000 words)
        long_query = " ".join(["word"] * 1000)

        result = conductor.run(
            history=[],
            user_text=long_query,
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
        )

        # Should handle gracefully (may truncate or error, but not crash)
        assert isinstance(result, AgentResult), "Should return AgentResult for very long query"
        assert result.text, "Should have response for long query"

        # Verify agent received the query (or truncated version)
        assert mock_recsys_agent.execute.called or any(
            agent.execute.called for agent in [mock_recsys_agent]
        ), "Some agent should be called"

    def test_malformed_history_handled(self, db_session, mock_agent_factory, mock_recsys_agent):
        """
        Verify malformed history doesn't crash system.

        Edge case: history has missing or extra fields.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        # History with missing 'a' field
        malformed_history = [
            {"u": "first message"},  # Missing 'a'
            {"u": "second message", "a": "response"},
        ]

        result = conductor.run(
            history=malformed_history,
            user_text="test",
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
        )

        # Should handle gracefully
        assert isinstance(result, AgentResult), "Should return AgentResult with malformed history"

        # System should not crash - that's the key test

    def test_database_none_for_agent_requiring_db(self, mock_agent_factory, mock_recsys_agent):
        """
        Verify recsys agent handles db=None appropriately.

        Recsys requires database but might be called with db=None.
        Should either work without db or fail clearly.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
            history=[],
            user_text="recommend books",
            use_profile=False,
            db=None,  # Recsys needs db
            current_user=None,
            user_num_ratings=10,
            force_target="recsys",  # Force recsys
        )

        # Should return an AgentResult (success may vary)
        assert isinstance(result, AgentResult), "Should return AgentResult when db=None"

        # Should have some response text
        assert result.text, "Should have response text when db required but missing"

        # Verify recsys agent was called (with db=None in context)
        assert mock_recsys_agent.execute.called, "Recsys agent should be called even with db=None"
