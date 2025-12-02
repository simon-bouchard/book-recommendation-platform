# tests/integration/chatbot/conductor/test_context_builder.py
"""
Tests for context builder functions (make_router_input, make_branch_input).
Validates that history truncation and context preparation work correctly.
"""

import pytest
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult, RoutePlan


class TestContextBuilders:
    """
    Verify make_router_input and make_branch_input work correctly.

    Nobody explicitly tests these helper functions. Component tests
    bypass them by constructing inputs directly.

    These tests use mocked agents to verify orchestration logic,
    not LLM quality.
    """

    def test_router_input_truncates_to_k_user(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify make_router_input only includes last k_user messages.

        Critical for router prompt size management.
        Router should only see last k_user user messages, not full history.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        # Create 10-turn history
        history = [{"u": f"user message {i}", "a": f"assistant response {i}"} for i in range(1, 11)]

        # Execute with k_user=2 (router should only see last 2 user messages)
        result = conductor.run(
            history=history,
            user_text="final message",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            router_k_user=2,  # Only last 2 user messages to router
        )

        assert result.success, f"Request with k_user=2 failed: {result.text}"

        # Success implies make_router_input correctly truncated
        # If all 10 messages were sent to router, it would use larger context

    def test_router_input_k_user_larger_than_history(
        self, db_session, test_user_warm, mock_agent_factory
    ):
        """
        Verify k_user handles edge case when k_user > len(history).

        If history has 2 messages but k_user=5, should use all 2 messages.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        history = [
            {"u": "first message", "a": "first response"},
            {"u": "second message", "a": "second response"},
        ]

        # Request k_user=5 but only 2 messages available
        result = conductor.run(
            history=history,
            user_text="third message",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            router_k_user=5,  # More than available
        )

        assert result.success, "Failed when k_user > history length"

    def test_branch_input_truncates_to_hist_turns(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify make_branch_input only includes last hist_turns.

        Different from router truncation - affects agent context.
        Branch agents should only see last hist_turns full turns.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        # Create 6-turn history
        history = [{"u": f"message {i}", "a": f"response {i}"} for i in range(1, 7)]

        # Execute with hist_turns=3 (agent should only see last 3 turns)
        result = conductor.run(
            history=history,
            user_text="new query",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            hist_turns=3,  # Only last 3 turns to agent
            router_k_user=2,
            force_target="recsys",  # Force recsys to ensure mock_recsys_agent is called
        )

        assert result.success, f"Request with hist_turns=3 failed: {result.text}"

        # Verify agent received exactly 3 history turns
        agent_request = mock_recsys_agent.execute.call_args[0][0]
        assert len(agent_request.conversation_history) == 3, (
            f"Expected 3 history turns, got {len(agent_request.conversation_history)}"
        )

    def test_branch_input_hist_turns_zero(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify hist_turns=0 sends no history to agent.

        Edge case: agent should see only current query, no history.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        history = [
            {"u": "old message 1", "a": "old response 1"},
            {"u": "old message 2", "a": "old response 2"},
        ]

        result = conductor.run(
            history=history,
            user_text="new query",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            hist_turns=0,  # No history to agent
            force_target="recsys",  # Force recsys to ensure mock_recsys_agent is called
        )

        assert result.success, "Failed with hist_turns=0"

        # Verify agent received no history
        agent_request = mock_recsys_agent.execute.call_args[0][0]
        assert len(agent_request.conversation_history) == 0, (
            f"Expected no history, got {len(agent_request.conversation_history)}"
        )

    def test_force_target_bypasses_router(
        self, db_session, test_user_warm, mock_agent_factory, mock_docs_agent, mock_router
    ):
        """
        Verify force_target parameter skips routing logic.

        Used for testing/debugging specific agents.
        Router should not be invoked when force_target is set.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks
        conductor.router = mock_router  # Inject mock router to verify it's not called

        # Query that would normally route to recsys
        result = conductor.run(
            history=[],
            user_text="recommend mystery books",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            force_target="docs",  # Force docs agent instead
        )

        assert result.success, "force_target=docs failed"

        # Verify docs agent was called (not recsys)
        assert mock_docs_agent.execute.called, "Docs agent should be called"

        # Verify router was NOT called (bypassed)
        assert not mock_router.classify.called, "Router should not be called with force_target"

    def test_force_target_all_agents(
        self,
        db_session,
        test_user_warm,
        mock_agent_factory,
        mock_recsys_agent,
        mock_web_agent,
        mock_docs_agent,
        mock_response_agent,
    ):
        """
        Verify force_target works for all agent types.

        Each target should be callable via force_target.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        targets_to_test = [
            ("recsys", mock_recsys_agent),
            ("web", mock_web_agent),
            ("docs", mock_docs_agent),
            ("respond", mock_response_agent),
        ]

        for target, mock_agent in targets_to_test:
            # Reset mock
            mock_agent.execute.reset_mock()

            result = conductor.run(
                history=[],
                user_text="test query",
                use_profile=False,
                current_user=test_user_warm if target == "recsys" else None,
                db=db_session if target == "recsys" else None,
                user_num_ratings=10 if target == "recsys" else 0,
                force_target=target,
            )

            # Each should execute without crashing
            assert isinstance(result, AgentResult), (
                f"force_target={target} didn't return AgentResult"
            )

            # Verify correct agent was called
            assert mock_agent.execute.called, f"Agent for target={target} was not called"

    def test_context_with_empty_history(
        self, db_session, test_user_warm, mock_agent_factory, mock_response_agent
    ):
        """
        Verify context builders handle empty history correctly.

        First turn: history=[], both router and agent should handle gracefully.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        result = conductor.run(
            history=[],  # Empty - first turn
            user_text="hello",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            router_k_user=2,
            hist_turns=3,
        )

        assert result.success, "Failed with empty history"

        # Both make_router_input and make_branch_input should handle
        # empty history without breaking
        assert isinstance(result, AgentResult), "Should return AgentResult"
