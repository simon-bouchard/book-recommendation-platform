# tests/integration/chatbot/conductor/test_multi_turn_state.py
"""
Tests for multi-turn conversation state management in Conductor.
Validates that history persists, truncates correctly, and doesn't leak between conversations.
"""

import pytest
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult


class TestMultiTurnState:
    """
    Verify conversation state management across multiple turns.

    Component tests use static history. These tests verify that
    state actually persists correctly in sequential Conductor calls.

    These tests use mocked agents to verify orchestration logic,
    not LLM quality.
    """

    def test_history_accumulates_across_turns(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent, mock_docs_agent
    ):
        """
        Verify conversation history persists and accumulates correctly.

        Tests 3-turn conversation:
        1. Initial query → recsys
        2. Follow-up → recsys (continuation)
        3. Topic switch → docs (new intent)
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        # Turn 1: Initial query
        history = []
        result1 = conductor.run(
            history=history,
            user_text="recommend sci-fi books",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
        )

        assert result1.success, f"Turn 1 failed: {result1.text}"
        assert result1.text, "Turn 1: no response text"

        # Build history for turn 2
        history.append({"u": "recommend sci-fi books", "a": result1.text})

        # Turn 2: Follow-up (should maintain recsys context)
        mock_recsys_agent.execute.reset_mock()  # Reset call count

        result2 = conductor.run(
            history=history,
            user_text="more like that please",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
        )

        assert result2.success, f"Turn 2 failed: {result2.text}"
        assert result2.text, "Turn 2: no response text"

        # Verify agent received history from turn 1
        agent_request_turn2 = mock_recsys_agent.execute.call_args[0][0]
        assert len(agent_request_turn2.conversation_history) >= 1, (
            "Turn 2 should have history from turn 1"
        )

        # Turn 3: Topic switch (should route to docs)
        history.append({"u": "more like that please", "a": result2.text})

        result3 = conductor.run(
            history=history,
            user_text="how do I rate a book?",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
        )

        assert result3.success, f"Turn 3 failed: {result3.text}"
        assert result3.text, "Turn 3: no response text"

        # Verify docs agent was called (topic switched)
        assert mock_docs_agent.execute.called, "Turn 3 should route to docs agent"

        # All turns should succeed with proper history management

    def test_history_truncation_with_hist_turns(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify hist_turns parameter correctly limits history to branch agents.

        Creates 5-turn conversation, then executes with hist_turns=3.
        Agent should only see last 3 turns, not all 5.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        # Build 5-turn conversation history
        history = [
            {"u": "turn 1 query", "a": "turn 1 response"},
            {"u": "turn 2 query", "a": "turn 2 response"},
            {"u": "turn 3 query", "a": "turn 3 response"},
            {"u": "turn 4 query", "a": "turn 4 response"},
            {"u": "turn 5 query", "a": "turn 5 response"},
        ]

        # Execute with hist_turns=3 (should only see last 3 turns)
        result = conductor.run(
            history=history,
            user_text="turn 6 query",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            hist_turns=3,  # Only last 3 turns to agent
            router_k_user=2,  # Router sees last 2 user messages
        )

        assert result.success, f"Truncation test failed: {result.text}"
        assert result.text, "No response with truncated history"

        # Verify agent received exactly 3 history turns
        agent_request = mock_recsys_agent.execute.call_args[0][0]
        assert len(agent_request.conversation_history) == 3, (
            f"Expected 3 history turns, got {len(agent_request.conversation_history)}"
        )

        # Verify the last 3 turns were sent (not first 3)
        first_turn_in_history = agent_request.conversation_history[0]
        assert "turn 3" in first_turn_in_history["u"], (
            "Should have turns 3, 4, 5 (last 3), not turns 1, 2, 3"
        )

    def test_history_truncation_edge_case_fewer_than_hist_turns(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify hist_turns handles edge case when history < hist_turns.

        If history has 2 turns but hist_turns=3, should use all 2 turns.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        history = [
            {"u": "first query", "a": "first response"},
            {"u": "second query", "a": "second response"},
        ]

        # Request hist_turns=5 but only 2 turns available
        result = conductor.run(
            history=history,
            user_text="third query",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            hist_turns=5,  # More than available
        )

        # Should complete without crashing
        assert isinstance(result, AgentResult), (
            "Failed to return AgentResult with hist_turns > len(history)"
        )
        assert result.text, "No response when hist_turns > history length"

        # Verify agent received all available history (2 turns)
        agent_request = mock_recsys_agent.execute.call_args[0][0]
        assert len(agent_request.conversation_history) == 2, (
            f"Expected 2 history turns (all available), got {len(agent_request.conversation_history)}"
        )

    def test_conversations_are_isolated(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent
    ):
        """
        Verify multiple conversations don't leak state.

        Same Conductor instance with different conv_id should be isolated.
        Tests that history for conv A doesn't affect conv B.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory  # Inject mocks

        # Conversation A
        history_a = []
        result_a1 = conductor.run(
            history=history_a,
            user_text="recommend fantasy",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_a",
        )

        assert result_a1.success, "Conv A turn 1 failed"

        history_a.append({"u": "recommend fantasy", "a": result_a1.text})

        # Reset mock to track conv B separately
        mock_recsys_agent.execute.reset_mock()

        # Conversation B (separate conv_id, empty history)
        history_b = []
        result_b1 = conductor.run(
            history=history_b,  # Empty - fresh conversation
            user_text="recommend thriller",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_b",
        )

        assert result_b1.success, "Conv B turn 1 failed"

        # Verify conv B agent received empty history (not conv A's history)
        agent_request_b = mock_recsys_agent.execute.call_args[0][0]
        assert len(agent_request_b.conversation_history) == 0, (
            "Conv B should have empty history, not conv A's history"
        )

        # Reset mock for conv A continuation
        mock_recsys_agent.execute.reset_mock()

        # Continue conversation A
        result_a2 = conductor.run(
            history=history_a,
            user_text="more epic fantasy",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_a",
        )

        assert result_a2.success, "Conv A turn 2 failed"

        # Verify conv A agent received its own history (1 turn)
        agent_request_a2 = mock_recsys_agent.execute.call_args[0][0]
        assert len(agent_request_a2.conversation_history) >= 1, (
            "Conv A turn 2 should have history from turn 1"
        )

        # Both conversations should succeed independently

    def test_empty_history_first_turn(
        self, db_session, test_user_warm, mock_agent_factory, mock_response_agent
    ):
        """
        Verify empty history (first turn) works correctly.

        Edge case: no conversation history yet.
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
        )

        assert isinstance(result, AgentResult), (
            "First turn with empty history didn't return AgentResult"
        )
        assert result.text, "No response on first turn"

        # Verify agent received empty history
        assert mock_response_agent.execute.called, "Some agent should be called on first turn"
