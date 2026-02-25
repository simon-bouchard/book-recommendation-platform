# tests/integration/chatbot/conductor/test_multi_turn_state.py
"""
Tests for multi-turn conversation state management in Conductor.
Validates that history persists, truncates correctly, and doesn't leak between conversations.
"""

import pytest
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult

pytestmark = pytest.mark.asyncio


class TestMultiTurnState:
    """
    Verify conversation state management across multiple turns.

    Component tests use static history. These tests verify that
    state actually persists correctly in sequential Conductor calls.

    These tests use mocked agents to verify orchestration logic,
    not LLM quality.
    """

    async def test_history_accumulates_across_turns(
        self,
        db_session,
        test_user_warm,
        mock_agent_factory,
        mock_recsys_agent,
        mock_docs_agent,
        collect_result,
    ):
        """
        Verify conversation history persists and accumulates correctly.

        Tests 3-turn conversation:
        1. Initial query → recsys
        2. Follow-up → recsys (continuation)
        3. Topic switch → docs (new intent)
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

        # Turn 1: Initial query
        history = []
        result1 = await collect_result(
            conductor,
            history=history,
            user_text="recommend sci-fi books",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
        )

        assert result1.success, f"Turn 1 failed: {result1.text}"
        assert result1.text, "Turn 1: no response text"

        history.append({"u": "recommend sci-fi books", "a": result1.text})

        # Turn 2: Follow-up (should maintain recsys context)
        mock_recsys_agent.execute_stream.reset_mock()

        result2 = await collect_result(
            conductor,
            history=history,
            user_text="more like that please",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
        )

        assert result2.success, f"Turn 2 failed: {result2.text}"
        assert result2.text, "Turn 2: no response text"

        agent_request_turn2 = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request_turn2.conversation_history) >= 1, (
            "Turn 2 should have history from turn 1"
        )

        history.append({"u": "more like that please", "a": result2.text})

        # Turn 3: Topic switch (should route to docs)
        result3 = await collect_result(
            conductor,
            history=history,
            user_text="how do I rate a book?",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
        )

        assert result3.success, f"Turn 3 failed: {result3.text}"
        assert result3.text, "Turn 3: no response text"

        assert mock_docs_agent.execute_stream.called, "Turn 3 should route to docs agent"

    async def test_history_truncation_with_hist_turns(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent, collect_result
    ):
        """
        Verify hist_turns parameter correctly limits history to branch agents.

        Creates 5-turn conversation, then executes with hist_turns=3.
        Agent should only see last 3 turns, not all 5.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

        history = [
            {"u": "turn 1 query", "a": "turn 1 response"},
            {"u": "turn 2 query", "a": "turn 2 response"},
            {"u": "turn 3 query", "a": "turn 3 response"},
            {"u": "turn 4 query", "a": "turn 4 response"},
            {"u": "turn 5 query", "a": "turn 5 response"},
        ]

        result = await collect_result(
            conductor,
            history=history,
            user_text="turn 6 query",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            hist_turns=3,
            router_k_user=2,
            force_target="recsys",
        )

        assert result.success, f"Truncation test failed: {result.text}"
        assert result.text, "No response with truncated history"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request.conversation_history) == 3, (
            f"Expected 3 history turns, got {len(agent_request.conversation_history)}"
        )

        first_turn_in_history = agent_request.conversation_history[0]
        assert "turn 3" in first_turn_in_history["u"], (
            "Should have turns 3, 4, 5 (last 3), not turns 1, 2, 3"
        )

    async def test_history_truncation_edge_case_fewer_than_hist_turns(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent, collect_result
    ):
        """
        Verify hist_turns handles edge case when history < hist_turns.

        If history has 2 turns but hist_turns=3, should use all 2 turns.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

        history = [
            {"u": "first query", "a": "first response"},
            {"u": "second query", "a": "second response"},
        ]

        result = await collect_result(
            conductor,
            history=history,
            user_text="third query",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            hist_turns=5,
            force_target="recsys",
        )

        assert isinstance(result, AgentResult), (
            "Failed to return AgentResult with hist_turns > len(history)"
        )
        assert result.text, "No response when hist_turns > history length"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request.conversation_history) == 2, (
            f"Expected 2 history turns (all available), "
            f"got {len(agent_request.conversation_history)}"
        )

    async def test_conversations_are_isolated(
        self, db_session, test_user_warm, mock_agent_factory, mock_recsys_agent, collect_result
    ):
        """
        Verify multiple conversations don't leak state.

        Same Conductor instance with different conv_id should be isolated.
        Tests that history for conv A doesn't affect conv B.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

        # Conversation A
        history_a = []
        result_a1 = await collect_result(
            conductor,
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

        mock_recsys_agent.execute_stream.reset_mock()

        # Conversation B (separate conv_id, empty history)
        history_b = []
        result_b1 = await collect_result(
            conductor,
            history=history_b,
            user_text="recommend thriller",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_b",
        )

        assert result_b1.success, "Conv B turn 1 failed"

        agent_request_b = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request_b.conversation_history) == 0, (
            "Conv B should have empty history, not conv A's history"
        )

        mock_recsys_agent.execute_stream.reset_mock()

        # Continue conversation A
        result_a2 = await collect_result(
            conductor,
            history=history_a,
            user_text="more epic fantasy",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_a",
        )

        assert result_a2.success, "Conv A turn 2 failed"

        agent_request_a2 = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request_a2.conversation_history) >= 1, (
            "Conv A turn 2 should have history from turn 1"
        )

    async def test_empty_history_first_turn(
        self, db_session, test_user_warm, mock_agent_factory, mock_response_agent, collect_result
    ):
        """
        Verify empty history (first turn) works correctly.

        Edge case: no conversation history yet.
        """
        conductor = Conductor()
        conductor.factory = mock_agent_factory

        result = await collect_result(
            conductor,
            history=[],
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

        assert mock_response_agent.execute_stream.called, (
            "Some agent should be called on first turn"
        )
