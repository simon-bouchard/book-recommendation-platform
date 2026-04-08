# tests/integration/chatbot/conductor/test_multi_turn_state.py
"""
Tests for multi-turn conversation state management in Conductor.
Validates that history persists, truncates correctly, and doesn't leak between conversations.
"""

import pytest

from app.agents.schemas import AgentResult

pytestmark = pytest.mark.asyncio


class TestMultiTurnState:
    """
    Verify conversation state management across multiple turns.

    Component tests use static history. These tests verify that
    state actually persists correctly in sequential Conductor calls.

    All routing is controlled via force_target so these tests remain
    deterministic and free of LLM calls. Routing decisions — which agent
    a given message should reach — are evaluated separately in evaluation
    tests, not asserted here.
    """

    async def test_history_accumulates_across_turns(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify conversation history persists and accumulates across turns.

        Tests a 3-turn conversation where each subsequent turn receives
        the history of all prior turns. All turns use force_target="recsys"
        so the test is purely about history accumulation, not routing.
        """
        # Turn 1: no prior history
        history: list = []
        result1 = await collect_result(
            conductor,
            history=history,
            user_text="recommend sci-fi books",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            force_target="recsys",
        )

        assert result1.success, f"Turn 1 failed: {result1.text}"

        agent_request_turn1 = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request_turn1.conversation_history) == 0, (
            "Turn 1 should have no prior history"
        )

        history.append({"u": "recommend sci-fi books", "a": result1.text})
        mock_recsys_agent.execute_stream.reset_mock()

        # Turn 2: should see turn 1 in history
        result2 = await collect_result(
            conductor,
            history=history,
            user_text="more like that please",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            force_target="recsys",
        )

        assert result2.success, f"Turn 2 failed: {result2.text}"

        agent_request_turn2 = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request_turn2.conversation_history) == 1, (
            "Turn 2 should have 1 turn of history (from turn 1)"
        )

        history.append({"u": "more like that please", "a": result2.text})
        mock_recsys_agent.execute_stream.reset_mock()

        # Turn 3: should see turns 1 and 2 in history
        result3 = await collect_result(
            conductor,
            history=history,
            user_text="any award winners in that genre?",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            force_target="recsys",
        )

        assert result3.success, f"Turn 3 failed: {result3.text}"

        agent_request_turn3 = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request_turn3.conversation_history) == 2, (
            "Turn 3 should have 2 turns of history (from turns 1 and 2)"
        )

    async def test_history_truncation_with_hist_turns(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify hist_turns parameter correctly limits history sent to branch agents.

        Creates 5-turn history, executes with hist_turns=3.
        Agent should only see the last 3 turns, not all 5.
        """
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

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request.conversation_history) == 3, (
            f"Expected 3 history turns, got {len(agent_request.conversation_history)}"
        )

        # Should be turns 3, 4, 5 — the last 3
        first_turn_in_history = agent_request.conversation_history[0]
        assert "turn 3" in first_turn_in_history["u"], (
            "Should have turns 3, 4, 5 (last 3), not turns 1, 2, 3"
        )

    async def test_history_truncation_edge_case_fewer_than_hist_turns(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify hist_turns handles edge case when history < hist_turns.

        If history has 2 turns but hist_turns=5, should use all 2 turns
        without error.
        """
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

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request.conversation_history) == 2, (
            f"Expected 2 history turns (all available), "
            f"got {len(agent_request.conversation_history)}"
        )

    async def test_conversations_are_isolated(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify multiple conversations don't leak state.

        Same Conductor instance, different conv_ids. History provided for
        conv A must not appear in conv B.
        """
        # Conversation A, turn 1
        history_a: list = []
        result_a1 = await collect_result(
            conductor,
            history=history_a,
            user_text="recommend fantasy",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_a",
            force_target="recsys",
        )

        assert result_a1.success, "Conv A turn 1 failed"
        history_a.append({"u": "recommend fantasy", "a": result_a1.text})
        mock_recsys_agent.execute_stream.reset_mock()

        # Conversation B, turn 1 (separate conv_id, empty history)
        history_b: list = []
        result_b1 = await collect_result(
            conductor,
            history=history_b,
            user_text="recommend thriller",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_b",
            force_target="recsys",
        )

        assert result_b1.success, "Conv B turn 1 failed"

        agent_request_b = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request_b.conversation_history) == 0, (
            "Conv B should have empty history, not conv A's history"
        )
        mock_recsys_agent.execute_stream.reset_mock()

        # Conversation A, turn 2 — should still have its own history
        result_a2 = await collect_result(
            conductor,
            history=history_a,
            user_text="more epic fantasy",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_a",
            force_target="recsys",
        )

        assert result_a2.success, "Conv A turn 2 failed"

        agent_request_a2 = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request_a2.conversation_history) == 1, (
            "Conv A turn 2 should have 1 turn of history from turn 1"
        )

    async def test_empty_history_first_turn(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_response_agent,
        collect_result,
    ):
        """
        Verify empty history (first turn) works correctly.

        force_target="respond" exercises the conversational agent path
        without relying on routing intelligence.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="hello",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            force_target="respond",
        )

        assert isinstance(result, AgentResult), (
            "First turn with empty history didn't return AgentResult"
        )
        assert result.text, "No response on first turn"
        assert mock_response_agent.execute_stream.called, (
            "Response agent should be called on first turn"
        )
