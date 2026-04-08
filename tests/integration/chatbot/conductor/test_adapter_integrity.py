# tests/integration/chatbot/conductor/test_adapter_integrity.py
"""
Tests for adapter layer data integrity in Conductor.
Validates that data survives TurnInput→AgentRequest and AgentResponse→AgentResult conversions.
"""

import pytest

from app.agents.schemas import AgentResult

pytestmark = pytest.mark.asyncio


class TestAdapterDataIntegrity:
    """
    Verify adapter conversions preserve all data fields.

    Component tests construct AgentRequest directly, so they don't
    validate that the adapter layer correctly converts TurnInput.

    All routing is controlled via the pre-wired mock_router (default: recsys),
    so no LLM calls are made.
    """

    async def test_turn_input_to_request_preserves_data(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify adapter.turn_input_to_request() doesn't lose data.

        Tests that all Conductor.run_stream() parameters correctly flow through
        the adapter layer to reach the agent.
        """
        result = await collect_result(
            conductor,
            history=[
                {"u": "recommend sci-fi", "a": "Here are some sci-fi books..."},
                {"u": "more recent ones", "a": "Try these newer releases..."},
            ],
            user_text="what about fantasy",
            use_profile=True,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            hist_turns=2,
            conv_id="test_conv_123",
            uid=test_user_warm.user_id,
            router_k_user=2,
        )

        assert isinstance(result, AgentResult)
        assert result.text, "Response text is missing"
        assert result.policy_version, "Policy version should be set"

        assert mock_recsys_agent.execute_stream.called, "Agent execute_stream() was not called"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert agent_request.user_text == "what about fantasy", "User query didn't reach agent"
        assert len(agent_request.conversation_history) == 2, (
            f"Expected 2 history turns (hist_turns=2), got "
            f"{len(agent_request.conversation_history)}"
        )
        assert agent_request.context.user_preferences["profile_allowed"] is True, (
            "Profile access flag not preserved"
        )
        assert agent_request.context.user_preferences["num_ratings"] == 15, (
            "User rating count not preserved"
        )

    async def test_response_to_result_preserves_metadata(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify adapter.response_to_agent_result() preserves all fields.

        Critical for recsys agent which returns rich metadata:
        book_ids, tool_calls, citations, etc.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend mystery novels with strong female leads",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=12,
            hist_turns=3,
            router_k_user=2,
        )

        assert isinstance(result, AgentResult), "Result is not AgentResult"
        assert result.success, "Result should be successful"

        # Verify book_ids survived conversion (mock returns 5 books)
        assert result.book_ids is not None, "book_ids should not be None"
        assert len(result.book_ids) == 5, f"Expected 5 books from mock, got {len(result.book_ids)}"

        for book_id in result.book_ids:
            assert isinstance(book_id, int), f"book_id {book_id} is not an integer"

        assert result.text and len(result.text) > 20, "Response text missing or too short"

        assert len(result.tool_calls) > 0, "Tool calls not preserved"
        assert result.tool_calls[0].name == "als_recommendations", "Tool call name not preserved"

    async def test_profile_access_propagates_through_layers(
        self,
        db_session,
        test_user_with_profile,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify use_profile flag reaches agent through all conversions.

        Flow: Conductor → TurnInput.profile_allowed → AgentRequest.context → Tools

        Privacy-critical: must not leak profile when use_profile=False.
        """
        # Test with profile ENABLED
        result_with = await collect_result(
            conductor,
            history=[],
            user_text="recommend something for me",
            use_profile=True,
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=15,
        )

        assert isinstance(result_with, AgentResult), (
            "Profile-enabled request didn't return AgentResult"
        )
        assert result_with.text, "No response text with profile enabled"

        agent_request_with = mock_recsys_agent.execute_stream.call_args[0][0]
        assert agent_request_with.context.user_preferences["profile_allowed"] is True, (
            "Profile access flag should be True"
        )

        mock_recsys_agent.execute_stream.reset_mock()

        # Test with profile DISABLED
        result_without = await collect_result(
            conductor,
            history=[],
            user_text="recommend something for me",
            use_profile=False,
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=15,
        )

        assert isinstance(result_without, AgentResult), (
            "Profile-disabled request didn't return AgentResult"
        )
        assert result_without.text, "No response text with profile disabled"

        agent_request_without = mock_recsys_agent.execute_stream.call_args[0][0]
        assert agent_request_without.context.user_preferences["profile_allowed"] is False, (
            "Profile access flag should be False"
        )

    async def test_empty_history_handled_correctly(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify empty history doesn't break adapter conversions.

        Edge case: first turn in conversation (history=[]).
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend fantasy books",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
        )

        assert isinstance(result, AgentResult), "Didn't return AgentResult"
        assert result.text, "No response with empty history"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request.conversation_history) == 0, (
            "History should be empty for first turn"
        )

        assert result.book_ids is not None, "Should have book_ids"
        assert len(result.book_ids) == 5, (
            f"Expected 5 books with empty history, got {len(result.book_ids)}"
        )
