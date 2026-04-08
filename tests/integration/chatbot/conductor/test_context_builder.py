# tests/integration/chatbot/conductor/test_context_builder.py
"""
Tests for context builder functions (make_router_input, make_branch_input).
Validates that history truncation and context preparation work correctly.
"""

import pytest

from app.agents.schemas import AgentResult

pytestmark = pytest.mark.asyncio


class TestContextBuilders:
    """
    Verify make_router_input and make_branch_input work correctly.

    Nobody explicitly tests these helper functions. Component tests
    bypass them by constructing inputs directly.

    All routing is controlled via the pre-wired mock_router (default: recsys),
    so no LLM calls are made.
    """

    async def test_router_input_truncates_to_k_user(
        self,
        db_session,
        test_user_warm,
        conductor,
        collect_result,
    ):
        """
        Verify make_router_input only includes last k_user messages.

        The router context builder should truncate history to k_user
        user messages. This test verifies the system doesn't error with
        a long history and a small k_user window.
        """
        history = [{"u": f"user message {i}", "a": f"assistant response {i}"} for i in range(1, 11)]

        result = await collect_result(
            conductor,
            history=history,
            user_text="final message",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            router_k_user=2,
        )

        assert result.success, f"Request with k_user=2 failed: {result.text}"

    async def test_router_input_k_user_larger_than_history(
        self,
        db_session,
        test_user_warm,
        conductor,
        collect_result,
    ):
        """
        Verify k_user handles edge case when k_user > len(history).

        If history has 2 turns but k_user=5, should use all 2 turns
        without error.
        """
        history = [
            {"u": "first message", "a": "first response"},
            {"u": "second message", "a": "second response"},
        ]

        result = await collect_result(
            conductor,
            history=history,
            user_text="third message",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            router_k_user=5,
        )

        assert result.success, "Failed when k_user > history length"

    async def test_branch_input_truncates_to_hist_turns(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify make_branch_input only includes last hist_turns.

        Branch agents should only see last hist_turns full turns,
        not the complete history.
        """
        history = [{"u": f"message {i}", "a": f"response {i}"} for i in range(1, 7)]

        result = await collect_result(
            conductor,
            history=history,
            user_text="new query",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            hist_turns=3,
            router_k_user=2,
            force_target="recsys",
        )

        assert result.success, f"Request with hist_turns=3 failed: {result.text}"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request.conversation_history) == 3, (
            f"Expected 3 history turns, got {len(agent_request.conversation_history)}"
        )

    async def test_branch_input_hist_turns_zero(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify hist_turns=0 sends no history to agent.

        Agent should see only the current query, no prior history.
        """
        history = [
            {"u": "old message 1", "a": "old response 1"},
            {"u": "old message 2", "a": "old response 2"},
        ]

        result = await collect_result(
            conductor,
            history=history,
            user_text="new query",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            hist_turns=0,
            force_target="recsys",
        )

        assert result.success, "Failed with hist_turns=0"

        agent_request = mock_recsys_agent.execute_stream.call_args[0][0]
        assert len(agent_request.conversation_history) == 0, (
            f"Expected no history, got {len(agent_request.conversation_history)}"
        )

    async def test_force_target_bypasses_router(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_docs_agent,
        mock_router,
        collect_result,
    ):
        """
        Verify force_target parameter skips routing logic entirely.

        Router should not be invoked when force_target is set.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend mystery books",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            force_target="docs",
        )

        assert result.success, "force_target=docs failed"
        assert mock_docs_agent.execute_stream.called, "Docs agent should be called"

        # Router must not be invoked when force_target is set
        assert not mock_router.classify.called, "Router should not be called with force_target"

    async def test_force_target_all_agents(
        self,
        db_session,
        test_user_warm,
        conductor,
        mock_recsys_agent,
        mock_web_agent,
        mock_docs_agent,
        mock_response_agent,
        collect_result,
    ):
        """
        Verify force_target works for all four agent types.
        """
        targets_to_test = [
            ("recsys", mock_recsys_agent),
            ("web", mock_web_agent),
            ("docs", mock_docs_agent),
            ("respond", mock_response_agent),
        ]

        for target, mock_agent in targets_to_test:
            mock_agent.execute_stream.reset_mock()

            result = await collect_result(
                conductor,
                history=[],
                user_text="test query",
                use_profile=False,
                current_user=test_user_warm if target == "recsys" else None,
                db=db_session if target == "recsys" else None,
                user_num_ratings=10 if target == "recsys" else 0,
                force_target=target,
            )

            assert isinstance(result, AgentResult), (
                f"force_target={target} didn't return AgentResult"
            )
            assert mock_agent.execute_stream.called, f"Agent for target={target} was not called"

    async def test_context_with_empty_history(
        self,
        db_session,
        test_user_warm,
        conductor,
        collect_result,
    ):
        """
        Verify context builders handle empty history correctly.

        First turn: history=[], both router context and branch context
        should handle an empty list without error.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="hello",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            router_k_user=2,
            hist_turns=3,
        )

        assert result.success, "Failed with empty history"
        assert isinstance(result, AgentResult), "Should return AgentResult"
