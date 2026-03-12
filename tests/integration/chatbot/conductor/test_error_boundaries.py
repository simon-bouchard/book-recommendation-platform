# tests/integration/chatbot/conductor/test_error_boundaries.py
"""
Tests for error handling and resilience in Conductor.
Validates that failures are caught and handled gracefully at orchestration level.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult, RoutePlan

pytestmark = pytest.mark.asyncio


class TestErrorBoundaries:
    """
    Verify Conductor handles failures gracefully.

    Component tests assume happy path. These test failure modes
    and verify system doesn't crash or leak exceptions.

    All routing is controlled via the pre-wired mock_router (default: recsys),
    so no LLM calls are made.
    """

    async def test_agent_execution_failure_returns_error_result(
        self,
        db_session,
        mock_router,
        collect_result,
    ):
        """
        Verify agent execution failures are caught and return error result.

        Conductor catches exceptions and returns a 'complete' chunk with
        success=False. Users get a graceful error message, not a crash.
        """

        # Async generator that raises immediately — simulates a broken agent.
        async def _failing_execute_stream(*args, **kwargs):
            raise RuntimeError("Simulated agent execution failure")
            yield  # pragma: no cover  (makes this an async generator function)

        failing_agent = Mock()
        failing_agent.execute_stream = _failing_execute_stream

        failing_factory = Mock()
        failing_factory.create_agent = Mock(return_value=failing_agent)

        # Inject both the failing factory and the mock router so no real LLM
        # is called before the agent itself raises.
        conductor = Conductor()
        conductor.factory = failing_factory
        conductor.router = mock_router

        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend books",
            use_profile=False,
            db=db_session,
            user_num_ratings=10,
            force_target="recsys",
        )

        assert isinstance(result, AgentResult), "Should return AgentResult on error"
        assert result.success is False, "Result should indicate failure"
        assert result.text, "Should have error message"

    async def test_router_classification_failure_returns_error_result(
        self,
        db_session,
        conductor,
        collect_result,
    ):
        """
        Verify router classification failures are caught and return error result.

        Conductor catches exceptions and returns a 'complete' chunk with
        success=False.
        """
        failing_router = Mock()
        failing_router.classify = AsyncMock(
            side_effect=RuntimeError("Router classification failed")
        )
        conductor.router = failing_router

        result = await collect_result(
            conductor,
            history=[],
            user_text="test query",
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
        )

        assert isinstance(result, AgentResult), "Should return AgentResult on error"
        assert result.success is False, "Result should indicate failure"
        assert result.text, "Should have error message"

    async def test_empty_query_handled_gracefully(
        self,
        db_session,
        conductor,
        mock_response_agent,
        collect_result,
    ):
        """
        Verify empty user_text doesn't break the system.

        Edge case: user submits an empty string.
        force_target="respond" ensures the response agent path is exercised
        without relying on routing intelligence.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="",
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
            force_target="respond",
        )

        assert isinstance(result, AgentResult), "Should return AgentResult for empty query"
        assert result.text, "Should have response even for empty query"
        assert mock_response_agent.execute_stream.called, "Response agent should handle empty query"

    async def test_whitespace_only_query_handled(
        self,
        db_session,
        conductor,
        mock_response_agent,
        collect_result,
    ):
        """
        Verify whitespace-only query is handled.

        Edge case: user submits only spaces/newlines.
        force_target="respond" ensures the response agent path is exercised
        without relying on routing intelligence.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="   \n\t  ",
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
            force_target="respond",
        )

        assert isinstance(result, AgentResult), "Should return AgentResult for whitespace query"
        assert result.text, "Should have response for whitespace query"
        assert mock_response_agent.execute_stream.called, (
            "Response agent should handle whitespace query"
        )

    async def test_very_long_query_handled(
        self,
        db_session,
        conductor,
        collect_result,
    ):
        """
        Verify extremely long queries don't break the system.

        Edge case: user submits a massive query (potential token overflow).
        """
        long_query = " ".join(["word"] * 1000)

        result = await collect_result(
            conductor,
            history=[],
            user_text=long_query,
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
            force_target="recsys",
        )

        assert isinstance(result, AgentResult), "Should return AgentResult for very long query"
        assert result.text, "Should have response for long query"

    async def test_malformed_history_handled(
        self,
        db_session,
        conductor,
        collect_result,
    ):
        """
        Verify malformed history doesn't crash the system.

        Edge case: history has missing fields (no 'a' key on first turn).
        """
        malformed_history = [
            {"u": "first message"},  # Missing 'a'
            {"u": "second message", "a": "response"},
        ]

        result = await collect_result(
            conductor,
            history=malformed_history,
            user_text="test",
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
        )

        assert isinstance(result, AgentResult), "Should return AgentResult with malformed history"

    async def test_database_none_for_agent_requiring_db(
        self,
        conductor,
        mock_recsys_agent,
        collect_result,
    ):
        """
        Verify the recsys agent path handles db=None.

        The mock agent itself does not need db, so this tests that the
        adapter and factory layers don't crash before the agent is reached.
        """
        result = await collect_result(
            conductor,
            history=[],
            user_text="recommend books",
            use_profile=False,
            db=None,
            current_user=None,
            user_num_ratings=10,
            force_target="recsys",
        )

        assert isinstance(result, AgentResult), "Should return AgentResult when db=None"
        assert result.text, "Should have response text"
        assert mock_recsys_agent.execute_stream.called, (
            "Recsys agent should be called even with db=None"
        )
