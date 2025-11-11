# tests/integration/chatbot/test_error_boundaries.py
"""
Tests for error handling and resilience in Conductor.
Validates that failures are caught and handled gracefully at orchestration level.
"""
import pytest
from unittest.mock import Mock, patch
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult, Target, RoutePlan


class TestErrorBoundaries:
    """
    Verify Conductor handles failures gracefully.
    
    Component tests assume happy path. These test failure modes
    and verify system doesn't crash or leak exceptions.
    """
    
    def test_agent_execution_failure_returns_error_result(self, db_session):
        """
        Verify Conductor catches agent exceptions and returns error AgentResult.
        
        Agent.execute() can raise exceptions (timeout, LLM failure, etc.).
        Conductor should catch and return failure result, not crash.
        """
        conductor = Conductor()
        
        # Mock factory to return agent that raises exception
        original_create = conductor.factory.create_agent
        
        def failing_factory(*args, **kwargs):
            agent = original_create(*args, **kwargs)
            original_execute = agent.execute
            
            def failing_execute(*exec_args, **exec_kwargs):
                raise RuntimeError("Simulated agent execution failure")
            
            agent.execute = failing_execute
            return agent
        
        with patch.object(conductor.factory, 'create_agent', failing_factory):
            result = conductor.run(
                history=[],
                user_text="recommend books",
                use_profile=False,
                db=db_session,
                user_num_ratings=10,
            )
        
        # Should return error result, not raise exception
        assert isinstance(result, AgentResult), \
            "Should return AgentResult even on failure"
        assert not result.success, \
            "Result should indicate failure"
        
        # Should have user-friendly error message
        assert result.text, "Error result should have text"
        error_keywords = ["error", "trouble", "unable", "sorry"]
        assert any(kw in result.text.lower() for kw in error_keywords), \
            f"Error message not user-friendly: {result.text}"
    
    def test_router_classification_failure_handled(self, db_session):
        """
        Verify Conductor handles router failures gracefully.
        
        Router.classify() can fail (LLM timeout, malformed response).
        Conductor should handle gracefully or fallback to response agent.
        """
        conductor = Conductor()
        
        # Mock router to raise exception
        def failing_classify(*args, **kwargs):
            raise RuntimeError("Router classification failed")
        
        with patch.object(conductor.router, 'classify', failing_classify):
            result = conductor.run(
                history=[],
                user_text="test query",
                use_profile=False,
                db=db_session,
                user_num_ratings=0,
            )
        
        # Should handle gracefully
        assert isinstance(result, AgentResult), \
            "Should return AgentResult on router failure"
        
        # Might fallback to response agent or return error
        # Either way, should not crash
    
    def test_empty_query_handled_gracefully(self, db_session):
        """
        Verify empty user_text doesn't break system.
        
        Edge case: user submits empty string.
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],
            user_text="",  # Empty query
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
        )
        
        # Should handle gracefully (route to response agent)
        assert isinstance(result, AgentResult), \
            "Should return AgentResult for empty query"
        assert result.text, "Should have response even for empty query"
    
    def test_whitespace_only_query_handled(self, db_session):
        """
        Verify whitespace-only query is handled.
        
        Edge case: user submits only spaces/newlines.
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],
            user_text="   \n\t  ",  # Whitespace only
            use_profile=False,
            db=db_session,
            user_num_ratings=0,
        )
        
        assert isinstance(result, AgentResult), \
            "Should return AgentResult for whitespace query"
        assert result.text, "Should have response for whitespace query"
    
    def test_very_long_query_handled(self, db_session):
        """
        Verify extremely long queries don't break system.
        
        Edge case: user submits massive query (potential token overflow).
        """
        conductor = Conductor()
        
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
        assert isinstance(result, AgentResult), \
            "Should return AgentResult for very long query"
        assert result.text, "Should have response for long query"
    
    def test_malformed_history_handled(self, db_session):
        """
        Verify malformed history doesn't crash system.
        
        Edge case: history has missing or extra fields.
        """
        conductor = Conductor()
        
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
        assert isinstance(result, AgentResult), \
            "Should return AgentResult with malformed history"
    
    def test_database_none_for_agent_requiring_db(self):
        """
        Verify recsys agent handles db=None appropriately.
        
        Recsys requires database but might be called with db=None.
        Should fail gracefully with clear error.
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],
            user_text="recommend books",
            use_profile=False,
            db=None,  # Recsys needs db
            current_user=None,
            user_num_ratings=10,
            force_target=Target.RECSYS,  # Force recsys
        )
        
        # Should handle gracefully (either succeed without db or error clearly)
        assert isinstance(result, AgentResult), \
            "Should return AgentResult when db=None"
        
        # If it fails (expected), should have error message
        if not result.success:
            assert result.text, "Should have error message when db required"
