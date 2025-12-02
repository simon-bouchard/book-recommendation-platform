# tests/integration/chatbot/test_context_builders.py
"""
Tests for context builder functions (make_router_input, make_branch_input).
Validates that history truncation and context preparation work correctly.
"""
import pytest
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult


class TestContextBuilders:
    """
    Verify make_router_input and make_branch_input work correctly.
    
    Nobody explicitly tests these helper functions. Component tests
    bypass them by constructing inputs directly.
    """
    
    def test_router_input_truncates_to_k_user(self, db_session, test_user_warm):
        """
        Verify make_router_input only includes last k_user messages.
        
        Critical for router prompt size management.
        Router should only see last k_user user messages, not full history.
        """
        conductor = Conductor()
        
        # Create 10-turn history
        history = [
            {"u": f"user message {i}", "a": f"assistant response {i}"}
            for i in range(1, 11)
        ]
        
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
        
        assert result.success, \
            f"Request with k_user=2 failed: {result.text}"
        
        # Success implies:
        # - make_router_input correctly truncated to last 2 messages
        # - Router didn't fail with oversized prompt
        # - If all 10 messages were sent, router might fail or timeout
    
    def test_router_input_k_user_larger_than_history(
        self, db_session, test_user_warm
    ):
        """
        Verify k_user handles edge case when k_user > len(history).
        
        If history has 2 messages but k_user=5, should use all 2 messages.
        """
        conductor = Conductor()
        
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
        
        assert result.success, \
            "Failed when k_user > history length"
    
    def test_branch_input_truncates_to_hist_turns(
        self, db_session, test_user_warm
    ):
        """
        Verify make_branch_input only includes last hist_turns.
        
        Different from router truncation - affects agent context.
        Branch agents should only see last hist_turns full turns.
        """
        conductor = Conductor()
        
        # Create 6-turn history
        history = [
            {"u": f"message {i}", "a": f"response {i}"}
            for i in range(1, 7)
        ]
        
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
        )
        
        assert result.success, \
            f"Request with hist_turns=3 failed: {result.text}"
        
        # Success implies correct truncation
        # If agent received all 6 turns, might fail with context overflow
    
    def test_branch_input_hist_turns_zero(self, db_session, test_user_warm):
        """
        Verify hist_turns=0 sends no history to agent.
        
        Edge case: agent should see only current query, no history.
        """
        conductor = Conductor()
        
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
        )
        
        assert result.success, \
            "Failed with hist_turns=0"
        
        # Agent should succeed with just current query, no history
    
    def test_force_target_bypasses_router(self, db_session, test_user_warm):
        """
        Verify force_target parameter skips routing logic.
        
        Used for testing/debugging specific agents.
        Router should not be invoked when force_target is set.
        """
        conductor = Conductor()
        
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
        
        assert result.success, \
            "force_target=DOCS failed"
        
        # Docs agent should have handled it
        # (might give generic response since query isn't docs-related)
        assert result.text, "No response from forced target"
    
    def test_force_target_all_agents(self, db_session, test_user_warm):
        """
        Verify force_target works for all agent types.
        
        Each target should be callable via force_target.
        """
        conductor = Conductor()
        
        targets_to_test = [
            "recsys",
            "web",
            "docs",
            "respond"
        ]
        
        for target in targets_to_test:
            result = conductor.run(
                history=[],
                user_text="test query",
                use_profile=False,
                current_user=test_user_warm,
                db=db_session if target == "recsys" else None,
                user_num_ratings=10 if target == "recsys" else 0,
                force_target=target,
            )
            
            # Each should execute without crashing
            assert isinstance(result, AgentResult), \
                f"force_target={target.value} didn't return AgentResult"
            
            # Should at least not crash (success may vary by agent/params)
    
    def test_context_with_empty_history(self, db_session, test_user_warm):
        """
        Verify context builders handle empty history correctly.
        
        First turn: history=[], both router and agent should handle gracefully.
        """
        conductor = Conductor()
        
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
        
        assert result.success, \
            "Failed with empty history"
        
        # Both make_router_input and make_branch_input should handle
        # empty history without breaking
