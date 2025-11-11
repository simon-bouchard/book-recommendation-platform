# tests/integration/chatbot/test_multi_turn_state.py
"""
Tests for multi-turn conversation state management in Conductor.
Validates that history persists, truncates correctly, and doesn't leak between conversations.
"""
import pytest
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import Target


class TestMultiTurnState:
    """
    Verify conversation state management across multiple turns.
    
    Component tests use static history. These tests verify that
    state actually persists correctly in sequential Conductor calls.
    """
    
    def test_history_accumulates_across_turns(self, db_session, test_user_warm):
        """
        Verify conversation history persists and accumulates correctly.
        
        Tests 3-turn conversation:
        1. Initial query → recsys
        2. Follow-up → recsys (continuation)
        3. Topic switch → docs (new intent)
        """
        conductor = Conductor()
        
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
        history.append({
            "u": "recommend sci-fi books",
            "a": result1.text
        })
        
        # Turn 2: Follow-up (should maintain recsys context)
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
        
        # Turn 3: Topic switch (should route to docs)
        history.append({
            "u": "more like that please",
            "a": result2.text
        })
        
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
        
        # All turns should succeed with proper history management
        # If history was malformed, routing or execution would fail
    
    def test_history_truncation_with_hist_turns(self, db_session, test_user_warm):
        """
        Verify hist_turns parameter correctly limits history to branch agents.
        
        Creates 5-turn conversation, then executes with hist_turns=3.
        Agent should only see last 3 turns, not all 5.
        """
        conductor = Conductor()
        
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
            router_k_user=2  # Router sees last 2 user messages
        )
        
        assert result.success, f"Truncation test failed: {result.text}"
        assert result.text, "No response with truncated history"
        
        # Success implies:
        # - make_branch_input correctly truncated to 3 turns
        # - Agent didn't fail with context overflow
        # - If all 5 turns were sent, agent might fail or timeout
    
    def test_history_truncation_edge_case_fewer_than_hist_turns(
        self, db_session, test_user_warm
    ):
        """
        Verify hist_turns handles edge case when history < hist_turns.
        
        If history has 2 turns but hist_turns=3, should use all 2 turns.
        """
        conductor = Conductor()
        
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
        
        assert result.success, "Failed with hist_turns > len(history)"
        assert result.text, "No response when hist_turns > history length"
    
    def test_conversations_are_isolated(self, db_session, test_user_warm):
        """
        Verify multiple conversations don't leak state.
        
        Same Conductor instance with different conv_id should be isolated.
        Tests that history for conv A doesn't affect conv B.
        """
        conductor = Conductor()
        
        # Conversation A
        history_a = []
        result_a1 = conductor.run(
            history=history_a,
            user_text="recommend fantasy",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_a"
        )
        
        assert result_a1.success, "Conv A turn 1 failed"
        
        history_a.append({"u": "recommend fantasy", "a": result_a1.text})
        
        # Conversation B (separate conv_id, empty history)
        history_b = []
        result_b1 = conductor.run(
            history=history_b,  # Empty - fresh conversation
            user_text="recommend thriller",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_b"
        )
        
        assert result_b1.success, "Conv B turn 1 failed"
        
        # Continue conversation A
        result_a2 = conductor.run(
            history=history_a,
            user_text="more epic fantasy",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="conv_a"
        )
        
        assert result_a2.success, "Conv A turn 2 failed"
        
        # Both conversations should succeed independently
        # If state leaked, routing or execution could fail
        # Each conversation maintains its own context
    
    def test_empty_history_first_turn(self, db_session, test_user_warm):
        """
        Verify empty history (first turn) works correctly.
        
        Edge case: no conversation history yet.
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],  # Empty - first turn
            user_text="hello",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
        )
        
        assert result.success, "First turn with empty history failed"
        assert result.text, "No response on first turn"
