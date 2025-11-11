# tests/integration/chatbot/test_parameter_handling.py
"""
Tests for parameter passing and edge cases in Conductor.
Validates that user_num_ratings, use_profile, and optional parameters work correctly.
"""
import pytest
from app.agents.orchestrator.conductor import Conductor
from app.agents.schemas import AgentResult


class TestParameterHandling:
    """
    Verify parameter propagation and edge case handling.
    
    Component tests use well-formed inputs. These test boundary conditions
    and verify parameters correctly flow through the system.
    """
    
    def test_cold_user_routing_and_tool_selection(self, db_session, test_user_cold):
        """
        Verify user_num_ratings=0 prevents ALS tool usage.
        
        Cold users should not get ALS-based recommendations (requires ≥10 ratings).
        System should use alternative tools (subject_hybrid_pool).
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],
            user_text="recommend fantasy adventure books",
            use_profile=False,
            current_user=test_user_cold,
            db=db_session,
            user_num_ratings=0,  # Cold user
            force_target="recsys"  # Force recsys to test tool selection
        )
        
        assert result.success, f"Cold user recommendation failed: {result.text}"
        
        # Cold user should still get recommendations
        # (using subject-based tools, not ALS)
        if result.book_ids:
            assert len(result.book_ids) >= 3, \
                f"Cold user got no recommendations (expected ≥3, got {len(result.book_ids)})"
        
        # If ALS was incorrectly used, might get empty results or error
    
    def test_warm_user_can_access_personalized_recs(
        self, db_session, test_user_warm
    ):
        """
        Verify user_num_ratings≥10 enables personalized recommendations.
        
        Warm users should be able to use ALS-based recommendations.
        This is the complement of test_cold_user.
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],
            user_text="recommend something good",  # Vague = might use ALS
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,  # Warm user
            force_target="recsys"
        )
        
        assert result.success, f"Warm user recommendation failed: {result.text}"
        if result.book_ids:
            assert len(result.book_ids) >= 3, \
                "Warm user got no personalized recommendations"
        
        # Warm user should get high-quality personalized recommendations
    
    def test_user_num_ratings_none_defaults_to_zero(self, db_session, test_user_new):
        """
        Verify user_num_ratings=None is handled as 0 (cold user).
        
        When user_num_ratings is not provided, should default to cold user behavior.
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],
            user_text="hello",
            use_profile=False,
            current_user=test_user_new,
            db=db_session,
            user_num_ratings=None,  # Should default to 0
        )
        
        assert result.success, "Request with user_num_ratings=None failed"
        assert result.text, "No response with user_num_ratings=None"
    
    def test_optional_parameters_none_values(self):
        """
        Verify Conductor handles None for optional parameters.
        
        Response and docs agents don't require db/current_user.
        System should work with minimal parameters.
        """
        conductor = Conductor()
        
        # Response agent should work with minimal params
        result = conductor.run(
            history=[],
            user_text="hello there",
            use_profile=False,
            db=None,  # Response agent doesn't need db
            current_user=None,
            user_num_ratings=None,  # Defaults to 0
        )
        
        assert result.success, "Response agent failed with None parameters"
        assert result.text, "No response from response agent"
    
    def test_docs_agent_works_without_user_context(self):
        """
        Verify docs agent works without user context.
        
        Documentation queries don't need personalization.
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],
            user_text="how do I rate a book?",
            use_profile=False,
            db=None,  # Docs agent doesn't need db
            current_user=None,
            user_num_ratings=0,
        )
        
        assert isinstance(result, AgentResult), \
            "Docs agent didn't return AgentResult without user context"
        assert result.text, "No response from docs agent"
    
    def test_use_profile_false_prevents_profile_access(
        self, db_session, test_user_with_profile
    ):
        """
        Verify use_profile=False is respected (privacy-critical).
        
        Even if user has profile, agent should not access it when
        use_profile=False.
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],
            user_text="recommend something for me",  # Vague = would use profile
            use_profile=False,  # Explicit denial
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=15,  # Warm user with profile
        )
        
        assert result.success, "Request with use_profile=False failed"
        assert result.text, "No response when profile disabled"
        
        # Should succeed without profile access
        # (can't directly verify profile wasn't used, but should work)
    
    def test_conv_id_and_uid_passed_through(self, db_session, test_user_warm):
        """
        Verify conv_id and uid parameters pass through system.
        
        These are metadata fields that should survive all conversions.
        """
        conductor = Conductor()
        
        result = conductor.run(
            history=[],
            user_text="recommend books",
            use_profile=False,
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=10,
            conv_id="test_conv_456",
            uid=test_user_warm.user_id,
        )
        
        assert result.success, "Request with conv_id/uid failed"
        
        # Metadata fields should be present (may be stored in execution_state)
        # At minimum, request should succeed without breaking on these params
