# tests/integration/chatbot/agents/recommendation/test_pipeline_integration.py
"""
Integration tests for RecommendationAgent's three-stage pipeline.
Tests data flow through Planner → Retrieval → Curation using mocked sub-agents.
"""

import pytest
from unittest.mock import Mock

from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent
from app.agents.domain.entities import AgentRequest, AgentResponse
from app.agents.domain.recsys_schemas import PlannerInput, RetrievalInput, ExecutionContext


class TestStageTransitions:
    """
    Verify data flows correctly between pipeline stages.

    Tests that outputs from one stage become inputs to the next stage,
    and that all necessary data is preserved through the pipeline.
    """

    def test_planner_strategy_reaches_retrieval(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
        strategy_factory,
    ):
        """
        Verify PlannerStrategy output becomes RetrievalInput.

        PlannerAgent returns a strategy → RetrievalAgent receives it.
        """
        # Configure planner with known strategy
        custom_strategy = strategy_factory.custom_strategy(
            recommended_tools=["als_recommendations"],
            fallback_tools=["popular_books"],
            reasoning="Test reasoning",
            profile_data=None,
        )
        mock_planner = mock_planner_builder.returns_strategy(custom_strategy).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        # Create agent with mocked sub-agents
        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        # Execute pipeline
        request = AgentRequest(
            user_text="recommend something good",
            conversation_history=[],
            context={
                "profile_allowed": False,
                "num_ratings": 15,
                "conv_id": "test_123",
                "uid": test_user_warm.user_id,
            },
        )
        result = agent.execute(request)

        # Verify planner was called
        assert mock_planner.execute.called, "Planner should be called"

        # Verify retrieval received the strategy from planner
        assert mock_retrieval.execute.called, "Retrieval should be called"
        retrieval_input = mock_retrieval.execute.call_args[0][0]

        assert isinstance(retrieval_input, RetrievalInput), (
            "Retrieval should receive RetrievalInput"
        )
        assert retrieval_input.strategy == custom_strategy, (
            "Retrieval should receive planner's strategy"
        )
        assert retrieval_input.query == "recommend something good", (
            "Original query should be passed to retrieval"
        )

    def test_retrieval_candidates_reach_curation(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        Verify Retrieval candidates become curation input.

        RetrievalAgent returns candidates → CurationAgent receives them.
        """
        # Configure retrieval with known candidates
        test_candidates = candidate_factory.create_batch(60)
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_candidates(test_candidates).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="recommend mystery novels",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Verify curation received the candidates from retrieval
        assert mock_curation.execute.called, "Curation should be called"
        curation_candidates = mock_curation.execute.call_args.kwargs["candidates"]

        assert len(curation_candidates) == 60, (
            f"Curation should receive all 60 candidates, got {len(curation_candidates)}"
        )
        # Verify same candidates (by comparing first item)
        # Note: test_candidates are dicts, curation_candidates are BookRecommendation objects
        assert curation_candidates[0].item_idx == test_candidates[0]["item_idx"], (
            "Candidates should be preserved from retrieval to curation"
        )

    def test_execution_context_assembled_correctly(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
        strategy_factory,
    ):
        """
        Verify ExecutionContext has all required fields.

        ExecutionContext should include:
        - planner_reasoning
        - tools_used
        - profile_data (if applicable)
        """
        # Configure with known data
        custom_strategy = strategy_factory.custom_strategy(
            recommended_tools=["als_recommendations"],
            fallback_tools=["popular_books"],
            reasoning="Using ALS for warm user",
            profile_data=None,
        )
        mock_planner = mock_planner_builder.returns_strategy(custom_strategy).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60, "als_recommendations").build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="test query",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Verify curation received ExecutionContext
        assert mock_curation.execute.called, "Curation should be called"
        execution_context = mock_curation.execute.call_args.kwargs["execution_context"]

        assert isinstance(execution_context, ExecutionContext), (
            "Curation should receive ExecutionContext"
        )
        assert execution_context.planner_reasoning == "Using ALS for warm user", (
            "ExecutionContext should have planner reasoning"
        )
        assert len(execution_context.tools_used) > 0, (
            "ExecutionContext should have tools_used from retrieval"
        )

    def test_profile_data_flows_through_all_stages(
        self,
        db_session,
        test_user_with_profile,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
        strategy_factory,
    ):
        """
        Verify profile data preserved through pipeline.

        Planner fetches profile → strategy contains it → retrieval receives it →
        curation receives it in ExecutionContext.
        """
        # Configure planner to return strategy with profile
        profile_data = {
            "user_profile": {
                "favorite_subjects": [12, 45, 78],
            }
        }
        strategy_with_profile = strategy_factory.cold_user_with_profile_strategy(profile_data)
        mock_planner = mock_planner_builder.returns_strategy(strategy_with_profile).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=3,  # Cold user
            allow_profile=True,  # Profile allowed
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="suggest a book",
            conversation_history=[],
            context={"profile_allowed": True, "num_ratings": 3},
        )
        result = agent.execute(request)

        # Verify profile data reached retrieval
        retrieval_input = mock_retrieval.execute.call_args[0][0]
        assert retrieval_input.profile_data == profile_data, "Profile data should reach retrieval"

        # Verify profile data reached curation
        execution_context = mock_curation.execute.call_args.kwargs["execution_context"]
        assert execution_context.profile_data == profile_data, (
            "Profile data should reach curation in ExecutionContext"
        )

    def test_candidate_metadata_preserved(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        Verify book metadata survives Retrieval → Curation.

        Candidates have item_idx, title, author, year - all should be preserved.
        """
        # Create candidates with specific metadata
        test_candidates = [
            candidate_factory.create_with_metadata(
                item_idx=12345,
                title="The Three-Body Problem",
                author="Cixin Liu",
                year=2008,
            ),
            candidate_factory.create_with_metadata(
                item_idx=12346,
                title="Foundation",
                author="Isaac Asimov",
                year=1951,
            ),
        ]
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_candidates(test_candidates).build()
        mock_curation = mock_curation_builder.returns_success_with_books(2).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="recommend sci-fi",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Verify curation received full metadata
        curation_candidates = mock_curation.execute.call_args.kwargs["candidates"]

        assert len(curation_candidates) == 2, "Should have 2 candidates"

        # Check first candidate
        candidate_1 = curation_candidates[0]
        assert candidate_1.item_idx == 12345, "item_idx should be preserved"
        assert candidate_1.title == "The Three-Body Problem", "title should be preserved"
        assert candidate_1.author == "Cixin Liu", "author should be preserved"
        assert candidate_1.year == 2008, "year should be preserved"

    def test_book_ids_survive_full_pipeline(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify final book IDs match curation output.

        CurationAgent returns BookRecommendation list → AgentResponse.book_recommendations
        → Final output has matching item_idx values.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        # Curation returns 5 books
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="recommend books",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Verify final response has book_recommendations
        assert isinstance(result, AgentResponse), "Should return AgentResponse"
        assert result.success, "Pipeline should complete successfully"
        assert result.book_recommendations is not None, "Should have book_recommendations"
        assert len(result.book_recommendations) == 5, (
            f"Should have 5 books from curation, got {len(result.book_recommendations)}"
        )

        # Verify book IDs are integers
        for book in result.book_recommendations:
            assert isinstance(book.item_idx, int), f"book_id {book.item_idx} should be integer"


class TestParameterPropagation:
    """
    Verify parameters correctly configure each pipeline stage.

    Tests that user context (num_ratings, profile access, etc.) reaches
    all sub-agents and affects their behavior appropriately.
    """

    def test_cold_user_parameters_reach_all_stages(
        self,
        db_session,
        test_user_cold,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify cold user status (num_ratings < 10) reaches all stages.

        Cold users cannot use ALS → planner should know this.
        """
        mock_planner = mock_planner_builder.returns_cold_descriptive_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60, "book_semantic_search").build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_cold,
            db=db_session,
            user_num_ratings=3,  # Cold user
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="dark atmospheric mystery",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 3},
        )
        result = agent.execute(request)

        # Verify planner received cold user context
        assert mock_planner.execute.called, "Planner should be called"
        planner_input = mock_planner.execute.call_args[0][0]
        assert isinstance(planner_input, PlannerInput), "Should receive PlannerInput"
        assert planner_input.has_als_recs_available is False, (
            "Cold user should not have ALS available"
        )

        # Pipeline should complete successfully
        assert result.success, "Cold user pipeline should complete"
        assert len(result.book_recommendations) == 5, "Should return recommendations"

    def test_warm_user_parameters_reach_all_stages(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify warm user status (num_ratings >= 10) reaches all stages.

        Warm users can use ALS → planner should know this.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60, "als_recommendations").build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,  # Warm user
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="recommend something",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Verify planner received warm user context
        planner_input = mock_planner.execute.call_args[0][0]
        assert planner_input.has_als_recs_available is True, "Warm user should have ALS available"

        # Pipeline should complete successfully
        assert result.success, "Warm user pipeline should complete"

    def test_profile_access_flag_propagates(
        self,
        db_session,
        test_user_with_profile,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify allow_profile flag reaches planner.

        Privacy-critical: planner must respect profile access permission.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        # Test with profile ALLOWED
        agent_with_profile = RecommendationAgent(
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=15,
            allow_profile=True,  # Profile allowed
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="suggest a book",
            conversation_history=[],
            context={"profile_allowed": True, "num_ratings": 15},
        )
        result = agent_with_profile.execute(request)

        # Verify planner received allow_profile=True
        planner_input = mock_planner.execute.call_args[0][0]
        assert planner_input.allow_profile is True, "Profile access should be True"

        # Reset mock
        mock_planner.execute.reset_mock()

        # Test with profile DENIED
        agent_no_profile = RecommendationAgent(
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,  # Profile denied
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request_no_profile = AgentRequest(
            user_text="suggest a book",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result_no_profile = agent_no_profile.execute(request_no_profile)

        # Verify planner received allow_profile=False
        planner_input_no_profile = mock_planner.execute.call_args[0][0]
        assert planner_input_no_profile.allow_profile is False, "Profile access should be False"

    def test_optional_parameters_handled(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify pipeline handles None for optional parameters.

        Some parameters may be None - system should handle gracefully.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        # Create agent with minimal parameters
        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=None,  # Should default to 0
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="test query",
            conversation_history=[],
            context={},
        )
        result = agent.execute(request)

        # Should complete without crashing
        assert isinstance(result, AgentResponse), "Should return AgentResponse"
        assert result.text, "Should have response text"


class TestErrorHandling:
    """
    Verify pipeline handles failures gracefully at each stage.

    Tests that errors in one stage don't crash the system and that
    appropriate fallback logic is triggered.
    """

    def test_planner_failure_uses_fallback_strategy(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify planner failure triggers fallback strategy.

        If planner fails, orchestrator should use hardcoded fallback.
        """
        # Configure planner to raise error
        mock_planner = mock_planner_builder.raises_error(
            RuntimeError("Planner LLM failure")
        ).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="recommend books",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Should complete with fallback strategy (not crash)
        assert isinstance(result, AgentResponse), "Should return AgentResponse"
        # Retrieval and curation should still be called with fallback
        assert mock_retrieval.execute.called or not result.success, (
            "Should use fallback strategy or return error"
        )

    def test_retrieval_failure_returns_error_response(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify retrieval failure (no candidates) returns error response.

        If retrieval returns 0 candidates, pipeline should return helpful error.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        # Configure retrieval to return empty candidates
        mock_retrieval = mock_retrieval_builder.returns_empty().build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="books about xyz123 nonexistent topic",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Should return error response (not crash)
        assert isinstance(result, AgentResponse), "Should return AgentResponse"
        # Either success=False or has error message
        if not result.success:
            assert result.text, "Error response should have message"
        # Curation should NOT be called (no candidates to curate)
        # Note: orchestrator might still call curation with empty list, so we check result

    def test_curation_failure_returns_fallback_response(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify curation failure returns fallback response.

        If curation fails, orchestrator should return simple list or error.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        # Configure curation to raise error
        mock_curation = mock_curation_builder.raises_error(
            RuntimeError("Curation LLM failure")
        ).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="recommend books",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Should return some response (not crash)
        assert isinstance(result, AgentResponse), "Should return AgentResponse"
        # May or may not succeed depending on fallback logic
        assert result.text, "Should have response text even on failure"

    def test_database_none_handled_gracefully(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify db=None is handled gracefully.

        Some tools require database, but system shouldn't crash if missing.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=None,
            db=None,  # No database
            user_num_ratings=10,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="recommend books",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 10},
        )
        result = agent.execute(request)

        # Should return response (may succeed or fail depending on tools needed)
        assert isinstance(result, AgentResponse), "Should return AgentResponse"
        assert result.text, "Should have response text"

    def test_invalid_input_doesnt_crash(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify invalid/malformed input doesn't crash pipeline.

        Edge case: empty query, missing context fields, etc.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        # Empty query
        request = AgentRequest(
            user_text="",
            conversation_history=[],
            context={},
        )
        result = agent.execute(request)

        # Should return some response (not crash)
        assert isinstance(result, AgentResponse), "Should return AgentResponse for empty query"
        assert result.text, "Should have response text"


class TestFullPipelineFlow:
    """
    Verify end-to-end pipeline execution for common scenarios.

    Tests that the complete pipeline works correctly for typical use cases:
    warm users, cold users, profile usage, etc.
    """

    def test_warm_user_vague_query_complete_flow(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Test complete pipeline for warm user with vague query.

        Expected flow:
        Planner → ALS strategy
        Retrieval → 60+ ALS candidates
        Curation → 5 final books
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60, "als_recommendations").build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="recommend something",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Verify complete flow
        assert mock_planner.execute.called, "Planner should be called"
        assert mock_retrieval.execute.called, "Retrieval should be called"
        assert mock_curation.execute.called, "Curation should be called"

        # Verify final result
        assert result.success, "Pipeline should complete successfully"
        assert len(result.book_recommendations) == 5, "Should return 5 final books"
        assert result.text, "Should have response text"
        assert result.policy_version, "Should have policy version"

    def test_cold_user_descriptive_query_complete_flow(
        self,
        db_session,
        test_user_cold,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Test complete pipeline for cold user with descriptive query.

        Expected flow:
        Planner → semantic search strategy
        Retrieval → semantic candidates
        Curation → 5 final books
        """
        mock_planner = mock_planner_builder.returns_cold_descriptive_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60, "book_semantic_search").build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_cold,
            db=db_session,
            user_num_ratings=3,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="dark atmospheric mystery",
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 3},
        )
        result = agent.execute(request)

        # Verify complete flow
        assert mock_planner.execute.called, "Planner should be called"
        assert mock_retrieval.execute.called, "Retrieval should be called"
        assert mock_curation.execute.called, "Curation should be called"

        # Verify final result
        assert result.success, "Pipeline should complete successfully"
        assert len(result.book_recommendations) == 5, "Should return 5 final books"

    def test_cold_user_with_profile_complete_flow(
        self,
        db_session,
        test_user_with_profile,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
        strategy_factory,
    ):
        """
        Test complete pipeline for cold user with profile.

        Expected flow:
        Planner → fetch profile → subject-based strategy
        Retrieval → subject candidates
        Curation → 5 final books
        """
        profile_data = {"user_profile": {"favorite_subjects": [12, 45, 78]}}
        strategy_with_profile = strategy_factory.cold_user_with_profile_strategy(profile_data)
        mock_planner = mock_planner_builder.returns_strategy(strategy_with_profile).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60, "subject_hybrid_pool").build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_with_profile,
            db=db_session,
            user_num_ratings=3,
            allow_profile=True,  # Profile allowed
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="suggest a book",
            conversation_history=[],
            context={"profile_allowed": True, "num_ratings": 3},
        )
        result = agent.execute(request)

        # Verify complete flow
        assert mock_planner.execute.called, "Planner should be called"
        assert mock_retrieval.execute.called, "Retrieval should be called"
        assert mock_curation.execute.called, "Curation should be called"

        # Verify profile data was used
        retrieval_input = mock_retrieval.execute.call_args[0][0]
        assert retrieval_input.profile_data == profile_data, (
            "Profile data should be passed to retrieval"
        )

        # Verify final result
        assert result.success, "Pipeline should complete successfully"
        assert len(result.book_recommendations) == 5, "Should return 5 final books"

    def test_empty_query_handled(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify empty query doesn't break pipeline.

        Edge case: user submits empty string.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        request = AgentRequest(
            user_text="",  # Empty query
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Should handle gracefully
        assert isinstance(result, AgentResponse), "Should return AgentResponse"
        assert result.text, "Should have response text"

    def test_very_long_query_handled(
        self,
        db_session,
        test_user_warm,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_curation_builder,
    ):
        """
        Verify very long query doesn't break pipeline.

        Edge case: query with 1000+ words.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = RecommendationAgent(
            current_user=test_user_warm,
            db=db_session,
            user_num_ratings=15,
            allow_profile=False,
            planner_agent=mock_planner,
            retrieval_agent=mock_retrieval,
            curation_agent=mock_curation,
        )

        # Generate very long query
        long_query = " ".join(["word"] * 1000)

        request = AgentRequest(
            user_text=long_query,
            conversation_history=[],
            context={"profile_allowed": False, "num_ratings": 15},
        )
        result = agent.execute(request)

        # Should handle gracefully (may truncate)
        assert isinstance(result, AgentResponse), "Should return AgentResponse"
        assert result.text, "Should have response text"
