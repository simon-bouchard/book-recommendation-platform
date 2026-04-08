# tests/integration/chatbot/agents/recommendation/test_pipeline_integration.py
"""
Integration tests for the four-stage RecommendationAgent pipeline.

All tests drive execute_stream() and assert on the StreamChunk sequence.
Sub-agents are replaced with mocks from conftest.py — no LLM calls or
database queries are made.

Chunk collection pattern used throughout:
    chunks, complete, tokens, statuses = await _run(agent, request)
"""

from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.domain.entities import AgentRequest, BookRecommendation
from app.agents.domain.recsys_schemas import (
    ExecutionContext,
    PlannerInput,
    RetrievalInput,
    RetrievalOutput,
)
from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent
from app.agents.schemas import StreamChunk

# ==============================================================================
# Test Helpers
# ==============================================================================


async def _run(
    agent: RecommendationAgent,
    request: AgentRequest,
) -> Tuple[List[StreamChunk], StreamChunk, List[StreamChunk], List[StreamChunk]]:
    """
    Drive execute_stream() and partition the resulting chunks.

    Returns:
        (all_chunks, complete_chunk, token_chunks, status_chunks)

    Raises:
        AssertionError: If the pipeline does not emit exactly one complete chunk.
    """
    chunks: List[StreamChunk] = []
    async for chunk in agent.execute_stream(request):
        chunks.append(chunk)

    complete_chunks = [c for c in chunks if c.type == "complete"]
    assert len(complete_chunks) == 1, (
        f"Pipeline must emit exactly one complete chunk, got {len(complete_chunks)}"
    )
    return (
        chunks,
        complete_chunks[0],
        [c for c in chunks if c.type == "token"],
        [c for c in chunks if c.type == "status"],
    )


def _request(query: str = "recommend something", history: list | None = None) -> AgentRequest:
    """Minimal AgentRequest for test use."""
    return AgentRequest(
        user_text=query,
        conversation_history=history or [],
        context={"profile_allowed": False},
    )


def _agent(
    mock_planner,
    mock_retrieval,
    mock_selection,
    mock_curation,
    *,
    num_ratings: int = 15,
    allow_profile: bool = False,
    current_user=None,
    db=None,
) -> RecommendationAgent:
    """Build a RecommendationAgent with all four sub-agents injected."""
    return RecommendationAgent(
        current_user=current_user,
        db=db,
        user_num_ratings=num_ratings,
        allow_profile=allow_profile,
        planner_agent=mock_planner,
        retrieval_agent=mock_retrieval,
        selection_agent=mock_selection,
        curation_agent=mock_curation,
    )


# ==============================================================================
# TestStageTransitions
# ==============================================================================


class TestStageTransitions:
    """Verify data flows correctly from one stage to the next."""

    @pytest.mark.asyncio
    async def test_planner_strategy_reaches_retrieval(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        strategy_factory,
    ):
        """PlannerAgent output becomes the strategy field of RetrievalInput."""
        known_strategy = strategy_factory.custom_strategy(
            recommended_tools=["als_recs"],
            fallback_tools=["popular_books"],
            reasoning="Known test reasoning",
        )
        mock_planner = mock_planner_builder.returns_strategy(known_strategy).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        await _run(agent, _request("recommend books"))

        assert mock_retrieval.execute.called
        retrieval_input: RetrievalInput = mock_retrieval.execute.call_args[0][0]
        assert isinstance(retrieval_input, RetrievalInput)
        assert retrieval_input.strategy is known_strategy
        assert retrieval_input.query == "recommend books"

    @pytest.mark.asyncio
    async def test_retrieval_candidates_reach_selection(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        Full retrieval pool (as BookRecommendation objects) is passed to SelectionAgent.

        The orchestrator converts raw dicts to BookRecommendation via result_processor
        before calling selection; we verify count and item_idx are preserved.
        """
        raw_candidates = candidate_factory.create_batch(60, start_idx=10000)
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_candidates(raw_candidates).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        await _run(agent, _request())

        assert mock_selection.execute.called
        call_kwargs = mock_selection.execute.call_args.kwargs
        selection_candidates: List[BookRecommendation] = call_kwargs["candidates"]

        assert len(selection_candidates) == 60
        assert selection_candidates[0].item_idx == 10000

    @pytest.mark.asyncio
    async def test_selection_output_reaches_curation(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        SelectionAgent output (filtered subset) is what CurationAgent receives.

        Retrieval returns 60 books; selection returns 10 specific ones; curation
        must receive those 10 — not the full 60.
        """
        selected = candidate_factory.create_book_recommendations(10, start_idx=20000)
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_books(selected).build()

        # Track candidates received by curation
        received_by_curation: List[List[BookRecommendation]] = []

        async def _curation_stream(*args, **kwargs):
            received_by_curation.append(kwargs.get("candidates", []))
            yield StreamChunk(type="status", content="Curating...")
            yield StreamChunk(type="token", content="Here are your books.")
            yield StreamChunk(
                type="complete",
                data={
                    "target": "recsys",
                    "success": True,
                    "book_ids": [c.item_idx for c in kwargs.get("candidates", [])],
                    "tool_calls": [],
                    "elapsed_ms": 50,
                },
            )

        mock_curation = MagicMock()
        mock_curation.execute_stream = MagicMock(
            side_effect=lambda *a, **kw: _curation_stream(*a, **kw)
        )

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        await _run(agent, _request())

        assert len(received_by_curation) == 1
        curation_input = received_by_curation[0]
        assert len(curation_input) == 10
        assert all(c.item_idx >= 20000 for c in curation_input), (
            "Curation should receive selection output, not the raw retrieval pool"
        )

    @pytest.mark.asyncio
    async def test_execution_context_assembled_correctly(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        strategy_factory,
    ):
        """ExecutionContext passed to selection and curation includes planner reasoning."""
        known_strategy = strategy_factory.custom_strategy(
            recommended_tools=["als_recs"],
            fallback_tools=["popular_books"],
            reasoning="Using ALS for warm user",
        )
        mock_planner = mock_planner_builder.returns_strategy(known_strategy).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60, source_tool="als_recs").build()
        mock_selection = mock_selection_builder.returns_batch(10).build()

        received_contexts: List[ExecutionContext] = []

        async def _curation_stream(*args, **kwargs):
            received_contexts.append(kwargs.get("execution_context"))
            yield StreamChunk(type="token", content="Books.")
            yield StreamChunk(
                type="complete",
                data={"target": "recsys", "success": True, "book_ids": [], "elapsed_ms": 50},
            )

        mock_curation = MagicMock()
        mock_curation.execute_stream = MagicMock(
            side_effect=lambda *a, **kw: _curation_stream(*a, **kw)
        )

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        await _run(agent, _request())

        assert len(received_contexts) == 1
        ctx = received_contexts[0]
        assert isinstance(ctx, ExecutionContext)
        assert ctx.planner_reasoning == "Using ALS for warm user"
        assert "als_recs" in ctx.tools_used

    @pytest.mark.asyncio
    async def test_profile_data_flows_through_all_stages(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        strategy_factory,
    ):
        """Profile data set by planner propagates to retrieval and into ExecutionContext."""
        profile_data = {"user_profile": {"favorite_subjects": [12, 45, 78]}}
        strategy_with_profile = strategy_factory.cold_user_with_profile_strategy(profile_data)
        mock_planner = mock_planner_builder.returns_strategy(strategy_with_profile).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()

        received_contexts: List[ExecutionContext] = []

        async def _curation_stream(*args, **kwargs):
            received_contexts.append(kwargs.get("execution_context"))
            yield StreamChunk(type="token", content="Books.")
            yield StreamChunk(
                type="complete",
                data={"target": "recsys", "success": True, "book_ids": [], "elapsed_ms": 50},
            )

        mock_curation = MagicMock()
        mock_curation.execute_stream = MagicMock(
            side_effect=lambda *a, **kw: _curation_stream(*a, **kw)
        )

        agent = _agent(
            mock_planner,
            mock_retrieval,
            mock_selection,
            mock_curation,
            num_ratings=3,
            allow_profile=True,
        )
        await _run(agent, _request("suggest a book"))

        # Profile data reached retrieval
        retrieval_input: RetrievalInput = mock_retrieval.execute.call_args[0][0]
        assert retrieval_input.profile_data == profile_data

        # Profile data reached curation via ExecutionContext
        assert len(received_contexts) == 1
        assert received_contexts[0].profile_data == profile_data

    @pytest.mark.asyncio
    async def test_candidate_metadata_preserved_through_pipeline(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """Book title, author, and year survive the dict→BookRecommendation conversion."""
        raw_candidates = [
            candidate_factory.create_with_metadata(
                item_idx=12345,
                title="The Three-Body Problem",
                author="Cixin Liu",
                year=2008,
            ),
        ]
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_candidates(raw_candidates).build()
        # Pass through unchanged (return the single book as selected)
        mock_selection = mock_selection_builder.returns_batch(1, start_idx=12345).build()
        mock_curation_builder.returns_success_with_books(1).build()

        received: List[List[BookRecommendation]] = []

        async def _capture_stream(*args, **kwargs):
            received.append(list(kwargs.get("candidates", [])))
            yield StreamChunk(type="token", content="A book.")
            yield StreamChunk(
                type="complete",
                data={"target": "recsys", "success": True, "book_ids": [], "elapsed_ms": 50},
            )

        mock_curation_capture = MagicMock()
        mock_curation_capture.execute_stream = MagicMock(
            side_effect=lambda *a, **kw: _capture_stream(*a, **kw)
        )

        # We want to assert metadata on selection input since selection receives the converted objects
        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation_capture)
        await _run(agent, _request())

        assert mock_selection.execute.called
        selection_candidates = mock_selection.execute.call_args.kwargs["candidates"]
        book = selection_candidates[0]
        assert book.item_idx == 12345
        assert book.title == "The Three-Body Problem"
        assert book.author == "Cixin Liu"
        assert book.year == 2008


# ==============================================================================
# TestParameterPropagation
# ==============================================================================


class TestParameterPropagation:
    """Verify user context parameters configure each stage correctly."""

    @pytest.mark.asyncio
    async def test_warm_user_has_als_available(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """Warm user (num_ratings >= 10) gives planner als_recs in available tools."""
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation, num_ratings=15)
        await _run(agent, _request())

        planner_input: PlannerInput = mock_planner.execute.call_args[0][0]
        assert planner_input.has_als_recs_available is True
        assert "als_recs" in planner_input.available_retrieval_tools

    @pytest.mark.asyncio
    async def test_cold_user_has_no_als(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """Cold user (num_ratings < 10) does not have als_recs in available tools."""
        mock_planner = mock_planner_builder.returns_cold_descriptive_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(
            60, source_tool="book_semantic_search"
        ).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation, num_ratings=3)
        await _run(agent, _request())

        planner_input: PlannerInput = mock_planner.execute.call_args[0][0]
        assert planner_input.has_als_recs_available is False
        assert "als_recs" not in planner_input.available_retrieval_tools

    @pytest.mark.asyncio
    async def test_none_num_ratings_treated_as_cold(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """user_num_ratings=None defaults to 0 (cold user, no ALS)."""
        mock_planner = mock_planner_builder.returns_cold_descriptive_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(
            mock_planner, mock_retrieval, mock_selection, mock_curation, num_ratings=None
        )
        await _run(agent, _request())

        assert agent._has_als_recs is False
        planner_input: PlannerInput = mock_planner.execute.call_args[0][0]
        assert planner_input.has_als_recs_available is False

    @pytest.mark.asyncio
    async def test_allow_profile_propagates_to_planner(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """allow_profile flag reaches PlannerInput.allow_profile."""
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(
            mock_planner, mock_retrieval, mock_selection, mock_curation, allow_profile=True
        )
        await _run(agent, _request())

        planner_input: PlannerInput = mock_planner.execute.call_args[0][0]
        assert planner_input.allow_profile is True


# ==============================================================================
# TestErrorHandling  (fallback tests)
# ==============================================================================


class TestErrorHandling:
    """
    Verify each stage's fallback is triggered and the pipeline still completes.

    The invariant under test in every case: execute_stream() always emits
    exactly one complete chunk regardless of which stage fails.
    """

    # ------------------------------------------------------------------
    # Planning fallback
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_planning_failure_pipeline_still_completes(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """
        When PlannerAgent raises, the orchestrator falls back to a hardcoded
        strategy and the pipeline continues through all remaining stages.
        """
        mock_planner = mock_planner_builder.raises_error(RuntimeError("LLM timeout")).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, complete, _, _ = await _run(agent, _request("recommend books"))

        # Pipeline must complete successfully despite planner failure
        assert complete.data["success"] is True
        # Retrieval and selection must still be called
        assert mock_retrieval.execute.called
        assert mock_selection.execute.called

    @pytest.mark.asyncio
    async def test_planning_failure_warm_user_uses_als_strategy(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """
        Hardcoded fallback for a warm user recommends als_recs as primary tool.
        RetrievalAgent receives that strategy.
        """
        mock_planner = mock_planner_builder.raises_error(Exception("timeout")).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation, num_ratings=15)
        await _run(agent, _request())

        retrieval_input: RetrievalInput = mock_retrieval.execute.call_args[0][0]
        assert retrieval_input.strategy.recommended_tools == ["als_recs"]
        assert retrieval_input.strategy.fallback_tools == ["popular_books"]

    @pytest.mark.asyncio
    async def test_planning_failure_cold_user_uses_popular_books_strategy(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """
        Hardcoded fallback for a cold user (no ALS) uses popular_books as primary.
        """
        mock_planner = mock_planner_builder.raises_error(Exception("timeout")).build()
        mock_retrieval = mock_retrieval_builder.returns_batch(
            60, source_tool="popular_books"
        ).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation, num_ratings=3)
        await _run(agent, _request())

        retrieval_input: RetrievalInput = mock_retrieval.execute.call_args[0][0]
        assert retrieval_input.strategy.recommended_tools == ["popular_books"]

    # ------------------------------------------------------------------
    # Retrieval fallback
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_retrieval_failure_invokes_direct_tool_fallback(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        When RetrievalAgent raises, orchestrator calls _retrieve_fallback_candidates()
        directly. The pipeline continues and completes successfully.
        """
        fallback_candidates = candidate_factory.create_batch(30, start_idx=99000)
        fallback_output = RetrievalOutput(
            candidates=fallback_candidates,
            execution_context=ExecutionContext(
                planner_reasoning="Retrieval fallback",
                tools_used=["als_recs"],
            ),
            reasoning="Direct fallback call",
        )

        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.raises_error(
            RuntimeError("RetrievalAgent LLM failure")
        ).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)

        # Patch the internal fallback method so no real tool calls are made
        with patch.object(
            agent, "_retrieve_fallback_candidates", new=AsyncMock(return_value=fallback_output)
        ) as mock_fallback:
            _, complete, _, _ = await _run(agent, _request("recommend books"))

        assert mock_fallback.called, "_retrieve_fallback_candidates should be invoked"
        assert complete.data["success"] is True
        assert mock_selection.execute.called

    @pytest.mark.asyncio
    async def test_retrieval_total_failure_yields_error_complete_chunk(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """
        When both RetrievalAgent and _retrieve_fallback_candidates fail,
        the pipeline emits a terminal error complete chunk and stops.
        Selection and curation must not be called.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.raises_error(
            RuntimeError("RetrievalAgent failure")
        ).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)

        with patch.object(
            agent,
            "_retrieve_fallback_candidates",
            new=AsyncMock(side_effect=RuntimeError("Fallback tool also unavailable")),
        ):
            _, complete, _, _ = await _run(agent, _request())

        assert complete.data["success"] is False
        assert not mock_selection.execute.called
        assert not mock_curation.execute_stream.called

    # ------------------------------------------------------------------
    # No candidates
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_candidates_yields_terminal_error_chunk(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """
        When retrieval returns zero candidates, pipeline stops with a
        success=False complete chunk. Selection and curation are not called.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_empty().build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, complete, _, _ = await _run(agent, _request("some obscure query xyz123"))

        assert complete.data["success"] is False
        assert complete.data["book_ids"] == []
        assert not mock_selection.execute.called
        assert not mock_curation.execute_stream.called

    @pytest.mark.asyncio
    async def test_no_candidates_with_year_query_includes_catalog_hint(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """
        When the query contains a recent year and there are no candidates,
        the error message mentions the catalog cutoff.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_empty().build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, complete, _, _ = await _run(agent, _request("books published in 2022"))

        assert complete.data["success"] is False
        assert "2004" in complete.data["text"], (
            "Error message should mention catalog cutoff year for recency queries"
        )

    @pytest.mark.asyncio
    async def test_no_candidates_with_recent_keyword_includes_catalog_hint(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """Query containing 'recent' also triggers the catalog-limit hint."""
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_empty().build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, complete, _, _ = await _run(agent, _request("recent sci-fi novels"))

        assert complete.data["success"] is False
        assert "2004" in complete.data["text"]

    # ------------------------------------------------------------------
    # Selection fallback
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_selection_failure_forwards_top10_to_curation(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        When SelectionAgent raises, the orchestrator forwards the first 10
        candidates from the retrieval pool directly to CurationAgent.
        """
        raw_candidates = candidate_factory.create_batch(60, start_idx=10000)
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_candidates(raw_candidates).build()
        mock_selection = mock_selection_builder.raises_error(
            RuntimeError("SelectionAgent LLM failure")
        ).build()

        received_by_curation: List[List[BookRecommendation]] = []

        async def _curation_stream(*args, **kwargs):
            received_by_curation.append(list(kwargs.get("candidates", [])))
            yield StreamChunk(type="token", content="Books.")
            yield StreamChunk(
                type="complete",
                data={"target": "recsys", "success": True, "book_ids": [], "elapsed_ms": 50},
            )

        mock_curation = MagicMock()
        mock_curation.execute_stream = MagicMock(
            side_effect=lambda *a, **kw: _curation_stream(*a, **kw)
        )

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, complete, _, _ = await _run(agent, _request())

        # Curation must still be called
        assert len(received_by_curation) == 1
        fallback_candidates = received_by_curation[0]
        assert len(fallback_candidates) == 10, (
            f"Selection fallback should forward top-10, got {len(fallback_candidates)}"
        )
        # They should be the first 10 from the retrieval pool
        assert fallback_candidates[0].item_idx == 10000

    @pytest.mark.asyncio
    async def test_selection_empty_result_forwards_top10_to_curation(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        When SelectionAgent returns an empty list (no books pass its filter),
        the orchestrator also falls back to the top-10 retrieval candidates.
        """
        raw_candidates = candidate_factory.create_batch(60, start_idx=30000)
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_candidates(raw_candidates).build()
        mock_selection = mock_selection_builder.returns_empty().build()

        received_by_curation: List[List[BookRecommendation]] = []

        async def _curation_stream(*args, **kwargs):
            received_by_curation.append(list(kwargs.get("candidates", [])))
            yield StreamChunk(type="token", content="Books.")
            yield StreamChunk(
                type="complete",
                data={"target": "recsys", "success": True, "book_ids": [], "elapsed_ms": 50},
            )

        mock_curation = MagicMock()
        mock_curation.execute_stream = MagicMock(
            side_effect=lambda *a, **kw: _curation_stream(*a, **kw)
        )

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        await _run(agent, _request())

        assert len(received_by_curation) == 1
        assert len(received_by_curation[0]) == 10

    # ------------------------------------------------------------------
    # Curation fallback
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_curation_failure_yields_fallback_prose_token(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        When CurationAgent.execute_stream() raises, the orchestrator emits a
        single token chunk containing markdown-formatted prose and then closes
        the stream with a complete chunk.
        """
        selected = candidate_factory.create_book_recommendations(5, start_idx=50000)
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_books(selected).build()
        mock_curation = mock_curation_builder.raises_error_on_stream(
            RuntimeError("Curation LLM failure")
        ).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, complete, tokens, _ = await _run(agent, _request())

        assert len(tokens) >= 1, "Fallback prose must be emitted as at least one token chunk"
        full_text = "".join(c.content for c in tokens)
        assert "Here are some book recommendations" in full_text

    @pytest.mark.asyncio
    async def test_curation_failure_complete_chunk_has_book_ids(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        The complete chunk after a curation failure must contain the book_ids
        taken from the selected candidates, so the frontend can render book cards.
        """
        selected = candidate_factory.create_book_recommendations(5, start_idx=50000)
        expected_ids = [c.item_idx for c in selected]

        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_books(selected).build()
        mock_curation = mock_curation_builder.raises_error_on_stream(
            RuntimeError("Curation LLM failure")
        ).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, complete, _, _ = await _run(agent, _request())

        assert sorted(complete.data["book_ids"]) == sorted(expected_ids)

    @pytest.mark.asyncio
    async def test_curation_failure_complete_chunk_marks_success_false(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """Complete chunk on curation failure must have success=False (degraded mode)."""
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.raises_error_on_stream(
            RuntimeError("Curation LLM failure")
        ).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, complete, _, _ = await _run(agent, _request())

        assert complete.data["success"] is False

    @pytest.mark.asyncio
    async def test_curation_failure_fallback_prose_contains_titles(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
        candidate_factory,
    ):
        """
        The fallback prose token includes book titles and authors so the
        response is human-readable without the frontend rendering book cards.
        """
        selected = [
            BookRecommendation(item_idx=1001, title="Dune", author="Frank Herbert", year=1965),
            BookRecommendation(
                item_idx=1002, title="Neuromancer", author="William Gibson", year=1984
            ),
        ]
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_books(selected).build()
        mock_curation = mock_curation_builder.raises_error_on_stream(
            RuntimeError("Curation failure")
        ).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, _, tokens, _ = await _run(agent, _request())

        full_text = "".join(c.content for c in tokens)
        assert "Dune" in full_text
        assert "Frank Herbert" in full_text
        assert "Neuromancer" in full_text


# ==============================================================================
# TestFullPipelineFlow
# ==============================================================================


class TestFullPipelineFlow:
    """End-to-end chunk sequence checks for the happy path."""

    @pytest.mark.asyncio
    async def test_chunk_sequence_order(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """
        Verify status chunks appear before token chunks and complete is last.

        Expected order: status(s)... → token(s)... → complete
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        chunks, complete, tokens, statuses = await _run(agent, _request())

        assert len(statuses) >= 1, "At least one status chunk should be emitted"
        assert len(tokens) >= 1, "At least one token chunk should be emitted"
        assert chunks[-1].type == "complete", "Complete chunk must be last"

        # All status chunks should come before the first token
        if tokens:
            first_token_idx = chunks.index(tokens[0])
            for status in statuses:
                assert chunks.index(status) < first_token_idx, (
                    "Status chunks should precede token chunks"
                )

    @pytest.mark.asyncio
    async def test_complete_chunk_has_book_ids(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """Complete chunk must include book_ids so the frontend can render cards."""
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation)
        _, complete, _, _ = await _run(agent, _request())

        assert "book_ids" in complete.data
        assert isinstance(complete.data["book_ids"], list)
        assert len(complete.data["book_ids"]) > 0
        assert all(isinstance(bid, int) for bid in complete.data["book_ids"])

    @pytest.mark.asyncio
    async def test_complete_chunk_annotated_with_tools_used(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """
        The orchestrator annotates the complete chunk with tools_used from
        the ExecutionContext so callers can audit what retrieval method was used.
        """
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60, source_tool="als_recs").build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation, num_ratings=15)
        _, complete, _, _ = await _run(agent, _request())

        assert "tools_used" in complete.data
        assert "als_recs" in complete.data["tools_used"]

    @pytest.mark.asyncio
    async def test_warm_user_complete_flow(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """Full happy-path: warm user → all four stages called → success."""
        mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(60, source_tool="als_recs").build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation, num_ratings=15)
        _, complete, _, _ = await _run(agent, _request("recommend something"))

        assert mock_planner.execute.called
        assert mock_retrieval.execute.called
        assert mock_selection.execute.called
        assert mock_curation.execute_stream.called
        assert complete.data["success"] is True

    @pytest.mark.asyncio
    async def test_cold_user_complete_flow(
        self,
        mock_planner_builder,
        mock_retrieval_builder,
        mock_selection_builder,
        mock_curation_builder,
    ):
        """Full happy-path: cold user → semantic search strategy → success."""
        mock_planner = mock_planner_builder.returns_cold_descriptive_strategy().build()
        mock_retrieval = mock_retrieval_builder.returns_batch(
            60, source_tool="book_semantic_search"
        ).build()
        mock_selection = mock_selection_builder.returns_batch(10).build()
        mock_curation = mock_curation_builder.returns_success_with_books(5).build()

        agent = _agent(mock_planner, mock_retrieval, mock_selection, mock_curation, num_ratings=3)
        _, complete, _, _ = await _run(agent, _request("dark atmospheric mystery"))

        assert complete.data["success"] is True
