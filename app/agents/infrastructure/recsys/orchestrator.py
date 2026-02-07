# app/agents/infrastructure/recsys/orchestrator.py
"""
Three-stage recommendation agent: Planning -> Retrieval -> Curation
Orchestrates the complete pipeline with both sync and streaming execution.
"""

from typing import Optional, Any, AsyncGenerator
import time

from app.agents.domain.entities import (
    AgentRequest,
    AgentResponse,
    AgentConfiguration,
    AgentCapability,
)
from app.agents.domain.recsys_schemas import (
    PlannerInput,
    RetrievalInput,
)
from app.agents.domain.interfaces import BaseAgent
from app.agents.domain.services import StandardResultProcessor
from app.agents.schemas import StreamChunk
from app.agents.logging import append_chatbot_log

from .planner_agent import PlannerAgent
from .retrieval_agent import RetrievalAgent
from .curation_agent import CurationAgent


class RecommendationAgent(BaseAgent):
    """
    Orchestrates three-stage recommendation pipeline.

    Stage 1 (PlannerAgent): Analyze query and determine strategy
    Stage 2 (RetrievalAgent): Execute strategy, gather 60-120 candidates
    Stage 3 (CurationAgent): Rank, filter, and generate prose with citations

    Supports both synchronous and streaming execution.
    """

    def __init__(
        self,
        current_user: Any = None,
        db: Any = None,
        user_num_ratings: Optional[int] = None,
        warm_threshold: int = 10,
        allow_profile: bool = False,
        planner_agent: Optional["PlannerAgent"] = None,
        retrieval_agent: Optional["RetrievalAgent"] = None,
        curation_agent: Optional["CurationAgent"] = None,
    ):
        """
        Initialize orchestrator with context for sub-agents.

        Args:
            current_user: Current user object (for internal tools)
            db: Database session (for internal tools)
            user_num_ratings: Number of ratings user has made
            warm_threshold: Minimum ratings for "warm" user status
            allow_profile: Whether user granted profile access consent
            planner_agent: Optional planner instance (for testing)
            retrieval_agent: Optional retrieval instance (for testing)
            curation_agent: Optional curation instance (for testing)
        """
        # Build configuration for metadata
        configuration = AgentConfiguration(
            policy_name="recsys.orchestrator",
            capabilities=frozenset([AgentCapability.INTERNAL_TOOLS]),
            llm_tier="large",
            timeout_seconds=120,
            max_iterations=0,
        )

        super().__init__(configuration)

        # Store context
        self._current_user = current_user
        self._db = db
        self._user_num_ratings = user_num_ratings or 0
        self._warm_threshold = warm_threshold
        self._allow_profile = allow_profile

        # Determine if ALS is available
        self._has_als_recs = self._user_num_ratings >= warm_threshold

        # Initialize sub-agents (use injected OR create default)
        self.planner_agent = (
            planner_agent
            if planner_agent is not None
            else PlannerAgent(
                current_user=current_user,
                db=db,
                user_num_ratings=self._user_num_ratings,
                has_als_recs_available=self._has_als_recs,
                allow_profile=allow_profile,
            )
        )

        self.retrieval_agent = (
            retrieval_agent
            if retrieval_agent is not None
            else RetrievalAgent(
                current_user=current_user,
                db=db,
                user_num_ratings=self._user_num_ratings,
                has_als_recs_available=self._has_als_recs,
            )
        )

        self.curation_agent = curation_agent if curation_agent is not None else CurationAgent()

        # Service for converting tool results to BookRecommendations
        self.result_processor = StandardResultProcessor()

        append_chatbot_log(
            f"Initialized RecommendationAgent orchestrator "
            f"(warm={self._has_als_recs}, profile={allow_profile})"
        )

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute three-stage recommendation pipeline asynchronously.

        Args:
            request: User request with query and context

        Returns:
            Final agent response with ordered books and prose
        """
        append_chatbot_log(
            f"\n{'=' * 60}\n"
            f"RECOMMENDATION ORCHESTRATOR START\n"
            f"Query: {request.user_text[:100]}...\n"
            f"{'=' * 60}"
        )

        # ============================================================
        # STAGE 1: PLANNING
        # ============================================================
        append_chatbot_log("\n=== STAGE 1: PLANNING ===")
        planning_start = time.time()

        try:
            # Determine available retrieval tools
            available_tools = [
                "book_semantic_search",
                "subject_hybrid_pool",
                "subject_id_search",
                "popular_books",
            ]
            if self._has_als_recs:
                available_tools.insert(0, "als_recs")

            planner_input = PlannerInput(
                query=request.user_text,
                has_als_recs_available=self._has_als_recs,
                allow_profile=self._allow_profile,
                available_retrieval_tools=available_tools,
            )

            # Execute planner (await if async)
            import inspect

            if inspect.iscoroutinefunction(self.planner_agent.execute):
                strategy = await self.planner_agent.execute(planner_input)
            else:
                strategy = self.planner_agent.execute(planner_input)

        except Exception as e:
            planning_time = int((time.time() - planning_start) * 1000)
            append_chatbot_log(f"[PLANNING ERROR after {planning_time}ms] {e}")
            return self._planning_error_fallback(request, str(e))

        planning_time = int((time.time() - planning_start) * 1000)

        append_chatbot_log(
            f"Planning: {planning_time}ms\n"
            f"Strategy: {strategy.reasoning[:150]}...\n"
            f"Recommended: {', '.join(strategy.recommended_tools)}\n"
            f"Fallback: {', '.join(strategy.fallback_tools)}"
        )

        # ============================================================
        # STAGE 2: RETRIEVAL (CANDIDATE GENERATION)
        # ============================================================
        append_chatbot_log("\n=== STAGE 2: RETRIEVAL ===")
        retrieval_start = time.time()

        try:
            retrieval_input = RetrievalInput(
                query=request.user_text,
                strategy=strategy,
                profile_data=strategy.profile_data,
            )

            # Execute retrieval (async)
            retrieval_output = await self.retrieval_agent.execute(retrieval_input)

        except Exception as e:
            retrieval_time = int((time.time() - retrieval_start) * 1000)
            append_chatbot_log(f"[RETRIEVAL ERROR after {retrieval_time}ms] {e}")
            return self._retrieval_error_fallback(request, str(e))

        retrieval_time = int((time.time() - retrieval_start) * 1000)

        # Convert candidates (dicts) to BookRecommendations
        candidates = self.result_processor._build_recommendations_from_objects(
            retrieval_output.candidates
        )

        append_chatbot_log(
            f"Retrieval: {len(candidates)} candidates in {retrieval_time}ms\n"
            f"Tools used: {', '.join(retrieval_output.execution_context.tools_used)}"
        )

        # Check if we got any candidates
        if not candidates:
            append_chatbot_log("[NO CANDIDATES] Returning empty results response")
            return self._no_candidates_fallback(request)

        # Log candidate quality
        rich_metadata_count = sum(
            1 for c in candidates if hasattr(c, "has_rich_metadata") and c.has_rich_metadata()
        )
        append_chatbot_log(f"Metadata: {rich_metadata_count}/{len(candidates)} with enrichment")

        # ============================================================
        # STAGE 3: CURATION
        # ============================================================
        append_chatbot_log("\n=== STAGE 3: CURATION ===")
        curation_start = time.time()

        try:
            # Execute curation (async)
            final_response = await self.curation_agent.execute(
                request=request,
                candidates=candidates,
                execution_context=retrieval_output.execution_context,
            )

        except Exception as e:
            curation_time = int((time.time() - curation_start) * 1000)
            append_chatbot_log(f"[CURATION ERROR after {curation_time}ms] {e}")
            return self._curation_error_fallback(request, candidates, str(e))

        curation_time = int((time.time() - curation_start) * 1000)

        # ============================================================
        # VALIDATION
        # ============================================================

        # Basic sanity check
        if not final_response.book_recommendations:
            append_chatbot_log("[VALIDATION ERROR] Curation returned no books, using fallback")
            return self._curation_empty_fallback(request, candidates)

        if not final_response.text or len(final_response.text.strip()) < 10:
            append_chatbot_log(
                "[VALIDATION WARNING] Curation returned minimal text, but proceeding"
            )

        # Log final stats
        total_time = planning_time + retrieval_time + curation_time

        append_chatbot_log(
            f"Curation: {len(final_response.book_recommendations)} books in {curation_time}ms"
        )
        append_chatbot_log(
            f"\n{'=' * 60}\n"
            f"RECOMMENDATION COMPLETE\n"
            f"Total: {total_time}ms (planning: {planning_time}ms, "
            f"retrieval: {retrieval_time}ms, curation: {curation_time}ms)\n"
            f"Books: {len(candidates)} -> {len(final_response.book_recommendations)}\n"
            f"{'=' * 60}\n"
        )

        return final_response

    async def execute_stream(self, request: AgentRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Execute three-stage pipeline with streaming output.

        Yields:
            StreamChunk objects with types:
            - "status": Progress updates from each stage
            - "token": Individual words from curation prose
            - "complete": Final result with book IDs from citations
        """
        append_chatbot_log(
            f"\n{'=' * 60}\n"
            f"RECOMMENDATION ORCHESTRATOR STREAMING\n"
            f"Query: {request.user_text[:100]}...\n"
            f"{'=' * 60}"
        )

        start_time = time.time()

        try:
            # ============================================================
            # STAGE 1: PLANNING
            # ============================================================
            yield StreamChunk(type="status", content="Analyzing your request...")

            append_chatbot_log("\n=== STAGE 1: PLANNING ===")
            planning_start = time.time()

            # Determine available retrieval tools
            available_tools = [
                "book_semantic_search",
                "subject_hybrid_pool",
                "subject_id_search",
                "popular_books",
            ]
            if self._has_als_recs:
                available_tools.insert(0, "als_recs")

            planner_input = PlannerInput(
                query=request.user_text,
                has_als_recs_available=self._has_als_recs,
                allow_profile=self._allow_profile,
                available_retrieval_tools=available_tools,
            )

            # Execute planner (will be async if planner is async)
            if hasattr(self.planner_agent.execute, "__call__"):
                # Check if it's a coroutine
                import inspect

                if inspect.iscoroutinefunction(self.planner_agent.execute):
                    strategy = await self.planner_agent.execute(planner_input)
                else:
                    strategy = self.planner_agent.execute(planner_input)
            else:
                strategy = self.planner_agent.execute(planner_input)

            planning_time = int((time.time() - planning_start) * 1000)

            append_chatbot_log(
                f"Planning: {planning_time}ms\n"
                f"Strategy: {strategy.reasoning[:150]}...\n"
                f"Recommended: {', '.join(strategy.recommended_tools)}"
            )

            # ============================================================
            # STAGE 2: RETRIEVAL
            # ============================================================
            yield StreamChunk(type="status", content="Gathering book candidates...")

            append_chatbot_log("\n=== STAGE 2: RETRIEVAL ===")
            retrieval_start = time.time()

            retrieval_input = RetrievalInput(
                query=request.user_text,
                strategy=strategy,
                profile_data=strategy.profile_data,
            )

            # Execute retrieval (async, non-blocking)
            retrieval_output = await self.retrieval_agent.execute(retrieval_input)

            # Convert candidates
            candidates = self.result_processor._build_recommendations_from_objects(
                retrieval_output.candidates
            )
            execution_context = retrieval_output.execution_context

            retrieval_time = int((time.time() - retrieval_start) * 1000)

            append_chatbot_log(f"Retrieval: {len(candidates)} candidates in {retrieval_time}ms")

            # Check if we got any candidates
            if not candidates:
                append_chatbot_log("[NO CANDIDATES] Returning empty results")
                yield StreamChunk(
                    type="complete",
                    data={
                        "target": "recsys",
                        "success": False,
                        "text": "I couldn't find books matching your request.",
                        "book_ids": [],
                        "elapsed_ms": int((time.time() - start_time) * 1000),
                    },
                )
                return

            # ============================================================
            # STAGE 3: CURATION
            # ============================================================
            append_chatbot_log("\n=== STAGE 3: CURATION ===")
            curation_start = time.time()

            # Stream tokens from curation
            async for chunk in self.curation_agent.execute_stream(
                request=request,
                candidates=candidates,
                execution_context=execution_context,
            ):
                # Forward all chunks (status, tokens, complete)
                yield chunk

            curation_time = int((time.time() - curation_start) * 1000)
            total_time = planning_time + retrieval_time + curation_time

            append_chatbot_log(
                f"\n{'=' * 60}\n"
                f"RECOMMENDATION COMPLETE (STREAMING)\n"
                f"Total: {total_time}ms\n"
                f"{'=' * 60}\n"
            )

        except Exception as e:
            append_chatbot_log(f"[ORCHESTRATOR ERROR] {type(e).__name__}: {e}")

            # Yield error completion
            yield StreamChunk(
                type="complete",
                data={
                    "target": "recsys",
                    "success": False,
                    "text": "I encountered an error generating recommendations.",
                    "error": str(e),
                    "elapsed_ms": int((time.time() - start_time) * 1000),
                },
            )

    # ================================================================
    # FALLBACK HANDLERS
    # ================================================================

    def _planning_error_fallback(self, request: AgentRequest, error_msg: str) -> AgentResponse:
        """Fallback when planning stage fails."""
        return AgentResponse(
            text="I'm having trouble analyzing your request. Please try rephrasing your query.",
            target_category="recsys",
            success=False,
            policy_version="recsys.orchestrator.planning_error",
        )

    def _retrieval_error_fallback(self, request: AgentRequest, error_msg: str) -> AgentResponse:
        """Fallback when retrieval stage fails."""
        return AgentResponse(
            text="I'm having trouble finding book recommendations right now. Please try again.",
            target_category="recsys",
            success=False,
            policy_version="recsys.orchestrator.retrieval_error",
        )

    def _no_candidates_fallback(self, request: AgentRequest) -> AgentResponse:
        """Fallback when retrieval returns no candidates."""
        # Check if query mentions dates/years
        query_lower = request.user_text.lower()
        has_year = any(str(year) in query_lower for year in range(2004, 2026))

        if has_year or "recent" in query_lower or "new" in query_lower:
            text = (
                "Our catalog only includes books published before 2004. "
                "Try searching for older books in similar genres or themes?"
            )
        else:
            text = (
                "I couldn't find books matching your specific request in our catalog. "
                "Try broader search terms or different subjects?"
            )

        return AgentResponse(
            text=text,
            target_category="recsys",
            success=False,
            policy_version="recsys.orchestrator.no_candidates",
        )

    def _curation_error_fallback(
        self, request: AgentRequest, candidates: list, error_msg: str
    ) -> AgentResponse:
        """Fallback when curation stage fails - return top candidates without prose."""
        # Take top 10 candidates as fallback
        top_books = candidates[:10]

        return AgentResponse(
            text="Here are some book recommendations (I had trouble generating descriptions):",
            target_category="recsys",
            success=True,
            book_recommendations=top_books,
            policy_version="recsys.orchestrator.curation_error",
        )

    def _curation_empty_fallback(self, request: AgentRequest, candidates: list) -> AgentResponse:
        """Fallback when curation returns empty results - return top candidates."""
        top_books = candidates[:10]

        return AgentResponse(
            text="Here are some book recommendations for you:",
            target_category="recsys",
            success=True,
            book_recommendations=top_books,
            policy_version="recsys.orchestrator.curation_empty",
        )
