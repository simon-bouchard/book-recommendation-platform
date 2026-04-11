# app/agents/infrastructure/recsys/orchestrator.py
"""
Four-stage recommendation pipeline: Planning → Retrieval → Selection → Curation.
Streaming-first execution with independent per-stage fallback recovery.
"""

import asyncio
import time
from typing import Any, AsyncGenerator, List, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

from app.agents.domain.entities import (
    AgentCapability,
    AgentConfiguration,
    AgentRequest,
    BookRecommendation,
)
from app.agents.domain.interfaces import BaseAgent
from app.agents.domain.recsys_schemas import (
    ExecutionContext,
    PlannerInput,
    PlannerStrategy,
    RetrievalInput,
    RetrievalOutput,
)
from app.agents.domain.services import StandardResultProcessor
from app.agents.logging import append_chatbot_log
from app.agents.schemas import StreamChunk
from app.agents.tools.registry import InternalToolGates, ToolRegistry

from .curation_agent import CurationAgent
from .planner_agent import PlannerAgent
from .retrieval_agent import RetrievalAgent
from .selection_agent import SelectionAgent


class RecommendationAgent(BaseAgent):
    """
    Orchestrates the four-stage recommendation pipeline.

    Stage 1 (PlannerAgent):   Analyze query and determine retrieval strategy.
    Stage 2 (RetrievalAgent): Execute strategy, gather 60–120 candidate books.
    Stage 3 (SelectionAgent): Filter and rank candidates to a validated 6–30 subset.
    Stage 4 (CurationAgent):  Write personalized prose with inline citations (streaming).

    Each stage has an independent fallback so a single failure does not
    cascade to the user. Only Curation output is streamed; earlier stages
    complete fully before the next begins.

    Failure contract:
        Planning failure  → hardcoded strategy (ALS for warm, popular_books for cold),
                            pipeline continues.
        Retrieval failure → single direct tool call bypassing the agent,
                            pipeline continues.
        No candidates     → terminal error chunk, pipeline stops.
        Selection failure → top-10 raw candidates forwarded to Curation,
                            pipeline continues.
        Curation failure  → fallback prose built from candidates, sent as a
                            single token chunk for minimal time-to-first-token.
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
        selection_agent: Optional["SelectionAgent"] = None,
        curation_agent: Optional["CurationAgent"] = None,
    ):
        """
        Initialize orchestrator with context for sub-agents.

        Args:
            current_user: SQLAlchemy user object (for retrieval tools).
            db: SQLAlchemy session (for retrieval tools).
            user_num_ratings: Number of ratings the user has made.
            warm_threshold: Minimum ratings for warm-user status (ALS eligible).
            allow_profile: Whether the user granted profile-access consent.
            planner_agent: Override PlannerAgent (primarily for testing).
            retrieval_agent: Override RetrievalAgent (primarily for testing).
            selection_agent: Override SelectionAgent (primarily for testing).
            curation_agent: Override CurationAgent (primarily for testing).
        """
        configuration = AgentConfiguration(
            policy_name="recsys.orchestrator",
            capabilities=frozenset([AgentCapability.INTERNAL_TOOLS]),
            llm_tier="large",
            timeout_seconds=120,
            max_iterations=0,
        )
        super().__init__(configuration)

        self._current_user = current_user
        self._db = db
        self._user_num_ratings = user_num_ratings or 0
        self._warm_threshold = warm_threshold
        self._allow_profile = allow_profile
        self._has_als_recs = self._user_num_ratings >= warm_threshold

        self.planner_agent = planner_agent or PlannerAgent(
            current_user=current_user,
            db=db,
            user_num_ratings=self._user_num_ratings,
            has_als_recs_available=self._has_als_recs,
            allow_profile=allow_profile,
        )
        self.retrieval_agent = retrieval_agent or RetrievalAgent(
            current_user=current_user,
            db=db,
            user_num_ratings=self._user_num_ratings,
            has_als_recs_available=self._has_als_recs,
        )
        self.selection_agent = selection_agent or SelectionAgent()
        self.curation_agent = curation_agent or CurationAgent()
        self.result_processor = StandardResultProcessor()

        append_chatbot_log(
            f"Initialized RecommendationAgent (warm={self._has_als_recs}, profile={allow_profile})"
        )

    # ================================================================
    # PRIMARY ENTRY POINT
    # ================================================================

    async def execute_stream(self, request: AgentRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Execute the four-stage pipeline with streaming output.

        Yields StreamChunk objects in this order:
            status  — one per stage transition (always yielded).
            token   — individual LLM prose tokens from Curation, or a single
                      fallback string when Curation fails.
            complete — exactly one final chunk with book_ids and metadata.

        Args:
            request: User request carrying the query and conversation history.
        """
        start_time = time.time()

        # ============================================================
        # STAGE 1: PLANNING
        # ============================================================
        yield StreamChunk(type="status", content="Building a search strategy...")

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

        with tracer.start_as_current_span("chat.recsys.planning") as plan_span:
            try:
                append_chatbot_log("\n=== STAGE 1: PLANNING ===")
                planning_start = time.time()

                strategy = await self.planner_agent.execute(planner_input)

                plan_span.set_attribute("recsys.planning.recommended_tools", str(strategy.recommended_tools))
                plan_span.set_attribute("recsys.planning.fallback_tools", str(strategy.fallback_tools))
                append_chatbot_log(
                    f"Planning: {int((time.time() - planning_start) * 1000)}ms | "
                    f"Recommended: {strategy.recommended_tools} | "
                    f"Fallback: {strategy.fallback_tools}"
                )

            except Exception as e:
                plan_span.record_exception(e)
                plan_span.set_attribute("recsys.planning.used_fallback", True)
                append_chatbot_log(
                    f"[PLANNING FAILED] {type(e).__name__}: {e} — using hardcoded fallback strategy"
                )
                strategy = self._build_fallback_strategy()

        # ============================================================
        # STAGE 2: RETRIEVAL
        # ============================================================
        yield StreamChunk(type="status", content="Gathering book candidates...")

        retrieval_input = RetrievalInput(
            query=request.user_text,
            strategy=strategy,
            profile_data=strategy.profile_data,
        )

        with tracer.start_as_current_span("chat.recsys.retrieval") as ret_span:
            try:
                append_chatbot_log("\n=== STAGE 2: RETRIEVAL ===")
                retrieval_start = time.time()

                retrieval_output = await self.retrieval_agent.execute(retrieval_input)
                candidates = self.result_processor._build_recommendations_from_objects(
                    retrieval_output.candidates
                )
                execution_context = retrieval_output.execution_context

                ret_span.set_attribute("recsys.retrieval.candidate_count", len(candidates))
                ret_span.set_attribute("recsys.retrieval.tools_used", str(execution_context.tools_used))
                append_chatbot_log(
                    f"Retrieval: {int((time.time() - retrieval_start) * 1000)}ms | "
                    f"{len(candidates)} candidates | "
                    f"Tools: {execution_context.tools_used}"
                )

            except Exception as e:
                ret_span.record_exception(e)
                append_chatbot_log(
                    f"[RETRIEVAL FAILED] {type(e).__name__}: {e} — attempting direct tool fallback"
                )
                try:
                    retrieval_output = await self._retrieve_fallback_candidates()
                    candidates = self.result_processor._build_recommendations_from_objects(
                        retrieval_output.candidates
                    )
                    execution_context = retrieval_output.execution_context

                    ret_span.set_attribute("recsys.retrieval.candidate_count", len(candidates))
                    ret_span.set_attribute("recsys.retrieval.used_fallback", True)
                    ret_span.set_attribute("recsys.retrieval.tools_used", str(execution_context.tools_used))
                    append_chatbot_log(
                        f"Retrieval fallback: {len(candidates)} candidates via "
                        f"{execution_context.tools_used}"
                    )

                except Exception as fallback_err:
                    ret_span.set_attribute("recsys.retrieval.total_failure", True)
                    append_chatbot_log(f"[RETRIEVAL FALLBACK FAILED] {fallback_err}")
                    yield self._error_complete_chunk(
                        "I'm having trouble finding book recommendations right now. Please try again.",
                        start_time,
                    )
                    return

        if not candidates:
            append_chatbot_log("[NO CANDIDATES] Returning empty results response")
            yield self._no_candidates_complete_chunk(request.user_text, start_time)
            return

        # ============================================================
        # STAGE 3: SELECTION
        # ============================================================
        yield StreamChunk(type="status", content="Selecting best matches...")

        append_chatbot_log("\n=== STAGE 3: SELECTION ===")
        selection_start = time.time()
        selected_candidates: Optional[List[BookRecommendation]] = None

        with tracer.start_as_current_span("chat.recsys.selection") as sel_span:
            try:
                result = await self.selection_agent.execute(
                    request=request,
                    candidates=candidates,
                    execution_context=execution_context,
                )
                if result:
                    selected_candidates = result

            except Exception as e:
                sel_span.record_exception(e)
                sel_span.set_attribute("recsys.selection.used_fallback", True)
                append_chatbot_log(
                    f"[SELECTION FAILED] {type(e).__name__}: {e} — falling back to top-10 candidates"
                )

            if not selected_candidates:
                append_chatbot_log("Selection fallback: forwarding top-10 raw candidates to curation")
                selected_candidates = candidates[:10]
                sel_span.set_attribute("recsys.selection.used_fallback", True)
            else:
                append_chatbot_log(
                    f"Selection: {int((time.time() - selection_start) * 1000)}ms | "
                    f"{len(selected_candidates)} books selected"
                )
            sel_span.set_attribute("recsys.selection.selected_count", len(selected_candidates))

        # ============================================================
        # STAGE 4: CURATION
        # ============================================================
        append_chatbot_log("\n=== STAGE 4: CURATION ===")

        curation_tokens_sent = 0
        curation_complete_sent = False

        with tracer.start_as_current_span("chat.recsys.curation") as cur_span:
            try:
                async for chunk in self.curation_agent.execute_stream(
                    request=request,
                    candidates=selected_candidates,
                    execution_context=execution_context,
                ):
                    if chunk.type == "token":
                        curation_tokens_sent += 1

                    elif chunk.type == "complete":
                        # Annotate with upstream metadata before forwarding.
                        chunk.data["tools_used"] = execution_context.tools_used
                        chunk.data["selection_count"] = len(selected_candidates)
                        curation_complete_sent = True

                    yield chunk

                cur_span.set_attribute("recsys.curation.tokens_sent", curation_tokens_sent)
                cur_span.set_attribute("recsys.curation.succeeded", True)

            except Exception as e:
                cur_span.record_exception(e)
                cur_span.set_attribute("recsys.curation.succeeded", False)
                append_chatbot_log(
                    f"[CURATION FAILED] {type(e).__name__}: {e} — yielding fallback prose"
                )

                # Only send fallback text if no tokens reached the client yet.
                if not curation_tokens_sent:
                    fallback_text = self._build_curation_fallback_text(selected_candidates)
                    yield StreamChunk(type="token", content=fallback_text)

                # Always close the stream with a complete chunk.
                if not curation_complete_sent:
                    yield self._curation_fallback_complete_chunk(
                        selected_candidates, execution_context, start_time
                    )

    # ================================================================
    # FALLBACK HELPERS
    # ================================================================

    def _build_fallback_strategy(self) -> PlannerStrategy:
        """
        Hardcoded strategy used when PlannerAgent fails.

        Warm users (ALS available) are directed to collaborative filtering.
        Cold users fall back to popular books.
        Profile data is always omitted — a fallback should be simple and fast.
        """
        if self._has_als_recs:
            return PlannerStrategy(
                recommended_tools=["als_recs"],
                fallback_tools=["popular_books"],
                reasoning="Hardcoded fallback — planning stage unavailable",
                profile_data=None,
            )
        return PlannerStrategy(
            recommended_tools=["popular_books"],
            fallback_tools=["book_semantic_search"],
            reasoning="Hardcoded fallback — planning stage unavailable, cold user",
            profile_data=None,
        )

    async def _retrieve_fallback_candidates(self) -> RetrievalOutput:
        """
        Bypass RetrievalAgent and call a single retrieval tool directly.

        Uses ALS for warm users (personalized), popular_books for cold users.
        tool.invoke() is synchronous; asyncio.to_thread() prevents blocking
        the event loop while the tool query runs.

        Returns:
            RetrievalOutput with raw candidates and a minimal execution context.

        Raises:
            RuntimeError: If the target fallback tool is not available in the registry.
        """
        gates = InternalToolGates(
            user_num_ratings=self._user_num_ratings,
            warm_threshold=self._warm_threshold,
            profile_allowed=False,
        )
        registry = ToolRegistry.for_retrieval(
            gates=gates,
            ctx_user=self._current_user,
            ctx_db=self._db,
        )

        tool_name = "als_recs" if self._has_als_recs else "popular_books"
        tool = registry.get_tool(tool_name)

        if tool is None:
            raise RuntimeError(f"Fallback tool '{tool_name}' not available in registry")

        raw = await asyncio.to_thread(tool.invoke, {"top_k": 60})

        return RetrievalOutput(
            candidates=raw if isinstance(raw, list) else [],
            execution_context=ExecutionContext(
                planner_reasoning="Retrieval fallback — agent unavailable",
                tools_used=[tool_name],
                profile_data=None,
            ),
            reasoning=f"Retrieval fallback: called {tool_name} directly.",
        )

    def _build_curation_fallback_text(self, candidates: List[BookRecommendation]) -> str:
        """
        Build minimal recommendation prose when CurationAgent fails.

        Deliberately uses the [Title](item_idx) citation format so the
        frontend can still render book cards from the plain-text output.

        Args:
            candidates: Books to list (the selected subset, or top-10 raw).

        Returns:
            Formatted multi-line recommendation string.
        """
        lines = ["Here are some book recommendations:\n"]
        for book in candidates:
            author_part = f" by {book.author}" if book.author else ""
            lines.append(f"[{book.title}]({book.item_idx}){author_part}")
        return "\n".join(lines)

    def _error_complete_chunk(self, message: str, start_time: float) -> StreamChunk:
        """
        Build a terminal error complete chunk for hard failures (e.g. retrieval total failure).
        """
        return StreamChunk(
            type="complete",
            data={
                "target": "recsys",
                "success": False,
                "text": message,
                "book_ids": [],
                "elapsed_ms": int((time.time() - start_time) * 1000),
            },
        )

    def _no_candidates_complete_chunk(self, query: str, start_time: float) -> StreamChunk:
        """
        Build a terminal complete chunk when retrieval returns no candidates.

        If the query references a year (2004–2025) or recency keywords,
        includes a catalog-limit hint (catalog ends at 2004).
        """
        query_lower = query.lower()
        mentions_recent = any(str(year) in query_lower for year in range(2004, 2026)) or any(
            w in query_lower for w in ("recent", "new", "latest")
        )

        if mentions_recent:
            text = (
                "Our catalog only includes books published before 2004. "
                "Try searching for older books in similar genres or themes?"
            )
        else:
            text = (
                "I couldn't find books matching your request in our catalog. "
                "Try broader search terms or different subjects?"
            )

        return StreamChunk(
            type="complete",
            data={
                "target": "recsys",
                "success": False,
                "text": text,
                "book_ids": [],
                "elapsed_ms": int((time.time() - start_time) * 1000),
            },
        )

    def _curation_fallback_complete_chunk(
        self,
        candidates: List[BookRecommendation],
        execution_context: ExecutionContext,
        start_time: float,
    ) -> StreamChunk:
        """
        Build the complete chunk for the curation fallback path.

        Book IDs are taken directly from the candidates list since we wrote
        the citations ourselves in _build_curation_fallback_text().
        """
        book_ids = [c.item_idx for c in candidates]
        books = [
            {
                "item_idx": c.item_idx,
                "title": c.title,
                "author": c.author,
                "cover_id": c.cover_id,
                "year": c.year,
            }
            for c in candidates
        ]
        return StreamChunk(
            type="complete",
            data={
                "target": "recsys",
                "success": False,
                "book_ids": book_ids,
                "books": books,
                "tools_used": execution_context.tools_used,
                "selection_count": len(candidates),
                "elapsed_ms": int((time.time() - start_time) * 1000),
            },
        )
