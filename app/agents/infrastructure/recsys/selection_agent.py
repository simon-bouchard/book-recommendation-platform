# app/agents/infrastructure/recsys/selection_agent.py
"""
SelectionAgent for filtering and ranking book candidates before prose generation.
Stage 3a of the recommendation pipeline: Planner -> Retrieval -> Selection -> Curation.

JSON-mode agent that commits to a validated, ranked list of item_idx values
before any prose is written, eliminating ID hallucination in curation.
"""

import json
from typing import List, Optional

from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.agents.domain.entities import (
    AgentConfiguration,
    AgentRequest,
    BookRecommendation,
)
from app.agents.domain.recsys_schemas import ExecutionContext
from app.agents.infrastructure.base_langgraph_agent import BaseLangGraphAgent
from app.agents.tools.registry import ToolRegistry, InternalToolGates
from app.agents.prompts.loader import read_prompt
from app.agents.logging import append_chatbot_log


class SelectionResult(BaseModel):
    """Structured output from SelectionAgent — validated list of ranked book IDs."""

    selected_ids: List[int]
    """
    Ranked item_idx values from the candidate pool, best first.
    Must contain between 6 and 30 entries.
    All values must be item_idx values from the provided candidates.
    """


class SelectionAgent(BaseLangGraphAgent):
    """
    Filters and ranks candidate books, returning a validated ordered list of IDs.

    Single JSON-mode LLM call — no ReAct loop, no streaming.
    Applies the same filtering and ranking logic as the old CurationAgent:
    - Quality filtering (non-English, corrupted, missing metadata)
    - Negative constraint filtering from query
    - Relevance scoring against query
    - Diversity (avoid 3+ same author)
    - Outputs 6-30 ranked item_idx values

    All IDs are validated against the candidate pool before being returned,
    so CurationAgent receives only real books with real IDs.
    """

    def __init__(self):
        """Initialize SelectionAgent in JSON mode."""
        configuration = AgentConfiguration(
            policy_name="recsys.selection.md",
            capabilities=frozenset(),
            allowed_tools=frozenset(),
            llm_tier="medium",
            timeout_seconds=30,
            max_iterations=1,
        )

        super().__init__(configuration, mode="json")

        append_chatbot_log("Initialized SelectionAgent (json mode, no tools)")

    def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry:
        """Empty registry — selection uses no tools."""
        gates = InternalToolGates(
            user_num_ratings=0,
            warm_threshold=10,
            profile_allowed=False,
        )
        return ToolRegistry(
            web=False,
            docs=False,
            retrieval=False,
            context=False,
            gates=gates,
            ctx_user=ctx_user,
            ctx_db=ctx_db,
        )

    def _get_system_prompt(self) -> str:
        """Load selection system prompt."""
        return read_prompt("recsys.selection.md")

    def _get_target_category(self) -> str:
        return "recsys_selection"

    def _get_start_status(self) -> str:
        return "Selecting best matches..."

    def _add_context_messages(self, **context) -> List:
        """
        Inject candidates and execution context for selection.

        Args:
            **context: Must contain 'candidates' and 'execution_context'

        Returns:
            List containing single HumanMessage with formatted context
        """
        candidates: List[BookRecommendation] = context.get("candidates", [])
        execution_context: Optional[ExecutionContext] = context.get("execution_context")

        if not candidates:
            return []

        context_parts = []

        # Execution context so agent understands retrieval strategy
        if execution_context:
            context_parts.append(self._format_execution_context(execution_context))
            context_parts.append("")

        # Full candidate list — agent selects from these IDs only
        prepared = self._prepare_candidates(candidates)
        context_parts.append(f"CANDIDATES ({len(prepared)} books):")
        context_parts.append(json.dumps(prepared, indent=2, ensure_ascii=False))

        return [HumanMessage(content="\n".join(context_parts))]

    async def execute(
        self,
        request: AgentRequest,
        candidates: List[BookRecommendation],
        execution_context: ExecutionContext,
    ) -> List[BookRecommendation]:
        """
        Select and rank candidates, returning a validated ordered subset.

        Process:
        1. Single JSON LLM call to filter, score, and rank candidates
        2. Validate every returned ID exists in the candidate pool
        3. Silently drop any ID not in candidates (should not happen with a good prompt)
        4. Return ordered BookRecommendation objects ready for CurationAgent

        Args:
            request: Original user request (carries the query)
            candidates: Full pool from RetrievalAgent (60-120 books)
            execution_context: Strategy context from retrieval stage

        Returns:
            Ordered list of 6-30 BookRecommendation objects, all IDs validated
        """
        append_chatbot_log(
            f"\n{'=' * 60}\n"
            f"=== SELECTION AGENT ===\n"
            f"Input: {len(candidates)} candidates\n"
            f"Query: {request.user_text[:80]}...\n"
            f"{'=' * 60}"
        )

        candidate_map = {c.item_idx: c for c in candidates}

        result: SelectionResult = await self.execute_json(
            request,
            SelectionResult,
            candidates=candidates,
            execution_context=execution_context,
        )

        # Validate every returned ID against the candidate pool
        valid = []
        invalid = []
        for item_idx in result.selected_ids:
            if item_idx in candidate_map:
                valid.append(item_idx)
            else:
                invalid.append(item_idx)

        if invalid:
            append_chatbot_log(
                f"[SELECTION WARNING] {len(invalid)} IDs not in candidates, dropping: {invalid}"
            )

        # Enforce 6-30 range after validation
        valid = valid[:30]

        append_chatbot_log(
            f"Selection complete: {len(valid)} books selected ({len(invalid)} invalid IDs dropped)"
        )

        return [candidate_map[item_idx] for item_idx in valid]

    def _prepare_candidates(self, candidates: List[BookRecommendation]) -> List[dict]:
        """
        Prepare candidates for LLM context.

        Reuses the same dict structure as CurationAgent so the prompt
        sees identical fields in both selection and curation stages.

        Args:
            candidates: List of candidate BookRecommendation objects

        Returns:
            List of dicts with available metadata, empty fields omitted
        """
        prepared = []

        for book in candidates:
            if hasattr(book, "to_curation_dict"):
                book_dict = book.to_curation_dict()
            else:
                book_dict = {
                    "item_idx": book.item_idx,
                    "title": book.title or "",
                    "author": book.author or "",
                    "year": book.year or "",
                    "num_ratings": book.num_ratings or 0,
                }
                if book.subjects:
                    book_dict["subjects"] = book.subjects
                if book.tones:
                    book_dict["tones"] = book.tones
                if book.genre:
                    book_dict["genre"] = book.genre
                if book.vibe:
                    book_dict["vibe"] = book.vibe

            # Truncate vibe for token efficiency
            if "vibe" in book_dict and book_dict["vibe"]:
                if len(book_dict["vibe"]) > 200:
                    book_dict["vibe"] = book_dict["vibe"][:200] + "..."

            prepared.append(book_dict)

        append_chatbot_log(f"DEBUG first 3 candidates: {prepared[:3]}")

        return prepared

    def _format_execution_context(self, execution_context: ExecutionContext) -> str:
        """
        Format execution context for selection prompt.

        Args:
            execution_context: Context from retrieval stage

        Returns:
            Formatted string with strategy and tool info
        """
        lines = ["EXECUTION CONTEXT:", ""]
        lines.append(f"STRATEGY: {execution_context.planner_reasoning}")
        lines.append("")

        if execution_context.tools_used:
            lines.append("TOOLS USED:")
            for i, tool in enumerate(execution_context.tools_used, 1):
                lines.append(f"  {i}. {tool}")
            lines.append("")

        if execution_context.profile_data:
            lines.append("PROFILE DATA:")
            for key, value in execution_context.profile_data.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        return "\n".join(lines)
