# app/agents/infrastructure/recsys/curation_agent.py
"""
Curation agent for ranking and explaining book recommendations with inline citations.
Stage 3 of three-stage recommendation pipeline (Planner -> Retrieval -> Curation).
"""

import json
import re
from typing import List, Dict, AsyncGenerator
import os

from langchain_core.messages import HumanMessage

from app.agents.domain.entities import (
    AgentConfiguration,
    AgentCapability,
    AgentRequest,
    AgentResponse,
    BookRecommendation,
)
from app.agents.domain.recsys_schemas import ExecutionContext
from app.agents.infrastructure.base_langgraph_agent import BaseLangGraphAgent
from app.agents.schemas import StreamChunk
from app.agents.tools.registry import ToolRegistry, InternalToolGates
from app.agents.prompts.loader import read_prompt
from app.agents.logging import append_chatbot_log, log_data_transform, is_debug_mode
from app.agents.logging_modes import should_log_component, LoggingConfig


class CurationAgent(BaseLangGraphAgent):
    """
    Ranks candidate books and generates recommendation prose with inline citations.

    Uses BaseLangGraphAgent pattern with:
    - No tools (prose generation only via allowed_tools=frozenset())
    - Single-pass execution (max_iterations=1)
    - Natural language output with [Title](item_idx) markdown citations
    - Context injection via _add_context_messages()
    """

    def __init__(self):
        """Initialize curation agent with no-tools configuration."""
        configuration = AgentConfiguration(
            policy_name="recsys.curation.md",
            capabilities=frozenset(),  # No special capabilities needed
            allowed_tools=frozenset(),  # No tools - prose only
            llm_tier="large",
            timeout_seconds=30,
            max_iterations=1,  # Single LLM call
        )

        super().__init__(configuration)

        append_chatbot_log("Initialized CurationAgent with BaseLangGraphAgent (no tools)")

    def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry:
        """
        Create empty tool registry (curation uses no tools).

        Required by BaseLangGraphAgent but not used since allowed_tools is empty.
        """
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
        """Load and adapt system prompt for citation-based output."""
        base_prompt = read_prompt("recsys.curation.md")

        # Replace JSON output instructions with citation-based prose instructions
        adapted_prompt = base_prompt.replace(
            "## Output Format\n\nReturn JSON:",
            "## Output Format\n\nWrite natural prose with inline citations:",
        )

        # Remove JSON-specific instructions
        adapted_prompt = re.sub(r"```json.*?```", "", adapted_prompt, flags=re.DOTALL)

        # Add citation format reminder
        adapted_prompt += """

CRITICAL OUTPUT FORMAT:
- Write natural, conversational prose
- Use markdown-style citations: [Book Title](item_idx)
- Example: "I recommend [The Name of the Rose](702) by Umberto Eco for its masterful mystery."
- DO NOT output JSON
- DO NOT output book_ids list separately
- The citations in your prose ARE the book recommendations
"""

        return adapted_prompt

    def _get_target_category(self) -> str:
        """Return target category for curation responses."""
        return "recsys"

    def _get_start_status(self) -> str:
        """Custom initial status for curation stage."""
        return "Curating personalized recommendations..."

    def _add_context_messages(self, **context) -> List:
        """
        Inject candidates and execution context into message history.

        This is called by BaseLangGraphAgent._build_messages() between
        system prompt and conversation history.

        Args:
            **context: Must contain 'candidates' and 'execution_context'

        Returns:
            List containing single HumanMessage with formatted context
        """
        candidates = context.get("candidates", [])
        execution_context = context.get("execution_context")

        if not candidates or not execution_context:
            return []

        # Build context sections
        context_parts = []

        # 1. Execution context (strategy, tools, profile)
        context_parts.append(self._format_execution_context(execution_context))
        context_parts.append("")

        # 2. Candidates with metadata
        prepared_candidates = self._prepare_candidates(candidates)
        context_parts.append(f"CANDIDATES ({len(prepared_candidates)} books):")
        context_parts.append(json.dumps(prepared_candidates, indent=2, ensure_ascii=False))

        # Create single context message
        context_content = "\n".join(context_parts)

        if is_debug_mode():
            log_data_transform(
                "curation_context",
                {"num_candidates": len(candidates)},
                {"context_size": len(context_content)},
                "Injected context into messages",
            )

        return [HumanMessage(content=context_content)]

    async def execute(
        self,
        request: AgentRequest,
        candidates: List[BookRecommendation],
        execution_context: ExecutionContext,
    ) -> AgentResponse:
        """
        Curate candidates and generate response with inline citations.

        Args:
            request: Original user request
            candidates: Books from retrieval stage (60-120 unfiltered)
            execution_context: Context about retrieval strategy and execution

        Returns:
            AgentResponse with prose containing [Title](item_idx) citations
        """
        append_chatbot_log(f"\n{'=' * 60}")
        append_chatbot_log(f"=== CURATION START ===")
        append_chatbot_log(
            f"Input: {len(candidates)} candidates, Query: {request.user_text[:80]}..."
        )

        # Log stats in verbose mode
        if should_log_component("verbose"):
            if execution_context.tools_used:
                append_chatbot_log(f"Tools: {', '.join(execution_context.tools_used)}")

        # Execute via base class, passing context
        response = await super().execute(
            request,
            candidates=candidates,
            execution_context=execution_context,
        )

        # Parse citations to extract book IDs and build ordered list
        book_ids = self._extract_book_ids_from_citations(response.text)
        ordered_books = self._order_books_by_citations(candidates, book_ids)

        # Validate that cited books were in retrieved candidates
        candidate_ids_set = {c.item_idx for c in candidates}
        valid_citations = [bid for bid in book_ids if bid in candidate_ids_set]
        invalid_citations = [bid for bid in book_ids if bid not in candidate_ids_set]

        # Log validation results
        append_chatbot_log(
            f"Citations: {len(book_ids)} unique books referenced "
            f"({len(valid_citations)} valid, {len(invalid_citations)} invalid)"
        )

        if invalid_citations:
            append_chatbot_log(f"⚠️  WARNING: Cited books NOT in candidates: {invalid_citations}")

        if is_debug_mode():
            candidate_ids = [c.item_idx for c in candidates[:10]]
            append_chatbot_log(f"DEBUG - All cited IDs: {book_ids}")
            append_chatbot_log(f"DEBUG - Valid cited IDs: {valid_citations}")
            append_chatbot_log(f"DEBUG - First 10 candidate IDs: {candidate_ids}")
            append_chatbot_log(f"DEBUG - Response preview: {response.text[:500]}")

        # Update response with ordered books
        response.book_recommendations = ordered_books

        return response

    async def execute_stream(
        self,
        request: AgentRequest,
        candidates: List[BookRecommendation],
        execution_context: ExecutionContext,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream curation with status updates and token-by-token prose.

        Args:
            request: Original user request
            candidates: Books from retrieval stage
            execution_context: Context about retrieval

        Yields:
            StreamChunk objects:
            - type="status": "Curating personalized recommendations..."
            - type="token": Individual words/tokens of response
            - type="complete": Final result with book IDs from citations
        """
        append_chatbot_log(f"\n{'=' * 60}")
        append_chatbot_log(f"=== CURATION STREAMING ===")
        append_chatbot_log(f"Candidates: {len(candidates)}")

        # Track accumulated text for citation parsing
        accumulated_text = []

        # Stream via base class
        async for chunk in super().execute_stream(
            request,
            candidates=candidates,
            execution_context=execution_context,
        ):
            # Capture tokens for citation parsing
            if chunk.type == "token":
                accumulated_text.append(chunk.content)
                yield chunk

            # Intercept completion to add book IDs from citations
            elif chunk.type == "complete":
                full_text = "".join(accumulated_text)
                book_ids = self._extract_book_ids_from_citations(full_text)

                candidate_map = {c.item_idx: c for c in candidates}

                # Validate citations against candidates
                candidate_ids_set = {c.item_idx for c in candidates}
                valid_citations = [bid for bid in book_ids if bid in candidate_ids_set]
                invalid_citations = [bid for bid in book_ids if bid not in candidate_ids_set]

                append_chatbot_log(
                    f"Citations: {len(book_ids)} books cited "
                    f"({len(valid_citations)} valid, {len(invalid_citations)} invalid)"
                )

                if invalid_citations:
                    append_chatbot_log(
                        f"⚠️  WARNING: Cited books NOT in candidates: {invalid_citations}"
                    )

                if is_debug_mode():
                    candidate_ids = [c.item_idx for c in candidates[:10]]
                    append_chatbot_log(f"DEBUG - All cited IDs: {book_ids}")
                    append_chatbot_log(f"DEBUG - Valid cited IDs: {valid_citations}")
                    append_chatbot_log(f"DEBUG - First 10 candidate IDs: {candidate_ids}")
                    append_chatbot_log(f"DEBUG - Response preview: {full_text[:500]}")

                # Update completion data with book IDs
                chunk.data["book_ids"] = book_ids
                chunk.data["books"] = [
                    {
                        "item_idx": c.item_idx,
                        "title": c.title,
                        "author": c.author,
                        "cover_id": c.cover_id,
                        "year": c.year,
                    }
                    for bid in book_ids
                    if (c := candidate_map.get(bid)) is not None
                ]
                yield chunk

            # Pass through other chunks (status, etc.)
            else:
                yield chunk

    def _format_execution_context(self, execution_context: ExecutionContext) -> str:
        """
        Format execution context from planner and retrieval stages.

        Args:
            execution_context: Context about how candidates were generated

        Returns:
            Formatted multi-line string
        """
        lines = ["EXECUTION CONTEXT:"]
        lines.append("")

        # Strategy reasoning from planner
        lines.append(f"STRATEGY: {execution_context.planner_reasoning}")
        lines.append("")

        # Tools used in order
        if execution_context.tools_used:
            lines.append("TOOLS USED:")
            for i, tool in enumerate(execution_context.tools_used, 1):
                lines.append(f"  {i}. {tool}")
            lines.append("")

        # Profile data if available
        if execution_context.profile_data:
            lines.append("PROFILE DATA:")
            for key, value in execution_context.profile_data.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Guidance on personalization
        if execution_context.tools_used:
            if "als_recs" in execution_context.tools_used:
                lines.append("Note: Results are personalized based on collaborative filtering.")
            elif execution_context.profile_data:
                lines.append("Note: Results are informed by user profile data.")
            elif "popular_books" in execution_context.tools_used:
                lines.append("Note: Results are based on popular books (cold user).")

        return "\n".join(lines)

    def _prepare_candidates(self, candidates: List[BookRecommendation]) -> List[Dict]:
        """
        Prepare candidates for LLM context.

        Uses to_curation_dict() which excludes empty fields.
        Truncates vibe descriptions for token limits.

        Args:
            candidates: List of candidate books

        Returns:
            List of dict representations
        """
        prepared = []

        for book in candidates:
            if hasattr(book, "to_curation_dict"):
                book_dict = book.to_curation_dict()
            else:
                # Fallback manual extraction
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

            # Truncate vibe if too long (token limit)
            if "vibe" in book_dict and book_dict["vibe"]:
                vibe = book_dict["vibe"]
                if len(vibe) > 200:
                    book_dict["vibe"] = vibe[:200] + "..."

            prepared.append(book_dict)

        return prepared

    def _extract_book_ids_from_citations(self, text: str) -> List[int]:
        """
        Extract book IDs from markdown-style citations in text.

        Pattern: [Book Title](item_idx)
        Example: "[The Hobbit](4521)" -> 4521

        Args:
            text: Response text with citations

        Returns:
            List of unique book IDs in order of first appearance
        """
        # Pattern: [any text](digits)
        pattern = r"\[([^\]]+)\]\((\d+)\)"
        matches = re.findall(pattern, text)

        # Extract IDs, preserving order and removing duplicates
        book_ids = []
        seen = set()

        for title, item_idx in matches:
            item_idx = int(item_idx)
            if item_idx not in seen:
                book_ids.append(item_idx)
                seen.add(item_idx)

        return book_ids

    def _order_books_by_citations(
        self,
        candidates: List[BookRecommendation],
        cited_ids: List[int],
    ) -> List[BookRecommendation]:
        """
        Reorder candidates to match citation order.

        Args:
            candidates: Original candidate list
            cited_ids: Book IDs in citation order

        Returns:
            Ordered list of BookRecommendation objects
        """
        # Build lookup
        id_to_book = {book.item_idx: book for book in candidates}

        # Reorder based on citations
        ordered = []
        for book_id in cited_ids:
            if book_id in id_to_book:
                ordered.append(id_to_book[book_id])

        return ordered
