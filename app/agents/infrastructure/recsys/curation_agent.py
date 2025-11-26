# app/agents/infrastructure/recsys/curation_agent.py
"""
Curation agent for ranking and selecting final book recommendations.
Stage 3 of three-stage recommendation pipeline (Planner â†’ Retrieval â†’ Curation).
"""

import json
from typing import List, Dict

from langchain_core.messages import HumanMessage

from app.agents.domain.entities import (
    AgentRequest,
    AgentResponse,
    BookRecommendation,
)
from app.agents.domain.recsys_schemas import ExecutionContext
from app.agents.domain.parsers import InlineReferenceParser
from app.agents.settings import get_llm
from app.agents.prompts.loader import read_prompt
from app.agents.logging import append_chatbot_log, log_data_transform, is_debug_mode
from app.agents.logging_modes import should_log_component

from app.agents.logging_modes import debug_mode

debug_mode()


class CurationAgent:
    """
    Ranks candidate books and generates recommendation prose.

    No tool calling - just LLM reasoning over candidates.
    Single call with JSON output.
    """

    def __init__(self):
        self.llm = get_llm(tier="large", temperature=0.2, json_mode=True)

    def execute(
        self,
        request: AgentRequest,
        candidates: List[BookRecommendation],
        execution_context: ExecutionContext,
    ) -> AgentResponse:
        """
        Curate candidates and generate response.

        Args:
            request: Original user request
            candidates: Books from retrieval stage (unfiltered)
            execution_context: Context about retrieval strategy and execution

        Returns:
            Final agent response with ordered books and prose
        """
        append_chatbot_log(f"\n{'=' * 60}")
        append_chatbot_log(f"=== CURATION START ===")
        append_chatbot_log(
            f"Input: {len(candidates)} candidates, Query: {request.user_text[:80]}..."
        )

        # Log detailed stats in verbose mode
        if should_log_component("verbose"):
            rich_count = sum(
                1 for c in candidates if hasattr(c, "has_rich_metadata") and c.has_rich_metadata()
            )
            append_chatbot_log(f"Metadata: {rich_count}/{len(candidates)} with enrichment")
            if execution_context.tools_used:
                append_chatbot_log(f"Tools: {', '.join(execution_context.tools_used)}")

        # Prepare candidates for LLM (truncate descriptions only)
        prepared = self._prepare_candidates(candidates)

        # Log data transform in debug mode
        if is_debug_mode():
            log_data_transform(
                "curation_prepare",
                [{"item_idx": c.item_idx, "title": c.title} for c in candidates[:3]],
                prepared[:3],
                f"Prepared {len(prepared)} candidates for LLM",
            )

        # Build prompt
        prompt = self._build_curation_prompt(
            query=request.user_text,
            candidates=prepared,
            execution_context=execution_context,
        )

        # Log prompt size in debug mode
        if is_debug_mode():
            append_chatbot_log(f"Prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

        # Single LLM call
        try:
            import time

            start = time.time()

            response = self.llm.invoke([HumanMessage(content=prompt)])

            elapsed_ms = int((time.time() - start) * 1000)
            append_chatbot_log(f"LLM: {elapsed_ms}ms")

            decision = json.loads(response.content)

            # Extract results
            book_ids = decision.get("book_ids", [])
            response_text = decision.get("response_text", "")
            reasoning = decision.get("reasoning", "")

            append_chatbot_log(f"Output: {len(book_ids)} books selected")

            # Log reasoning in verbose mode
            if reasoning and should_log_component("verbose"):
                append_chatbot_log(f"Reasoning: {reasoning[:150]}...")

            # Log full decision in debug mode
            if is_debug_mode():
                log_data_transform(
                    "curation_decision",
                    {"query": request.user_text[:50], "candidates": len(candidates)},
                    {
                        "book_ids": book_ids,
                        "response_preview": response_text[:200],
                        "reasoning": reasoning[:200] if reasoning else None,
                    },
                    "Curation LLM output",
                )

            # Reorder candidates to match LLM's ordering
            ordered_books = self._order_books(candidates, book_ids)

            # Validate inline book references (backend quality control)
            errors, warnings = InlineReferenceParser.validate_references(
                text=response_text, book_recommendations=ordered_books
            )

            # Log validation results
            if errors:
                for error in errors:
                    append_chatbot_log(f"[INLINE REF ERROR] {error}")

            if warnings:
                for warning in warnings:
                    append_chatbot_log(f"[INLINE REF WARNING] {warning}")

            # Log inline reference stats
            inline_ids = InlineReferenceParser.get_inline_book_ids(response_text)
            if inline_ids:
                append_chatbot_log(
                    f"Inline references: {len(set(inline_ids))} unique books "
                    f"mentioned in prose (out of {len(book_ids)} total)"
                )
            else:
                append_chatbot_log("No inline book references found - using card-only display")

            return AgentResponse(
                text=response_text,
                target_category="recsys",
                book_recommendations=ordered_books,
                success=True,
                policy_version="recsys.curation.md",
            )

        except json.JSONDecodeError as e:
            append_chatbot_log(f"Curation JSON parse error: {e}")
            raise
        except Exception as e:
            append_chatbot_log(f"Curation failed: {e}")
            raise

    def _prepare_candidates(self, candidates: List[BookRecommendation]) -> List[Dict]:
        """
        Prepare candidates for LLM.

        Uses to_curation_dict() which now excludes empty fields.
        Only truncates vibe descriptions for token limits.
        """
        prepared = []

        for book in candidates:
            # Use optimized conversion that excludes empty fields
            if hasattr(book, "to_curation_dict"):
                book_dict = book.to_curation_dict()
            else:
                # Fallback: manual extraction (shouldn't happen with current code)
                book_dict = {
                    "item_idx": book.item_idx,
                    "title": book.title or "",
                    "author": book.author or "",
                    "year": book.year or "",
                    "num_ratings": book.num_ratings or 0,
                }
                # Only add non-empty enrichment fields
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

    def _build_curation_prompt(
        self, query: str, candidates: List[Dict], execution_context: ExecutionContext
    ) -> str:
        """Build the curation prompt with execution context."""
        base_prompt = read_prompt("recsys.curation.md")

        # Format execution context
        context = self._format_execution_context(execution_context)

        # Build full prompt
        return f"""{base_prompt}

USER QUERY: {query}

{context}

CANDIDATES ({len(candidates)} books):
{json.dumps(candidates, indent=2, ensure_ascii=False)}

Respond with JSON only.
"""

    def _format_execution_context(self, execution_context: ExecutionContext) -> str:
        """
        Format execution context from planner and retrieval stages.

        Shows the strategy reasoning, which tools were used, and profile data
        to help the curator understand how these candidates were generated.
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

    def _order_books(
        self, candidates: List[BookRecommendation], ordered_ids: List[int]
    ) -> List[BookRecommendation]:
        """
        Reorder candidate list to match LLM's ordering.

        Args:
            candidates: Original candidate list
            ordered_ids: IDs in desired order from LLM

        Returns:
            Ordered list of BookRecommendation objects
        """
        # Build lookup
        id_to_book = {book.item_idx: book for book in candidates}

        # Reorder
        ordered = []
        for book_id in ordered_ids:
            if book_id in id_to_book:
                ordered.append(id_to_book[book_id])

        return ordered
