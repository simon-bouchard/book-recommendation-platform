# app/agents/domain/services.py
"""
Domain services that implement business logic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .entities import AgentExecutionState, BookRecommendation


@dataclass
class BookExtractionResult:
    """Result of book ID extraction process."""

    book_ids: List[int]
    extraction_method: str
    raw_data: Any = None


class StandardResultProcessor:
    """Standard implementation of result processing logic."""

    def extract_book_recommendations(self, state: AgentExecutionState) -> List[BookRecommendation]:
        """
        Extract book recommendations with full metadata from execution state.

        Priority order:
        1. book_objects (full metadata from tools) - PREFERRED
        2. return_book_ids tool result (just IDs)
        3. Other extraction methods (fallback)
        """
        # Check for full book objects first (new method)
        book_objects = state.intermediate_outputs.get("book_objects", [])
        if book_objects:
            return self._build_recommendations_from_objects(book_objects)

        # Fall back to old ID-only extraction
        extraction_result = self._extract_book_ids_with_confidence(state)

        recommendations = []
        for item_idx in extraction_result.book_ids:
            recommendations.append(
                BookRecommendation(
                    item_idx=item_idx,
                    recommendation_reason=f"Extracted via {extraction_result.extraction_method}",
                )
            )

        return recommendations

    def _build_recommendations_from_objects(
        self, book_objects: List[Dict[str, Any]]
    ) -> List[BookRecommendation]:
        """
        Build BookRecommendation objects from tool result dictionaries.

        Args:
            book_objects: List of book dicts from tool results

        Returns:
            List of BookRecommendation objects with full metadata
        """
        recommendations = []

        for obj in book_objects:
            if not isinstance(obj, dict):
                continue

            item_idx = obj.get("item_idx")
            if item_idx is None:
                continue

            # Extract all available fields
            rec = BookRecommendation(
                item_idx=int(item_idx),
                title=obj.get("title"),
                author=obj.get("author"),
                year=self._safe_int(obj.get("year")),
                cover_id=obj.get("cover_id"),
                num_ratings=self._safe_int(obj.get("num_ratings")),
            )

            # Add extended metadata if available (not in base BookRecommendation)
            # Store in a dict for passing to curation
            if "subjects" in obj or "tones" in obj or "vibe" in obj or "genre" in obj:
                # We need to extend BookRecommendation or store metadata separately
                # For now, let's add these as attributes dynamically
                rec.subjects = obj.get("subjects", [])
                rec.tones = obj.get("tones", [])
                rec.vibe = obj.get("vibe")
                rec.genre = obj.get("genre")

            recommendations.append(rec)

        return recommendations

    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int."""
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def _extract_book_ids_with_confidence(self, state: AgentExecutionState) -> BookExtractionResult:
        """
        Extract book IDs using multiple strategies, returning the first successful result.
        Strategies are ordered by reliability.
        """
        strategies = [
            ("return_book_ids_tool", self._extract_from_return_book_ids_tool),
            ("intermediate_outputs", self._extract_from_intermediate_outputs),
            ("tool_outputs", self._extract_from_tool_outputs),
            ("reasoning_steps", self._extract_from_reasoning_steps),
        ]

        for method_name, strategy in strategies:
            try:
                result = strategy(state)
                if result.book_ids:  # Found books - use this method
                    return result
            except Exception:
                # Log but don't fail - try next strategy
                continue

        # No books found by any method
        return BookExtractionResult(book_ids=[], extraction_method="none")

    def _extract_from_return_book_ids_tool(
        self, state: AgentExecutionState
    ) -> BookExtractionResult:
        """Extract from return_book_ids tool calls (most reliable method)."""
        for execution in reversed(state.tool_executions):  # Check most recent first
            if execution.tool_name.lower() == "return_book_ids" and execution.succeeded:
                # Try parsing result first
                if isinstance(execution.result, str):
                    try:
                        data = json.loads(execution.result)
                        if "book_ids" in data:
                            ids = [int(x) for x in data["book_ids"] if str(x).strip()]
                            return BookExtractionResult(
                                book_ids=ids,
                                extraction_method="return_book_ids_result",
                                raw_data=data,
                            )
                    except (json.JSONDecodeError, ValueError):
                        pass

                # Try parsing arguments as fallback
                if isinstance(execution.arguments, dict):
                    if "book_ids" in execution.arguments:
                        try:
                            ids = [
                                int(x) for x in execution.arguments["book_ids"] if str(x).strip()
                            ]
                            return BookExtractionResult(
                                book_ids=ids,
                                extraction_method="return_book_ids_args",
                                raw_data=execution.arguments,
                            )
                        except (ValueError, TypeError):
                            pass

        return BookExtractionResult(book_ids=[], extraction_method="return_book_ids_not_found")

    def _extract_from_tool_outputs(self, state: AgentExecutionState) -> BookExtractionResult:
        """Extract book IDs from other tool outputs (fallback method)."""
        book_ids = []

        for execution in state.tool_executions:
            if not execution.succeeded:
                continue

            # Look for book IDs in results
            result_text = str(execution.result or "")
            ids_in_result = self._find_book_ids_in_text(result_text)
            book_ids.extend(ids_in_result)

        # Remove duplicates while preserving order
        unique_ids = list(dict.fromkeys(book_ids))

        return BookExtractionResult(book_ids=unique_ids, extraction_method="tool_outputs")

    def _extract_from_reasoning_steps(self, state: AgentExecutionState) -> BookExtractionResult:
        """Extract book IDs mentioned in reasoning steps (least reliable fallback)."""
        book_ids = []

        for step in state.reasoning_steps:
            ids_in_step = self._find_book_ids_in_text(step)
            book_ids.extend(ids_in_step)

        unique_ids = list(dict.fromkeys(book_ids))

        return BookExtractionResult(book_ids=unique_ids, extraction_method="reasoning_steps")

    def _extract_from_intermediate_outputs(
        self, state: AgentExecutionState
    ) -> BookExtractionResult:
        """Extract from intermediate outputs stored in state (fallback method)."""
        book_ids = []

        # Check for explicitly stored book recommendations
        if "book_recommendations" in state.intermediate_outputs:
            recs = state.intermediate_outputs["book_recommendations"]
            if isinstance(recs, list):
                for rec in recs:
                    if isinstance(rec, dict):
                        # Try both item_idx (standard) and book_id (fallback)
                        id_value = rec.get("item_idx") or rec.get("book_id")
                        if id_value is not None:
                            try:
                                book_ids.append(int(id_value))
                            except (ValueError, TypeError):
                                pass

        # Check other intermediate data
        for key, value in state.intermediate_outputs.items():
            if "book" in key.lower() or "id" in key.lower():
                text_value = str(value)
                ids_in_value = self._find_book_ids_in_text(text_value)
                book_ids.extend(ids_in_value)

        unique_ids = list(dict.fromkeys(book_ids))

        return BookExtractionResult(book_ids=unique_ids, extraction_method="intermediate_outputs")

    def _find_book_ids_in_text(self, text: str) -> List[int]:
        """Find potential book IDs in text using patterns."""
        if not isinstance(text, str):
            return []

        book_ids = []

        # Look for patterns like "book_id: 123", "id: 456", etc.
        patterns = [
            r'book_id[\'"]?\s*:\s*(\d+)',
            r'item_idx[\'"]?\s*:\s*(\d+)',
            r'\bid[\'"]?\s*:\s*(\d+)',
            r"\[(\d+(?:\s*,\s*\d+)*)\]",  # [123, 456, 789]
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Handle comma-separated list in brackets
                    if "," in match.group(1):
                        ids = [int(x.strip()) for x in match.group(1).split(",")]
                        book_ids.extend(ids)
                    else:
                        book_ids.append(int(match.group(1)))
                except (ValueError, IndexError):
                    continue

        return book_ids

    def format_response_text(self, state: AgentExecutionState) -> str:
        """Format final response text from execution state."""
        # Look for final answer in intermediate outputs first
        if "final_answer" in state.intermediate_outputs:
            text = str(state.intermediate_outputs["final_answer"]).strip()
            return self._normalize_final_answer(text)

        # Fallback to last reasoning step if available
        if state.reasoning_steps:
            last_step = state.reasoning_steps[-1].strip()
            return self._normalize_final_answer(last_step)

        # Final fallback
        return "I apologize, but I couldn't generate a proper response."

    def _normalize_final_answer(self, text: str) -> str:
        """Normalize final answer text by removing common prefixes."""
        text = text.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "final answer:",
            "answer:",
            "response:",
            "result:",
        ]

        text_lower = text.lower()
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                text = text[len(prefix) :].strip()
                break

        return text

    def extract_citations(self, state: AgentExecutionState) -> List[Dict[str, Any]]:
        """Extract citations from execution state."""
        citations = []

        # Check intermediate outputs for citations
        if "citations" in state.intermediate_outputs:
            stored_citations = state.intermediate_outputs["citations"]
            if isinstance(stored_citations, list):
                citations.extend(stored_citations)

        # Extract citations from web search tools
        for execution in state.tool_executions:
            if execution.tool_name.lower() in ["web_search", "search_web"] and execution.succeeded:
                citation = {
                    "source": "web",
                    "ref": f"Search: {execution.arguments.get('query', 'Unknown')}",
                    "meta": {"tool": execution.tool_name, "timestamp": execution.timestamp},
                }
                citations.append(citation)

        return citations
