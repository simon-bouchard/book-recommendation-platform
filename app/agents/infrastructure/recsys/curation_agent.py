# app/agents/infrastructure/recsys/curation_agent.py
# Currently broken!

"""
Curation agent for ranking and selecting final book recommendations.
Stage 2 of two-stage recommendation pipeline.
"""
import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage

from app.agents.domain.entities import (
    AgentRequest,
    AgentResponse,
    BookRecommendation,
    AgentExecutionState,
)
from app.agents.settings import get_llm
from app.agents.prompts.loader import read_prompt
from app.agents.logging import append_chatbot_log


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
        retrieval_summary: Optional[List[Dict]] = None
    ) -> AgentResponse:
        """
        Curate candidates and generate response.
        
        Args:
            request: Original user request
            candidates: Books from retrieval stage (unfiltered)
            retrieval_summary: Context about which tools were called
            
        Returns:
            Final agent response with ordered books and prose
        """
        append_chatbot_log(f"=== CURATION START ===")
        append_chatbot_log(f"Candidates: {len(candidates)}, Query: {request.user_text[:80]}")
        
        # Prepare candidates for LLM (truncate descriptions only)
        prepared = self._prepare_candidates(candidates)
        
        # Build prompt
        prompt = self._build_curation_prompt(
            query=request.user_text,
            candidates=prepared,
            retrieval_summary=retrieval_summary or []
        )
        
        # Single LLM call
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            decision = json.loads(response.content)
            
            # Extract results
            book_ids = decision.get("book_ids", [])
            response_text = decision.get("response_text", "")
            reasoning = decision.get("reasoning", "")
            
            if reasoning:
                append_chatbot_log(f"Curation reasoning: {reasoning[:200]}")
            
            append_chatbot_log(f"Selected {len(book_ids)} books")
            
            # Reorder candidates to match LLM's ordering
            ordered_books = self._order_books(candidates, book_ids)
            
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
        Prepare candidates for LLM with minimal processing.
        Only truncate descriptions for token limits.
        """
        prepared = []
        
        for book in candidates:
            # Check if we have rich metadata
            if hasattr(book, 'to_curation_dict'):
                # Use the built-in conversion method
                book_dict = book.to_curation_dict()
            else:
                # Fallback: manual extraction
                book_dict = {
                    "item_idx": book.item_idx,
                    "title": book.title or "",
                    "author": book.author or "",
                    "year": book.year or "",
                    "subjects": getattr(book, 'subjects', []) or [],
                    "tones": getattr(book, 'tones', []) or [],
                    "genre": getattr(book, 'genre', "") or "",
                    "description": getattr(book, 'description', "") or "",
                    "score": book.recommendation_score,
                }
            
            # Truncate description if too long (token limit)
            description = book_dict.get("description", "")
            if description and len(description) > 200:
                book_dict["description"] = description[:200] + "..."
            
            prepared.append(book_dict)
        
        return prepared
    
    def _build_curation_prompt(
        self,
        query: str,
        candidates: List[Dict],
        retrieval_summary: List[Dict]
    ) -> str:
        """Build the curation prompt."""
        base_prompt = read_prompt("recsys.curation.md")
        
        # Format retrieval context
        context = self._format_retrieval_context(retrieval_summary)
        
        # Build full prompt
        return f"""{base_prompt}

USER QUERY: {query}

{context}

CANDIDATES ({len(candidates)} books):
{json.dumps(candidates, indent=2, ensure_ascii=False)}

Respond with JSON only.
"""
    
    def _format_retrieval_context(self, retrieval_summary: List[Dict]) -> str:
        """Format tool execution history."""
        if not retrieval_summary:
            return "RETRIEVAL HISTORY: No context available"
        
        lines = ["RETRIEVAL HISTORY:"]
        for entry in retrieval_summary:
            tool = entry.get("tool", "unknown")
            count = entry.get("book_count", 0)
            order = entry.get("order", 0)
            args = entry.get("arguments", {})
            
            top_k = args.get("top_k", "")
            lines.append(f"{order}. {tool} (top_k={top_k}) → {count} books")
        
        if len(retrieval_summary) > 1:
            lines.append(
                "\nNote: Later tool calls may be refinements. "
                "Consider recency when evaluating candidates."
            )
        
        return "\n".join(lines)
    
    def _order_books(
        self,
        candidates: List[BookRecommendation],
        ordered_ids: List[int]
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