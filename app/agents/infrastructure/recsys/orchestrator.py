# app/agents/infrastructure/recsys/orchestrator.py
"""
Two-stage recommendation agent: Retrieval → Curation
Orchestrates the pipeline without implementing agent logic itself.
"""
from typing import Optional, Any

from app.agents.domain.entities import (
    AgentRequest,
    AgentResponse,
    AgentConfiguration,
    AgentCapability,
)
from app.agents.domain.interfaces import BaseAgent
from app.agents.logging import append_chatbot_log

from .retrieval_agent import RetrievalAgent
from .curation_agent import CurationAgent


class RecommendationAgent(BaseAgent):
    """
    Orchestrates two-stage recommendation pipeline.
    
    Stage 1 (RetrievalAgent): Gather 60-120 candidate books using tools
    Stage 2 (CurationAgent): Rank, filter, and generate prose
    
    This class handles coordination and fallback logic only.
    """
    
    def __init__(
        self,
        current_user: Any = None,
        db: Any = None,
        user_num_ratings: Optional[int] = None,
        warm_threshold: int = 10,
        allow_profile: bool = False,
    ):
        """
        Initialize orchestrator with context for sub-agents.
        
        Args:
            current_user: Current user object (for internal tools)
            db: Database session (for internal tools)
            user_num_ratings: Number of ratings user has made
            warm_threshold: Minimum ratings for "warm" user status
            allow_profile: Whether user granted profile access consent
        """
        # Build configuration for metadata
        configuration = AgentConfiguration(
            policy_name="recsys.orchestrator",
            capabilities=frozenset([AgentCapability.INTERNAL_TOOLS]),
            llm_tier="large",
            timeout_seconds=90,  # Combined timeout for both stages
            max_iterations=0,  # Orchestrator doesn't iterate
        )
        
        super().__init__(configuration)
        
        # Store context
        self._current_user = current_user
        self._db = db
        self._user_num_ratings = user_num_ratings or 0
        self._warm_threshold = warm_threshold
        self._allow_profile = allow_profile
        
        # Initialize sub-agents
        self.retrieval_agent = RetrievalAgent(
            current_user=current_user,
            db=db,
            user_num_ratings=user_num_ratings or 0,
            allow_profile=allow_profile,
        )
        
        self.curation_agent = CurationAgent()
        
        append_chatbot_log(
            f"Initialized RecommendationAgent orchestrator "
            f"(warm={self._user_num_ratings >= self._warm_threshold}, "
            f"profile={allow_profile})"
        )
    
    def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute two-stage recommendation pipeline.
        
        Args:
            request: User request with query and context
            
        Returns:
            Final agent response with ordered books and prose
        """
        append_chatbot_log(
            f"\n{'='*60}\n"
            f"RECOMMENDATION ORCHESTRATOR START\n"
            f"Query: {request.user_text[:100]}...\n"
            f"{'='*60}"
        )
        
        # ============================================================
        # STAGE 1: RETRIEVAL
        # ============================================================
        try:
            append_chatbot_log("\n>>> STAGE 1: RETRIEVAL <<<")
            retrieval_response = self.retrieval_agent.execute(request)
            
        except Exception as e:
            append_chatbot_log(f"[RETRIEVAL ERROR] {e}")
            return self._retrieval_error_fallback(request, str(e))
        
        # Extract candidates
        candidates = retrieval_response.book_recommendations
        
        # Safely get execution info
        exec_state = retrieval_response.execution_state
        tool_count = len(exec_state.tool_executions) if exec_state else 0
        exec_time = exec_state.execution_time_ms if exec_state else 0
        
        append_chatbot_log(
            f"Retrieval complete: {len(candidates)} candidates, "
            f"{tool_count} tool calls, "
            f"{exec_time}ms"
        )
        
        # Check if we got any candidates
        if not candidates:
            append_chatbot_log("[NO CANDIDATES] Returning empty results response")
            return self._no_candidates_fallback(request)
        
        # Log candidate quality
        rich_metadata_count = sum(
            1 for c in candidates 
            if hasattr(c, 'has_rich_metadata') and c.has_rich_metadata()
        )
        append_chatbot_log(
            f"Candidate metadata: {rich_metadata_count}/{len(candidates)} "
            f"have rich metadata (subjects/tones/description)"
        )
        
        # ============================================================
        # STAGE 2: CURATION
        # ============================================================
        try:
            append_chatbot_log("\n>>> STAGE 2: CURATION <<<")
            
            # Extract retrieval context for curation agent
            retrieval_summary = []
            if retrieval_response.execution_state:
                retrieval_summary = retrieval_response.execution_state.intermediate_outputs.get(
                    "retrieval_summary", []
                )
            
            final_response = self.curation_agent.execute(
                request=request,
                candidates=candidates,
                retrieval_summary=retrieval_summary,
            )
            
        except Exception as e:
            append_chatbot_log(f"[CURATION ERROR] {e}")
            return self._curation_error_fallback(request, candidates, str(e))
        
        # ============================================================
        # VALIDATION
        # ============================================================
        
        # Basic sanity check
        if not final_response.book_recommendations:
            append_chatbot_log(
                "[VALIDATION ERROR] Curation returned no books, "
                "using fallback"
            )
            return self._curation_empty_fallback(request, candidates)
        
        if not final_response.text or len(final_response.text.strip()) < 10:
            append_chatbot_log(
                "[VALIDATION WARNING] Curation returned minimal text, "
                "but proceeding"
            )
        
        # Log final stats
        retrieval_time = retrieval_response.execution_time_ms or 0
        curation_time = final_response.execution_time_ms or 0
        
        append_chatbot_log(
            f"\n{'='*60}\n"
            f"RECOMMENDATION COMPLETE\n"
            f"Retrieval: {len(candidates)} candidates\n"
            f"Curation: {len(final_response.book_recommendations)} final books\n"
            f"Total time: {retrieval_time + curation_time}ms\n"
            f"{'='*60}\n"
        )
        
        return final_response
    
    # ================================================================
    # FALLBACK HANDLERS
    # ================================================================
    
    def _retrieval_error_fallback(
        self, 
        request: AgentRequest, 
        error: str
    ) -> AgentResponse:
        """
        Handle complete retrieval failure.
        
        This means tools didn't work or agent crashed entirely.
        """
        append_chatbot_log(f"Using retrieval error fallback: {error[:100]}")
        
        return AgentResponse(
            text=(
                "I had trouble accessing the book catalog. "
                "Please try again in a moment."
            ),
            target_category="recsys",
            success=False,
            policy_version="recsys.orchestrator",
        )
    
    def _no_candidates_fallback(self, request: AgentRequest) -> AgentResponse:
        """
        Handle case where retrieval found zero books.
        
        This typically means query is too specific or outside catalog scope.
        """
        # Check if query mentions dates/years
        query_lower = request.user_text.lower()
        has_year = any(
            year_str in query_lower 
            for year_str in ['2004', '2005', '2010', '2015', '2020', '2024', '2025']
        )
        
        if has_year or 'recent' in query_lower or 'new' in query_lower:
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
            policy_version="recsys.orchestrator",
        )
    
    def _curation_error_fallback(
        self,
        request: AgentRequest,
        candidates: list,
        error: str,
    ) -> AgentResponse:
        """
        Handle curation failure with valid candidates.
        
        Return first 10 candidates with generic prose.
        """
        append_chatbot_log(f"Using curation error fallback: {error[:100]}")
        
        # Take first 10 candidates
        fallback_books = candidates[:10]
        
        # Generate simple prose
        if self._user_num_ratings >= self._warm_threshold:
            text = (
                "Here are some books from our catalog that might interest you "
                "based on your reading history."
            )
        else:
            text = (
                "Here are some popular books from our catalog. "
                "Rate a few books to get personalized recommendations!"
            )
        
        return AgentResponse(
            text=text,
            target_category="recsys",
            book_recommendations=fallback_books,
            success=True,  # Partial success
            policy_version="recsys.orchestrator",
        )
    
    def _curation_empty_fallback(
        self,
        request: AgentRequest,
        candidates: list,
    ) -> AgentResponse:
        """
        Handle case where curation returned no books despite having candidates.
        
        This shouldn't happen often - means curation filtered everything out.
        """
        append_chatbot_log(
            f"Curation filtered out all {len(candidates)} candidates, "
            f"using first 8"
        )
        
        # Just take first 8 candidates
        fallback_books = candidates[:8]
        
        text = (
            "Here are some books from our catalog. "
            "Let me know if you'd like different recommendations!"
        )
        
        return AgentResponse(
            text=text,
            target_category="recsys",
            book_recommendations=fallback_books,
            success=True,  # Partial success
            policy_version="recsys.orchestrator",
        )