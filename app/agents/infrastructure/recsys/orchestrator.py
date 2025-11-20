# app/agents/infrastructure/recsys/orchestrator.py
"""
Two-stage recommendation agent: Retrieval → Curation
Orchestrates the pipeline without implementing agent logic itself.
"""
from typing import Optional, Any
import time

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
        append_chatbot_log("\n=== STAGE 1: RETRIEVAL ===")
        retrieval_start = time.time()
        
        try:
            retrieval_response = self.retrieval_agent.execute(request)
            
        except Exception as e:
            retrieval_time = int((time.time() - retrieval_start) * 1000)
            append_chatbot_log(f"[RETRIEVAL ERROR after {retrieval_time}ms] {e}")
            return self._retrieval_error_fallback(request, str(e))
        
        # Extract candidates
        candidates = retrieval_response.book_recommendations
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        
        # Safely get execution info
        exec_state = retrieval_response.execution_state
        tool_count = len(exec_state.tool_executions) if exec_state else 0
        
        append_chatbot_log(
            f"Retrieval: {len(candidates)} candidates in {retrieval_time}ms "
            f"({tool_count} tool calls)"
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
            f"Metadata: {rich_metadata_count}/{len(candidates)} with enrichment"
        )
        
        # ============================================================
        # STAGE 2: CURATION
        # ============================================================
        append_chatbot_log("\n=== STAGE 2: CURATION ===")
        curation_start = time.time()
        
        try:
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
            curation_time = int((time.time() - curation_start) * 1000)
            append_chatbot_log(f"[CURATION ERROR after {curation_time}ms] {e}")
            return self._curation_error_fallback(request, candidates, str(e))
        
        curation_time = int((time.time() - curation_start) * 1000)
        
        # ============================================================
        # VALIDATION
        # ============================================================
        
        # Basic sanity check
        if not final_response.book_recommendations:
            append_chatbot_log(
                "[VALIDATION ERROR] Curation returned no books, using fallback"
            )
            return self._curation_empty_fallback(request, candidates)
        
        if not final_response.text or len(final_response.text.strip()) < 10:
            append_chatbot_log(
                "[VALIDATION WARNING] Curation returned minimal text, but proceeding"
            )
        
        # Log final stats
        total_time = retrieval_time + curation_time
        
        append_chatbot_log(
            f"Curation: {len(final_response.book_recommendations)} books in {curation_time}ms"
        )
        append_chatbot_log(
            f"\n{'='*60}\n"
            f"RECOMMENDATION COMPLETE\n"
            f"Total: {total_time}ms (retrieval: {retrieval_time}ms, curation: {curation_time}ms)\n"
            f"Books: {len(candidates)} → {len(final_response.book_recommendations)}\n"
            f"{'='*60}\n"
        )
        
        return final_response
    
    # ================================================================
    # FALLBACK HANDLERS
    # ================================================================
    
    def _retrieval_error_fallback(
        self, request: AgentRequest, error_msg: str
    ) -> AgentResponse:
        """Fallback when retrieval stage fails."""
        return AgentResponse(
            text="I'm having trouble finding book recommendations right now. Please try again.",
            target_category="recsys",
            success=False,
            policy_version="recsys.orchestrator.retrieval_error",
        )
    
    def _no_candidates_fallback(self, request: AgentRequest) -> AgentResponse:
        """Fallback when retrieval returns no candidates."""
        return AgentResponse(
            text="I couldn't find any books matching your request. Could you try rephrasing or being more specific?",
            target_category="recsys",
            success=True,
            policy_version="recsys.orchestrator.no_candidates",
        )
    
    def _curation_error_fallback(
        self, 
        request: AgentRequest, 
        candidates: list, 
        error_msg: str
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
    
    def _curation_empty_fallback(
        self, request: AgentRequest, candidates: list
    ) -> AgentResponse:
        """Fallback when curation returns empty results - return top candidates."""
        top_books = candidates[:10]
        
        return AgentResponse(
            text="Here are some book recommendations for you:",
            target_category="recsys",
            success=True,
            book_recommendations=top_books,
            policy_version="recsys.orchestrator.curation_empty",
        )
