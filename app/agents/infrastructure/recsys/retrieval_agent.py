# app/agents/infrastructure/recsys/retrieval_agent.py
"""
Retrieval agent for gathering book candidates using internal tools.
Stage 1 of two-stage recommendation pipeline.
"""
from typing import Dict, Any

from app.agents.domain.entities import AgentConfiguration, AgentCapability, AgentExecutionState
from app.agents.prompts.loader import read_prompt
from app.agents.infrastructure.base_langgraph_agent import BaseLangGraphAgent
from app.agents.logging import append_chatbot_log


class RetrievalAgent(BaseLangGraphAgent):
    """
    Retrieves candidate books using internal recommendation tools.
    
    Decides which tools to call, when to stop, and accumulates all results.
    Does not filter, rank, or generate prose - just gathers candidates.
    """
    
    def __init__(self, current_user, db, user_num_ratings, allow_profile):
        # Build configuration
        configuration = AgentConfiguration(
            policy_name="recsys.retrieval.md",
            capabilities=frozenset([AgentCapability.INTERNAL_TOOLS]),
            allowed_tools=frozenset([
                "als_recs",
                "subject_hybrid_pool",
                "subject_id_search",
                "book_semantic_search",
                "user_profile",
                "recent_interactions"
            ]),
            llm_tier="large",  # Need good reasoning for tool selection
            timeout_seconds=60,
            max_iterations=7
        )
        
        self._user_num_ratings = user_num_ratings or 0
        self._allow_profile = allow_profile
        
        super().__init__(configuration, ctx_user=current_user, ctx_db=db)
    
    def _create_tool_registry(self, ctx_user, ctx_db):
        """Override to set proper gates."""
        from app.agents.tools.registry import ToolRegistry, InternalToolGates
        
        gates = InternalToolGates(
            user_num_ratings=self._user_num_ratings,
            warm_threshold=10,
            profile_allowed=self._allow_profile
        )
        
        return ToolRegistry(
            web=False,
            docs=False,
            internal=True,
            gates=gates,
            ctx_user=ctx_user,
            ctx_db=ctx_db,
        )
    
    def _get_system_prompt(self) -> str:
        """Load retrieval-specific prompt."""
        return read_prompt("recsys.retrieval.md")
    
    def _get_target_category(self) -> str:
        return "recsys"
    
    def _finalize_node(self, state):
        """
        Override to capture retrieval summary for curation stage.
        No prose generation - just package candidates and context.
        """
        append_chatbot_log("=== RETRIEVAL FINALIZE ===")
        
        # Build summary of what tools were called
        tool_summary = []
        for exec in state.tool_executions:
            if exec.succeeded and self.tool_executor.is_book_recommendation_tool(exec.tool_name):
                book_ids = self.tool_executor.extract_book_ids_from_result(exec.result)
                tool_summary.append({
                    "tool": exec.tool_name,
                    "arguments": exec.arguments,
                    "book_count": len(book_ids),
                    "order": len(tool_summary) + 1
                })
        
        state.intermediate_outputs["retrieval_summary"] = tool_summary
        
        # Mark as retrieval-only (no prose needed)
        state.intermediate_outputs["retrieval_only"] = True
        
        state.mark_completed()
        
        # Count book_objects (the rich metadata), not just IDs
        total_books = len(state.intermediate_outputs.get("book_objects", []))
        append_chatbot_log(
            f"Retrieval complete: {total_books} candidates with metadata from "
            f"{len(tool_summary)} tool calls"
        )
        
        return state

        def _process_decision(self, state: AgentExecutionState, decision: Dict[str, Any]) -> None:
            """
            Override to handle retrieval-specific completion.
            
            If LLM provides "answer" action, ignore the text and just finalize.
            This handles the case where examples show "answer" but we don't want prose.
            """
            action = decision.get("action")
            
            if action == "tool_call":
                # Standard tool call - use parent implementation
                super()._process_decision(state, decision)
                
            elif action == "answer":
                # Retrieval is done - mark for finalization WITHOUT storing text
                state.intermediate_outputs["next_action"] = {"type": "finalize"}
                
                # Log reasoning if present
                if "reasoning" in decision:
                    append_chatbot_log(f"Stopping reasoning: {decision['reasoning']}")
                
                # DO NOT store final_answer - curation will handle prose
                
            else:
                # Fallback to parent
                super()._process_decision(state, decision)
        
        return state