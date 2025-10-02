# app/agents/infrastructure/recommendation_agent.py
"""
Recommendation agent with access to internal book recommendation tools.
"""
from app.agents.domain.entities import AgentConfiguration, AgentCapability
from app.agents.prompts.loader import read_prompt
from app.agents.tools.registry import InternalToolGates
from ..base_langgraph_agent import BaseLangGraphAgent


class RecommendationAgent(BaseLangGraphAgent):
    """
    Agent specialized in book recommendations.
    
    Has access to:
    - ALS collaborative filtering
    - Semantic search
    - Subject-based search
    - User profile data (with permission)
    """
    
    def __init__(
        self,
        current_user=None,
        db=None,
        user_num_ratings=None,
        warm_threshold=10,
        allow_profile=False
    ):
        # Build configuration
        configuration = AgentConfiguration(
            policy_name="recsys.system.md",
            capabilities=frozenset([AgentCapability.INTERNAL_TOOLS]),
            allowed_tools=frozenset([
                "als_recs",
                "subject_hybrid_pool",
                "subject_id_search",
                "book_semantic_search",
                "return_book_ids",
                "user_profile",
                "recent_interactions"
            ]),
            llm_tier="large",
            timeout_seconds=60,
            max_iterations=7  # Allow multi-step reasoning
        )
        
        # Store context for tool registry
        self._ctx_user = current_user
        self._ctx_db = db
        self._user_num_ratings = user_num_ratings or 0
        self._allow_profile = allow_profile
        
        super().__init__(configuration, ctx_user=current_user, ctx_db=db)
    
    def _create_tool_registry(self, ctx_user, ctx_db):
        """Override to set proper gates for recommendation tools."""
        from app.agents.tools.registry import ToolRegistry
        
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
        """Load recommendation-specific system prompt."""
        persona = read_prompt("persona.system.md")
        policy = read_prompt("recsys.system.md")
        return f"{persona}\n\n{policy}".strip()
    
    def _get_target_category(self) -> str:
        return "recsys"