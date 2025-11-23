
"""
PlannerAgent for recommendation system query analysis and strategy planning.
First stage of the multi-agent pipeline: analyzes queries and determines retrieval strategy.
"""
import json
from typing import Optional

from app.agents.domain.entities import AgentConfiguration, AgentCapability, AgentRequest
from app.agents.domain.recsys_schemas import PlannerInput, PlannerStrategy
from app.agents.prompts.loader import read_prompt
from app.agents.tools.registry import InternalToolGates, ToolRegistry
from app.agents.logging import append_chatbot_log
from ..base_langgraph_agent import BaseLangGraphAgent


class PlannerAgent(BaseLangGraphAgent):
    """
    Query analysis and strategy planning agent (ReAct-based).
    
    Analyzes user queries to determine the best retrieval strategy.
    Classifies query type (vague/descriptive/genre/complex) and recommends
    which retrieval tools to use with fallback options.
    
    May call profile tools (user_profile, recent_interactions) if allowed
    to gather context for cold users with vague queries.
    
    Uses medium-tier LLM with tight iteration limits (max 3):
    - Iteration 1: Call user_profile (if needed)
    - Iteration 2: Call recent_interactions (if needed)  
    - Iteration 3: Return strategy as JSON
    
    Not a full ReAct exploration loop - focused strategy determination.
    """
    
    def __init__(
        self,
        current_user=None,
        db=None,
        user_num_ratings: int = 0,
        has_als_recs_available: bool = False,
        allow_profile: bool = False,
    ):
        """
        Initialize PlannerAgent.
        
        Args:
            current_user: Current user object (for profile tools)
            db: Database session (for profile tools)
            user_num_ratings: Number of ratings user has (for context)
            has_als_recs_available: Whether ALS collaborative filtering is available
            allow_profile: Whether agent can access user profile tools
        """
        configuration = AgentConfiguration(
            policy_name="recsys.planner.md",
            capabilities=frozenset([AgentCapability.INTERNAL_TOOLS]),
            allowed_tools=frozenset(["user_profile", "recent_interactions"]),
            llm_tier="medium",  # 70B - good balance for query classification
            timeout_seconds=30,
            max_iterations=3  # user_profile call + recent_interactions call + final answer
        )
        
        self._ctx_user = current_user
        self._ctx_db = db
        self._user_num_ratings = user_num_ratings
        self._has_als_recs_available = has_als_recs_available
        self._allow_profile = allow_profile
        
        super().__init__(configuration, ctx_user=current_user, ctx_db=db)
    
    def _create_tool_registry(self, ctx_user, ctx_db):
        """
        Create registry with ONLY context tools (user_profile, recent_interactions).
        
        Retrieval tools are NOT available to planner - only used by RetrievalAgent.
        """
        gates = InternalToolGates(
            user_num_ratings=self._user_num_ratings,
            warm_threshold=10,
            profile_allowed=self._allow_profile
        )
        
        return ToolRegistry.for_context(
            gates=gates,
            ctx_user=ctx_user,
            ctx_db=ctx_db,
        )
    
    def _get_system_prompt(self) -> str:
        """Load planner-specific system prompt with strategy logic."""
        return read_prompt("recsys.planner.md")
    
    def _get_target_category(self) -> str:
        """Target category for agent adapter."""
        return "recsys_planner"

    def _build_current_situation(self, state) -> str:
        """
        Build current query context for the planner.
        
        Shows only the immediate context (query + available tools),
        not the strategic framework (which is in system prompt).
        """
        parts = [
            "CURRENT QUERY CONTEXT:",
            "",
            f"Query: {state.input_text}",
            "",
            f"User Context:",
            f"- ALS collaborative filtering available: {self._has_als_recs_available}",
            f"- Profile access allowed: {self._allow_profile}",
            f"- User has {self._user_num_ratings} ratings",
            "",
            "Your Decision Process:",
            "1. If vague query AND profile allowed → call user_profile first",
            "2. Analyze query type and user context",
            "3. Decide on 1-2 primary retrieval tools",
            "4. Decide on 1-2 fallback tools",
            "5. Return your final strategy JSON (see Output Format in system prompt)",
            "",
            "IMPORTANT: You can call profile tools (user_profile, recent_interactions)",
            "but NEVER call retrieval tools directly. Only RECOMMEND which should be used.",
        ]
        
        return "\n".join(parts)
    
    def execute(self, planner_input: PlannerInput) -> PlannerStrategy:
        """
        Analyze query and determine retrieval strategy.
        
        Process:
        1. Classifies query type (vague, descriptive, genre, complex)
        2. May call profile tools if allowed and needed
        3. Recommends 1-2 primary retrieval tools
        4. Recommends 1-2 fallback tools
        5. Provides reasoning for strategy
        6. Detects negative constraints for logging
        
        Args:
            planner_input: Query and context about available tools
            
        Returns:
            PlannerStrategy with tool recommendations and reasoning
            
        Raises:
            ValueError: If JSON parsing fails or required fields missing
            RuntimeError: If agent execution fails
        """
        append_chatbot_log("\n=== PLANNER AGENT ===")
        append_chatbot_log(f"Query: {planner_input.query[:100]}...")
        append_chatbot_log(f"Available tools: {', '.join(planner_input.available_retrieval_tools)}")
        
        # Build the user message with query and available tools
        context_lines = [
            f"Available retrieval tools to recommend:",
        ]
        
        for tool_name in planner_input.available_retrieval_tools:
            context_lines.append(f"- {tool_name}")
        
        context_lines.append("")
        context_lines.append(f"User Query: {planner_input.query}")
        
        user_message = "\n".join(context_lines)
        
        # Create agent request
        request = AgentRequest(
            user_text=user_message,
            conversation_history=[],
        )
        
        # Execute agent (may call profile tools)
        try:
            response = super().execute(request)
        except Exception as e:
            append_chatbot_log(f"Planner execution failed: {e}")
            raise RuntimeError(f"PlannerAgent execution failed: {e}") from e
        
        # Parse JSON response into PlannerStrategy
        try:
            strategy_data = self._parse_strategy_response(response.text)
            append_chatbot_log(f"Strategy: {strategy_data.reasoning[:150]}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            append_chatbot_log(f"Failed to parse strategy: {e}")
            raise ValueError(f"Failed to parse planner strategy: {e}") from e
        
        return strategy_data
    
    def _parse_strategy_response(self, response_text: str) -> PlannerStrategy:
        """
        Parse LLM response into PlannerStrategy.
        
        Handles common issues:
        - JSON wrapped in markdown code blocks
        - Extra whitespace
        - Missing optional fields
        
        Args:
            response_text: Raw LLM response (should be JSON)
            
        Returns:
            Parsed PlannerStrategy object
            
        Raises:
            json.JSONDecodeError: If response is not valid JSON
            KeyError: If required fields are missing
        """
        # Strip markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Parse JSON
        data = json.loads(text)
        
        # Validate required fields
        required_fields = ["recommended_tools", "fallback_tools", "reasoning"]
        for field in required_fields:
            if field not in data:
                raise KeyError(f"Missing required field: {field}")
        
        # Build PlannerStrategy with optional fields
        return PlannerStrategy(
            recommended_tools=data["recommended_tools"],
            fallback_tools=data["fallback_tools"],
            reasoning=data["reasoning"],
            profile_data=data.get("profile_data"),
            negative_constraints=data.get("negative_constraints"),
        )
