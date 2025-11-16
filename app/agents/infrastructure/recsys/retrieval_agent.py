# app/agents/infrastructure/recsys/retrieval_agent.py
"""
RetrievalAgent - executes retrieval strategy and gathers 60-120 book candidates.
Second stage of multi-agent recommendation pipeline.
"""
import json
from typing import Optional

from app.agents.domain.entities import (
    AgentConfiguration, 
    AgentCapability,
    AgentRequest,
    AgentResponse,
    AgentExecutionState,
)
from app.agents.domain.recsys_schemas import (
    RetrievalInput,
    RetrievalOutput,
    ExecutionContext,
)
from app.agents.prompts.loader import read_prompt
from app.agents.tools.registry import InternalToolGates, ToolRegistry
from app.agents.logging import append_chatbot_log
from ..base_langgraph_agent import BaseLangGraphAgent


class RetrievalAgent(BaseLangGraphAgent):
    """
    Strategy execution and candidate gathering agent.
    
    Receives strategy from PlannerAgent and executes it adaptively:
    - Calls recommended tools with appropriate parameters
    - Accumulates book candidates with metadata
    - Applies fallback logic when primary tools underperform
    - Stops when 60-120 candidates gathered or all tools exhausted
    
    Uses large-tier LLM (405B) with max 6 iterations for adaptive execution.
    This is a TACTICAL agent - follows strategy but adapts to execution reality.
    """
    
    def __init__(
        self,
        current_user=None,
        db=None,
        user_num_ratings: int = 0,
        has_als_recs_available: bool = False,
    ):
        """
        Initialize RetrievalAgent.
        
        Args:
            current_user: Current user object (for retrieval tools)
            db: Database session (for retrieval tools)
            user_num_ratings: Number of ratings user has
            has_als_recs_available: Whether ALS collaborative filtering is available
        """
        # Build configuration
        configuration = AgentConfiguration(
            policy_name="recsys.retrieval.md",
            capabilities=frozenset([AgentCapability.INTERNAL_TOOLS]),
            allowed_tools=frozenset([
                "als_recs",
                "book_semantic_search",
                "subject_hybrid_pool",
                "subject_id_search",
                "popular_books",
            ]),
            llm_tier="large",  # Need strong reasoning for adaptive execution
            timeout_seconds=60,
            max_iterations=6  # Enough for primary + fallback + refinement
        )
        
        # Store context
        self._ctx_user = current_user
        self._ctx_db = db
        self._user_num_ratings = user_num_ratings
        self._has_als_recs_available = has_als_recs_available
        
        super().__init__(configuration, ctx_user=current_user, ctx_db=db)
    
    def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry:
        """
        Create registry with ONLY retrieval tools.
        
        Context tools (user_profile, recent_interactions) are NOT available here -
        those are for PlannerAgent only. Uses for_retrieval() factory.
        """
        gates = InternalToolGates(
            user_num_ratings=self._user_num_ratings,
            warm_threshold=10,
            profile_allowed=False,  # No profile access - Planner handles that
        )
        
        return ToolRegistry.for_retrieval(
            gates=gates,
            ctx_user=ctx_user,
            ctx_db=ctx_db,
        )
    
    def _get_system_prompt(self) -> str:
        """Load candidate generator system prompt."""
        return read_prompt("recsys.retrieval.md")
    
    def _get_target_category(self) -> str:
        """Target category for agent adapter."""
        return "recsys_generator"
    
    def _build_current_situation(self, state: AgentExecutionState) -> str:
        """
        Build tactical situation description with strategy and execution feedback.
        
        Shows:
        - Original query
        - Strategy from Planner
        - Execution history (what tools called, how many books)
        - Current candidate count
        - Adaptive guidance (continue / try fallback / stop)
        """
        # Extract strategy and execution state
        strategy = state.intermediate_outputs.get("strategy")
        profile_data = state.intermediate_outputs.get("profile_data")
        candidates = state.intermediate_outputs.get("book_objects", [])
        tool_execs = state.tool_executions
        
        # Build situation description
        situation_parts = [
            "# Current Retrieval Status",
            "",
            f"**Query**: {state.input_text}",
            "",
        ]
        
        # Show strategy from Planner
        if strategy:
            situation_parts.extend([
                "**Strategy from Planner**:",
                f"- Recommended tools: {', '.join(strategy.recommended_tools)}",
                f"- Fallback tools: {', '.join(strategy.fallback_tools)}",
                f"- Reasoning: {strategy.reasoning}",
                "",
            ])
            
            # Show profile data if available
            if profile_data:
                situation_parts.append(f"- Profile data available: {json.dumps(profile_data)}")
                situation_parts.append("")
            
            # Show negative constraints if detected
            if strategy.negative_constraints:
                situation_parts.extend([
                    f"- Negative constraints detected: {', '.join(strategy.negative_constraints)}",
                    "  (For semantic search: use positive terms only. Curator will filter.)",
                    "",
                ])
        
        # Show execution history
        if tool_execs:
            situation_parts.append("**Tools Executed So Far**:")
            for i, exec_record in enumerate(tool_execs, 1):
                book_count = len([b for b in candidates if b.get('tool_source') == exec_record.tool_name]) if candidates else 0
                status = "✓" if exec_record.succeeded else "✗"
                situation_parts.append(
                    f"{i}. {status} {exec_record.tool_name}({exec_record.arguments}) "
                    f"→ {book_count} books"
                )
            situation_parts.append("")
        
        # Show current candidate pool status
        total_candidates = len(candidates)
        situation_parts.append(f"**Current Candidate Pool**: {total_candidates} books")
        situation_parts.append("")
        
        # Provide adaptive guidance
        situation_parts.append("**Next Steps**:")
        
        if total_candidates >= 120:
            situation_parts.append("✓ Sufficient candidates (120+) - consider stopping")
        elif total_candidates >= 60:
            situation_parts.append("✓ Good candidate count (60+) - can stop or refine")
        elif total_candidates >= 30:
            situation_parts.append("⚠ Moderate candidates (30-59) - may need more tools")
        else:
            situation_parts.append("⚠ Low candidates (<30) - try fallback tools if available")
        
        # Check if all tools have been tried
        if strategy:
            all_tools = set(strategy.recommended_tools + strategy.fallback_tools)
            tried_tools = {exec.tool_name for exec in tool_execs}
            remaining_tools = all_tools - tried_tools
            
            if remaining_tools:
                situation_parts.append(f"- Remaining tools: {', '.join(remaining_tools)}")
            else:
                situation_parts.append("- All recommended/fallback tools have been tried")
        
        return "\n".join(situation_parts)
    
    def execute(
        self, 
        generator_input: RetrievalInput
    ) -> RetrievalOutput:
        """
        Execute retrieval strategy and gather candidates.
        
        Process:
        1. Inject strategy into agent state
        2. Run guided ReAct loop
        3. Extract candidates and build execution context
        4. Return output for CurationAgent
        
        Args:
            generator_input: Strategy and query from PlannerAgent
            
        Returns:
            RetrievalOutput with 60-120 candidates and context
            
        Raises:
            RuntimeError: If agent execution fails
        """
        append_chatbot_log(
            f"\n{'='*60}\n"
            f"RETRIEVAL AGENT EXECUTION\n"
            f"Query: {generator_input.query}\n"
            f"Strategy: {generator_input.strategy.reasoning}\n"
            f"{'='*60}"
        )
        
        # Build agent request
        request = AgentRequest(
            user_text=generator_input.query,
            conversation_history=[],
        )
        
        # Create initial state with strategy injected
        initial_state = AgentExecutionState(
            input_text=generator_input.query,
            conversation_history=[],
        )
        
        # Inject strategy and profile data into intermediate outputs
        initial_state.intermediate_outputs["strategy"] = generator_input.strategy
        initial_state.intermediate_outputs["profile_data"] = generator_input.profile_data
        
        # Execute ReAct loop with base agent
        try:
            response = self._execute_with_initial_state(request, initial_state)
        except Exception as e:
            append_chatbot_log(f"Generator execution failed: {e}")
            raise RuntimeError(f"RetrievalAgent execution failed: {e}") from e
        
        # Extract candidates from execution state
        candidates = response.execution_state.intermediate_outputs.get("book_objects", [])
        
        # Build execution context for Curator
        tools_used = [exec.tool_name for exec in response.execution_state.tool_executions]
        
        execution_context = ExecutionContext(
            planner_reasoning=generator_input.strategy.reasoning,
            tools_used=tools_used,
            profile_data=generator_input.profile_data,
        )
        
        # Build output
        output = RetrievalOutput(
            candidates=candidates,
            execution_context=execution_context,
            reasoning=self._build_reasoning_summary(response.execution_state),
        )
        
        append_chatbot_log(
            f"Generator complete: {len(candidates)} candidates, "
            f"{len(tools_used)} tools used"
        )
        
        return output
    
    def _execute_with_initial_state(
        self, 
        request: AgentRequest,
        initial_state: AgentExecutionState
    ) -> AgentResponse:
        """
        Execute agent with pre-populated initial state.
        
        This is a workaround since BaseLangGraphAgent.execute() doesn't
        accept initial_state parameter. We temporarily inject state before
        graph execution.
        """
        # Store original execute method
        original_graph_invoke = self.graph.invoke
        
        # Wrap graph invoke to use our initial state
        def invoke_with_state(state_dict):
            # Override with our initial state values
            merged = {**state_dict}
            merged["intermediate_outputs"] = initial_state.intermediate_outputs
            return original_graph_invoke(merged)
        
        # Temporarily replace invoke method
        self.graph.invoke = invoke_with_state
        
        try:
            # Call base execute
            return super().execute(request)
        finally:
            # Restore original invoke
            self.graph.invoke = original_graph_invoke
    
    def _build_reasoning_summary(self, state: AgentExecutionState) -> str:
        """
        Build human-readable summary of execution decisions.
        
        Args:
            state: Final execution state
            
        Returns:
            Summary string explaining what happened
        """
        candidates = state.intermediate_outputs.get("book_objects", [])
        tool_execs = state.tool_executions
        
        summary_parts = []
        
        # Summary of tool executions
        if tool_execs:
            summary_parts.append(f"Executed {len(tool_execs)} tool(s):")
            for exec in tool_execs:
                if exec.succeeded:
                    summary_parts.append(f"  - {exec.tool_name}: success")
                else:
                    summary_parts.append(f"  - {exec.tool_name}: failed ({exec.error})")
        
        # Summary of results
        summary_parts.append(f"Gathered {len(candidates)} candidate books.")
        
        # Stopping reason
        if len(candidates) >= 60:
            summary_parts.append("Stopped: sufficient candidates for curation.")
        elif len(candidates) < 60 and len(tool_execs) >= 4:
            summary_parts.append("Stopped: exhausted available tools.")
        else:
            summary_parts.append("Stopped: execution limit reached.")
        
        return " ".join(summary_parts)
