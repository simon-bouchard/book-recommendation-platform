# app/agents/infrastructure/recsys/retrieval_agent.py
"""
RetrievalAgent - executes retrieval strategy and gathers 60-120 book candidates.
Second stage of multi-agent recommendation pipeline.
"""

import json
from typing import List
import time

from langchain_core.messages import HumanMessage

from app.agents.domain.entities import (
    AgentConfiguration,
    AgentCapability,
    AgentRequest,
)
from app.agents.domain.recsys_schemas import (
    RetrievalInput,
    RetrievalOutput,
    ExecutionContext,
    ToolExecutionSummary,
)
from app.agents.infrastructure.base_langgraph_agent import BaseLangGraphAgent
from app.agents.tools.registry import InternalToolGates, ToolRegistry
from app.agents.prompts.loader import read_prompt
from app.agents.logging import append_chatbot_log
from app.agents.domain.entities import AgentExecutionState, ExecutionStatus
from app.agents.utils.retrieval_logging_callback import RetrievalLoggingCallback


class RetrievalAgent(BaseLangGraphAgent):
    """
    Strategy execution and candidate gathering agent.

    Receives strategy from PlannerAgent and executes it adaptively:
    - Calls recommended tools with appropriate parameters
    - Accumulates book candidates with metadata
    - Applies fallback logic when primary tools underperform
    - Stops when 60-120 candidates gathered or all tools exhausted

    Uses native function calling with adaptive ReAct loop.
    Inherits streaming support from BaseLangGraphAgent.
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
            allowed_tools=frozenset(
                [
                    "als_recs",
                    "book_semantic_search",
                    "subject_hybrid_pool",
                    "subject_id_search",
                    "popular_books",
                ]
            ),
            llm_tier="large",
            timeout_seconds=60,
            max_iterations=6,  # Enough for primary + fallback + refinement
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
        those are for PlannerAgent only.
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
        """Load retrieval system prompt."""
        return read_prompt("recsys.retrieval.md")

    def _get_target_category(self) -> str:
        """Target category for agent adapter."""
        return "recsys_retrieval"

    def _get_start_status(self) -> str:
        """Initial status when retrieval starts."""
        return "Gathering book candidates..."

    def _add_context_messages(self, **context) -> List:
        """
        Inject strategy and progress tracking into message history.

        Args:
            **context: Must contain 'strategy' and 'profile_data'

        Returns:
            List containing HumanMessage with strategy and progress
        """
        strategy = context.get("strategy")
        profile_data = context.get("profile_data")
        current_state = context.get("current_state")

        if not strategy:
            return []

        # Build context sections
        context_parts = ["# Retrieval Strategy", ""]

        # Strategy from Planner
        context_parts.extend(
            [
                f"**Recommended Tools**: {', '.join(strategy.recommended_tools)}",
                f"**Fallback Tools**: {', '.join(strategy.fallback_tools)}",
                f"**Reasoning**: {strategy.reasoning}",
                "",
            ]
        )

        # Profile data if available
        if profile_data:
            context_parts.append(f"**Profile Data**: {json.dumps(profile_data)}")
            context_parts.append("")

        # Progress tracking (if state available)
        if current_state:
            book_objects = current_state.intermediate_outputs.get("book_objects", [])
            tool_execs = current_state.tool_executions

            if tool_execs:
                context_parts.append("**Tools Executed So Far**:")
                for exec_record in tool_execs:
                    status = "Success" if exec_record.succeeded else "Failed"
                    context_parts.append(f"- {exec_record.tool_name}: {status}")
                context_parts.append("")

            context_parts.append(f"**Current Candidates**: {len(book_objects)} books")
            context_parts.append("")

        # Stopping criteria reminder
        context_parts.extend(
            [
                "**Your Goal**: Gather 60-120 book candidates",
                "**When to Stop**:",
                "- You have 60-120 candidates (IDEAL)",
                "- You have 120+ candidates (sufficient, stop immediately)",
                "- All recommended AND fallback tools have been tried",
                "",
                "When stopping, respond with ONLY: 'Complete: [N] candidates gathered.'",
            ]
        )

        context_message = "\n".join(context_parts)
        return [HumanMessage(content=context_message)]

    async def execute(self, retrieval_input: RetrievalInput) -> RetrievalOutput:
        """
        Execute retrieval strategy and gather candidates asynchronously.

        Process:
        1. Inject strategy into context
        2. Run guided ReAct loop (non-blocking)
        3. Extract candidates even if agent times out or fails
        4. Return output for CurationAgent

        Args:
            retrieval_input: Strategy and query from PlannerAgent

        Returns:
            RetrievalOutput with candidates and execution context
            Always returns candidates even on partial failure
        """
        append_chatbot_log(
            f"\n{'=' * 60}\n"
            f"RETRIEVAL AGENT EXECUTION\n"
            f"Query: {retrieval_input.query}\n"
            f"Strategy: {retrieval_input.strategy.reasoning[:100]}...\n"
            f"{'=' * 60}"
        )

        # Build agent request
        request = AgentRequest(
            user_text=retrieval_input.query,
            conversation_history=[],
        )

        # Build messages with strategy context
        messages = self._build_messages(
            request,
            strategy=retrieval_input.strategy,
            profile_data=retrieval_input.profile_data,
        )

        logging_callback = RetrievalLoggingCallback()

        # Execute graph asynchronously (non-blocking)
        try:
            start_time = time.time()

            result = await self.graph.ainvoke(
                {"messages": messages}, config={"callbacks": [logging_callback]}
            )

            state = self._extract_execution_state(result, retrieval_input.query, start_time)

        except Exception as e:
            append_chatbot_log(f"[RETRIEVAL ERROR] {e}")
            # Create minimal fallback state

            state = AgentExecutionState(
                input_text=retrieval_input.query,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
            )

        # Extract candidates from execution state (even on failure/timeout)
        candidates = state.intermediate_outputs.get("book_objects", [])

        append_chatbot_log(
            f"Retrieval complete: {len(candidates)} candidates gathered (status: {state.status})"
        )

        # Build execution context for Curation
        tools_used = [exec.tool_name for exec in state.tool_executions]

        execution_context = ExecutionContext(
            planner_reasoning=retrieval_input.strategy.reasoning,
            tools_used=tools_used,
            profile_data=retrieval_input.profile_data,
        )

        # Build lightweight tool execution summaries
        tool_execution_summaries = [
            ToolExecutionSummary(
                tool_name=exec.tool_name,
                arguments=exec.arguments,
                succeeded=exec.succeeded,
                execution_time_ms=exec.execution_time_ms,
            )
            for exec in state.tool_executions
        ]

        # Build output
        output = RetrievalOutput(
            candidates=candidates,
            execution_context=execution_context,
            reasoning=self._build_reasoning_summary(state),
            tool_executions=tool_execution_summaries,
        )

        return output

    def _extract_execution_state(self, result: dict, query: str, start_time: float):
        """
        Extract execution state from graph result.

        Args:
            result: Graph execution result with messages
            query: Original query
            start_time: Execution start time

        Returns:
            AgentExecutionState with tool executions and book objects
        """
        from app.agents.domain.entities import (
            AgentExecutionState,
            ExecutionStatus,
            ToolExecution,
        )
        import time

        state = AgentExecutionState(
            input_text=query,
            conversation_history=[],
            status=ExecutionStatus.COMPLETED,
        )
        state.start_time = start_time
        state.end_time = time.time()

        # Extract messages from graph result
        messages = result.get("messages", [])

        # Extract tool executions by pairing tool calls with tool results
        tool_executions = []
        for i, msg in enumerate(messages):
            # AI message with tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_id = tool_call.get("id")
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})

                    tool_result = None
                    tool_error = None
                    for next_msg in messages[i + 1 :]:
                        if hasattr(next_msg, "tool_call_id") and next_msg.tool_call_id == tool_id:
                            tool_result = next_msg.content
                            if not tool_result:
                                tool_error = "No result returned"
                            break

                    if tool_result is None:
                        tool_error = "Tool result not found in message history"

                    tool_exec = ToolExecution(
                        tool_name=tool_name,
                        arguments=tool_args,
                        result=tool_result,
                        error=tool_error,
                        execution_time_ms=0,
                    )
                    tool_executions.append(tool_exec)

                    state.tool_executions = tool_executions

        # Use result processor to extract and build book recommendations
        books = self.result_processor.extract_book_recommendations(state)

        # Convert to dicts for RetrievalOutput compatibility
        book_objects = []
        for book in books:
            if hasattr(book, "to_dict"):
                book_objects.append(book.to_dict())
            elif hasattr(book, "__dict__"):
                book_objects.append(book.__dict__)

        state.intermediate_outputs["book_objects"] = book_objects

        return state

    def _build_reasoning_summary(self, state) -> str:
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
                    error_msg = exec.error[:50] if exec.error else "unknown error"
                    summary_parts.append(f"  - {exec.tool_name}: failed ({error_msg})")

        # Summary of results
        summary_parts.append(f"Gathered {len(candidates)} candidate books.")

        # Stopping reason
        from app.agents.domain.entities import ExecutionStatus

        if state.status == ExecutionStatus.FAILED:
            summary_parts.append("Stopped: execution error.")
        elif len(candidates) >= 60:
            summary_parts.append("Stopped: sufficient candidates for curation.")
        elif len(tool_execs) >= 4:
            summary_parts.append("Stopped: exhausted available tools.")
        else:
            summary_parts.append("Stopped: iteration limit reached.")

        return " ".join(summary_parts)
