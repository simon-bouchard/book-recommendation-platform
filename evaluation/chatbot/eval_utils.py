# evaluation/chatbot/eval_utils.py
"""
Shared utilities for agent evaluation scripts.
Provides helpers for executing agents and reconstructing responses.
"""

from typing import Any
from pathlib import Path
import sys

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.agents.domain.entities import (
    AgentRequest,
    AgentResponse,
    AgentExecutionState,
    ExecutionStatus,
    ToolExecution,
)
from app.agents.infrastructure.base_langgraph_agent import BaseLangGraphAgent


async def execute_with_streaming(agent: BaseLangGraphAgent, request: AgentRequest) -> AgentResponse:
    """
    Execute agent with streaming and reconstruct full response.

    Makes streaming transparent to evaluation logic by:
    1. Consuming the async stream
    2. Accumulating text chunks
    3. Reconstructing AgentResponse from final chunk data
    4. Converting tool calls from dict format to ToolExecution objects

    This allows evaluation scripts to work with a complete AgentResponse
    without worrying about streaming mechanics.

    Args:
        agent: Any agent inheriting from BaseLangGraphAgent
        request: AgentRequest to execute

    Returns:
        AgentResponse with accumulated text and full execution state

    Raises:
        RuntimeError: If streaming completes without a final chunk

    Example:
        ```python
        agent = DocsAgent()
        request = AgentRequest(user_text="What is ALS?", ...)

        # Execute with streaming (async)
        response = await execute_with_streaming(agent, request)

        # Now work with complete response
        print(response.text)
        print(response.execution_state.tool_executions)
        ```
    """
    text_chunks = []
    final_data = None

    # Consume the stream
    async for chunk in agent.execute_stream(request):
        if chunk.type == "token":
            text_chunks.append(chunk.content)
        elif chunk.type == "complete":
            final_data = chunk.data

    # Ensure we got a final chunk
    if not final_data:
        raise RuntimeError("Streaming completed without final chunk")

    # Extract tool calls from final data
    # BaseLangGraphAgent includes tool_calls in the complete chunk
    tool_calls_raw = final_data.get("tool_calls", [])

    # Convert dict format to ToolExecution objects for consistency
    # This matches the format used by non-streaming execute()
    tool_executions = [
        ToolExecution(
            tool_name=tc["tool_name"],
            arguments=tc["arguments"],
            result=tc["result"],
        )
        for tc in tool_calls_raw
    ]

    # Build execution state with tool executions
    exec_state = AgentExecutionState(
        input_text=request.user_text,
        conversation_history=request.conversation_history,
        status=(ExecutionStatus.COMPLETED if final_data.get("success") else ExecutionStatus.FAILED),
        tool_executions=tool_executions,
    )

    # Reconstruct full response
    accumulated_text = "".join(text_chunks) if text_chunks else final_data.get("text", "")

    # Strip query prefix if present (LangGraph sometimes echoes the query)
    # Pattern: query text appears at start without separator
    query_text = request.user_text
    if accumulated_text.startswith(query_text):
        accumulated_text = accumulated_text[len(query_text) :].lstrip()

    response = AgentResponse(
        text=accumulated_text,
        success=final_data.get("success", True),
        target_category=final_data.get("target", "respond"),
        execution_state=exec_state,
        policy_version=final_data.get("policy_version"),
    )

    return response
