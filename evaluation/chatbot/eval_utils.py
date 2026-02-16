# evaluation/chatbot/eval_utils.py
"""
Shared utilities for agent evaluation scripts.
Provides helpers for executing agents and reconstructing responses.
"""

from typing import Any
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.agents.domain.entities import (
    AgentRequest,
    AgentResponse,
    AgentExecutionState,
    ExecutionStatus,
    ToolExecution,
    BookRecommendation,
)
from app.agents.infrastructure.base_langgraph_agent import BaseLangGraphAgent


async def execute_with_streaming(
    agent: BaseLangGraphAgent, request: AgentRequest, **context
) -> AgentResponse:
    """
    Execute agent with streaming and reconstruct full response.

    Makes streaming transparent to evaluation logic by:
    1. Consuming the async stream
    2. Accumulating text chunks
    3. Reconstructing AgentResponse from final chunk data
    4. Converting tool calls from dict format to ToolExecution objects
    5. Extracting book recommendations from curation agent's book_ids

    This allows evaluation scripts to work with a complete AgentResponse
    without worrying about streaming mechanics.

    Args:
        agent: Any agent inheriting from BaseLangGraphAgent
        request: AgentRequest to execute
        **context: Additional context for agent execution (e.g., candidates, execution_context)

    Returns:
        AgentResponse with accumulated text and full execution state

    Raises:
        RuntimeError: If streaming completes without a final chunk

    Example:
        ```python
        agent = CurationAgent()
        request = AgentRequest(user_text="recommend fantasy", ...)

        # Execute with streaming and context
        response = await execute_with_streaming(
            agent,
            request,
            candidates=candidates,
            execution_context=context
        )

        # Now work with complete response
        print(response.text)
        print(response.book_recommendations)
        ```
    """
    text_chunks = []
    final_data = None

    async for chunk in agent.execute_stream(request, **context):
        if chunk.type == "token":
            text_chunks.append(chunk.content)
        elif chunk.type == "complete":
            final_data = chunk.data

    if not final_data:
        raise RuntimeError("Streaming completed without final chunk")

    tool_calls_raw = final_data.get("tool_calls", [])

    tool_executions = [
        ToolExecution(
            tool_name=tc["tool_name"],
            arguments=tc["arguments"],
            result=tc["result"],
        )
        for tc in tool_calls_raw
    ]

    exec_state = AgentExecutionState(
        input_text=request.user_text,
        conversation_history=request.conversation_history,
        status=(ExecutionStatus.COMPLETED if final_data.get("success") else ExecutionStatus.FAILED),
        tool_executions=tool_executions,
    )

    accumulated_text = "".join(text_chunks) if text_chunks else final_data.get("text", "")

    query_text = request.user_text
    if accumulated_text.startswith(query_text):
        accumulated_text = accumulated_text[len(query_text) :].lstrip()

    book_recommendations = []
    book_ids = final_data.get("book_ids", [])

    # Store raw book_ids for validation before filtering (curation-specific)
    if book_ids:
        exec_state.intermediate_outputs["cited_book_ids"] = book_ids

    if book_ids and "candidates" in context:
        candidates = context["candidates"]
        candidates_by_id = {c.item_idx: c for c in candidates}

        for book_id in book_ids:
            if book_id in candidates_by_id:
                book_recommendations.append(candidates_by_id[book_id])

    response = AgentResponse(
        text=accumulated_text,
        success=final_data.get("success", True),
        target_category=final_data.get("target", "respond"),
        execution_state=exec_state,
        policy_version=final_data.get("policy_version"),
        book_recommendations=book_recommendations,
    )

    return response
