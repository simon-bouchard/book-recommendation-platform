# app/agents/infrastructure/agent_adapter.py
"""
Adapter layer that converts between legacy schemas and domain entities.
Allows new domain-based agents to work with existing route/orchestrator code.
"""
from typing import List, Dict, Any

from app.agents.schemas import TurnInput, AgentResult, ToolCall as LegacyToolCall, BookOut
from app.agents.runtime import _safe_str
from app.agents.domain.entities import (
    AgentRequest,
    AgentResponse,
    BookRecommendation,
    ExecutionContext,
)


class AgentAdapter:
    """
    Converts between legacy schemas (TurnInput/AgentResult) and domain entities
    (AgentRequest/AgentResponse).
    """
    
    @staticmethod
    def turn_input_to_request(turn_input: TurnInput) -> AgentRequest:
        """Convert TurnInput to AgentRequest."""
        # Build execution context from TurnInput.ctx
        ctx_data = turn_input.ctx or {}
        
        context = ExecutionContext(
            user_id=ctx_data.get("uid"),
            conversation_id=ctx_data.get("conv_id"),
            session_data={
                "db": ctx_data.get("db"),
                "current_user": ctx_data.get("current_user"),
            },
            user_preferences={
                "profile_allowed": turn_input.profile_allowed,
                "num_ratings": turn_input.user_num_ratings,
            }
        )
        
        return AgentRequest(
            user_text=turn_input.user_text,
            conversation_history=turn_input.full_history,
            context=context
        )
    
    @staticmethod
    def response_to_agent_result(response: AgentResponse) -> AgentResult:
        """Convert AgentResponse to legacy AgentResult."""
        # Convert book recommendations to BookOut format
        book_ids = None
        if response.book_recommendations:
            book_ids = [rec.item_idx for rec in response.book_recommendations]
        
        # Convert tool executions to legacy ToolCall format
        tool_calls = []
        if response.execution_state:
            for exec in response.execution_state.tool_executions:
                tool_calls.append(LegacyToolCall(
                    name=exec.tool_name,
                    args=exec.arguments,
                    ok=exec.succeeded,
                    elapsed_ms=exec.execution_time_ms
                ))
        
        return AgentResult(
            target=response.target_category,
            text=response.text,
            success=response.success,
            book_ids=book_ids,
            tool_calls=tool_calls,
            citations=response.citations,
            policy_version=response.policy_version,
        )
    
    @staticmethod
    def book_recommendations_to_book_out(recommendations: List[BookRecommendation]) -> List[BookOut]:
        """Convert domain BookRecommendations to legacy BookOut schema."""
        return [
            BookOut(
                item_idx=rec.item_idx,
                title=_safe_str(rec.title),
                author=_safe_str(rec.author),
                year=_safe_str(rec.year),
                cover_id=_safe_str(rec.cover_id),
            )
            for rec in recommendations
        ]