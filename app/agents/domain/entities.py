# app/agents/domain/entities.py
"""
Domain entities for the agent system.
These represent core business concepts and are framework-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentCapability(Enum):
    """Agent capabilities that determine what an agent can do."""

    WEB_SEARCH = "web_search"
    DOCUMENT_SEARCH = "docs_search"
    INTERNAL_TOOLS = "internal_tools"  # Includes recommendation tools
    CONVERSATIONAL = "conversational"


class ExecutionStatus(Enum):
    """Status of agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class AgentConfiguration:
    """Immutable configuration for an agent."""

    policy_name: str
    capabilities: frozenset[AgentCapability]
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    llm_tier: str = "medium"
    timeout_seconds: int = 60
    max_iterations: int = 10

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if tool is allowed."""
        return not self.allowed_tools or tool_name in self.allowed_tools


@dataclass
class ExecutionContext:
    """Context information available during agent execution."""

    user_id: Optional[int] = None
    conversation_id: Optional[str] = None
    session_data: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecution:
    """Record of a single tool execution."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def succeeded(self) -> bool:
        """Check if tool execution was successful."""
        return self.error is None


@dataclass
class AgentExecutionState:
    """Current state of agent execution."""

    status: ExecutionStatus = ExecutionStatus.PENDING
    input_text: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    tool_executions: List[ToolExecution] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    intermediate_outputs: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def execution_time_ms(self) -> Optional[int]:
        """Calculate execution time in milliseconds."""
        if self.end_time is None:
            return None
        return int((self.end_time - self.start_time) * 1000)

    def add_tool_execution(self, execution: ToolExecution) -> None:
        """Add a tool execution record."""
        self.tool_executions.append(execution)

    def add_reasoning_step(self, step: str) -> None:
        """Add a reasoning step."""
        self.reasoning_steps.append(step)

    def mark_completed(self) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED
        self.end_time = time.time()

    def mark_failed(self, error_message: str) -> None:
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.error_message = error_message
        self.end_time = time.time()


@dataclass
class AgentRequest:
    """Request to execute an agent."""

    user_text: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    context: ExecutionContext = field(default_factory=ExecutionContext)
    configuration_overrides: Dict[str, Any] = field(default_factory=dict)

    def with_context(self, **kwargs) -> AgentRequest:
        """Create a new request with updated context."""
        new_context = ExecutionContext(
            user_id=kwargs.get("user_id", self.context.user_id),
            conversation_id=kwargs.get("conversation_id", self.context.conversation_id),
            session_data={
                **self.context.session_data,
                **kwargs.get("session_data", {}),
            },
            user_preferences={
                **self.context.user_preferences,
                **kwargs.get("user_preferences", {}),
            },
            execution_metadata={
                **self.context.execution_metadata,
                **kwargs.get("execution_metadata", {}),
            },
        )
        return AgentRequest(
            user_text=self.user_text,
            conversation_history=self.conversation_history,
            context=new_context,
            configuration_overrides=self.configuration_overrides,
        )


@dataclass
class BookRecommendation:
    """Domain entity for a book recommendation with full metadata."""

    item_idx: int  # Matches existing BookOut schema naming
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    cover_id: Optional[str] = None
    num_ratings: Optional[int] = None  # Rating count for social proof
    recommendation_reason: Optional[str] = None

    # Extended metadata for curation (from tool results)
    subjects: Optional[List[str]] = None
    tones: Optional[List[str]] = None
    vibe: Optional[str] = None
    genre: Optional[str] = None

    def is_complete(self) -> bool:
        """Check if recommendation has minimum required data."""
        return self.item_idx is not None and (self.title is not None or self.author is not None)

    def has_rich_metadata(self) -> bool:
        """Check if recommendation has extended metadata for curation."""
        return bool(self.subjects or self.tones or self.vibe or self.genre)

    def to_curation_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for curation agent.

        Returns dict with all available metadata, ready for JSON serialization.
        """
        result = {
            "item_idx": self.item_idx,
            "title": self.title or "",
            "author": self.author or "",
            "year": self.year or "",
            "num_ratings": self.num_ratings or 0,
        }

        # Only add enrichment fields if they have content
        if self.subjects:
            result["subjects"] = self.subjects
        if self.tones:
            result["tones"] = self.tones
        if self.genre:
            result["genre"] = self.genre
        if self.vibe:
            result["vibe"] = self.vibe

        return result


@dataclass
class AgentResponse:
    """Response from agent execution."""

    text: str
    target_category: str = "respond"  # recsys, web, docs, respond
    success: bool = True
    book_recommendations: List[BookRecommendation] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    execution_state: Optional[AgentExecutionState] = None
    policy_version: Optional[str] = None

    @property
    def execution_time_ms(self) -> Optional[int]:
        """Get execution time from state if available."""
        return self.execution_state.execution_time_ms if self.execution_state else None

    @property
    def tool_calls_count(self) -> int:
        """Get number of tool calls made."""
        return len(self.execution_state.tool_executions) if self.execution_state else 0

    def get_book_ids(self) -> List[int]:
        """Extract book IDs from recommendations."""
        return [rec.item_idx for rec in self.book_recommendations if rec.item_idx is not None]
