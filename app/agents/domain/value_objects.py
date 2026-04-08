# app/agents/domain/value_objects.py
"""
Value objects for the agent domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AgentType(Enum):
    """Types of agents in the system."""

    RECOMMENDATION = "recommendation"
    WEB_SEARCH = "web_search"
    DOCUMENT_SEARCH = "document_search"
    CONVERSATIONAL = "conversational"


@dataclass(frozen=True)
class ToolPermissions:
    """Immutable permissions for tool access."""

    allowed_tools: frozenset[str]
    web_access: bool = False
    docs_access: bool = False
    internal_access: bool = False

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if tool is allowed."""
        return tool_name in self.allowed_tools

    def with_additional_tools(self, *tool_names: str) -> ToolPermissions:
        """Create new permissions with additional tools."""
        new_tools = self.allowed_tools | frozenset(tool_names)
        return ToolPermissions(
            allowed_tools=new_tools,
            web_access=self.web_access,
            docs_access=self.docs_access,
            internal_access=self.internal_access,
        )


@dataclass(frozen=True)
class ModelConfiguration:
    """Configuration for LLM model usage."""

    tier: str = "medium"  # small, medium, large
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout_seconds: int = 30

    def for_creativity(self) -> ModelConfiguration:
        """Create config optimized for creative tasks."""
        return ModelConfiguration(
            tier=self.tier,
            temperature=0.7,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )

    def for_analysis(self) -> ModelConfiguration:
        """Create config optimized for analytical tasks."""
        return ModelConfiguration(
            tier="large",
            temperature=0.0,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds * 2,  # More time for analysis
        )


@dataclass(frozen=True)
class ExecutionLimits:
    """Limits for agent execution."""

    max_iterations: int = 10
    timeout_seconds: int = 60
    max_tool_calls: int = 20
    max_reasoning_steps: int = 50

    def is_within_limits(
        self, iterations: int, elapsed_seconds: float, tool_calls: int, reasoning_steps: int
    ) -> bool:
        """Check if execution is within all limits."""
        return (
            iterations <= self.max_iterations
            and elapsed_seconds <= self.timeout_seconds
            and tool_calls <= self.max_tool_calls
            and reasoning_steps <= self.max_reasoning_steps
        )
