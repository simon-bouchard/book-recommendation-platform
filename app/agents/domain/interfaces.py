# app/agents/domain/interfaces.py
"""
Abstract interfaces that define what agents should do (not how).
These are technology-agnostic contracts.
"""

from abc import ABC
from typing import Any, Dict, List, Protocol, runtime_checkable

from .entities import (
    AgentConfiguration,
    AgentExecutionState,
    AgentRequest,
    AgentResponse,
    BookRecommendation,
    ExecutionContext,
)


@runtime_checkable
class Agent(Protocol):
    """Protocol defining the agent interface."""

    def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute agent with the given request."""
        ...

    @property
    def configuration(self) -> AgentConfiguration:
        """Get agent configuration."""
        ...


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, configuration: AgentConfiguration):
        self._configuration = configuration

    @property
    def configuration(self) -> AgentConfiguration:
        """Get agent configuration."""
        return self._configuration

    def can_handle_request(self, request: AgentRequest) -> bool:
        """Check if this agent can handle the request."""
        return True  # Override in subclasses for specific logic


class ToolProvider(Protocol):
    """Protocol for tool providers."""

    def get_available_tools(self, context: ExecutionContext) -> List[str]:
        """Get list of available tool names for the context."""
        ...

    def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Execute a tool with given arguments."""
        ...

    def is_tool_allowed(self, tool_name: str, context: ExecutionContext) -> bool:
        """Check if tool is allowed in the context."""
        ...


class ResultProcessor(Protocol):
    """Protocol for processing agent execution results."""

    def extract_book_recommendations(self, state: AgentExecutionState) -> List[BookRecommendation]:
        """Extract book recommendations from execution state."""
        ...

    def format_response_text(self, state: AgentExecutionState) -> str:
        """Format final response text from execution state."""
        ...

    def extract_citations(self, state: AgentExecutionState) -> List[Dict[str, Any]]:
        """Extract citations from execution state."""
        ...
