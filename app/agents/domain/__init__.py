# app/agents/domain/__init__.py
"""
Domain layer for the agent system.
Clean architecture: framework-agnostic business logic.
"""

# Entities
from .entities import (
    AgentCapability,
    ExecutionStatus,
    AgentConfiguration,
    ExecutionContext,
    ToolExecution,
    AgentExecutionState,
    AgentRequest,
    BookRecommendation,
    AgentResponse,
)

# Interfaces
from .interfaces import (
    Agent,
    BaseAgent,
    ToolProvider,
    ResultProcessor,
)

# Services
from .services import (
    BookExtractionResult,
    StandardResultProcessor,
)

# Value Objects
from .value_objects import (
    AgentType,
    ToolPermissions,
    ModelConfiguration,
    ExecutionLimits,
)

__all__ = [
    # Entities
    "AgentCapability",
    "ExecutionStatus",
    "AgentConfiguration",
    "ExecutionContext",
    "ToolExecution",
    "AgentExecutionState",
    "AgentRequest",
    "BookRecommendation",
    "AgentResponse",
    # Interfaces
    "Agent",
    "BaseAgent",
    "ToolProvider",
    "ResultProcessor",
    # Services
    "BookExtractionResult",
    "StandardResultProcessor",
    # Value Objects
    "AgentType",
    "ToolPermissions",
    "ModelConfiguration",
    "ExecutionLimits",
]