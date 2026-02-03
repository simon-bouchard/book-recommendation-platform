# app/agents/tools/native_tool.py
"""
Native tool interface - no LangChain dependencies.
Simple, type-safe, and LangGraph-compatible.
"""
from __future__ import annotations
from typing import Protocol, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import inspect


class ToolCategory(Enum):
    """Tool categories for filtering and permissions."""
    WEB = "web"
    DOCS = "docs"
    INTERNAL = "internal"


@dataclass
class ToolMetadata:
    """Metadata describing a tool's interface and behavior."""
    name: str
    description: str
    category: ToolCategory
    requires_auth: bool = False
    requires_db: bool = False
    is_deterministic: bool = True
    
    def to_llm_description(self) -> str:
        """Format for LLM prompt."""
        return f"{self.name}: {self.description}"


class NativeTool(Protocol):
    """
    Protocol for native tools - no wrapper complexity.
    
    Tools are just callables with metadata attached.
    """
    metadata: ToolMetadata
    
    def __call__(self, **kwargs) -> Any:
        """Execute the tool with keyword arguments."""
        ...


@dataclass
class ToolDefinition:
    """
    Complete tool definition with function and metadata.
    
    This is what ToolRegistry stores and ToolExecutor calls.
    """
    func: Callable
    metadata: ToolMetadata
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def category(self) -> ToolCategory:
        return self.metadata.category
    
    @property
    def description(self) -> str:
        return self.metadata.description
    
    def get_signature(self) -> inspect.Signature:
        """Get function signature for validation."""
        return inspect.signature(self.func)
    
    def execute(self, **kwargs) -> Any:
        """Execute with keyword arguments."""
        return self.func(**kwargs)
    
    def to_llm_description(self) -> str:
        """Format for LLM consumption."""
        sig = self.get_signature()
        params = []
        for name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                type_str = getattr(param.annotation, '__name__', str(param.annotation))
                params.append(f"{name}: {type_str}")
            else:
                params.append(name)
        
        param_str = ", ".join(params) if params else "no parameters"
        return f"{self.name}({param_str}) - {self.description}"


def tool(
    name: str,
    description: str,
    category: ToolCategory,
    requires_auth: bool = False,
    requires_db: bool = False,
    is_deterministic: bool = True,
):
    """
    Decorator to create native tools from functions.
    
    Example:
        @tool("book_search", "Search for books", ToolCategory.INTERNAL)
        def search_books(query: str, limit: int = 10) -> list[dict]:
            return [...]
    """
    def decorator(func: Callable) -> ToolDefinition:
        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            requires_auth=requires_auth,
            requires_db=requires_db,
            is_deterministic=is_deterministic,
        )
        return ToolDefinition(func=func, metadata=metadata)
    
    return decorator