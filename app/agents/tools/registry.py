# app/agents/tools/registry.py
"""
Modernized tool registry with native tool support.
No LangChain dependencies - clean separation of concerns.
"""
from dataclasses import dataclass
from typing import Optional, Any

from sqlalchemy.orm import Session

from .native_tool import ToolDefinition, ToolCategory
from .external.web_tools import WebTools, WebToolState
from .docs_tools import DocsTools
from .recsys.native_tools import InternalTools


@dataclass
class InternalToolGates:
    """
    Controls for internal tool availability based on user state.
    
    This encapsulates all the business logic for determining which
    internal tools should be available to a given user.
    """
    user_num_ratings: Optional[int] = None
    warm_threshold: int = 10
    profile_allowed: bool = False
    
    @property
    def is_warm_user(self) -> bool:
        """Check if user has enough ratings for warm recommendations."""
        if self.user_num_ratings is None:
            return False
        return self.user_num_ratings >= self.warm_threshold


class ToolRegistry:
    """
    Unified tool registry with native tool support.
    
    Manages tool discovery, filtering, and access control based on:
    - Agent capabilities (web, docs, internal)
    - User authentication state
    - User rating history (warm/cold)
    - Profile access consent
    
    This is the single source of truth for what tools are available
    to a given agent in a given context.
    """
    
    def __init__(
        self,
        *,
        web: bool = True,
        docs: bool = True,
        internal: bool = False,
        gates: Optional[InternalToolGates] = None,
        ctx_user: Any = None,
        ctx_db: Optional[Session] = None,
    ):
        """
        Initialize registry with capability flags and context.
        
        Args:
            web: Enable web tools
            docs: Enable documentation tools
            internal: Enable internal recommendation tools
            gates: Access control for internal tools
            ctx_user: Current user object (for internal tools)
            ctx_db: Database session (for internal tools)
        """
        self.web_enabled = web
        self.docs_enabled = docs
        self.internal_enabled = internal
        self.gates = gates or InternalToolGates()
        self.ctx_user = ctx_user
        self.ctx_db = ctx_db
        
        # Tool factories
        self._web_tools: Optional[WebTools] = None
        self._docs_tools: Optional[DocsTools] = None
        self._internal_tools: Optional[InternalTools] = None
        
        # Cached tool list
        self._tools_cache: Optional[list[ToolDefinition]] = None
    
    def get_tools(self) -> list[ToolDefinition]:
        """
        Get all enabled tools based on capabilities and gates.
        
        Tools are filtered based on:
        1. Category-level flags (web, docs, internal)
        2. Per-tool gates (authentication, warm/cold, consent)
        
        Returns:
            List of available tool definitions
        """
        if self._tools_cache is not None:
            return self._tools_cache
        
        tools: list[ToolDefinition] = []
        
        # Web tools
        if self.web_enabled:
            if self._web_tools is None:
                self._web_tools = WebTools()
            tools.extend(self._web_tools.get_tools())
        
        # Documentation tools
        if self.docs_enabled:
            if self._docs_tools is None:
                self._docs_tools = DocsTools()
            tools.extend(self._docs_tools.get_tools())
        
        # Internal recommendation tools
        if self.internal_enabled:
            if self._internal_tools is None:
                self._internal_tools = InternalTools(
                    current_user=self.ctx_user,
                    db=self.ctx_db,
                    user_num_ratings=self.gates.user_num_ratings or 0,
                    allow_profile=self.gates.profile_allowed,
                )
            tools.extend(self._internal_tools.get_tools(
                is_warm=self.gates.is_warm_user
            ))
        
        self._tools_cache = tools
        return tools
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a specific tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool definition if found and enabled, None otherwise
        """
        for tool in self.get_tools():
            if tool.name == name:
                return tool
        return None
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is available."""
        return self.get_tool(name) is not None
    
    def get_tools_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """Get all tools in a specific category."""
        return [t for t in self.get_tools() if t.category == category]
    
    def get_tool_names(self) -> list[str]:
        """Get list of all available tool names."""
        return [t.name for t in self.get_tools()]
    
    def format_for_llm(self) -> str:
        """
        Format available tools for LLM consumption.
        
        Returns:
            Formatted string describing all available tools
        """
        tools = self.get_tools()
        if not tools:
            return "No tools available."
        
        lines = ["Available Tools:"]
        for tool in tools:
            lines.append(f"- {tool.to_llm_description()}")
        
        return "\n".join(lines)
    
    def reset_web_state(self) -> None:
        """Reset web tool deduplication state (for new conversation turn)."""
        if self._web_tools:
            self._web_tools.state.reset()
    
    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics for debugging/monitoring."""
        tools = self.get_tools()
        
        return {
            "total_tools": len(tools),
            "web_enabled": self.web_enabled,
            "docs_enabled": self.docs_enabled,
            "internal_enabled": self.internal_enabled,
            "is_warm_user": self.gates.is_warm_user,
            "profile_allowed": self.gates.profile_allowed,
            "tools_by_category": {
                "web": len([t for t in tools if t.category == ToolCategory.WEB]),
                "docs": len([t for t in tools if t.category == ToolCategory.DOCS]),
                "internal": len([t for t in tools if t.category == ToolCategory.INTERNAL]),
            },
            "tool_names": self.get_tool_names(),
        }


# Backward compatibility adapter for legacy code
class ToolRegistry(ToolRegistry):
    """
    Backward-compatible wrapper that maintains the old ToolRegistry API.
    
    This allows gradual migration without breaking existing code.
    Maps the new native tools to look like LangChain tools where needed.
    """
    
    def get_tools(self) -> list:
        """
        Return tools in a format compatible with legacy code.
        
        For now, this returns native ToolDefinitions. If legacy code
        expects LangChain Tools, we can add a compatibility layer here.
        """
        return super().get_tools()