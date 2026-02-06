# app/agents/infrastructure/web_agent.py
"""
Web search agent using BaseLangGraphAgent.
Performs external web searches and synthesizes information with citations.
"""

from app.agents.infrastructure.base_langgraph_agent import BaseLangGraphAgent
from app.agents.domain.entities import AgentConfiguration, AgentCapability
from app.agents.prompts.loader import read_prompt
from app.agents.tools.registry import ToolRegistry, InternalToolGates


class WebAgent(BaseLangGraphAgent):
    """
    Web search agent for external information retrieval.

    Capabilities:
        - Search the web for current information
        - Fetch and analyze web pages
        - Synthesize findings with citations

    Tools:
        - web_search(query): Search the web
        - web_fetch(url): Fetch full page content
    """

    def __init__(self):
        """Initialize web agent with high-tier LLM and web tools."""
        configuration = AgentConfiguration(
            policy_name="web.system.md",
            capabilities=frozenset([AgentCapability.WEB_SEARCH]),
            allowed_tools=frozenset(["web_search", "web_fetch"]),
            llm_tier="large",
            timeout_seconds=60,
            max_iterations=10,
        )

        super().__init__(configuration)

    def _get_system_prompt(self) -> str:
        """
        Build system prompt from persona + web policy.

        Returns:
            Complete system prompt for web search
        """
        persona = read_prompt("persona.system.md")
        policy = read_prompt("web.system.md")

        return f"{persona}\n\n{policy}".strip()

    def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry:
        """
        Create tool registry with web tools enabled.

        Args:
            ctx_user: Not used for web agent
            ctx_db: Not used for web agent

        Returns:
            ToolRegistry with web tools
        """
        return ToolRegistry(
            web=True,
            docs=False,
            retrieval=False,
            context=False,
            gates=InternalToolGates(),
        )

    def _get_target_category(self) -> str:
        """
        Get target category for web agent.

        Returns:
            Category string "web"
        """
        return "web"

    def _get_start_status(self) -> str:
        """Initial status for web search."""
        return "Searching the web..."

    def _get_tool_complete_status(self) -> str:
        """Status after tool execution."""
        return "Analyzing results..."
