# app/agents/infrastructure/docs_agent.py
"""
Documentation search agent using BaseLangGraphAgent.
Searches internal help documentation and returns formatted answers.
"""

from app.agents.infrastructure.base_langgraph_agent import BaseLangGraphAgent
from app.agents.domain.entities import AgentConfiguration, AgentCapability
from app.agents.prompts.loader import read_prompt
from app.agents.tools.registry import ToolRegistry, InternalToolGates
from app.agents.tools.docs_tools import DocsTools


class DocsAgent(BaseLangGraphAgent):
    """
    Documentation search agent with internal help document access.

    Capabilities:
        - Search available documentation
        - Read specific help documents
        - Format answers from docs content

    Tools:
        - help_manifest(): List available help documents
        - help_read(doc_name): Read specific document content
    """

    def __init__(self):
        """Initialize documentation agent with medium-tier LLM and docs tools."""
        configuration = AgentConfiguration(
            policy_name="docs.system.md",
            capabilities=frozenset([AgentCapability.DOCUMENT_SEARCH]),
            allowed_tools=frozenset(["help_read"]),
            llm_tier="large",
            timeout_seconds=30,
            max_iterations=5,
        )

        super().__init__(configuration)

    def _get_system_prompt(self) -> str:
        """
        Build system prompt from persona + policy + manifest.

        Returns:
            Complete system prompt with available docs context
        """
        persona = read_prompt("persona.system.md")
        policy = read_prompt("docs.system.md")

        # Add manifest context if available
        try:
            manifest = DocsTools().render_manifest_for_prompt(max_items=10)
            if manifest:
                policy = f"{policy}\n\nAvailable Documentation:\n{manifest}"
        except Exception:
            # Manifest load failure shouldn't break agent
            pass

        return f"{persona}\n\n{policy}".strip()

    def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry:
        """
        Create tool registry with docs tools enabled.

        Args:
            ctx_user: Not used for docs agent
            ctx_db: Not used for docs agent

        Returns:
            ToolRegistry with docs tools
        """
        return ToolRegistry(
            web=False,
            docs=True,
            retrieval=False,
            context=False,
            gates=InternalToolGates(),
        )

    def _get_target_category(self) -> str:
        """
        Get target category for docs agent.

        Returns:
            Category string "docs"
        """
        return "docs"

    def _get_start_status(self) -> str:
        """Initial status for docs search."""
        return "Searching documentation..."

    def _get_tool_complete_status(self) -> str:
        """Status after tool execution."""
        return "Analyzing documentation..."
