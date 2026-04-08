# app/agents/infrastructure/response_agent.py
"""
Simple response agent using BaseLangGraphAgent.
Handles conversational responses without any tool usage.
"""

from app.agents.domain.entities import AgentCapability, AgentConfiguration
from app.agents.infrastructure.base_langgraph_agent import BaseLangGraphAgent
from app.agents.prompts.loader import read_prompt
from app.agents.tools.registry import InternalToolGates, ToolRegistry


class ResponseAgent(BaseLangGraphAgent):
    """
    Conversational response agent with no tools.

    Capabilities:
        - Handle greetings and acknowledgements
        - Provide simple conversational responses
        - No tool usage required

    Tools:
        None - purely conversational
    """

    def __init__(self):
        """Initialize response agent with medium-tier LLM and no tools."""
        configuration = AgentConfiguration(
            policy_name="persona.system.md",
            capabilities=frozenset([AgentCapability.CONVERSATIONAL]),
            allowed_tools=frozenset(),  # No tools
            llm_tier="medium",
            timeout_seconds=20,
            max_iterations=1,  # Single turn only
        )

        super().__init__(configuration)

    def _get_system_prompt(self) -> str:
        """
        Build system prompt for conversational responses.

        Returns:
            System prompt emphasizing helpful, brief responses
        """
        persona = read_prompt("persona.system.md")

        # Add simple instruction for response mode
        instruction = (
            "Provide brief, helpful responses to acknowledgements and simple questions. "
            "Keep responses concise and friendly. No tools are available."
        )

        return f"{persona}\n\n{instruction}".strip()

    def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry:
        """
        Create tool registry with no tools enabled.

        Args:
            ctx_user: Not used for response agent
            ctx_db: Not used for response agent

        Returns:
            ToolRegistry with no tools
        """
        return ToolRegistry(
            web=False,
            docs=False,
            retrieval=False,
            context=False,
            gates=InternalToolGates(),
        )

    def _get_target_category(self) -> str:
        """
        Get target category for response agent.

        Returns:
            Category string "respond"
        """
        return "respond"

    def _get_start_status(self) -> str:
        """Initial status for response generation."""
        return "Responding..."

    def _get_tool_complete_status(self) -> str:
        """
        Status after tool execution.

        Note: This should never be called since ResponseAgent has no tools.
        """
        return "Processing..."
