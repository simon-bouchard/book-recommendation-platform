# app/agents/infrastructure/web_agent.py
"""
Web search agent with access to external search tools.
"""
from app.agents.domain.entities import AgentConfiguration, AgentCapability
from app.agents.prompts.loader import read_prompt
from .base_langgraph_agent import BaseLangGraphAgent


class WebAgent(BaseLangGraphAgent):
    """
    Agent specialized in web search and information retrieval.
    
    Has access to:
    - Web search
    - Web page fetching
    """
    
    def __init__(self):
        # Build configuration
        configuration = AgentConfiguration(
            policy_name="web.system.md",
            capabilities=frozenset([AgentCapability.WEB_SEARCH]),
            allowed_tools=frozenset([
                "web_search",
                "web_fetch"
            ]),
            llm_tier="medium",
            timeout_seconds=45,
            max_iterations=4  # Fewer iterations for web search
        )
        
        super().__init__(configuration)
    
    def _get_system_prompt(self) -> str:
        """Load web search specific system prompt."""
        persona = read_prompt("persona.system.md")
        policy = read_prompt("web.system.md")
        return f"{persona}\n\n{policy}".strip()
    
    def _get_target_category(self) -> str:
        return "web"