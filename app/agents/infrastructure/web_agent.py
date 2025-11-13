# app/agents/infrastructure/web_agent.py
"""
Web search agent with access to external search tools.
"""
from datetime import datetime
from app.agents.domain.entities import AgentConfiguration, AgentCapability, AgentExecutionState
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
            timeout_seconds=60,  # Increased from 45
            max_iterations=8  # Increased from 4 for complex queries
        )
        
        super().__init__(configuration)
    
    def _get_system_prompt(self) -> str:
        """Load web search specific system prompt."""
        persona = read_prompt("persona.system.md")
        policy = read_prompt("web.system.md")
        return f"{persona}\n\n{policy}".strip()
    
    def _get_target_category(self) -> str:
        return "web"
    
    def _build_current_situation(self, state: AgentExecutionState) -> str:
        """
        Override to add current date context for web agent.
        Critical for temporal reasoning about 'recent', 'current', 'latest'.
        """
        parts = []
        
        # CURRENT DATE CONTEXT (web agent specific)
        current_date = datetime.now().strftime("%B %d, %Y")
        parts.append(f"CURRENT DATE: {current_date}")
        parts.append("Use this to determine what is 'recent', 'current', 'latest', or 'new'.")
        parts.append("If search results contain information from 2024-2025, that IS current/recent.")
        parts.append("")
        
        # Get base situation from parent (tools, query, format)
        base_situation = super()._build_current_situation(state)
        parts.append(base_situation)
        
        return "\n".join(parts)
