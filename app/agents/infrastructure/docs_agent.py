# app/agents/infrastructure/docs_agent.py
"""
Documentation search agent with access to internal help docs.
"""
from app.agents.domain.entities import AgentConfiguration, AgentCapability
from app.agents.prompts.loader import read_prompt
from .base_langgraph_agent import BaseLangGraphAgent


class DocsAgent(BaseLangGraphAgent):
    """
    Agent specialized in searching internal documentation.
    
    Has access to:
    - help_manifest: List available help documents
    - help_read: Read a specific help document
    """
    
    def __init__(self):
        # Build configuration
        configuration = AgentConfiguration(
            policy_name="docs.system.md",
            capabilities=frozenset([AgentCapability.DOCUMENT_SEARCH]),
            allowed_tools=frozenset([
                "help_manifest",  # Lists all available docs with metadata
                "help_read"       # Reads a specific doc by alias/filename
            ]),
            llm_tier="medium",
            timeout_seconds=30,
            max_iterations=5  # Simple lookup tasks
        )
        
        super().__init__(configuration)
    
    def _get_system_prompt(self) -> str:
        """Load docs-specific system prompt."""
        persona = read_prompt("persona.system.md")
        policy = read_prompt("docs.system.md")
        
        # Get manifest if docs_tools available
        try:
            from app.agents.tools.docs_tools import DocsTools
            manifest = DocsTools().render_manifest_for_prompt(max_items=10)
            if manifest:
                policy = f"{policy}\n\nAvailable Documentation:\n{manifest}"
        except Exception:
            pass  # No manifest available
        
        return f"{persona}\n\n{policy}".strip()

    def _get_target_category(self) -> str:
        return "docs"