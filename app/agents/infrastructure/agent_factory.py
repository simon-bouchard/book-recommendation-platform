# app/agents/infrastructure/agent_factory.py
"""
Factory for creating appropriate agents based on routing target.
Supports dependency injection for testing.
"""

from typing import Any, Callable, Optional

from app.agents.domain.interfaces import Agent
from app.agents.logging import append_chatbot_log
from app.agents.schemas import Target

from .docs_agent import DocsAgent
from .recsys.orchestrator import RecommendationAgent
from .response_agent import ResponseAgent
from .web_agent import WebAgent


class AgentFactory:
    """
    Creates agent instances based on routing target and context.
    Supports provider injection for testing.
    """

    def __init__(
        self,
        recsys_provider: Optional[Callable[..., Agent]] = None,
        web_provider: Optional[Callable[..., Agent]] = None,
        docs_provider: Optional[Callable[..., Agent]] = None,
        respond_provider: Optional[Callable[..., Agent]] = None,
    ):
        """
        Initialize factory with optional agent providers.

        Args:
            recsys_provider: Optional callable returning recsys agent (for testing)
            web_provider: Optional callable returning web agent (for testing)
            docs_provider: Optional callable returning docs agent (for testing)
            respond_provider: Optional callable returning respond agent (for testing)
        """
        self._recsys_provider = (
            recsys_provider if recsys_provider is not None else self._create_recsys
        )
        self._web_provider = web_provider if web_provider is not None else self._create_web
        self._docs_provider = docs_provider if docs_provider is not None else self._create_docs
        self._respond_provider = (
            respond_provider if respond_provider is not None else self._create_respond
        )

    @staticmethod
    def _create_recsys(**kwargs) -> Agent:
        """Create default RecommendationAgent instance."""
        return RecommendationAgent(**kwargs)

    @staticmethod
    def _create_web(**kwargs) -> Agent:
        """Create default WebAgent instance."""
        return WebAgent(**kwargs)

    @staticmethod
    def _create_docs(**kwargs) -> Agent:
        """Create default DocsAgent instance."""
        return DocsAgent(**kwargs)

    @staticmethod
    def _create_respond(**kwargs) -> Agent:
        """Create default ResponseAgent instance."""
        return ResponseAgent(**kwargs)

    def create_agent(
        self,
        target: Target,
        *,
        current_user: Any = None,
        db: Any = None,
        user_num_ratings: Optional[int] = None,
        use_profile: bool = False,
    ) -> Agent:
        """
        Create the appropriate agent for the given target.

        Args:
            target: Routing target ("recsys", "web", "docs", "respond")
            current_user: Current user object (for recsys)
            db: Database session (for recsys)
            user_num_ratings: Number of user ratings (for recsys warm/cold detection)
            use_profile: Whether user has granted profile access

        Returns:
            Agent instance implementing the Agent protocol
        """
        append_chatbot_log(f"AgentFactory: Creating {target} agent")

        if target == "recsys":
            is_warm = (user_num_ratings or 0) >= 10
            append_chatbot_log(
                f"  - RecsysAgent: ratings={user_num_ratings}, warm={is_warm}, "
                f"profile={use_profile}"
            )
            return self._recsys_provider(
                current_user=current_user,
                db=db,
                user_num_ratings=user_num_ratings,
                warm_threshold=10,
                allow_profile=use_profile,
            )

        elif target == "web":
            append_chatbot_log("  - WebAgent: search enabled")
            return self._web_provider()

        elif target == "docs":
            append_chatbot_log("  - DocsAgent: document search")
            return self._docs_provider()

        else:  # "respond" or fallback
            append_chatbot_log("  - ResponseAgent: conversational")
            return self._respond_provider()
