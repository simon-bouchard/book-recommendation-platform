# app/agents/infrastructure/agent_factory.py
"""
Factory for creating appropriate agents based on routing target.
"""
from typing import Optional, Any

from app.agents.schemas import Target
from app.agents.domain.interfaces import Agent
from .recsys.recommendation_agent import RecommendationAgent
from .web_agent import WebAgent
from .docs_agent import DocsAgent
from .response_agent import ResponseAgent


class AgentFactory:
    """
    Creates agent instances based on routing target and context.
    """
    
    @staticmethod
    def create_agent(
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
        if target == "recsys":
            return RecommendationAgent(
                current_user=current_user,
                db=db,
                user_num_ratings=user_num_ratings,
                warm_threshold=10,
                allow_profile=use_profile,
            )
        
        elif target == "web":
            return WebAgent()
        
        elif target == "docs":
            return DocsAgent()
        
        else:  # "respond" or fallback
            return ResponseAgent()