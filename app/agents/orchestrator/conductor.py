# app/agents/orchestrator/conductor.py
"""
Multi-agent conductor using clean LangGraph infrastructure.
Routes requests to appropriate agents and returns standardized results.
"""
from __future__ import annotations
from typing import Any, Optional, List

from app.agents.schemas import AgentResult, TurnInput, Target
from app.agents.orchestrator.router import RouterLLM
from app.agents.context_builder import make_router_input, make_branch_input
from app.agents.infrastructure.agent_factory import AgentFactory
from app.agents.infrastructure.agent_adapter import AgentAdapter


class Conductor:
    """
    Multi-agent conductor with clean architecture:
      1) Route with RouterLLM
      2) Build AgentRequest from TurnInput
      3) Execute agent and convert result back to AgentResult
    """

    def __init__(self) -> None:
        self.router = RouterLLM()
        self.factory = AgentFactory()
        self.adapter = AgentAdapter()
        self.policy_version = "conductor.mas.v1"

    def run(
        self,
        *,
        history: List[dict],
        user_text: str,
        use_profile: bool,
        current_user: Any = None,
        db: Any = None,
        user_num_ratings: Optional[int] = None,
        hist_turns: int = 3,
        conv_id: Optional[str] = None,
        uid: Optional[int] = None,
        force_target: Optional[Target] = None,
        router_k_user: int = 2,
    ) -> AgentResult:
        """
        Execute one chat turn through the multi-agent system.
        
        Args:
            history: Rolling conversation history [{u,a}, ...]
            user_text: Current user input
            use_profile: User consent for profile access
            current_user: User object (optional)
            db: Database session (optional)
            user_num_ratings: Number of ratings for warm/cold detection
            hist_turns: Number of history turns for branch agents
            conv_id: Conversation ID (optional)
            uid: User ID (optional)
            force_target: Override routing decision (optional)
            router_k_user: Number of last user messages for router
            
        Returns:
            AgentResult with response and metadata
        """

        # 1) Route to determine target
        if force_target:
            target = force_target
            reason = "forced"
        else:
            router_input: TurnInput = make_router_input(
                history, user_text, k_user=router_k_user
            )
            route_plan = self.router.classify(router_input)
            target = route_plan.target
            reason = route_plan.reason

        # 2) Build branch input
        branch_input: TurnInput = make_branch_input(
            history=history,
            user_text=user_text,
            hist_turns=hist_turns,
            use_profile=use_profile,
            user_num_ratings=(user_num_ratings or 0),
            db=db,
            current_user=current_user,
            conv_id=conv_id,
            uid=uid,
        )

        # 3) Create agent using factory
        agent = self.factory.create_agent(
            target,
            current_user=current_user,
            db=db,
            user_num_ratings=user_num_ratings,
            use_profile=use_profile,
        )

        # 4) Convert to domain request
        request = self.adapter.turn_input_to_request(branch_input)

        # 5) Execute agent
        response = agent.execute(request)

        # 6) Convert back to legacy AgentResult
        result = self.adapter.response_to_agent_result(response)

        # Attach conductor metadata
        result.policy_version = result.policy_version or self.policy_version
        
        return result