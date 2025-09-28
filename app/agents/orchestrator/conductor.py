# app/agents/orchestrator/conductor.py
from __future__ import annotations
from typing import Any, Optional, List

from app.agents.schemas import AgentResult, TurnInput, Target
from app.agents.orchestrator.router import RouterLLM
from app.agents.branches.web_llm_agent import WebLLMAgent
from app.agents.branches.docs_llm_agent import DocsLLMAgent
from app.agents.branches.recsys_llm_agent import RecsysLLMAgent
from app.agents.branches.langgraph_recsys import RecsysLangGraphAgent
from app.agents.branches.respond_llm_agent import RespondLLMAgent
from app.agents.context_builder import make_router_input, make_branch_input


class Conductor:
    """
    Multi-agent conductor:
      1) Route the current user turn with RouterLLM (LLM-only, no heuristics).
      2) Build a uniform TurnInput for the chosen branch.
      3) Execute the selected branch agent and return AgentResult.
    """

    def __init__(self) -> None:
        self.router = RouterLLM()
        self.policy_version = "conductor.mas.v1"

    def _pick_agent(
        self,
        target: Target,
        *,
        current_user: Any,
        db: Any,
        user_num_ratings: Optional[int],
        use_profile: bool,
    ):
        if target == "web":
            return WebLLMAgent()
        if target == "docs":
            return DocsLLMAgent()
        if target == "recsys":
            return RecsysLangGraphAgent(
                current_user=current_user,
                db=db,
                user_num_ratings=user_num_ratings,
                warm_threshold=10,
                allow_profile=use_profile,
            )
        # default
        return RespondLLMAgent()

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
        Execute one chat turn through the MAS pipeline.
        - history: rolling [{u,a}, ...]
        - user_text: raw user input
        - use_profile: user consent toggle
        - current_user, db: contexts (may be None)
        - user_num_ratings: warm/cold signal (int or None)
        - hist_turns: branch history window
        - conv_id, uid: optional metadata for downstream tools
        - force_target: optional explicit override ("recsys"|"web"|"docs"|"respond")
        - router_k_user: number of last user messages to expose to the router
        """

        # 1) Route
        if force_target:
            target = force_target
            reason = "forced"
        else:
            router_input: TurnInput = make_router_input(history, user_text, k_user=router_k_user)
            route_plan = self.router.classify(router_input)
            target = route_plan.target  # "recsys"|"web"|"docs"|"respond"
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

        # 3) Pick and run agent
        agent = self._pick_agent(
            target,
            current_user=current_user,
            db=db,
            user_num_ratings=user_num_ratings,
            use_profile=use_profile,
        )
        result: AgentResult = agent.run(branch_input)

        # Conductor-level normalization (pass-through + attach policy hint)
        result.policy_version = result.policy_version or self.policy_version
        return result
