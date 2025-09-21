# app/agents/orchestrator/conductor.py
from __future__ import annotations
from typing import Any, Dict, Optional

# PR-1: delegate mode (no behavior change)
# We call the existing legacy agent exactly as before.
from app.agents.branches.legacy_single_agent import answer as legacy_answer  # do not move this in PR-1

class Conductor:
    """
    PR-1 scaffolding: thin wrapper that delegates to the legacy single agent.
    Signature and return format are preserved 1:1 so downstream logic is unchanged.
    """

    def __init__(self) -> None:
        # In PR-2 we'll add router, budgets, and tracing here.
        pass

    def run(
        self,
        question: str,
        current_user=None,
        db=None,
        user_num_ratings: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Delegate to the existing web_agent.answer() with the exact signature:
        def answer(question: str, current_user=None, db=None, user_num_ratings: Optional[int] = None) -> Dict[str, Any]
        Returns {"text": str, "intermediate_steps": list}.
        """
        return legacy_answer(
            question,
            current_user=current_user,
            db=db,
            user_num_ratings=user_num_ratings,
        )
