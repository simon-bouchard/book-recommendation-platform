# app/agents/orchestrator/conductor.py
from __future__ import annotations
from typing import Any, Dict, Optional, Iterable, List

# PR-1: delegate mode (no behavior change in routing)
# We call the existing legacy agent exactly as before.
from app.agents.branches.legacy_single_agent import answer as legacy_answer  # do not move this in PR-1
from app.agents.schemas import AgentResult, ToolCall


def _steps_to_tool_calls(steps: Iterable[Dict[str, Any]]) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for s in steps or []:
        # legacy format: {"tool": str, "input" or "tool_input": Any, "log": str?}
        name = s.get("tool") or ""
        raw_input = (
            s.get("tool_input")
            if "tool_input" in s
            else (s.get("input") if "input" in s else None)
        )
        args = raw_input if isinstance(raw_input, dict) else {"input": raw_input}
        calls.append(ToolCall(name=name, args=args or {}))
    return calls


class Conductor:
    """
    PR-1 scaffolding: thin wrapper that delegates to the legacy single agent
    but normalizes the return to AgentResult so downstream code can rely on
    the new schema immediately.
    """

    def __init__(self) -> None:
        self.policy_version = "legacy.delegate.v1"

    def run(
        self,
        question: str,
        current_user=None,
        db=None,
        user_num_ratings: Optional[int] = None,
    ) -> AgentResult:
        """
        Delegate to the existing web_agent.answer() with the exact signature:
        def answer(question: str, current_user=None, db=None, user_num_ratings: Optional[int] = None)
        Legacy returns {"text": str, "intermediate_steps": list}.

        We wrap that into AgentResult(schema v1).
        """
        legacy = legacy_answer(
            question,
            current_user=current_user,
            db=db,
            user_num_ratings=user_num_ratings,
        )

        text = (legacy.get("text") or "").strip()
        steps = legacy.get("intermediate_steps") or []
        tool_calls = _steps_to_tool_calls(steps)

        return AgentResult(
            target="respond",               # legacy is a single blended agent
            text=text,
            success=bool(text),
            tool_calls=tool_calls,
            policy_version=self.policy_version,
        )
