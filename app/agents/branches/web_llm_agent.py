# app/agents/branches/web_llm_agent.py
from __future__ import annotations
from typing import Any, Dict, List
from app.agents.base import BaseLLMAgent
from app.agents.schemas import AgentResult

class WebLLMAgent(BaseLLMAgent):
    def __init__(self) -> None:
        super().__init__(
            policy_name="web.system.md",
            web=True, docs=False, internal=False,
            allowed_names={"web-search", "Wikipedia", "openlibrary-search", "openlibrary-work"},
            llm_tier="large",
        )

    def finalize(self, input_text: str, raw_result: Dict[str, Any], steps: List[Dict[str, Any]]) -> AgentResult:
        text = (raw_result.get("output") or "").strip()
        calls = self._steps_to_tool_calls(steps)
        return AgentResult(
            target="web",
            text=text,
            success=bool(text),
            tool_calls=calls,
            policy_version=self.policy_version,
        )
