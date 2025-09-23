# app/agents/branches/docs_llm_agent.py
from __future__ import annotations
from typing import Any, Dict, List
from app.agents.base import BaseLLMAgent
from app.agents.schemas import AgentResult, ToolCall

class DocsLLMAgent(BaseLLMAgent):
    def __init__(self) -> None:
        super().__init__(
            policy_name="docs.system.md",
            web=False, docs=True, internal=False,
            allowed_names={"help-read"},
            docs_manifest=True,
            llm_tier="medium",
        )

    def finalize(self, input_text: str, raw_result: Dict[str, Any], steps: List[Dict[str, Any]]) -> AgentResult:
        text = (raw_result.get("output") or "").strip()
        calls = self._steps_to_tool_calls(steps)
        return AgentResult(
            target="docs",
            text=text,
            success=bool(text),
            tool_calls=calls,
            policy_version=self.policy_version,
        )
