from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.agents.prompts.loader import read_prompt
from app.agents.runtime import get_llm


ALLOWED_TARGETS = {"recsys", "web", "docs", "respond"}


@dataclass(frozen=True)
class RoutePlan:
    """
    Final router decision returned to the Conductor.
    """
    target: str
    reason: str


def _extract_first_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses the first JSON object found in a string.
    Handles accidental code fences and leading/trailing prose.
    Returns a dict if parsing succeeds, otherwise None.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class RouterLLM:
    """
    LLM-based router that selects one of: recsys | web | docs | respond.
    The model sees only the system router prompt and the raw user text.
    It never calls tools and never browses.
    """

    def __init__(self) -> None:
        """
        Loads the router system prompt and obtains an LLM instance from the runtime.
        """
        self.system_prompt = read_prompt("router.system.md")
        self.llm = get_llm(tier="small", json_mode=True, temperature=0, timeout=15)

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends messages to the LLM and returns the assistant string content.
        Supports LangChain-style .invoke or a callable returning a response object or string.
        """
        llm = self.llm
        resp = None
        if hasattr(llm, "invoke"):
            resp = llm.invoke(messages)
        elif callable(llm):
            resp = llm(messages)
        else:
            raise RuntimeError("LLM object is not invokable")
        if hasattr(resp, "content"):
            return str(resp.content)
        return str(resp)

    def _validate(self, obj: Dict[str, Any]) -> Optional[RoutePlan]:
        """
        Validates a parsed JSON object and returns a RoutePlan if valid.
        Ensures target is in the allowed set and normalizes the reason.
        """
        if not isinstance(obj, dict):
            return None
        target = str(obj.get("target", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
        if target not in ALLOWED_TARGETS:
            return None
        if not reason:
            reason = "router: no reason provided"
        return RoutePlan(target=target, reason=reason)

    def _repair_retry(self, user_text: str, prior_output: str) -> Optional[RoutePlan]:
        """
        Performs a single repair retry focused on output formatting.
        Asks the model to return strict JSON with only the required fields.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text or ""},
            {
                "role": "user",
                "content": (
                    "Repair your previous output to be STRICT JSON with only keys "
                    '{"target": one of ["recsys","web","docs","respond"], "reason": "<short rationale>"} '
                    "Do not add code fences or any extra text. Do not change your decision—only fix formatting."
                ),
            },
            {"role": "assistant", "content": prior_output or ""},
        ]
        content = self._chat(messages)
        obj = _extract_first_json_block(content)
        return self._validate(obj) if obj is not None else None

    def classify(self, text: str, user_num_ratings: Optional[int] = None) -> RoutePlan:
        """
        Routes a single user message to one target branch.
        Uses a primary LLM call and, if parsing fails, a single repair retry.
        Returns 'respond' with a parse-failed reason if both attempts fail.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text or ""},
        ]
        content = self._chat(messages)
        obj = _extract_first_json_block(content)
        plan = self._validate(obj) if obj is not None else None
        if plan is not None:
            return plan
        repaired = self._repair_retry(text or "", content or "")
        if repaired is not None:
            return repaired
        return RoutePlan(target="respond", reason="router: parse-failed-after-retry")
