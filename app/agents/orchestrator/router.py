# app/agents/orchestrator/router.py
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from app.agents.logging import capture_agent_console_and_httpx
from app.agents.prompts.loader import read_prompt
from app.agents.schemas import RoutePlan, TurnInput
from app.agents.settings import get_llm

ALLOWED_TARGETS = {"recsys", "web", "docs", "respond"}


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

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """
        Loads the router system prompt and obtains an LLM instance.

        Args:
                llm_client: Optional LLM client for dependency injection (testing)
        """
        self.system_prompt = read_prompt("router.system.md")
        self.llm = (
            llm_client
            if llm_client is not None
            else get_llm(tier="medium", json_mode=True, temperature=0, timeout=15)
        )

    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends messages to the LLM and returns the assistant string content.
        Supports LangChain-style .invoke or a callable returning a response object or string.
        """
        llm = self.llm
        with capture_agent_console_and_httpx():
            if hasattr(llm, "ainvoke"):
                resp = await llm.ainvoke(messages)
            elif hasattr(llm, "invoke"):
                resp = await asyncio.to_thread(llm.invoke, messages)
            elif callable(llm):
                resp = llm(messages)
            else:
                raise RuntimeError("LLM object is not invokable")

        if isinstance(resp, str):
            return resp
        content = getattr(resp, "content", None)  # LangChain BaseMessage
        if isinstance(content, str):
            return content
        if isinstance(resp, dict):
            # Fallbacks for odd providers
            for k in ("content", "text", "message"):
                if isinstance(resp.get(k), str):
                    return resp[k]
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

    async def _repair_retry(self, user_text: str, prior_output: str) -> Optional[RoutePlan]:
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
        content = await self._chat(messages)
        obj = _extract_first_json_block(content)
        return self._validate(obj) if obj is not None else None

    async def classify(self, inp: TurnInput) -> RoutePlan:
        """
        Routes a single user message to one target branch.
        Accepts TurnInput for a uniform contract.
        """
        user_text = (inp.user_text or "").strip()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        content = await self._chat(messages)
        obj = _extract_first_json_block(content)
        plan = self._validate(obj) if obj is not None else None
        if plan is not None:
            return plan
        repaired = await self._repair_retry(user_text, content or "")
        if repaired is not None:
            return repaired
        return RoutePlan(target="respond", reason="router: parse-failed-after-retry")
