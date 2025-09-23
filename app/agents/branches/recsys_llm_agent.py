from __future__ import annotations
from typing import Any, Dict, Optional, List
import json

from app.agents.base import BaseLLMAgent
from app.agents.tools.registry import InternalToolGates
from app.agents.schemas import AgentResult

class RecsysLLMAgent(BaseLLMAgent):
    """
    LLM-driven recommendations agent that operates over the internal catalog.
    Loads internal recommendation tools, enforces policy via the recsys prompt,
    and guarantees at most one repair attempt to finalize with return_book_ids.
    """

    def __init__(
        self,
        current_user: Any = None,
        db: Any = None,
        user_num_ratings: Optional[int] = None,
        warm_threshold: int = 10,
        allow_profile: bool = False,
    ) -> None:
        gates = InternalToolGates(
            user_num_ratings=user_num_ratings,
            warm_threshold=warm_threshold,
            profile_allowed=bool(allow_profile),
        )
        super().__init__(
            policy_name="recsys.system.md",
            web=False,
            docs=False,
            internal=True,
            gates=gates,
            ctx_user=current_user,
            ctx_db=db,
            llm_tier="large",
            allowed_names={
                "als_recs",
                "subject_hybrid_pool",
                "subject_id_search",
                "book_semantic_search",
                "return_book_ids",
                "user-profile", 
                "recent-interactions"
            },
        )

    @staticmethod
    def _extract_return_book_ids(steps: List[Dict[str, Any]]) -> List[int]:
        """
        Look for the most recent `return_book_ids` tool call and extract integer IDs.
        Accepts inputs as: {"book_ids": [...]}, or a bare list, or a JSON string.
        """
        if not steps:
            return []
        for s in reversed(steps):
            if (s or {}).get("tool") != "return_book_ids":
                continue
            raw = (s or {}).get("input")
            try:
                # If input is a JSON string, parse it
                if isinstance(raw, str):
                    raw = json.loads(raw)
                # If dict, prefer the 'book_ids' field
                if isinstance(raw, dict):
                    raw = raw.get("book_ids", raw)
                # Now raw should ideally be a list of ids
                if isinstance(raw, list):
                    ids = [int(x) for x in raw if str(x).strip()]
                    return ids
            except Exception:
                return []
        return []

    def finalize(self, input_text: str, raw_result: Dict[str, Any], steps: List[Dict[str, Any]]) -> AgentResult:
        """
        Finalization contract:
        - If `return_book_ids` already called, extract IDs and return.
        - Else, perform one repair attempt forcing the tool call, then extract IDs.
        """
        # First attempt: use current steps
        ids = self._extract_return_book_ids(steps)
        if ids:
            text = (raw_result.get("output") or "").strip()
            return AgentResult(
                target="recsys",
                text=text,
                success=bool(text),
                tool_calls=self._steps_to_tool_calls(steps),
                book_ids=ids,
                policy_version=self.policy_version,
            )

        # Retry once with explicit instruction to finalize
        retry = self.exec.invoke({
            "input": f"{input_text}\n\nImportant: finalize now by calling return_book_ids with 4–12 curated IDs.",
            "history": self._to_history_msgs(raw_result.get("history", []))
        })
        retry_steps = self._serialize_steps(retry.get("intermediate_steps"))
        ids = self._extract_return_book_ids(retry_steps)
        text = (retry.get("output") or "").strip()

        return AgentResult(
            target="recsys",
            text=text,
            success=bool(text),
            tool_calls=self._steps_to_tool_calls(retry_steps),
            book_ids=ids,
            policy_version=self.policy_version,
        )