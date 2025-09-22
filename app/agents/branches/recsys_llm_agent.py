from __future__ import annotations
from typing import Any, Dict, Optional

from app.agents.base import BaseLLMAgent
from app.agents.tools.registry import InternalToolGates
from app.agents.settings import settings

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

    def finalize(self, input_text: str, raw_result: Dict[str, Any], steps: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        If the model failed to call return_book_ids, append a direct instruction
        and retry once. Otherwise return result as-is.
        """
        if any(s.get("tool") == "return_book_ids" for s in steps):
            return {"text": raw_result.get("output", ""), "intermediate_steps": steps}
        retry_input = (
            f"{input_text}\n\nImportant: finalize now by calling return_book_ids with 4–12 curated IDs."
        )
        retry = self.exec.invoke({"input": retry_input})
        retry_steps = self._serialize_steps(retry.get("intermediate_steps"))
        return {"text": retry.get("output", ""), "intermediate_steps": retry_steps}

    def run(self, composed: str) -> Dict[str, Any]:
        return super().run(composed)
