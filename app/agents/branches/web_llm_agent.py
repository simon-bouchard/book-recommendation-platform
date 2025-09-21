from __future__ import annotations
from typing import Any, Dict

from app.agents.base import BaseLLMAgent


class WebLLMAgent(BaseLLMAgent):
    """
    LLM-driven web agent that handles factual and freshness-oriented queries.
    Exposes only external web tools and uses the web policy prompt.
    """

    def __init__(self) -> None:
        super().__init__(
            policy_name="web.system.md",
            web=True,
            docs=False,
            internal=False,
            # optional final filter — registry should already only give these
            allowed_names={"web-search", "Wikipedia", "openlibrary-search", "openlibrary-work"},
        )

    def run(self, text: str) -> Dict[str, Any]:
        return super().run(text)
