from __future__ import annotations
from typing import Any, Dict

from app.agents.base import BaseLLMAgent


class DocsLLMAgent(BaseLLMAgent):
    """
    LLM-driven Docs agent that consults internal help documents.
    System prompt includes an inline manifest; only the docs reader tool is exposed.
    """

    def __init__(self) -> None:
        super().__init__(
            policy_name="docs.system.md",
            web=False,
            docs=True,
            internal=False,
            allowed_names={"help-read"},
            docs_manifest=True,  # inject manifest into prompt once
        )

    def run(self, text: str) -> Dict[str, Any]:
        return super().run(text)
