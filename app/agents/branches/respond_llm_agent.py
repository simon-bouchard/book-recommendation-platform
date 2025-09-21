from __future__ import annotations
from typing import Any, Dict
from langchain.prompts import ChatPromptTemplate
from app.agents.prompts.loader import read_prompt
from app.agents.settings import get_llm

class RespondLLMAgent:
    """
    Tool-less responder that uses ONLY the shared persona as the system prompt.
    - No registry, no tools, no agent_scratchpad.
    - Returns the same contract: {'text', 'intermediate_steps'} (empty steps).
    """

    def __init__(self) -> None:
        persona = read_prompt("persona.system.md")  # same persona as other branches
        self.llm = get_llm(tier="medium")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", persona),
            ("user", "{input}"),
        ])

    def run(self, composed_input: str) -> Dict[str, Any]:
        msg = self.prompt.format_messages(input=composed_input or "")
        out = self.llm.invoke(msg)
        text = getattr(out, "content", str(out))
        return {"text": text, "intermediate_steps": []}
