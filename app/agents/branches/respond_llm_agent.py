from __future__ import annotations
from typing import Any
from langchain.prompts import ChatPromptTemplate
from app.agents.prompts.loader import read_prompt
from app.agents.settings import get_llm
from app.agents.schemas import AgentResult
from app.agents.logging import capture_agent_console_and_httpx

class RespondLLMAgent:
    """
    Tool-less responder that uses ONLY the shared persona as the system prompt.
    - No registry, no tools, no agent_scratchpad.
    - Returns AgentResult (schema v1) for consistency across branches.
    """

    def __init__(self) -> None:
        persona = read_prompt("persona.system.md")  # same persona as other branches
        self.llm = get_llm(tier="medium")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", persona),
            ("user", "{input}"),
        ])
        self.policy_version = "respond.v1"

    def run(self, composed_input: str) -> AgentResult:
        msg = self.prompt.format_messages(input=composed_input or "")
        with capture_agent_console_and_httpx():
            out = self.llm.invoke(msg)

        text = getattr(out, "content", str(out)) or ""
        return AgentResult(
            target="respond",
            text=text.strip(),
            success=bool(text.strip()),
            policy_version=self.policy_version,
        )
