from __future__ import annotations
from typing import Any
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.agents.prompts.loader import read_prompt
from app.agents.settings import get_llm
from app.agents.schemas import AgentResult, TurnInput
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
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.policy_version = "respond.v1"

    def run(self, inp: TurnInput) -> AgentResult:
        user_text = (inp.user_text or "").strip()
        history = []
        for t in inp.full_history or []:
            if (t.get("u") or "").strip(): history.append(("human", t["u"]))
            if (t.get("a") or "").strip(): history.append(("ai", t["a"]))
        msgs = self.prompt.format_messages(input=user_text, history=history)
        with capture_agent_console_and_httpx():
            out = self.llm.invoke(msgs)
        text = (getattr(out, "content", str(out)) or "").strip()
        return AgentResult(target="respond", text=text, success=bool(text), policy_version=self.policy_version)