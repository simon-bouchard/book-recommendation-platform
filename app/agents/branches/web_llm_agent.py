from __future__ import annotations

from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from app.agents.prompts.loader import read_prompt
from app.agents.tools.registry import ToolRegistry
from app.agents.runtime import get_llm


def _serialize_steps(steps: List) -> List[Dict[str, Any]]:
    """
    Convert LangChain intermediate_steps into a JSON-serializable list of dicts.
    Each entry contains the tool name, the tool input, and the observed output.
    """
    out: List[Dict[str, Any]] = []
    for (tool_invocation, observation) in steps or []:
        try:
            out.append({
                "tool": getattr(tool_invocation, "tool", str(tool_invocation)),
                "input": getattr(tool_invocation, "tool_input", None),
                "output": observation,
            })
        except Exception:
            out.append({"tool": str(tool_invocation), "output": str(observation)})
    return out


class WebLLMAgent:
    """
    LLM-driven web agent that handles factual and freshness-oriented queries.
    It exposes only external web tools and composes a ReAct agent with the shared persona
    and the web policy prompt.
    """

    def __init__(self) -> None:
        """
        Initialize the agent with web tools only and build a ReAct executor using
        the shared persona and the web system prompt.
        """
        registry = ToolRegistry(web=True, help=False, internal=False)
        all_tools: List[Tool] = registry.get_tools()
        allowed = {"web-search", "Wikipedia", "openlibrary-search", "openlibrary-work"}
        self.tools: List[Tool] = [t for t in all_tools if t.name in allowed]
        self.llm = get_llm()
        persona = read_prompt("persona.system.md")
        policy = read_prompt("web.system.md")
        system = f"{persona}\n\n{policy}".strip()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.exec = AgentExecutor(
            agent=agent,
            tools=self.tools,
            return_intermediate_steps=True,
        )

    def run(self, text: str) -> Dict[str, Any]:
        """
        Execute a single web query against the LLM agent.
        Returns a dict with 'text' and 'intermediate_steps' suitable for the existing route.
        """
        result = self.exec.invoke({"input": text or ""})
        steps = _serialize_steps(result.get("intermediate_steps"))
        return {"text": result.get("output", ""), "intermediate_steps": steps}
