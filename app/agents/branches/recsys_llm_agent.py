from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from app.agents.prompts.loader import read_prompt
from app.agents.runtime import get_llm
from app.agents.tools.registry import ToolRegistry, InternalToolGates


def _serialize_steps(steps: List) -> List[Dict[str, Any]]:
    """
    Converts LangChain intermediate_steps into a JSON-serializable list of dicts.
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


class RecsysLLMAgent:
    """
    LLM-driven recommendations agent that operates over the internal catalog.
    It loads only the internal recommendation tools, enforces policy via the recsys prompt,
    and expects the model to finalize by calling return_book_ids after curating candidates.
    """

    def __init__(
        self,
        current_user: Any = None,
        db: Any = None,
        user_num_ratings: Optional[int] = None,
        warm_threshold: int = 10,
    ) -> None:
        """
        Initializes the agent with internal tools only and builds a ReAct executor
        using the shared persona and the recsys system prompt.
        """
        gates = InternalToolGates(
            internal_enabled=True,
            user_num_ratings=user_num_ratings,
            warm_threshold=warm_threshold,
        )
        registry = ToolRegistry(web=False, help=False, gates=gates, ctx_user=current_user, ctx_db=db)
        all_tools: List[Tool] = registry.get_tools()
        allowed = {
            "als_recs",
            "subject_hybrid_pool",
            "subject_id_search",
            "book_semantic_search",
            "return_book_ids",
        }
        self.tools: List[Tool] = [t for t in all_tools if t.name in allowed]
        self.llm = get_llm()
        persona = read_prompt("persona.system.md")
        policy = read_prompt("recsys.system.md")
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

    def _ensure_finalization(self, composed: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a single repair pass if the model failed to call return_book_ids.
        Appends a direct instruction to finalize and reinvokes the agent once.
        """
        steps = _serialize_steps(result.get("intermediate_steps"))
        if any(s.get("tool") == "return_book_ids" for s in steps):
            return {"text": result.get("output", ""), "intermediate_steps": steps}
        retry_input = (
            f"{composed}\n\nImportant: finalize now by calling return_book_ids with 4–12 curated IDs."
        )
        retry = self.exec.invoke({"input": retry_input})
        retry_steps = _serialize_steps(retry.get("intermediate_steps"))
        return {"text": retry.get("output", ""), "intermediate_steps": retry_steps}

    def run(self, composed: str) -> Dict[str, Any]:
        """
        Executes a single recommendation turn against the LLM agent using the composed input.
        Returns a dict with 'text' and 'intermediate_steps' suitable for the existing route.
        Guarantees at most one repair attempt to obtain return_book_ids.
        """
        result = self.exec.invoke({"input": composed or ""})
        return self._ensure_finalization(composed or "", result)
