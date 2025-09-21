# app/agents/common/base.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from app.agents.prompts.loader import read_prompt
from app.agents.settings import get_llm
from app.agents.tools.registry import ToolRegistry, InternalToolGates
from app.agents.tools.help import SiteHelpToolkit  # used only when docs_manifest=True


def _render_manifest_lines(manifest_json: str) -> str:
    """
    Renders a compact manifest block for the prompt from a JSON string.
    Each line: alias | title | 3–5 keywords OR one-liner if keywords absent.
    """
    try:
        data = json.loads(manifest_json or "{}")
    except Exception:
        return ""
    lines: List[str] = []
    if isinstance(data, dict):
        for alias, meta in data.items():
            title = str(meta.get("title", alias)).strip()
            if isinstance(meta.get("keywords"), list) and meta.get("keywords"):
                kws = ", ".join(map(str, meta.get("keywords")))
                lines.append(f"{alias} | {title} | {kws}")
            else:
                desc = str(meta.get("description", "")).strip()
                hint = desc if desc else title
                lines.append(f"{alias} | {hint}")
    return "\n".join(lines)


class BaseLLMAgent:
    """
    Shared LLM agent scaffold:
      - builds tools via ToolRegistry(web/docs/internal + gates),
      - composes persona + policy system prompt (optional docs manifest injection),
      - creates a ReAct agent with `agent_scratchpad`,
      - returns {'text', 'intermediate_steps'}.
    Subclasses may override `finalize()` to perform a single post-pass (e.g., recsys).
    """

    def __init__(
        self,
        *,
        policy_name: str,
        web: bool = False,
        docs: bool = False,
        internal: bool = False,
        gates: Optional[InternalToolGates] = None,
        ctx_user: Any = None,
        ctx_db: Any = None,
        allowed_names: Optional[Iterable[str]] = None,
        docs_manifest: bool = False,
        llm_model: Optional[str] = None,
        llm_tier: Optional[str] = None,
        llm_temperature: float = 0.0,
        llm_timeout: int = 30,
        llm_json_mode: bool = False,
        llm_max_tokens: Optional[int] = None,
    ) -> None:

        # 1) Build toolset via the registry (registry is the single source of truth)
        self.registry = ToolRegistry(
            web=web,
            docs=docs,
            internal=internal,
            gates=gates,
            ctx_user=ctx_user,
            ctx_db=ctx_db,
        )
        tools = self.registry.get_tools()
        if allowed_names:
            allow: Set[str] = set(allowed_names)
            tools = [t for t in tools if t.name in allow]
        self.tools: List[Tool] = list(tools)

        # 2) Compose system prompt (persona + policy), with optional docs manifest injection
        persona = read_prompt("persona.system.md")
        policy = read_prompt(policy_name)

        if docs_manifest:
            # Fetch manifest once at construction time; at runtime only the docs reader should be exposed
            manifest_tool = next((t for t in SiteHelpToolkit.as_tools() if t.name == "help-manifest"), None)
            manifest_json = manifest_tool.func("") if manifest_tool else "{}"
            manifest_lines = _render_manifest_lines(manifest_json)
            policy = policy.replace(
                "[BEGIN_MANIFEST]\n… manifest content is injected here at runtime (alias | title | short keywords/one-liner per entry) …\n[END_MANIFEST]",
                f"[BEGIN_MANIFEST]\n{manifest_lines}\n[END_MANIFEST]"
            )

        system = f"{persona}\n\n{policy}".strip()

        # 3) Build ReAct prompt + executor (consistent across agents)
        self.llm = get_llm(
            model=llm_model,
            tier=llm_tier,
            temperature=llm_temperature,
            timeout=llm_timeout,
            json_mode=llm_json_mode,
            max_tokens=llm_max_tokens,
        )

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

    @staticmethod
    def _serialize_steps(steps: Sequence) -> List[Dict[str, Any]]:
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

    # --- Hook for subclasses (default: no-op) ---------------------------------

    def finalize(self, input_text: str, raw_result: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optional single-pass post-processing. Default returns raw output unchanged.
        Recsys can override to perform one repair retry (e.g., call `return_book_ids` once).
        """
        return {"text": raw_result.get("output", ""), "intermediate_steps": steps}

    # --- Public API -----------------------------------------------------------

    def run(self, text: str) -> Dict[str, Any]:
        """
        Executes a single turn with the prepared ReAct agent.
        Returns a dict with 'text' and 'intermediate_steps'.
        """
        result = self.exec.invoke({"input": text or ""})
        steps = self._serialize_steps(result.get("intermediate_steps"))
        return self.finalize(text or "", result, steps)
