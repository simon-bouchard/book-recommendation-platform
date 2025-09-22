from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from app.agents.prompts.loader import read_prompt
from app.agents.settings import get_llm
from app.agents.tools.registry import ToolRegistry, InternalToolGates
from app.agents.tools.help import SiteHelpToolkit

from app.agents.schemas import (
    AgentResult,
    ToolCall,
    TurnInput,
)

def _render_manifest_lines(manifest_json: str) -> str:
    try:
        import json
        data = json.loads(manifest_json or "{}")
    except Exception:
        data = {}
    items = data.get("items") or []
    lines = []
    for it in items:
        title = it.get("title", "").strip()
        desc = it.get("desc", "").strip()
        if title:
            lines.append(f"- {title}: {desc}")
    return "\n".join(lines)

class BaseLLMAgent:
    """
    Shared scaffold for all LLM agents.
    Subclasses declare: policy_name, category flags, allowed_names.
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
        llm_tier: Optional[str] = None,
        allowed_names: Optional[Set[str]] = None,
        docs_manifest: bool = False,
    ) -> None:
        self.policy_name = policy_name
        self.llm = get_llm(tier=llm_tier)
        self.allowed_names = allowed_names or set()

        # Build system prompt (persona + policy), optional docs manifest injection
        persona = read_prompt("persona.system.md")
        policy = read_prompt(policy_name)
        if docs_manifest:
            manifest_tool = next((t for t in SiteHelpToolkit.as_tools() if t.name == "help-manifest"), None)
            manifest_json = manifest_tool.func("") if manifest_tool else "{}"
            policy = policy.replace(
                "[BEGIN_MANIFEST]\n… manifest content is injected here …\n[END_MANIFEST]",
                f"[BEGIN_MANIFEST]\n{_render_manifest_lines(manifest_json)}\n[END_MANIFEST]"
            )
        system = f"{persona}\n\n{policy}".strip()
        self.policy_version = f"{policy_name}"

        # Build tools via registry and filter to allowed_names
        self.registry = ToolRegistry(
            web=web,
            docs=docs,
            internal=internal,
            gates=gates or InternalToolGates(),
            ctx_user=ctx_user,
            ctx_db=ctx_db,
        )
        tools: List[Tool] = []
        for t in self.registry.get_tools():
            if not self.allowed_names or t.name in self.allowed_names:
                tools.append(t)

        # Build ReAct prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                ("human", "{input}"),
            ]
        )
        agent = create_react_agent(self.llm, tools, prompt)
        self.exec: AgentExecutor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # --- Utilities ------------------------------------------------------------

    @staticmethod
    def _serialize_steps(steps: Iterable[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in steps or []:
            try:
                tool = s.get("tool") if isinstance(s, dict) else getattr(s, "tool", None)
                inp = s.get("tool_input") if isinstance(s, dict) else getattr(s, "tool_input", None)
                log = s.get("log") if isinstance(s, dict) else getattr(s, "log", None)
                out.append({"tool": tool, "input": inp, "log": log})
            except Exception:
                continue
        return out

    @staticmethod
    def _steps_to_tool_calls(steps: Iterable[Dict[str, Any]]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for s in steps or []:
            name = s.get("tool") or ""
            args = s.get("input") if isinstance(s.get("input"), dict) else {"input": s.get("input")}
            calls.append(ToolCall(name=name, args=args))
        return calls

    # --- Overridables ---------------------------------------------------------

    def finalize(self, input_text: str, raw_result: Dict[str, Any], steps: List[Dict[str, Any]]) -> AgentResult:
        """
        Default finalize: return the model text + serialized steps.
        Recsys can override to enforce a one-shot repair.
        """
        text = (raw_result.get("output") or "").strip()
        calls = self._steps_to_tool_calls(steps)
        return AgentResult(
            target="respond",  # subclasses should overwrite in run()
            text=text,
            success=bool(text),
            tool_calls=calls,
            policy_version=self.policy_version,
        )

    # --- Public API -----------------------------------------------------------

    def run(self, inp: Union[str, TurnInput]) -> AgentResult:
        """
        Execute a single turn. Accepts either a raw string (for compatibility)
        or a TurnInput. Returns AgentResult.
        """
        if isinstance(inp, str):
            text = inp
        else:
            text = inp.user_text

        result = self.exec.invoke({"input": text or ""})
        steps = self._serialize_steps(result.get("intermediate_steps"))
        return self.finalize(text or "", result, steps)
