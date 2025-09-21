from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from app.agents.prompts.loader import read_prompt
from app.agents.tools.registry import ToolRegistry
from app.agents.tools.help import SiteHelpToolkit
from app.agents.runtime import get_llm


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


def _render_manifest_lines(manifest_json: str) -> str:
    """
    Renders a compact manifest block for the prompt from a JSON string.
    Each line follows: alias | title | 3–5 keywords or a short one-liner if present.
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


def _build_docs_system_prompt() -> str:
    """
    Builds the Docs agent system prompt by concatenating the shared persona with the
    docs policy prompt and injecting a compact inline manifest between markers.
    The manifest is fetched once at construction via the existing help-manifest tool,
    but the agent itself will only have access to help-read at runtime.
    """
    persona = read_prompt("persona.system.md")
    policy = read_prompt("docs.system.md")
    manifest_tool = [t for t in SiteHelpToolkit.as_tools() if t.name == "help-manifest"][0]
    manifest_json = manifest_tool.func("")
    manifest_lines = _render_manifest_lines(manifest_json)
    system = policy.replace(
        "[BEGIN_MANIFEST]\n… manifest content is injected here at runtime (alias | title | short keywords/one-liner per entry) …\n[END_MANIFEST]",
        f"[BEGIN_MANIFEST]\n{manifest_lines}\n[END_MANIFEST]"
    )
    return f"{persona}\n\n{system}".strip()


class DocsLLMAgent:
    """
    LLM-driven Docs agent that consults internal help documents.
    The agent sees an inline manifest in its system prompt and may call the docs reader
    multiple times in one turn to gather sufficient coverage before answering.
    Only the docs reader tool is exposed to this agent.
    """

    def __init__(self) -> None:
        """
        Initializes the agent with a restricted tool surface (docs reader only),
        composes the system prompt with inline manifest, and prepares a ReAct executor.
        """
        registry = ToolRegistry(web=False, help=True, gates=None)
        all_tools: List[Tool] = registry.get_tools()
        self.tools: List[Tool] = [t for t in all_tools if t.name == "help-read"]
        self.llm = get_llm()
        system = _build_docs_system_prompt()
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
        Executes a single Docs query against the LLM agent.
        Returns a dict with 'text' and 'intermediate_steps' suitable for the existing route.
        """
        result = self.exec.invoke({"input": text or ""})
        steps = _serialize_steps(result.get("intermediate_steps"))
        return {"text": result.get("output", ""), "intermediate_steps": steps}
