# app/agents/base.py (updated sections)
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union
import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages

from app.agents.prompts.loader import read_prompt
from app.agents.settings import get_llm
from app.agents.tools.registry import ToolRegistry, InternalToolGates
from app.agents.tools.help import SiteHelpToolkit
from app.agents.logging import get_logger, capture_agent_console_and_httpx, LogCallbackHandler, append_chatbot_log
from langchain_core.callbacks import StdOutCallbackHandler

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
        final_answer = read_prompt("final_answer.system.md")

        if docs_manifest:
            manifest_tool = next((t for t in SiteHelpToolkit.as_tools() if t.name == "help-manifest"), None)
            manifest_json = manifest_tool.func("") if manifest_tool else "{}"
            manifest_lines = _render_manifest_lines(manifest_json) or "(no docs found)"

            placeholder = "[BEGIN_MANIFEST]\n… manifest content is injected here …\n[END_MANIFEST]"
            injected_section = f"[BEGIN_MANIFEST]\n{manifest_lines}\n[END_MANIFEST]"

            if placeholder in policy:
                policy = policy.replace(placeholder, injected_section)
            else:
                # Fallback: append a visible section at the end of the policy
                policy = f"{policy}\n\nAvailable documentation (aliases):\n{manifest_lines}"

        system = f"{persona}\n\n{policy}\n\n{final_answer}".strip()
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

        # Keep a human-readable tools block and names for prompt rendering + logging
        self._tool_str = "\n".join([f"{t.name}: {getattr(t, 'description', '')}" for t in tools])
        self._tool_names = ", ".join([t.name for t in tools])

        # Enhanced callbacks for better logging
        self._callbacks = [
            LogCallbackHandler(f"agent.{policy_name}"),
        ]
        
        # Add stdout callback only if LOG_VERBOSE is enabled
        if os.getenv("LOG_VERBOSE", "").lower() in ("1", "true", "yes"):
            self._callbacks.append(StdOutCallbackHandler())

        # 1) Flatten history to a single text block (like legacy)
        def _history_to_text(hist_list):
            lines = []
            for t in (hist_list or []):
                u = (t.get("u") or "").strip()
                a = (t.get("a") or "").strip()
                if u:
                    lines.append(f"User: {u}")
                if a:
                    lines.append(f"Assistant: {a}")
            return "\n".join(lines)

        self._history_to_text = _history_to_text  # stash helper on the instance

        # 2) Build a *text* PromptTemplate with a string scratchpad
        template = (
            f"{system}\n\n"
            "You can use the following tools:\n{tools}\n"
            "Tool names: {tool_names}\n"
            "When a tool is helpful, call it. If not, answer directly.\n\n"
            "Conversation so far:\n{history}\n\n"
            "Question: {input}\n\n"
            "{agent_scratchpad}"
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=["tools", "tool_names", "history", "input", "agent_scratchpad"],
        )

        self._prompt = prompt

        agent = create_react_agent(self.llm, tools, prompt)

        # 3) Plain AgentExecutor (no scratchpad formatter needed for string pad)
        callbacks = self._callbacks
        self.exec = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,  # This enables the Thought/Action/Observation logging
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            callbacks=callbacks,
            max_iterations=10,  # Prevent infinite loops
            max_execution_time=60,  # Timeout after 60 seconds
        )

    # --- Utilities ------------------------------------------------------------

    @staticmethod
    def _serialize_steps(steps: Iterable[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in steps or []:
            try:
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    # (AgentAction, observation) tuple format
                    action, observation = s[0], s[1]
                    tool = getattr(action, "tool", None) or getattr(action, "tool_name", None)
                    inp = getattr(action, "tool_input", None)
                    log = getattr(action, "log", None)
                    out.append({
                        "tool": tool,
                        "input": inp,
                        "log": log,
                        "observation": str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation)
                    })
                elif isinstance(s, dict):
                    # Dict format
                    tool = s.get("tool") or s.get("tool_name")
                    inp = s.get("tool_input")
                    log = s.get("log")
                    obs = s.get("observation", "")
                    out.append({"tool": tool, "input": inp, "log": log, "observation": str(obs)})
                else:
                    # Fallback for unknown formats
                    out.append({"tool": "unknown", "input": str(s), "log": "", "observation": ""})
            except Exception as e:
                out.append({"tool": "error", "input": str(e), "log": "", "observation": ""})
        return out

    @staticmethod
    def _steps_to_tool_calls(steps: Iterable[Dict[str, Any]]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for s in steps or []:
            name = s.get("tool") or ""
            args = s.get("input") if isinstance(s.get("input"), dict) else {"input": s.get("input")}
            calls.append(ToolCall(name=name, args=args))
        return calls

    @staticmethod
    def _to_history_msgs(hist: List[Dict[str, str]]) -> List[Any]:
        msgs = []
        for t in hist or []:
            u = (t.get("u") or "").strip()
            a = (t.get("a") or "").strip()
            if u:
                msgs.append(HumanMessage(content=u))
            if a:
                msgs.append(AIMessage(content=a))
        return msgs

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
        if isinstance(inp, str):
            text = inp
            hist_text = ""
        else:
            text = inp.user_text
            # Convert rolling history to the legacy text block for the prompt slot
            hist_text = self._history_to_text(inp.full_history)

        # Log the agent execution start
        append_chatbot_log(f"\n=== {self.policy_name.upper()} AGENT START ===")
        append_chatbot_log(f"User Input: {text}")
        if hist_text:
            append_chatbot_log(f"History: {hist_text}")

        # Optionally log the fully rendered prompt (enable with LOG_PROMPT=1)
        if self._prompt is not None and os.getenv("LOG_PROMPT", "").lower() in ("1", "true", "yes"):
            try:
                rendered = self._prompt.format(
                    tools=self._tool_str,
                    tool_names=self._tool_names,
                    history=hist_text,
                    input=text or "",
                    agent_scratchpad="",
                )
                append_chatbot_log(f"[FULL_PROMPT]\n{rendered}\n[/FULL_PROMPT]")
            except Exception as e:
                append_chatbot_log(f"[PROMPT_ERROR] {e}")

        # Capture verbose console + library logs into logs/chatbot.log
        try:
            with capture_agent_console_and_httpx():
                result = self.exec.invoke(
                    {"input": text or "", "history": hist_text},
                    config={"callbacks": self._callbacks}
                )
        except Exception as e:
            append_chatbot_log(f"[AGENT_ERROR] {str(e)}")
            # Return error result
            return AgentResult(
                target="respond",
                text=f"I encountered an error while processing your request: {str(e)}",
                success=False,
                tool_calls=[],
                policy_version=self.policy_version,
            )

        steps = self._serialize_steps(result.get("intermediate_steps"))
        
        # Log the agent execution end
        append_chatbot_log(f"Agent Output: {result.get('output', '')}")
        append_chatbot_log(f"Steps Count: {len(steps)}")
        append_chatbot_log(f"=== {self.policy_name.upper()} AGENT END ===\n")

        return self.finalize(text or "", result, steps)