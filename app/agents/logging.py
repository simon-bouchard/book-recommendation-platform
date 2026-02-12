# app/agents/logging.py
import logging
import os, json
from datetime import datetime
from typing import Any, Dict, Union
import io
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Dict, Any, Tuple, List, Optional, Union
from logging.handlers import RotatingFileHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue

# -------- Normal app logger (unchanged) --------
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_FMT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_root = logging.getLogger()
if not _root.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    _root.addHandler(h)
    _root.setLevel(_LOG_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger with unified formatting."""
    return logging.getLogger(name)


# -------- Dedicated chatbot raw log sink (verbatim text) --------
_LOG_DIR = os.path.join(os.getcwd(), "logs")
_CHATBOT_LOG = os.path.join(_LOG_DIR, "chatbot.log")


def _ensure_log_dir() -> None:
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
    except Exception:
        # fallback to CWD if mkdir fails
        global _CHATBOT_LOG
        _CHATBOT_LOG = os.path.join(os.getcwd(), "chatbot.log")


def append_chatbot_log(text: str) -> None:
    """Append raw text to the chatbot log (no formatting, no truncation)."""
    if not text:
        return
    _ensure_log_dir()
    try:
        with open(_CHATBOT_LOG, "a", encoding="utf-8") as f:
            # ensure trailing newline if missing
            if not text.endswith("\n"):
                f.write(text + "\n")
            else:
                f.write(text)
    except Exception as e:
        # Fallback to stderr if file write fails
        print(f"Failed to write to chatbot log: {e}", file=sys.stderr)


# Optional rotating handler (not used by the capture context, but you can tail it too)
def _setup_chatbot_rotating_handler() -> logging.Logger:
    logger = logging.getLogger("chatbot")
    if not logger.handlers:
        _ensure_log_dir()
        fh = RotatingFileHandler(
            _CHATBOT_LOG, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# Expose if you want a classic Logger API elsewhere
chatbot_logger = _setup_chatbot_rotating_handler()

# -------- Capture the exact verbose stream from Agent + httpx into chatbot.log --------
_LOGGER_NAMES_TO_CAPTURE: List[str] = [
    "langchain",
    "langchain_core",
    "langchain_community",
    "httpx",
    "openai",
    "urllib3",
    "agent",
    "agent.input",
    "chatbot",
]


def _should_capture_logger(logger_name: str) -> bool:
    """Check if we should capture logs from this logger based on config."""
    # Always capture errors
    if os.getenv("LOG_LEVEL", "INFO").upper() == "ERROR":
        return False

    # HTTP-related loggers
    if logger_name in ["httpx", "openai", "urllib3"]:
        return os.getenv("LOG_HTTP", "1").lower() in ("1", "true", "yes")

    # LangChain loggers
    if logger_name.startswith("langchain"):
        return os.getenv("LOG_CHAINS", "1").lower() in ("1", "true", "yes")

    # Default: capture if not silent mode
    return True


@contextmanager
def capture_agent_console_and_httpx():
    """
    Capture EVERYTHING the agent would normally print with verbose=True:
      - stdout / stderr (chain banners, Thought/Action/Observation, Finished chain)
      - logging from 'langchain', 'langchain_core', and 'httpx' at INFO and above
    Nothing leaks into normal app logs during capture. On exit, the buffered
    text is appended verbatim to logs/chatbot.log.
    """
    # Skip capture entirely in silent mode
    if os.getenv("LOG_LEVEL", "INFO").upper() == "ERROR":
        yield
        return

    buf = io.StringIO()

    # Prepare a formatter that matches your normal console format (so httpx lines look identical)
    fmt = logging.Formatter(_FMT, _DATEFMT)

    # Save and replace logger state for target libraries
    saved: Dict[str, Tuple[int, List[logging.Handler], bool]] = {}
    handlers: Dict[str, logging.Handler] = {}

    try:
        for name in _LOGGER_NAMES_TO_CAPTURE:
            if not _should_capture_logger(name):
                continue

            lg = logging.getLogger(name)
            saved[name] = (lg.level, list(lg.handlers), lg.propagate)

            # Replace handlers with one that writes into our buffer
            stream_handler = logging.StreamHandler(buf)
            stream_handler.setLevel(logging.DEBUG)  # Capture more detail
            stream_handler.setFormatter(fmt)

            lg.handlers = [stream_handler]
            lg.setLevel(logging.DEBUG)  # Lower threshold for more detail
            lg.propagate = False
            handlers[name] = stream_handler

        # Redirect stdout/stderr so AgentExecutor(verbose=True) banners & traces are captured
        # But only if verbose logging is enabled
        if os.getenv("LOG_VERBOSE", "1").lower() in ("1", "true", "yes"):
            with redirect_stdout(buf), redirect_stderr(buf):
                yield
        else:
            yield

    finally:
        # Restore all logger states
        for name, (level, old_handlers, propagate) in saved.items():
            lg = logging.getLogger(name)
            try:
                lg.handlers = old_handlers
                lg.setLevel(level)
                lg.propagate = propagate
            except Exception:
                pass

        # Flush buffer and append verbatim to chatbot log
        try:
            content = buf.getvalue()
            if content.strip():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_chatbot_log(f"\n=== AGENT EXECUTION {timestamp} ===")
                append_chatbot_log(content)
                append_chatbot_log("=== END AGENT EXECUTION ===\n")
        except Exception:
            # As a last resort, at least try to write something minimal
            try:
                append_chatbot_log("[chatbot-log] <failed to capture buffer>\n")
            except Exception:
                pass


class LogCallbackHandler(BaseCallbackHandler):
    """Enhanced callback handler that logs detailed ReAct execution steps."""

    def __init__(self, name: str = "agent"):
        super().__init__()
        self.name = name

    def _should_log_component(self, component: str) -> bool:
        """Check if a specific component should be logged based on env vars."""
        component_map = {
            "tools": "LOG_TOOLS",
            "chains": "LOG_CHAINS",
            "http": "LOG_HTTP",
            "verbose": "LOG_VERBOSE",
        }

        env_var = component_map.get(component.lower())
        if not env_var:
            return True  # Default to logging unknown components

        return os.getenv(env_var, "1").lower() in ("1", "true", "yes")

    def _log_event(self, event_type: str, content: str, component: str = "general") -> None:
        """Log an event with timestamp and formatting, respecting component flags."""
        if not self._should_log_component(component):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"{timestamp} INFO {self.name} | [{event_type.upper()}]\n{content}\n"
        append_chatbot_log(formatted)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Log when LLM starts processing."""
        if not self._should_log_component("http"):
            return

        try:
            # Handle different prompt formats
            if isinstance(prompts, list) and prompts:
                if all(isinstance(p, str) for p in prompts):
                    # List of strings
                    prompt_text = "\n---PROMPT---\n".join(prompts)
                else:
                    # List of prompt objects
                    prompt_texts = []
                    for p in prompts:
                        if hasattr(p, "text"):
                            prompt_texts.append(p.text)
                        else:
                            prompt_texts.append(str(p))
                    prompt_text = "\n---PROMPT---\n".join(prompt_texts)
            else:
                prompt_text = str(prompts) if prompts else "No prompts"

            # Truncate very long prompts
            if len(prompt_text) > 2000:
                prompt_text = prompt_text[:2000] + "...[truncated]"

            model_name = "unknown"
            if isinstance(serialized, dict):
                model_name = serialized.get("name", serialized.get("model", "unknown"))

            self._log_event("llm_start", f"Model: {model_name}\n{prompt_text}", "http")
        except Exception as e:
            self._log_event("llm_start", f"LLM start (parse error: {e})", "http")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log LLM response."""
        if not self._should_log_component("http"):
            return

        try:
            # Extract text from LLMResult - handle different structures
            texts = []

            if hasattr(response, "generations") and response.generations:
                for generation_list in response.generations:
                    if isinstance(generation_list, list):
                        for gen in generation_list:
                            if hasattr(gen, "text") and gen.text:
                                texts.append(gen.text)
                            elif hasattr(gen, "message") and hasattr(gen.message, "content"):
                                texts.append(gen.message.content)
                            else:
                                texts.append(str(gen))
                    else:
                        texts.append(str(generation_list))

            if texts:
                response_text = "\n---RESPONSE---\n".join(texts)
            else:
                # Fallback - try to get any text from the response
                response_text = str(response)

            # Truncate very long responses
            if len(response_text) > 1500:
                response_text = response_text[:1500] + "...[truncated]"

            self._log_event("llm_end", response_text, "http")
        except Exception as e:
            self._log_event(
                "llm_end", f"LLM end (parse error: {e})\nResponse: {str(response)[:500]}", "http"
            )

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Log LLM errors."""
        self._log_event("llm_error", str(error), "http")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log chain start."""
        if not self._should_log_component("chains"):
            return

        try:
            # Handle serialized being None or not a dict
            if isinstance(serialized, dict):
                chain_name = serialized.get("name", "unknown_chain")
            else:
                chain_name = str(serialized) if serialized else "unknown_chain"

            # Handle inputs being different types
            if isinstance(inputs, dict):
                input_str = "\n".join(
                    f"{k}: {str(v)[:200]}..." if len(str(v)) > 200 else f"{k}: {v}"
                    for k, v in inputs.items()
                )
            else:
                input_str = str(inputs)[:500] + "..." if len(str(inputs)) > 500 else str(inputs)

            self._log_event("chain_start", f"Chain: {chain_name}\nInputs:\n{input_str}", "chains")
        except Exception as e:
            # Fallback logging to avoid breaking execution
            self._log_event("chain_start", f"Chain start (parse error: {e})", "chains")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log chain end."""
        if not self._should_log_component("chains"):
            return

        try:
            # Handle different output types - could be dict, AgentFinish, StringPromptValue, etc.
            if isinstance(outputs, dict):
                output_str = "\n".join(
                    f"{k}: {str(v)[:200]}..." if len(str(v)) > 200 else f"{k}: {v}"
                    for k, v in outputs.items()
                )
            elif hasattr(outputs, "return_values"):
                # AgentFinish object
                output_str = f"AgentFinish: {outputs.return_values}"
            elif hasattr(outputs, "text"):
                # StringPromptValue or similar
                text = outputs.text[:500] + "..." if len(outputs.text) > 500 else outputs.text
                output_str = f"Prompt: {text}"
            else:
                # Fallback to string representation
                output_str = str(outputs)[:500] + "..." if len(str(outputs)) > 500 else str(outputs)

            self._log_event("chain_end", f"Outputs:\n{output_str}", "chains")
        except Exception as e:
            # Fallback logging to avoid breaking execution
            self._log_event("chain_end", f"Chain end (parse error: {e})", "chains")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Log chain errors."""
        self._log_event("chain_error", str(error), "chains")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Log tool execution start."""
        tool_name = serialized.get("name", "unknown_tool")
        self._log_event("tool_start", f"Tool: {tool_name}\nInput: {input_str}", "tools")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Log tool execution end."""
        # Truncate very long outputs for readability
        display_output = output[:1000] + "..." if len(output) > 1000 else output
        self._log_event("tool_end", f"Output: {display_output}", "tools")

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Log tool errors."""
        self._log_event("tool_error", str(error), "tools")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Log agent actions (Thought/Action/Action Input)."""
        content = f"Tool: {action.tool}\nTool Input: {action.tool_input}"
        if hasattr(action, "log") and action.log:
            content += f"\nFull Log:\n{action.log}"
        self._log_event("agent_action", content, "general")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Log agent finish."""
        content = f"Return Values: {finish.return_values}"
        if hasattr(finish, "log") and finish.log:
            content += f"\nFinal Log:\n{finish.log}"
        self._log_event("agent_finish", content, "general")

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Log arbitrary text (often Thought/Observation parts)."""
        if text.strip() and self._should_log_component("verbose"):
            self._log_event("text", text, "verbose")


def log_data_transform(
    stage: str, input_data: Any, output_data: Any, description: str = ""
) -> None:
    """
    Log data transformation in DEBUG mode only.
    Shows how data changes between stages for debugging.

    Args:
        stage: Name of the transformation stage
        input_data: Data before transformation
        output_data: Data after transformation
        description: Optional description of what's happening

    Usage:
        log_data_transform(
            "tool_to_BookRecommendation",
            raw_tool_results[:3],  # Sample
            recommendations[:3],
            f"Built {len(recommendations)} BookRecommendation objects"
        )
    """
    # Only log in DEBUG mode
    if not os.getenv("LOG_PROMPT", "0").lower() in ("1", "true", "yes"):
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format as JSON if possible
    try:
        if isinstance(input_data, (dict, list)):
            input_str = json.dumps(input_data, indent=2, default=str)
        else:
            input_str = str(input_data)

        if isinstance(output_data, (dict, list)):
            output_str = json.dumps(output_data, indent=2, default=str)
        else:
            output_str = str(output_data)
    except Exception:
        input_str = str(input_data)
        output_str = str(output_data)

    # Limit size to avoid huge logs
    if len(input_str) > 5000:
        input_str = input_str[:5000] + "\n...[truncated]"
    if len(output_str) > 5000:
        output_str = output_str[:5000] + "\n...[truncated]"

    log_content = f"""
{timestamp} DEBUG | [TRANSFORM: {stage}]
{description}

INPUT:
{input_str}

OUTPUT:
{output_str}
{"=" * 80}
"""
    append_chatbot_log(log_content)


def is_debug_mode() -> bool:
    """Check if DEBUG logging mode is enabled."""
    return os.getenv("LOG_PROMPT", "0").lower() in ("1", "true", "yes")


def suppress_noisy_loggers():
    """Suppress DEBUG logs from noisy third-party libraries."""
    noisy_loggers = [
        "httpcore",
        "httpcore.http11",
        "primp",
        "primp.utils",
        "rquest",
        "rquest.connect",
        "rquest.util.client.connect.dns",
        "rquest.util.client.connect.http",
        "rquest.util.client.pool",
        "rquest.client.http",
        "ddgs",
        "ddgs.ddgs",
        "cookie_store",
        "cookie_store.cookie_store",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Re-export everything from the original module
suppress_noisy_loggers()

__all__ = [
    "append_chatbot_log",
    "LogCallbackHandler",
    "capture_agent_console_and_httpx",
    "log_data_transform",
    "is_debug_mode",
]
