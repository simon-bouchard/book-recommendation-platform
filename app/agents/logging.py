# app/agents/logging.py
import logging
import os
import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Dict, Any, Tuple, List, Optional, Union
from logging.handlers import RotatingFileHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from datetime import datetime

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
        fh = RotatingFileHandler(_CHATBOT_LOG, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
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

@contextmanager
def capture_agent_console_and_httpx():
    """
    Capture EVERYTHING the agent would normally print with verbose=True:
      - stdout / stderr (chain banners, Thought/Action/Observation, Finished chain)
      - logging from 'langchain', 'langchain_core', and 'httpx' at INFO and above
    Nothing leaks into normal app logs during capture. On exit, the buffered
    text is appended verbatim to logs/chatbot.log.
    """
    buf = io.StringIO()

    # Prepare a formatter that matches your normal console format (so httpx lines look identical)
    fmt = logging.Formatter(_FMT, _DATEFMT)

    # Save and replace logger state for target libraries
    saved: Dict[str, Tuple[int, List[logging.Handler], bool]] = {}
    handlers: Dict[str, logging.Handler] = {}

    try:
        for name in _LOGGER_NAMES_TO_CAPTURE:
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
        with redirect_stdout(buf), redirect_stderr(buf):
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
        
    def _log_event(self, event_type: str, content: str) -> None:
        """Log an event with timestamp and formatting."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"{timestamp} INFO {self.name} | [{event_type.upper()}]\n{content}\n"
        append_chatbot_log(formatted)
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Log when LLM starts processing."""
        prompt_text = "\n---PROMPT---\n".join(prompts) if prompts else "No prompts"
        self._log_event("llm_start", f"Model: {serialized.get('name', 'unknown')}\n{prompt_text}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log LLM response."""
        try:
            # Extract text from LLMResult
            texts = []
            for generation in response.generations:
                for gen in generation:
                    if hasattr(gen, 'text'):
                        texts.append(gen.text)
                    elif hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                        texts.append(gen.message.content)
            
            response_text = "\n---RESPONSE---\n".join(texts) if texts else str(response)
            self._log_event("llm_end", response_text)
        except Exception as e:
            self._log_event("llm_end", f"Error extracting response: {e}\n{response}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Log LLM errors."""
        self._log_event("llm_error", str(error))

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log chain start."""
        chain_name = serialized.get("name", "unknown_chain")
        input_str = "\n".join(f"{k}: {v}" for k, v in inputs.items())
        self._log_event("chain_start", f"Chain: {chain_name}\nInputs:\n{input_str}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log chain end."""
        output_str = "\n".join(f"{k}: {v}" for k, v in outputs.items())
        self._log_event("chain_end", f"Outputs:\n{output_str}")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Log chain errors."""
        self._log_event("chain_error", str(error))

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Log tool execution start."""
        tool_name = serialized.get("name", "unknown_tool")
        self._log_event("tool_start", f"Tool: {tool_name}\nInput: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Log tool execution end."""
        # Truncate very long outputs for readability
        display_output = output[:1000] + "..." if len(output) > 1000 else output
        self._log_event("tool_end", f"Output: {display_output}")

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Log tool errors."""
        self._log_event("tool_error", str(error))

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Log agent actions (Thought/Action/Action Input)."""
        content = f"Tool: {action.tool}\nTool Input: {action.tool_input}"
        if hasattr(action, 'log') and action.log:
            content += f"\nFull Log:\n{action.log}"
        self._log_event("agent_action", content)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Log agent finish."""
        content = f"Return Values: {finish.return_values}"
        if hasattr(finish, 'log') and finish.log:
            content += f"\nFinal Log:\n{finish.log}"
        self._log_event("agent_finish", content)

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Log arbitrary text (often Thought/Observation parts)."""
        if text.strip():
            self._log_event("text", text)