# app/agents/logging.py
import logging
import os
import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Dict, Any, Tuple, List
from logging.handlers import RotatingFileHandler

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
    _ensure_log_dir()
    with open(_CHATBOT_LOG, "a", encoding="utf-8") as f:
        # ensure trailing newline if missing
        if text and not text.endswith("\n"):
            f.write(text + "\n")
        else:
            f.write(text)

# Optional rotating handler (not used by the capture context, but you can tail it too)
def _setup_chatbot_rotating_handler() -> logging.Logger:
    logger = logging.getLogger("chatbot")
    if not logger.handlers:
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
_LOGGER_NAMES_TO_CAPTURE: List[str] = ["langchain", "langchain_core", "httpx"]

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
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(fmt)

            lg.handlers = [stream_handler]
            lg.setLevel(logging.INFO)
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
            if content:
                append_chatbot_log(content)
        except Exception:
            # As a last resort, at least try to write something minimal
            try:
                append_chatbot_log("[chatbot-log] <failed to capture buffer>\n")
            except Exception:
                pass
