# app/agents/logging.py
import logging
import os

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_FMT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Initialize root once
_root = logging.getLogger()
if not _root.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    _root.addHandler(h)
    _root.setLevel(_LOG_LEVEL)

def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger with unified formatting."""
    return logging.getLogger(name)
