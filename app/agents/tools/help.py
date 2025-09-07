import os
import json
from typing import Dict
from langchain_core.tools import Tool
from pathlib import Path

# Base path for end-user help docs
_BASE = os.environ.get("HELP_DOCS_PATH")
if _BASE:
    _DOCS_PATH = _BASE
else:
    # help.py is at .../app/agents/tools/help.py
    # parents[3] = project root
    _DOCS_PATH = str(Path(__file__).resolve().parents[3] / "docs" / "help")

_MANIFEST_PATH = os.path.join(_DOCS_PATH, "help_manifest.json")

# ------------------------
# Manifest loading
# ------------------------
def _load_manifest() -> Dict[str, Dict]:
    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Basic normalization of keys/fields
        norm = {}
        for alias, meta in (data or {}).items():
            if not isinstance(meta, dict):
                continue
            a = (alias or "").strip().lower()
            file = (meta.get("file") or "").strip()
            if not a or not file:
                continue
            norm[a] = {
                "file": file,
                "title": meta.get("title", a),
                "description": meta.get("description", ""),
                "keywords": meta.get("keywords", []),
            }
        return norm
    except Exception:
        return {}

_MANIFEST = _load_manifest()

# ------------------------
# Helpers
# ------------------------
def _abs(path: str) -> str:
    return os.path.join(_DOCS_PATH, path)

def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def _resolve_to_filepath(name_or_alias: str) -> str:
    key = (name_or_alias or "").strip().lower()
    # Try alias via manifest
    if key in _MANIFEST:
        p = _abs(_MANIFEST[key]["file"])
        return p if os.path.isfile(p) else ""
    # Try literal filename
    if key.endswith(".md"):
        p = _abs(key)
        return p if os.path.isfile(p) else ""
    return ""

# ------------------------
# Tool funcs
# ------------------------
def _get_manifest(_: str = "") -> str:
    """
    Return the entire help manifest as a JSON string.
    The LLM can read titles/descriptions/keywords and decide which doc to open.
    """
    if not _MANIFEST:
        return "{}"
    return json.dumps(_MANIFEST, ensure_ascii=False, indent=2)

def _read_doc(name: str) -> str:
    """
    Read a help doc by alias or filename. Returns a compact slice to keep tokens under control.
    """
    path = _resolve_to_filepath(name)
    if not path:
        return f"[HelpDocs] Unknown doc '{name}'. Use 'help-manifest' to see available aliases."
    txt = _read_file(path)
    return txt[:5000] if txt else f"[HelpDocs] File empty or not readable: {os.path.basename(path)}"

# ------------------------
# Exposed tools
# ------------------------
help_tools = [
    Tool(
        name="help-manifest",
        func=_get_manifest,
        description="Return the full curated help manifest (aliases → {file, title, description, keywords})."
    ),
    Tool(
        name="help-read",
        func=_read_doc,
        description="Read an end-user help doc by alias or filename (e.g., 'overview' or 'overview.md')."
    ),
]

class SiteHelpToolkit:
    @staticmethod
    def as_tools():
        return list(help_tools)
