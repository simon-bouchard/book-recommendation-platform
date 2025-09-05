import os
import re
from typing import List, Tuple
from langchain_core.tools import Tool

# Base path for end-user help docs
_DOCS_PATH = os.path.join(os.path.dirname(__file__), "../../docs/help")

# ------------------------
# Low-level file helpers
# ------------------------
def _safe_listdir(path: str) -> List[str]:
    try:
        return [f for f in os.listdir(path) if f.endswith(".md")]
    except Exception:
        return []

def _read_raw(name: str) -> str:
    try:
        with open(os.path.join(_DOCS_PATH, name), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def _read_doc(name: str) -> str:
    """
    Read a specific help doc file from docs/help/.
    Return a compact slice to keep model tokens under control.
    """
    try:
        with open(os.path.join(_DOCS_PATH, name), "r", encoding="utf-8") as f:
            txt = f.read()
            return txt[:3000]  # keep compact, but enough for context
    except FileNotFoundError:
        return f"[HelpDocs] No file named {name}"

def _list_docs(_: str = "") -> str:
    """List all available help docs."""
    files = sorted(_safe_listdir(_DOCS_PATH))
    if not files:
        return "[HelpDocs] No docs found."
    return "\n".join(files)

# ------------------------
# Tiny section-aware search
# ------------------------
_SECTION_SPLIT = re.compile(r"^(#{1,3})\s+(.*)$", flags=re.M)

def _split_sections(filename: str, text: str) -> List[Tuple[str, str]]:
    """
    Split a markdown doc into (section_title, section_text) tuples.
    Falls back to a single section if no headers are present.
    """
    matches = list(_SECTION_SPLIT.finditer(text))
    if not matches:
        return [("Document", text)]
    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))
    return sections

def _score(query: str, title: str, body: str) -> float:
    """
    Simple keyword scorer:
    - tokenize query
    - count occurrences in title (x3 weight) and body (x1)
    - add small boost for exact phrase presence in title/body
    """
    q = query.strip().lower()
    if not q:
        return 0.0
    tokens = re.findall(r"\w+", q)
    if not tokens:
        return 0.0

    t_low = title.lower()
    b_low = body.lower()
    score = 0.0

    # token counts
    for tok in tokens:
        score += 3.0 * t_low.count(tok)
        score += 1.0 * b_low.count(tok)

    # phrase boosts
    if q in t_low:
        score += 4.0
    if q in b_low:
        score += 2.0

    # short-title normalization
    if len(title) <= 40:
        score *= 1.1

    return score

def _search_docs(query: str) -> str:
    """
    Search across all docs/help/*.md and return top 3 section hits with filename + section.
    Output is compact and deterministic. No embeddings, no external services.
    """
    files = _safe_listdir(_DOCS_PATH)
    if not files:
        return "[HelpDocs] No docs found."

    candidates: List[Tuple[float, str]] = []  # (score, formatted_snippet)

    for fname in files:
        raw = _read_raw(fname)
        if not raw:
            continue
        sections = _split_sections(fname, raw)
        for title, body in sections:
            s = _score(query, title, body)
            if s <= 0:
                continue
            # Build a small snippet
            snippet = body.strip().replace("\n", " ")
            if len(snippet) > 260:
                snippet = snippet[:260].rstrip() + "…"
            formatted = f"{fname}#{title}: {snippet}"
            candidates.append((s, formatted))

    if not candidates:
        return "[HelpDocs] No relevant section found."

    # Sort by score desc, then filename/title asc for stability
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top = [c[1] for c in candidates[:3]]
    return "\n".join(top)

# ------------------------
# Exposed tools
# ------------------------
help_tools = [
    Tool(
        name="help-list",
        func=_list_docs,
        description="List available end-user help docs for THIS website (docs/help). Use before help-read if unsure which file to open."
    ),
    Tool(
        name="help-read",
        func=_read_doc,
        description="Read an end-user help doc for THIS website (input: filename like 'overview.md', 'chatbot.md', 'faq.md', 'recommendations.md'). Use this for 'how does the site work?' questions."
    ),
    Tool(
        name="help-search",
        func=_search_docs,
        description="Keyword search across end-user help docs (docs/help). Input: a short question or keywords. Returns top sections as 'file#Section: snippet'."
    ),
]
