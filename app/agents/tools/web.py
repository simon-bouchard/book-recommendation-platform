# app/agents/web_tools.py
import re
import requests
from typing import Dict, Set, Optional, List

from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper

__all__ = ["build_web_tools", "WebToolState"]


class WebToolState:
    """
    Per-request or per-conversation dedupe state for web tools.
    Store one of these in your agent runtime and pass to build_web_tools(state).
    """
    def __init__(self) -> None:
        self.seen: Dict[str, Set[str]] = {
            "web-search": set(),
            "Wikipedia": set(),
            "openlibrary": set(),
            "ol-work": set(),
        }

    def reset(self) -> None:
        for k in self.seen:
            self.seen[k].clear()


def _norm(q: str) -> str:
    toks = re.findall(r"\w+", (q or "").lower())
    toks.sort()
    return " ".join(toks)


def build_web_tools(state: Optional[WebToolState] = None) -> List[Tool]:
    """
    Factory for the external (web/OpenLibrary) + local-help tool palette.
    Pass an optional WebToolState to enforce per-turn dedupe.
    """
    state = state or WebToolState()

    # Single sessions/utilities (cheap and deterministic)
    ddg = DuckDuckGoSearchResults(num_results=5)
    wiki = WikipediaAPIWrapper(lang="en")
    ol_session = requests.Session()

    # ---- Safe wrappers with dedupe ----
    def _safe_web(q: str) -> str:
        key = _norm(q)
        if key in state.seen["web-search"]:
            return "[Guardrail] Repeated web search. Use what you have and answer."
        state.seen["web-search"].add(key)

        out = ddg.run(q)
        return out if (out or "").strip() else "[Search] No results."

    def _safe_wiki(q: str) -> str:
        key = _norm(q)
        if key in state.seen["Wikipedia"]:
            return "[Guardrail] Repeated Wikipedia query. Use what you have and answer."
        state.seen["Wikipedia"].add(key)

        out = wiki.run(q)
        return out if (out or "").strip() else "[Wiki] No content returned."

    def _safe_ol_search(q: str) -> str:
        """Search Open Library works compactly and deterministically."""
        key = _norm(q)
        if key in state.seen["openlibrary"]:
            return "[Guardrail] Repeated Open Library search. Use prior results."
        state.seen["openlibrary"].add(key)

        try:
            r = ol_session.get(
                "https://openlibrary.org/search.json",
                params={
                    "q": q,
                    "fields": "title,author_name,first_publish_year,key,edition_count",
                    "limit": 5,
                },
                timeout=12,
            )
            r.raise_for_status()
            js = r.json()
            docs = (js.get("docs") or [])[:5]
            if not docs:
                return "[OpenLibrary] No results."
            lines = []
            for d in docs:
                title = d.get("title") or "Untitled"
                author = ", ".join((d.get("author_name") or [])[:2]) or "Unknown"
                year = d.get("first_publish_year")
                work = d.get("key")  # '/works/OLxxxxW'
                editions = d.get("edition_count")
                meta = []
                if year:
                    meta.append(str(year))
                if editions:
                    meta.append(f"{editions} eds")
                meta_str = f" ({', '.join(meta)})" if meta else ""
                lines.append(f"- {title} — {author}{meta_str}  [source: openlibrary.org{work}]")
            return "\n".join(lines)
        except Exception:
            return "[OpenLibrary] Error reaching API."

    def _safe_ol_work(work_key: str) -> str:
        """Fetch one work’s metadata/description by key like '/works/OL12345W'."""
        key = _norm(work_key)
        if key in state.seen["ol-work"]:
            return "[Guardrail] Repeated Open Library work fetch. Use prior results."
        state.seen["ol-work"].add(key)

        wk = work_key.strip()
        if not wk.startswith("/works/"):
            wk = f"/works/{wk}"

        try:
            r = ol_session.get(f"https://openlibrary.org{wk}.json", timeout=12)
            r.raise_for_status()
            w = r.json()
            title = w.get("title") or "Untitled"
            desc = w.get("description")
            if isinstance(desc, dict):
                desc = desc.get("value")
            if not desc:
                desc = "[No description available]"
            trunc = desc[:800] + ("…" if len(desc or "") > 800 else "")
            return f"Title: {title}\nWork: {wk}\nDescription: {trunc}\n[source: openlibrary.org{wk}]"
        except Exception:
            return "[OpenLibrary] Error fetching work details."

    # ---- Wrap as LangChain Tools ----
    tools = [
        Tool(
            name="web-search",
            func=_safe_web,
            description="General web search for fresh info, lists, comparisons, links.",
        ),
        Tool(
            name="Wikipedia",
            func=_safe_wiki,
            description="Background/definitions only; avoid for 'best/top/latest' lists.",
        ),
        Tool(
            name="openlibrary-search",
            func=_safe_ol_search,
            description="Search Open Library for real books (title/author/year). Input: plain query.",
        ),
        Tool(
            name="openlibrary-work",
            func=_safe_ol_work,
            description="Fetch details for a specific Open Library work. Input: '/works/OLXXXXW' or 'OLXXXXW'.",
        ),
    ]

    # Add local help-docs tools (already Tool objects)
    return tools
