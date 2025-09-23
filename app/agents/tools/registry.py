from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import Tool

from .external.web import build_web_tools, WebToolState
from .help import SiteHelpToolkit
from .recsys.internal_tools import (
    make_als_pool_tool,
    make_return_book_ids_tool,
    make_subject_hybrid_pool_tool,
)
from .recsys.subject_search import make_subject_id_search_tool
from .recsys.semantic_search import book_semantic_search
from .user_context import UserContextToolkit


@dataclass
class InternalToolGates:
    """
    Controls exposure and heuristics for internal tools.
    """
    user_num_ratings: Optional[int] = None
    warm_threshold: int = 10
    profile_allowed: bool = False


@dataclass
class ToolEntry:
    """
    Catalog entry with category and the concrete LangChain Tool.
    """
    name: str
    category: str  # "web" | "docs" | "internal"
    tool: Tool


class ToolRegistry:
    """
    Builds a categorized tool catalog and returns enabled tools according to flags and gates.

    Categories
      - "web": External browsing / public metadata
      - "docs": Site help documentation
      - "internal": Recommendation-system tools (incl. semantic search)

    Flags
      - web: enable/disable all "web" tools
      - docs: enable/disable all "docs" tools
      - internal: enable/disable all "internal" tools

    Gates
      - Further restricts which internal tools are exposed depending on warm/cold
        and available contexts (user/db).
    """

    def __init__(
        self,
        *,
        web: bool = True,
        docs: bool = True,
        internal: bool = False,
        gates: Optional[InternalToolGates] = None,
        ctx_user: Any = None,
        ctx_db: Any = None,
    ) -> None:
        """
        Initialize the registry with capability flags, gates, and optional contexts.
        """
        self.web_enabled = web
        self.docs_enabled = docs
        self.internal_enabled = internal
        self.gates = gates or InternalToolGates()
        self.ctx_user = ctx_user
        self.ctx_db = ctx_db
        self._web_state = WebToolState()
        self._catalog: List[ToolEntry] = []
        self._built = False

    def get_tools(self) -> List[Tool]:
        """
        Return the list of enabled Tool objects according to flags and gates.
        """
        if not self._built:
            self._catalog = self._build_catalog()
            self._built = True
        return [e.tool for e in self._catalog if self._is_enabled(e)]

    def get_catalog(self) -> List[Tuple[str, str]]:
        """
        Return (name, category) pairs for all cataloged tools.
        """
        if not self._built:
            self._catalog = self._build_catalog()
            self._built = True
        return [(e.name, e.category) for e in self._catalog]

    def _is_enabled(self, entry: ToolEntry) -> bool:
        """
        Evaluate category-level flags and per-tool internal gates.
        """
        if entry.category == "web":
            return self.web_enabled
        if entry.category == "docs":
            return self.docs_enabled
        if entry.category == "internal":
            return self.internal_enabled and self._gate_internal(entry.name)
        return False

    def _gate_internal(self, tool_name: str) -> bool:
        """
        Apply internal gating per tool name and available contexts.
        """
        is_warm = None
        if self.gates.user_num_ratings is not None:
            is_warm = self.gates.user_num_ratings >= self.gates.warm_threshold

        if tool_name == "als_recs":
            return bool(is_warm) and (self.ctx_user is not None) and (self.ctx_db is not None)

        if tool_name in {"subject_hybrid_pool", "subject_id_search"}:
            return (self.ctx_user is not None) and (self.ctx_db is not None)

        if tool_name in {"book_semantic_search", "return_book_ids"}:
            return True

        if tool_name in {"user-profile", "recent-interactions"}:
            return (
                self.gates.profile_allowed
                and (self.ctx_user is not None)
                and (self.ctx_db is not None)
            )

        return True

    def _build_catalog(self) -> List[ToolEntry]:
        """
        Construct the full tool catalog once, attaching categories to each entry.
        """
        catalog: List[ToolEntry] = []

        if self.web_enabled:
            for t in build_web_tools(self._web_state):
                catalog.append(ToolEntry(name=t.name, category="web", tool=t))

        if self.docs_enabled:
            for t in SiteHelpToolkit.as_tools():
                catalog.append(ToolEntry(name=t.name, category="docs", tool=t))

        if self.internal_enabled:
            catalog.extend(self._build_internal_entries())

        return catalog

    def _build_internal_entries(self) -> List[ToolEntry]:
        """
        Create internal tool entries (including semantic search).
        """
        entries: List[ToolEntry] = []

        # Semantic search is internal
        entries.append(ToolEntry(
            name="book_semantic_search",
            category="internal",
            tool=self._make_semantic_tool(),
        ))

        if self._gate_internal("als_recs"):
            entries.append(ToolEntry(
                name="als_recs",
                category="internal",
                tool=Tool(
                    name="als_recs",
                    func=make_als_pool_tool(self.ctx_user, self.ctx_db),
                    description=(
                        "ALS warm-user candidate pool. "
                        "Input: string 'top_k' (default 200). "
                        "Output: JSON array of book dicts "
                        "{item_idx,title,author,year,cover_id,isbn,book_avg_rating,book_num_ratings,score}."
                    ),
                ),
            ))

        if self._gate_internal("subject_hybrid_pool"):
            entries.append(ToolEntry(
                name="subject_hybrid_pool",
                category="internal",
                tool=Tool(
                    name="subject_hybrid_pool",
                    func=make_subject_hybrid_pool_tool(self.ctx_user, self.ctx_db),
                    description=(
                        "Subject+Bayesian mixed pool with exclude-read. "
                        "Input: '' to use profile or JSON "
                        "{'top_k':200,'fav_subjects_idxs':[...],'w':0.5..0.9}. "
                        "Output: JSON array of book dicts "
                        "{item_idx,title,author,year,cover_id,isbn,book_avg_rating,book_num_ratings,score}."
                    ),
                ),
            ))

        if self._gate_internal("subject_id_search"):
            entries.append(ToolEntry(
                name="subject_id_search",
                category="internal",
                tool=Tool(
                    name="subject_id_search",
                    func=make_subject_id_search_tool(self.ctx_db),
                    description=(
                        "Resolve free-text subject phrases to subject indices. "
                        "Input (JSON only): {'phrases': ['history','military history'], 'top_k': 4}. "
                        "Output (JSON): [{'phrase':'history','candidates':[{'subject_idx':1669,'subject':'History','score':0.96}, ...]}, ...]."
                    ),
                ),
            ))

        if self._gate_internal("return_book_ids"):
            entries.append(ToolEntry(
                name="return_book_ids",
                category="internal",
                tool=Tool(
                    name="return_book_ids",
                    func=make_return_book_ids_tool(),
                    description=(
                        "Finalize chosen recommendations. "
                        "Input: JSON list or comma-separated item_idx. "
                        "Output: {'book_ids':[...]}."
                    ),
                ),
            ))

        # User context tools (exposed only with consent + user/db context)
        if self._gate_internal("user-profile") or self._gate_internal("recent-interactions"):
            for t in UserContextToolkit.as_tools(
                ctx_user=self.ctx_user,
                ctx_db=self.ctx_db,
                consent=self.gates.profile_allowed,
            ):
                entries.append(ToolEntry(
                    name=t.name,
                    category="internal",
                    tool=t,
                ))

        return entries

    def _make_semantic_tool(self) -> Tool:
        """
        Create the internal semantic search tool with a flexible input wrapper.
        """
        def _semantic_wrapper(s: str):
            q, k = s, 200
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    q = obj.get("query", "")
                    k = int(obj.get("top_k", 200))
            except Exception:
                pass
            # Return from outer scope to keep Tool construction clean after parsing
            return book_semantic_search(query=q, top_k=k)

        return Tool(
            name="book_semantic_search",
            func=_semantic_wrapper,
            description=(
                "Semantic search over internal embeddings (seeding/identification). "
                "Input: JSON {'query': 'free text', 'top_k': 200} or plain string. "
                "Output: list of candidate book objects with similarity scores and metadata."
            ),
        )
