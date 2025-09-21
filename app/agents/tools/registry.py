# app/agents/tools/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import json

from langchain_core.tools import Tool

# External + docs
from .external.web import build_web_tools, WebToolState
from .help import SiteHelpToolkit
from .recsys.internal_tools import (
    make_als_pool_tool,
    make_return_book_ids_tool,
    make_subject_hybrid_pool_tool,
)
from .recsys.subject_search import make_subject_id_search_tool
# New semantic tool
from .recsys.semantic_search import book_semantic_search


@dataclass
class InternalToolGates:
    """
    Gating signals for internal tools (future use).
    - internal_enabled: global on/off switch (demo: keep False)
    - user_num_ratings: used to decide warm/cold for ALS user recs (>= warm_threshold => warm)
    - user_has_als: explicit flag when you know ALS exists for this user_id
    - book_has_als: explicit flag when you know ALS exists for a specific book
    - warm_threshold: ratings needed to be considered "warm"
    """
    internal_enabled: bool = False
    user_num_ratings: Optional[int] = None
    user_has_als: Optional[bool] = None
    book_has_als: Optional[bool] = None
    warm_threshold: int = 10


class ToolRegistry:
    """
    Central tool palette builder.

    Always returns:
      - Web tools (DuckDuckGo, Wikipedia, OpenLibrary)
      - Help tools (help-list, help-read, help-search)
      - Semantic search (book_semantic_search)

    Optionally returns (future):
      - Internal ML tools (ALS recs, ColdHybrid, subject/ALS/hybrid sim, fetch_book_meta)
        gated by InternalToolGates. For now, internal tools are not exposed unless
        `internal_enabled=True` is passed explicitly.
    """
    def __init__(self, web=True, help=True, gates=None, ctx_user=None, ctx_db=None):
        self.web_enabled = web
        self.help_enabled = help
        self.gates = gates or InternalToolGates()
        self._ctx_user = ctx_user
        self._ctx_db = ctx_db
        self._web_state = WebToolState()

    # -------- Public API --------
    def get_tools(self) -> List[Tool]:
        tools: List[Tool] = []

        if self.web_enabled:
            tools.extend(build_web_tools(self._web_state))

        if self.help_enabled:
            tools.extend(SiteHelpToolkit.as_tools())

        # --- Semantic index search (always available) ---
        # Accepts either:
        #  - JSON: {"query":"...", "top_k": 200}
        #  - raw string: treated as the query; top_k defaults to 200
        def _semantic_wrapper(s: str):
            q, k = s, 200
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    q = obj.get("query", "")
                    k = int(obj.get("top_k", 200))
            except Exception:
                pass
            return book_semantic_search(query=q, top_k=k)

        tools.append(Tool(
            name="book_semantic_search",
            func=_semantic_wrapper,
            description=(
                "Semantic search over enriched tags/vibes. "
                "Input: JSON {'query': 'free text', 'top_k': 200} OR plain string (the query). "
                "Returns a list of {'book_id', 'score', 'meta'}."
            ),
        ))

        if self._should_expose_internal():
            tools.extend(self._build_internal_tools())

        return tools

    # -------- Decision logic for internal tools --------
    def _should_expose_internal(self) -> bool:
        # Global kill-switch (demo mode keeps internal hidden)
        if not self.gates.internal_enabled:
            return False
        # If you later want to require any specific condition globally, add here.
        return True

    # -------- Internal tools (future; currently stubs if enabled) --------
    def _build_internal_tools(self) -> List[Tool]:
        """
        Build internal tool objects. These are INPUT-SCHEMA'D as plain strings
        for LangChain Tool API. Prefer compact, deterministic inputs.

        Expected inputs (JSON or pipe-format — your choice later):
          - als_recs:          "top_k"
          - cold_hybrid_recs:  "user_id|top_k|w|tiers" or "fav_subjects_idxs=[...]|top_k|w|tiers"
          - subject_sim:       "item_idx|top_k"
          - als_sim:           "item_idx|top_k"
          - hybrid_sim:        "item_idx|top_k|alpha|min_count"
          - fetch_book_meta:   "item_idx1,item_idx2,..."
        """
        gates = self.gates
        tools: List[Tool] = []

        # Helper: not-available placeholder
        def _na(name: str) -> str:
            return f"[Internal] {name} unavailable in this demo."

        # Helper: quick warm check
        def _is_warm() -> Optional[bool]:
            if gates.user_num_ratings is not None:
                return gates.user_num_ratings >= gates.warm_threshold
            return None  # unknown

        warm = _is_warm()

        # ---- User-centric recs ----
        if warm is True:
            if hasattr(self, "_ctx_user") and hasattr(self, "_ctx_db"):
                tools.append(Tool(
                    name="als_recs",
                    func=make_als_pool_tool(self._ctx_user, self._ctx_db),
                    description=(
                        "ALS warm-user pool. Input: 'top_k' (default 200). "
                        "Returns JSON array of book dicts with "
                        "{item_idx,title,author,year,cover_id,isbn,book_avg_rating,book_num_ratings,score}."
                    ),
                ))

        if hasattr(self, "_ctx_user") and hasattr(self, "_ctx_db"):
            tools.append(Tool(
                name="subject_hybrid_pool",
                func=make_subject_hybrid_pool_tool(self._ctx_user, self._ctx_db),
                description=(
                    "Subject+Bayesian mixed pool (exclude-read on). Input: '' (use profile) "
                    "or JSON with {'top_k':200,'fav_subjects_idxs':[...]} to override subjects. "
                    "Returns JSON array of book dicts with "
                    "{item_idx,title,author,year,cover_id,isbn,book_avg_rating,book_num_ratings,score}."
                ),
            ))
            tools.append(Tool(
                name="subject_id_search",
                func=make_subject_id_search_tool(self._ctx_db),
                description=(
                    "Resolve free-text subject phrases to subject indices (for use with subject_hybrid_pool). "
                    "Input (JSON only): {'phrases': ['history','military history'], 'top_k': 4}. "
                    "Output (JSON): [{'phrase':'history','candidates':[{'subject_idx':1669,'subject':'History','score':0.96}, ...]}, ...]. "
                    "Rules: Action Input must be valid JSON (no trailing words)."
                ),
            ))

        # ---- Item-centric sim ----
        tools.append(Tool(
            name="subject_sim",
            func=lambda s: _na("subject_sim"),
            description="Find similar books by subject-embedding similarity. Input: 'item_idx|top_k'."
        ))

        if gates.book_has_als is True:
            tools.append(Tool(
                name="als_sim",
                func=lambda s: _na("als_sim"),
                description="Find similar books by ALS behavioral similarity. Input: 'item_idx|top_k'."
            ))
            tools.append(Tool(
                name="hybrid_sim",
                func=lambda s: _na("hybrid_sim"),
                description="Blend subject and ALS similarities. Input: 'item_idx|top_k|alpha|min_count'."
            ))

        tools.append(Tool(
            name="return_book_ids",
            func=make_return_book_ids_tool(),
            description="Finalize chosen books. Input: JSON list or comma-separated item_idx. Returns {'book_ids':[...]}."
        ))

        return tools
