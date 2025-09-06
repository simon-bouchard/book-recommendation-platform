# app/agents/tool_registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.tools import Tool

# External + docs
from .web import build_web_tools, WebToolState
from .help import SiteHelpToolkit

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

    Optionally returns (future):
      - Internal ML tools (ALS recs, ColdHybrid, subject/ALS/hybrid sim, fetch_book_meta)
        gated by InternalToolGates. For now, internal tools are not exposed unless
        `internal_enabled=True` is passed explicitly.
    """

    def __init__(
        self,
        web: bool = True,
        help: bool = True,
        gates: Optional[InternalToolGates] = None,
    ) -> None:
        self.web_enabled = web
        self.help_enabled = help
        self.gates = gates or InternalToolGates()  # internal disabled by default

        # Per-request dedupe state for web tools
        self._web_state = WebToolState()

    # -------- Public API --------
    def get_tools(self) -> List[Tool]:
        tools: List[Tool] = []

        if self.web_enabled:
            tools.extend(build_web_tools(self._web_state))

        if self.help_enabled:
            tools.extend(SiteHelpToolkit.as_tools())
            
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
          - als_recs:          "user_id|top_k"
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
            if gates.user_has_als is not None:
                return bool(gates.user_has_als)
            if gates.user_num_ratings is not None:
                return gates.user_num_ratings >= gates.warm_threshold
            return None  # unknown

        warm = _is_warm()

        # ---- User-centric recs ----
        # ALS recs → only if warm is explicitly True
        if warm is True:
            tools.append(Tool(
                name="als_recs",
                func=lambda s: _na("als_recs"),
                description="Personalised recommendations for a warm user (ALS). Input: 'user_id|top_k'."
            ))
        # ColdHybrid recs → only if warm is explicitly False
        if warm is False:
            tools.append(Tool(
                name="cold_hybrid_recs",
                func=lambda s: _na("cold_hybrid_recs"),
                description="Cold-user recommendations via subject+Bayesian hybrid. Input: 'user_id|top_k|w|tiers' or 'fav_subjects_idxs=[...]|top_k|w|tiers'."
            ))

        # ---- Item-centric sim ----
        # subject_sim is always allowed (subject embeddings exist for all books)
        tools.append(Tool(
            name="subject_sim",
            func=lambda s: _na("subject_sim"),
            description="Find similar books by subject-embedding similarity. Input: 'item_idx|top_k'."
        ))

        # als_sim requires ALS for this book (when known)
        if gates.book_has_als is True:
            tools.append(Tool(
                name="als_sim",
                func=lambda s: _na("als_sim"),
                description="Find similar books by ALS behavioral similarity. Input: 'item_idx|top_k'."
            ))

        # hybrid_sim requires both (subject is assumed; ALS must be available if known)
        if gates.book_has_als is True:
            tools.append(Tool(
                name="hybrid_sim",
                func=lambda s: _na("hybrid_sim"),
                description="Blend subject and ALS similarities. Input: 'item_idx|top_k|alpha|min_count'."
            ))

        # ---- Meta join ----
        tools.append(Tool(
            name="fetch_book_meta",
            func=lambda s: _na("fetch_book_meta"),
            description="Return canonical {title, author, year, cover_url} for given item_idx list. Input: '1,2,3,...'."
        ))

        return tools
