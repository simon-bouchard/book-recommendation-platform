# app/agents/tools/user_context.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import Tool
from sqlalchemy.orm import Session

from app.agents.user_context import fetch_user_context  # returns {fav_subjects:[], interactions:[]}  # noqa: E501
# ^ uses your existing DB queries + truncation (70/140) and limit handling

# --- tiny parsing/guards ------------------------------------------------------

def _json_input(x: Any) -> Dict[str, Any]:
    """
    Accepts a dict or a JSON string; returns a dict (empty if parsing fails).
    Enforces JSON-only inputs for predictable tool behavior.
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}

def _require_ctx(ctx_user: Any, ctx_db: Optional[Session]) -> None:
    if ctx_user is None or ctx_db is None:
        raise ValueError("user-context tools require authenticated user and active DB session")

# --- tool implementations -----------------------------------------------------

def _tool_user_profile(ctx_user: Any, ctx_db: Session, raw: Any, *, max_favs: int = 5) -> str:
    """
    Input JSON: {"limit": 5}  (optional; capped server-side)
    Output JSON: {"fav_subjects": [str, ...]}  (<= max_favs; may be empty)
    """
    _require_ctx(ctx_user, ctx_db)
    args = _json_input(raw)
    # cap requested limit, but honor your schema: at most 5 favorites
    req_limit = int(args.get("limit", max_favs) or max_favs)
    cap = min(max(1, req_limit), max_favs)

    # Fetch once; your helper already does the joins
    ctx = fetch_user_context(ctx_db, int(getattr(ctx_user, "id", ctx_user)), limit=cap)
    favs: List[str] = list(ctx.get("fav_subjects") or [])[:cap]

    # Minimal JSON; no prose
    return json.dumps({"fav_subjects": favs}, ensure_ascii=False)

def _tool_recent_interactions(
    ctx_user: Any,
    ctx_db: Session,
    raw: Any,
    *,
    max_items: int = 5,
) -> str:
    """
    Input JSON: {"limit": 5, "include_comments": false}
    Output JSON: {"interactions":[{"title","rating","date","comment"}, ...]}  (<= max_items; may be empty)
    """
    _require_ctx(ctx_user, ctx_db)
    args = _json_input(raw)
    req_limit = int(args.get("limit", max_items) or max_items)
    cap = min(max(1, req_limit), max_items)
    include_comments = bool(args.get("include_comments", False))

    ctx = fetch_user_context(ctx_db, int(getattr(ctx_user, "id", ctx_user)), limit=cap)
    items: List[Dict[str, Any]] = []
    for x in (ctx.get("interactions") or [])[:cap]:
        # Keep structure compact; optionally blank out comments
        items.append({
            "title": x.get("title"),
            "rating": x.get("rating"),
            "date": x.get("date"),
            "comment": (x.get("comment") or "") if include_comments else "",
        })

    return json.dumps({"interactions": items}, ensure_ascii=False)

# --- public factory -----------------------------------------------------------

class UserContextToolkit:
    """
    Factory for user-context tools. Call as_tools(...) inside your ToolRegistry
    when internal tools are enabled and ctx_user/ctx_db are present.
    """

    @staticmethod
    def as_tools(
        *,
        ctx_user: Any,
        ctx_db: Optional[Session],
        consent: bool = True,
        max_favs: int = 5,
        max_interactions: int = 5,
    ) -> List[Tool]:
        """
        Build the 'user-profile' and 'recent-interactions' tools.
        Only construct when consent is True and both contexts are present.
        """
        if not consent or ctx_user is None or ctx_db is None:
            return []

        def _user_profile_wrapper(raw_input: Any) -> str:
            return _tool_user_profile(ctx_user, ctx_db, raw_input, max_favs=max_favs)

        def _recent_interactions_wrapper(raw_input: Any) -> str:
            return _tool_recent_interactions(ctx_user, ctx_db, raw_input, max_items=max_interactions)

        return [
            Tool(
                name="user-profile",
                description=(
                    "Return a minimal list of the user's favorite subjects as JSON. "
                    "Input must be JSON like {\"limit\": 5}. Output is {\"fav_subjects\": [...]} "
                    "(<= 5 items, possibly empty)."
                ),
                func=_user_profile_wrapper,
            ),
            Tool(
                name="recent-interactions",
                description=(
                    "Return a small, recent list of the user's rated items as JSON. "
                    "Input must be JSON like {\"limit\": 5, \"include_comments\": false}. "
                    "Output is {\"interactions\": [{\"title\",\"rating\",\"date\",\"comment\"}, ...]} "
                    "(<= 5 items, possibly empty)."
                ),
                func=_recent_interactions_wrapper,
            ),
        ]
