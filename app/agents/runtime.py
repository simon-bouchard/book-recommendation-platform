# app/agents/runtime.py
"""
Runtime utilities for chat agents: rate limiting, conversation history, and book data helpers.
"""

from __future__ import annotations

import json, time, uuid
from typing import Optional, Any, Callable, Dict, Iterable, List, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import re

import redis
from fastapi import Request, Response, HTTPException

from app.agents.schemas import BookOut
from app.agents.settings import settings
from models.data.loaders import load_book_meta

# Single Redis client (decode to str)
try:
    r = redis.from_url(settings.redis_url, decode_responses=True)
except Exception:
    r = None  # degrade to stateless if Redis is down

# ---- tiny helpers ----
def _epoch_minute(ts: Optional[float] = None) -> int:
    ts = ts or time.time()
    return int(ts // 60)

def _today_local_str(tz: ZoneInfo) -> str:
    return datetime.now(tz).strftime("%Y-%m-%d")

def _incr_with_ttl(client, key: str, ttl: int) -> int:
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.expire(key, ttl)
    try:
        val, _ = pipe.execute()
        return int(val)
    except Exception:
        return 0

def _ua_hash(user_agent: str | None) -> str:
    """
    Short, stable fingerprint for UA to strengthen anon identity.
    Returns 8-hex prefix of SHA1 over a normalized UA string.
    """
    ua = (user_agent or "").strip().lower()
    # normalize super-short/empty UAs
    if len(ua) < 5:
        ua = "na"
    return hashlib.sha1(ua.encode("utf-8")).hexdigest()[:8]

def _seconds_until_next_minute(now: datetime) -> int:
    nxt = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
    return max(1, int((nxt - now).total_seconds()))

def _seconds_until_tomorrow(now: datetime) -> int:
    tomorrow = (now.date() + timedelta(days=1))
    nxt = datetime.combine(tomorrow, datetime.min.time(), tzinfo=now.tzinfo)
    return max(1, int((nxt - now).total_seconds()))

# ---- Internal tool -----
_FA = "final answer:"

def normalize_visible_reply(raw_text: str) -> str:
    s = (raw_text or "").strip()
    lo = s.lower()
    if lo.startswith(_FA):
        return s[len(_FA):].lstrip()
    return s

def extract_book_ids_from_steps(steps: List[Any]) -> List[int]:
    if not isinstance(steps, list):
        return []

    def _get_action_and_obs(step):
        if isinstance(step, (list, tuple)) and len(step) == 2:
            return step[0], step[1]
        if isinstance(step, dict):
            return step.get("action"), step.get("observation")
        return None, None

    def _get_tool_and_input(action):
        if action is None:
            return None, None
        name = getattr(action, "tool", None) or getattr(action, "tool_name", None)
        inp  = getattr(action, "tool_input", None)
        if name is None and isinstance(action, dict):
            name = action.get("tool") or action.get("tool_name")
            inp  = action.get("tool_input")
        return (name.lower() if isinstance(name, str) else name), inp

    for step in reversed(steps):
        action, observation = _get_action_and_obs(step)
        tool_name, tool_input = _get_tool_and_input(action)
        if tool_name == "return_book_ids":
            if isinstance(observation, str):
                try:
                    obj = json.loads(observation)
                    ids = obj.get("book_ids", [])
                    return [int(x) for x in ids if str(x).strip() != ""]
                except Exception:
                    pass
            if isinstance(tool_input, str) and tool_input.strip().startswith("["):
                try:
                    ids = json.loads(tool_input)
                    return [int(x) for x in ids if str(x).strip() != ""]
                except Exception:
                    pass
            return []
    return []

def _safe_str(val) -> Optional[str]:
    import math
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    return s or None

def build_books_from_ids(ids: Iterable[int]) -> List[BookOut]:
    """
    Build list of BookOut objects from item_idx list, preserving order.
    
    Args:
        ids: Iterable of item_idx values
        
    Returns:
        List of BookOut objects with metadata from book_meta
    """
    ids = list(dict.fromkeys(int(i) for i in ids))  # de-dup, preserve order
    if not ids:
        return []

    # Load book metadata using refactored loader
    BOOK_META = load_book_meta(use_cache=True)  # pd.DataFrame indexed by item_idx
    
    # Select & keep order
    rows = BOOK_META.loc[BOOK_META.index.intersection(ids)].copy()
    rows["__sort"] = rows.index.map({idx: i for i, idx in enumerate(ids)})
    rows = rows.sort_values("__sort").drop(columns="__sort")

    out: List[BookOut] = []
    id_set = set(rows.index)

    # First, add all present rows (preserving the requested order)
    for i in ids:
        if i in id_set:
            r = rows.loc[i]
            year_val = r.get("year")
            out.append(BookOut(
                item_idx=int(i),
                title=_safe_str(r.get("title")),
                author=_safe_str(r.get("author")),
                year=_safe_str(r.get("year")),
                cover_id=_safe_str(r.get("cover_id")),
            ))
        else:
            # Still return a minimal object so the UI can render gracefully
            out.append(BookOut(item_idx=int(i), title=None, author=None, year=None, cover_id=None))

    return out

# ---- public API ----
def rate_limit_check(request: Request, user_id: Optional[int]) -> None:
    if r is None:
        return  # degrade open if Redis is down

    tz = ZoneInfo(settings.chat_local_tz)
    now = datetime.now(tz)

    ip = request.client.host if request.client else "0.0.0.0"
    conv_id = request.cookies.get("conv_id") or "anon"

    if user_id is not None:
        id_key  = f"user:{user_id}"
        day_lim = settings.chat_limits_per_day_user
        min_lim = settings.chat_limits_per_min_user
    else:
        # (optionally include UA hash here if you applied step #2 earlier)
        id_key  = f"conv:{conv_id}:ip:{ip}"
        day_lim = settings.chat_limits_per_day_fallback
        min_lim = settings.chat_limits_per_min_fallback

    today  = _today_local_str(tz)
    minute = _epoch_minute()

    k_day     = f"rl:{id_key}:d:{today}"
    k_min     = f"rl:{id_key}:m:{minute}"
    k_sys_day = f"rl:system:d:{today}"

    # System-wide daily
    sys_day_count = _incr_with_ttl(r, k_sys_day, 24*60*60 + 300)
    if sys_day_count > settings.chat_limits_per_day_system:
        headers = {
            "Retry-After": str(_seconds_until_tomorrow(now)),
            "X-RateLimit-Block-Reason": "system_day",
        }
        raise HTTPException(
            status_code=429,
            detail="The demo's daily quota has been reached. Please try again tomorrow.",
            headers=headers
        )

    # Per-identity daily
    day_count = _incr_with_ttl(r, k_day, 24*60*60 + 300)
    if day_count > day_lim:
        headers = {
            "Retry-After": str(_seconds_until_tomorrow(now)),
            "X-RateLimit-Block-Reason": "identity_day",
            "X-RateLimit-Day-Limit": str(day_lim),
            "X-RateLimit-Day-Count": str(day_count),
        }
        raise HTTPException(
            status_code=429,
            detail=f"Daily chat limit reached ({day_lim} messages). Please try again tomorrow.",
            headers=headers
        )

    # Per-identity per-minute
    min_count = _incr_with_ttl(r, k_min, 70)
    if min_count > min_lim:
        headers = {
            "Retry-After": str(_seconds_until_next_minute(now)),
            "X-RateLimit-Block-Reason": "identity_minute",
            "X-RateLimit-Min-Limit": str(min_lim),
            "X-RateLimit-Min-Count": str(min_count),
        }
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait ~60 seconds and try again.",
            headers=headers
        )

    # Success headers for UX/debug (what you already had)
    request.state.rl_headers = {
        "X-RateLimit-Identity": id_key,
        "X-RateLimit-Day-Count": str(day_count),
        "X-RateLimit-Day-Limit": str(day_lim),
        "X-RateLimit-Min-Count": str(min_count),
        "X-RateLimit-Min-Limit": str(min_lim),
        "X-RateLimit-System-Day-Count": str(sys_day_count),
        "X-RateLimit-System-Day-Limit": str(settings.chat_limits_per_day_system),
    }
    
def ensure_conv_cookie(request: Request, response: Response) -> str:
    """Return existing conv_id or set a new one (SameSite=Lax, 7d)."""
    conv_id = request.cookies.get("conv_id")
    if not conv_id:
        conv_id = uuid.uuid4().hex
        response.set_cookie("conv_id", conv_id, max_age=60 * 60 * 24 * 7, samesite="Lax")
    return conv_id


def load_history(conv_id: str, user_id: Optional[int] = None) -> List[dict]:
    """Return rolling history list from Redis (or [])."""
    if r is None:
        return []
    
    # Use user_id for logged-in users, conv_id for anonymous
    if user_id is not None:
        key = f"chat:user:{user_id}"
    else:
        key = f"chat:conv:{conv_id}"
    
    try:
        return json.loads(r.get(key) or "[]")
    except Exception:
        return []

def save_history(
    conv_id: str, 
    hist: List[dict], 
    user_id: Optional[int] = None
) -> None:
    """Persist last N exchanges with TTL."""
    if r is None:
        return
    
    # Use user_id for logged-in users, conv_id for anonymous
    if user_id is not None:
        key = f"chat:user:{user_id}"
    else:
        key = f"chat:conv:{conv_id}"
    
    try:
        r.setex(key, settings.chat_ttl_sec, json.dumps(hist[-settings.chat_hist_turns:]))
    except Exception:
        pass
