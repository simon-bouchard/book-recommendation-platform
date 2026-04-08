# app/agents/runtime.py
"""
Runtime utilities for chat agents: rate limiting, conversation history, and book data helpers.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import redis
from fastapi import HTTPException, Request, Response

from app.agents.settings import settings

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
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    return max(1, int((nxt - now).total_seconds()))


def _seconds_until_tomorrow(now: datetime) -> int:
    tomorrow = now.date() + timedelta(days=1)
    nxt = datetime.combine(tomorrow, datetime.min.time(), tzinfo=now.tzinfo)
    return max(1, int((nxt - now).total_seconds()))


# ---- Internal tool -----
_FA = "final answer:"


def normalize_visible_reply(raw_text: str) -> str:
    s = (raw_text or "").strip()
    lo = s.lower()
    if lo.startswith(_FA):
        return s[len(_FA) :].lstrip()
    return s


def _safe_str(val) -> Optional[str]:
    import math

    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    return s or None


# ---- public API ----
def rate_limit_check(request: Request, user_id: Optional[int]) -> None:
    if r is None:
        return  # degrade open if Redis is down

    tz = ZoneInfo(settings.chat_local_tz)
    now = datetime.now(tz)

    ip = request.client.host if request.client else "0.0.0.0"
    conv_id = request.cookies.get("conv_id") or "anon"

    if user_id is not None:
        id_key = f"user:{user_id}"
        day_lim = settings.chat_limits_per_day_user
        min_lim = settings.chat_limits_per_min_user
    else:
        # (optionally include UA hash here if you applied step #2 earlier)
        id_key = f"conv:{conv_id}:ip:{ip}"
        day_lim = settings.chat_limits_per_day_fallback
        min_lim = settings.chat_limits_per_min_fallback

    today = _today_local_str(tz)
    minute = _epoch_minute()

    k_day = f"rl:{id_key}:d:{today}"
    k_min = f"rl:{id_key}:m:{minute}"
    k_sys_day = f"rl:system:d:{today}"

    # System-wide daily
    sys_day_count = _incr_with_ttl(r, k_sys_day, 24 * 60 * 60 + 300)
    if sys_day_count > settings.chat_limits_per_day_system:
        headers = {
            "Retry-After": str(_seconds_until_tomorrow(now)),
            "X-RateLimit-Block-Reason": "system_day",
        }
        raise HTTPException(
            status_code=429,
            detail="The demo's daily quota has been reached. Please try again tomorrow.",
            headers=headers,
        )

    # Per-identity daily
    day_count = _incr_with_ttl(r, k_day, 24 * 60 * 60 + 300)
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
            headers=headers,
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
            headers=headers,
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


def save_history(conv_id: str, hist: List[dict], user_id: Optional[int] = None) -> None:
    """Persist last N exchanges with TTL."""
    if r is None:
        return

    # Use user_id for logged-in users, conv_id for anonymous
    if user_id is not None:
        key = f"chat:user:{user_id}"
    else:
        key = f"chat:conv:{conv_id}"

    try:
        r.setex(key, settings.chat_ttl_sec, json.dumps(hist[-settings.chat_hist_turns :]))
    except Exception:
        pass
