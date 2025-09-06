from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from app.agents.web_agent import answer
from app.auth import get_current_user
import os, json, uuid, time
import redis
from app.agents.user_context import fetch_user_context, format_user_context
from app.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow

# ---- Feature flag: require login for chat ----
REQUIRE_LOGIN = os.getenv("CHAT_REQUIRE_LOGIN", "true").lower() == "true"

# ---- Redis-backed conversation state (rolling history) ----
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    _r = redis.from_url(REDIS_URL, decode_responses=True)  # str in/out
except Exception:
    _r = None  # degrade to stateless if Redis is unavailable

CHAT_TTL = int(os.getenv("CHAT_TTL_SEC", "172800"))   # 2 days
HIST_TURNS = int(os.getenv("CHAT_HIST_TURNS", "3"))   # last 3 exchanges

# ---- Rate limit configuration (demo-friendly defaults) ----
# Per-user
RL_MIN_LIMIT_USER     = int(os.getenv("CHAT_LIMITS_PER_MIN_USER", "5"))    # 8/min
RL_DAY_LIMIT_USER     = int(os.getenv("CHAT_LIMITS_PER_DAY_USER", "40"))   # 40/day
# Anonymous fallback
RL_MIN_LIMIT_FALLBACK = int(os.getenv("CHAT_LIMITS_PER_MIN_FALLBACK", "3"))  # 3/min
RL_DAY_LIMIT_FALLBACK = int(os.getenv("CHAT_LIMITS_PER_DAY_FALLBACK", "10")) # 10/day
# System-wide daily cap
RL_DAY_LIMIT_SYSTEM   = int(os.getenv("CHAT_LIMITS_PER_DAY_SYSTEM", "250"))  # 250/day

# Daily window in America/Toronto (predictable reset for your users)
LOCAL_TZ = ZoneInfo(os.getenv("CHAT_LOCAL_TZ", "America/Toronto"))

def _epoch_minute(ts: float | None = None) -> int:
    ts = ts or time.time()
    return int(ts // 60)

def _today_local_str() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")

def _incr_with_ttl(r, key: str, ttl: int) -> int:
    """
    Atomic INCR with TTL; ensures TTL set even under concurrency.
    """
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, ttl)
    try:
        val, _ = pipe.execute()
        return int(val)
    except Exception:
        return 0

def _rate_limit_check(request: Request, user_id: int | None):
    """
    Enforce:
      - System-wide daily hard cap (all traffic)
      - Per-user (or anon fallback) daily hard cap
      - Per-user (or anon fallback) per-minute burst cap
    Returns None if allowed; raises HTTPException(429) if blocked.
    """
    if _r is None:
        return  # No Redis → skip limiting

    ip = request.client.host if request.client else "0.0.0.0"
    conv_id = request.cookies.get("conv_id") or "anon"

    # Identity & limits
    if user_id is not None:
        id_key   = f"user:{user_id}"
        day_lim  = RL_DAY_LIMIT_USER
        min_lim  = RL_MIN_LIMIT_USER
    else:
        id_key   = f"conv:{conv_id}:ip:{ip}"
        day_lim  = RL_DAY_LIMIT_FALLBACK
        min_lim  = RL_MIN_LIMIT_FALLBACK

    today  = _today_local_str()
    minute = _epoch_minute()

    # Keys
    k_day     = f"rl:{id_key}:d:{today}"
    k_min     = f"rl:{id_key}:m:{minute}"
    k_sys_day = f"rl:system:d:{today}"

    # INCR with TTLs (1 day + buffer; 70s minute bucket)
    sys_day_count = _incr_with_ttl(_r, k_sys_day, 24 * 60 * 60 + 300)
    if sys_day_count > RL_DAY_LIMIT_SYSTEM:
        raise HTTPException(
            status_code=429,
            detail="The demo's daily quota has been reached. Please try again tomorrow."
        )

    day_count = _incr_with_ttl(_r, k_day, 24 * 60 * 60 + 300)
    if day_count > day_lim:
        raise HTTPException(
            status_code=429,
            detail=f"Daily chat limit reached ({day_lim} messages). Please try again tomorrow."
        )

    min_count = _incr_with_ttl(_r, k_min, 70)
    if min_count > min_lim:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait ~60 seconds and try again."
        )

    # Optional: surface RL info to client for UX
    request.state.rl_headers = {
        "X-RateLimit-Identity": id_key,
        "X-RateLimit-Day-Count": str(day_count),
        "X-RateLimit-Day-Limit": str(day_lim),
        "X-RateLimit-Min-Count": str(min_count),
        "X-RateLimit-Min-Limit": str(min_lim),
        "X-RateLimit-System-Day-Count": str(sys_day_count),
        "X-RateLimit-System-Day-Limit": str(RL_DAY_LIMIT_SYSTEM),
    }

# ---------- Chat page ----------
@router.get("/chat")
def chat_page(request: Request, current_user = Depends(get_current_user)):
    # If login is required and user is not authenticated → redirect to login (with next)
    if REQUIRE_LOGIN and not current_user:
        request.session["flash_warning"] = "Please log in to use the chatbot."
        return RedirectResponse(url="/login?next=/chat", status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        "chatbot.html",
        {"request": request, "page": "chat"}
    )

# ---------- Chat agent API ----------
class ChatIn(BaseModel):
    message: str
    use_profile: bool = True
    restrict_to_catalog: bool = True

class BookOut(BaseModel):
    item_idx: int
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    cover_url: Optional[str] = None

class ChatOut(BaseModel):
    reply: str
    books: List[BookOut] = []

@router.post("/chat/agent", response_model=ChatOut)
def chat_agent(
    body: ChatIn,
    request: Request,
    response: Response,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Login gate for API: return 401 JSON (fetch won't navigate)
    if REQUIRE_LOGIN and not current_user:
        raise HTTPException(status_code=401, detail="Please log in to use the chatbot.")

    # ---- RATE LIMITS (user / anon / system) ----
    uid = getattr(current_user, "user_id", None) if current_user else None
    _rate_limit_check(request, uid)

    # Echo limit headers to client (optional UX)
    if hasattr(request.state, "rl_headers"):
        for k, v in request.state.rl_headers.items():
            response.headers[k] = v

    text = (body.message or "").strip()
    if not text:
        return ChatOut(reply="Ask me for book ideas or comparisons.", books=[])

    # Assign/get a conversation id cookie (issued only after auth passes, if required)
    conv_id = request.cookies.get("conv_id")
    if not conv_id:
        conv_id = uuid.uuid4().hex
        response.set_cookie("conv_id", conv_id, max_age=60 * 60 * 24 * 7, samesite="Lax")

    # Load short history from Redis
    hist: List[dict] = []
    key = f"chat:{conv_id}"
    if _r:
        try:
            hist = json.loads(_r.get(key) or "[]")
        except Exception:
            hist = []

    user_context_block = ""
    if body.use_profile and current_user:
        try:
            ctx = fetch_user_context(db, current_user.user_id, limit=15)
            block = format_user_context(ctx)
            user_context_block = f"(User profile context — may guide your answer)\n{block}\n\n"
        except Exception:
            user_context_block = ""

    # Build compact context (last N exchanges)
    ctx_lines: List[str] = []
    for turn in hist[-HIST_TURNS:]:
        u = turn.get("u")
        a = turn.get("a")
        if u:
            ctx_lines.append(f"User: {u}")
        if a:
            ctx_lines.append(f"Assistant: {a}")
    context = "\n".join(ctx_lines).strip()
    
    sections = []
    if user_context_block:
        sections.append(user_context_block)
    if context:
        sections.append(f"(Conversation so far — use only for context)\n{context}\n")
    sections.append(f"User: {text}")
    composed = "\n\n".join(sections)

    # Call the existing agent with composed input
    raw = answer(composed).strip()
    if raw.lower().startswith("final answer:"):
        raw = raw[len("final answer:"):].strip()

    # Save back last N exchanges with TTL
    hist.append({"u": text, "a": raw})
    if _r:
        try:
            _r.setex(key, CHAT_TTL, json.dumps(hist[-HIST_TURNS:]))
        except Exception:
            pass  # degrade silently if Redis hiccups

    return ChatOut(reply=raw, books=[])
