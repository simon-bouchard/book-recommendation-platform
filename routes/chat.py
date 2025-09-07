from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from datetime import datetime
from zoneinfo import ZoneInfo
from routes.auth import get_current_user
from sqlalchemy.orm import Session
from app.agents.web_agent import answer
from app.database import get_db
from app.agents.schemas import ChatIn, ChatOut, BookOut
from app.agents.context_builder import build_composed_input
from app.agents.runtime import rate_limit_check, ensure_conv_cookie, load_history, save_history
from app.agents.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow

from app.agents.settings import settings

REQUIRE_LOGIN = settings.chat_require_login
REDIS_URL = settings.redis_url
CHAT_TTL = settings.chat_ttl_sec
HIST_TURNS = settings.chat_hist_turns
RL_MIN_LIMIT_USER     = settings.chat_limits_per_min_user
RL_DAY_LIMIT_USER     = settings.chat_limits_per_day_user
RL_MIN_LIMIT_FALLBACK = settings.chat_limits_per_min_fallback
RL_DAY_LIMIT_FALLBACK = settings.chat_limits_per_day_fallback
RL_DAY_LIMIT_SYSTEM   = settings.chat_limits_per_day_system
LOCAL_TZ = ZoneInfo(settings.chat_local_tz)

# ---------- Chat page ----------
@router.get("/chat")
def chat_page(request: Request, current_user = Depends(get_current_user)):
    # If login is required and user is not authenticated → redirect to login (with next)
    if REQUIRE_LOGIN and not current_user:
        request.session["flash_warning"] = "Please log in to use the chatbot."
        return RedirectResponse(url="/login?next=/chat", status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        "chatbot.html",
        {"request": request, "page": "chat", "logged_in": bool(current_user)}
    )

# ---------- Chat agent API ----------
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

    # Assign/get a conversation id cookie (issued only after auth passes, if required)
    conv_id = ensure_conv_cookie(request, response)
    
    # ---- RATE LIMITS (user / anon / system) ----
    uid = getattr(current_user, "user_id", None) if current_user else None
    rate_limit_check(request, uid)

    # Echo limit headers to client (optional UX)
    if hasattr(request.state, "rl_headers"):
        for k, v in request.state.rl_headers.items():
            response.headers[k] = v

    text = (body.message or "").strip()
    if not text:
        return ChatOut(reply="Ask me for book ideas or comparisons.", books=[])

    # Load short history from Redis
    hist = load_history(conv_id)

    # Build the composed input (profile + history + new message)
    composed = build_composed_input(
        db=db,
        user_id=uid,
        use_profile=body.use_profile,
        history=hist,
        user_text=text,
        hist_turns=HIST_TURNS,
    )

    # Call the agent
    raw = answer(composed).strip()
    if raw[:13].lower() == "final answer:":
        result = raw[13:].lstrip()
    
    # Save back last N exchanges with TTL
    hist.append({"u": text, "a": raw})
    save_history(conv_id, hist)

    return ChatOut(reply=result, books=[])
