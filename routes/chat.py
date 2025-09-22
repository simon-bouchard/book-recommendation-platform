from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from datetime import datetime
from zoneinfo import ZoneInfo
from routes.auth import get_current_user
from sqlalchemy.orm import Session
from app.agents.orchestrator.conductor import Conductor
from app.database import get_db
from app.agents.schemas import ChatIn, ChatOut
from app.agents.runtime import rate_limit_check, ensure_conv_cookie, load_history, save_history, normalize_visible_reply, extract_book_ids_from_steps, build_books_from_ids
from app.agents.context_builder import build_composed_input
from app.agents.logging import get_logger, chatbot_logger
from models.shared_utils import get_user_num_ratings

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

@router.get("/chat")
def chat_page(request: Request, current_user = Depends(get_current_user)):
    if REQUIRE_LOGIN and not current_user:
        request.session["flash_warning"] = "Please log in to use the chatbot."
        return RedirectResponse(url="/login?next=/chat", status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        "chatbot.html",
        {"request": request, "page": "chat", "logged_in": bool(current_user)}
    )

@router.post("/chat/agent", response_model=ChatOut)
def chat_agent(
    body: ChatIn,
    request: Request,
    response: Response,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if REQUIRE_LOGIN and not current_user:
        raise HTTPException(status_code=401, detail="Please log in to use the chatbot.")

    conv_id = ensure_conv_cookie(request, response)

    uid = getattr(current_user, "user_id", None) if current_user else None
    rate_limit_check(request, uid)
    if hasattr(request.state, "rl_headers"):
        for k, v in request.state.rl_headers.items():
            response.headers[k] = v

    text = (body.message or "").strip()
    if not text:
        return ChatOut(reply="Ask me for book ideas or comparisons.", books=[])

    # request boundary in chatbot log
    chatbot_logger.info("request_start", extra={"extra": {
        "conv_id": conv_id, "uid": uid, "use_profile": bool(body.use_profile), "q_preview": text[:160]
    }})

    hist = load_history(conv_id)
    composed = build_composed_input(
        db=db,
        user_id=uid,
        use_profile=body.use_profile,
        history=hist,
        user_text=text,
        hist_turns=HIST_TURNS,
    )

    user_num_ratings = 0
    if current_user and hasattr(current_user, "user_id"):
        user_num_ratings = get_user_num_ratings(current_user.user_id)

    # --- FIX: no conv_id / uid passed to answer() ---
    res = Conductor().run(
        composed,
        current_user=current_user,
        db=db,
        user_num_ratings=user_num_ratings
    )

    raw_text = (res.get("text") or "").strip()
    steps = res.get("intermediate_steps") or []

    reply_text = normalize_visible_reply(raw_text)

    legacyish_steps = [{"tool": c.name, "input": c.args} for c in (res.tool_calls or [])]
    ids = extract_book_ids_from_steps(legacyish_steps) or []
    books = build_books_from_ids(ids)

    hist.append({"u": text, "a": reply_text})
    save_history(conv_id, hist)

    chatbot_logger.info("request_end", extra={"extra": {
        "conv_id": conv_id, "uid": uid, "num_books": len(books), "reply_len": len(reply_text), "ids": ids[:12]
    }})

    return ChatOut(
        target=res.target,
        text=reply_text,
        books=books,
        steps=res.tool_calls,
        citations=res.citations,
    )
