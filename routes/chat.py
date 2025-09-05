from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from app.agents.web_agent import answer
from app.auth import get_current_user  # <-- reuse existing auth helper
import os, json, uuid
import redis

router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow  # keep existing

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
def chat_agent(body: ChatIn, request: Request, response: Response, current_user = Depends(get_current_user)):
    # Hard gate: block anonymous calls when login is required
    if REQUIRE_LOGIN and not current_user:
        raise HTTPException(status_code=401, detail="Please log in to use the chatbot.")
    
    text = body.message.strip()
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
    composed = f"(Conversation so far — use only for context)\n{context}\n\nUser: {text}" if context else text

    # Call the existing agent with composed input
    raw = answer(composed).strip()

    # Strip the control prefix for display
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
