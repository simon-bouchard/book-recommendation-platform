from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime 
from app.agents.web_agent import answer 

router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow  # <-- add this

# ---------- Chat page ----------
@router.get("/chat")
def chat_page(request: Request):
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
def chat_agent(body: ChatIn):
    text = body.message.strip()
    if not text:
        return ChatOut(reply="Ask me for book ideas or comparisons.", books=[])

    raw = answer(text).strip()
    # Strip the control prefix for display
    if raw.lower().startswith("final answer:"):
        raw = raw[len("final answer:"):].strip()

    return ChatOut(reply=raw, books=[])
