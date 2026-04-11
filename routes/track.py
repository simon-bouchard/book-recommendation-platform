# routes/track.py
"""
Lightweight click-tracking endpoint for book recommendation CTR monitoring.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from metrics import BOOK_CLICK_TOTAL

router = APIRouter()

_VALID_SOURCES = {"recommendations", "similar", "chatbot"}
_VALID_MODES = {"subject", "als", "hybrid", "behavioral", "auto", "chatbot"}


class ClickEvent(BaseModel):
    item_idx: int
    source: str
    mode: str


@router.post("/track/click")
async def track_click(body: ClickEvent) -> dict:
    if body.source not in _VALID_SOURCES or body.mode not in _VALID_MODES:
        raise HTTPException(status_code=400, detail="Invalid source or mode")
    BOOK_CLICK_TOTAL.labels(source=body.source, mode=body.mode).inc()
    return {"ok": True}
