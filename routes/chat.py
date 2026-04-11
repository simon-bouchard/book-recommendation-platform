import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from sqlalchemy.orm import Session

from app.agents.logging import chatbot_logger, get_logger
from app.agents.orchestrator.conductor import Conductor
from app.agents.runtime import (
    ensure_conv_cookie,
    load_history,
    normalize_visible_reply,
    rate_limit_check,
    save_history,
)
from app.agents.schemas import ChatIn
from app.agents.settings import settings
from app.database import get_db
from metrics import BOOK_IMPRESSION_TOTAL, CHAT_LATENCY, CHAT_REQUESTS
from models.data.queries import get_user_num_ratings
from routes.auth import get_current_user

tracer = trace.get_tracer(__name__)
logger = get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env.globals["now"] = datetime.utcnow

REQUIRE_LOGIN = settings.chat_require_login
REDIS_URL = settings.redis_url
CHAT_TTL = settings.chat_ttl_sec
HIST_TURNS = settings.chat_hist_turns
RL_MIN_LIMIT_USER = settings.chat_limits_per_min_user
RL_DAY_LIMIT_USER = settings.chat_limits_per_day_user
RL_MIN_LIMIT_FALLBACK = settings.chat_limits_per_min_fallback
RL_DAY_LIMIT_FALLBACK = settings.chat_limits_per_day_fallback
RL_DAY_LIMIT_SYSTEM = settings.chat_limits_per_day_system
LOCAL_TZ = ZoneInfo(settings.chat_local_tz)


@router.get("/chat")
def chat_page(request: Request, current_user=Depends(get_current_user)):
    if REQUIRE_LOGIN and not current_user:
        request.session["flash_warning"] = "Please log in to use the chatbot."
        return RedirectResponse(url="/login?next=/chat", status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        "chat_shell.html", {"request": request, "logged_in": bool(current_user)}
    )


@router.post("/chat/stream")
async def chat_agent_stream(
    body: ChatIn,
    request: Request,
    response: Response,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Streaming version of chat_agent that yields Server-Sent Events.

    Yields events with data in JSON format:
        - {"type": "status", "content": "Processing..."}
        - {"type": "token", "content": "word"}
        - {"type": "complete", "data": {...}}
    """
    if REQUIRE_LOGIN and not current_user:
        raise HTTPException(status_code=401, detail="Please log in to use the chatbot.")

    conv_id = ensure_conv_cookie(request, response)

    uid = getattr(current_user, "user_id", None) if current_user else None
    rate_limit_check(request, uid)

    text = (body.user_text or "").strip()
    if not text:

        async def empty_response():
            yield f"data: {json.dumps({'type': 'token', 'content': 'Ask me for book ideas or comparisons.'})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'data': {'target': 'respond', 'success': True}})}\n\n"

        return StreamingResponse(
            empty_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    chatbot_logger.info(
        "stream_request_start",
        extra={
            "extra": {
                "conv_id": conv_id,
                "uid": uid,
                "use_profile": bool(body.use_profile),
                "q_preview": text[:160],
            }
        },
    )

    # Load rolling history
    hist = load_history(conv_id, user_id=uid)

    # Warm/cold signal
    user_num_ratings = 0
    if current_user and hasattr(current_user, "user_id"):
        user_num_ratings = get_user_num_ratings(current_user.user_id)

    # Capture the active OTEL context (contains the FastAPI root span) before
    # returning StreamingResponse. The root span closes when the response headers
    # are sent, so we must re-attach this context inside the generator to keep
    # chat.conductor correctly parented in Jaeger.
    captured_ctx = otel_context.get_current()

    async def generate_stream():
        """Generate Server-Sent Events stream."""
        accumulated_text = []
        final_data = None
        start_time = time.time()

        ctx_token = otel_context.attach(captured_ctx)
        try:
            with tracer.start_as_current_span("chat.conductor") as conductor_span:
                conductor_span.set_attribute("chat.conv_id", conv_id)
                conductor_span.set_attribute("chat.use_profile", bool(body.use_profile))
                if uid is not None:
                    conductor_span.set_attribute("chat.uid", uid)

                try:
                    # Stream from conductor
                    async for chunk in Conductor().run_stream(
                        history=hist,
                        user_text=text,
                        use_profile=bool(body.use_profile),
                        current_user=current_user,
                        db=db,
                        user_num_ratings=user_num_ratings,
                        hist_turns=HIST_TURNS,
                        conv_id=conv_id,
                        uid=uid,
                        force_target=body.force_target,
                    ):
                        # Send chunk as SSE
                        chunk_data = {
                            "type": chunk.type,
                            "content": chunk.content,
                            "data": chunk.data,
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                        # Accumulate text tokens
                        if chunk.type == "token" and chunk.content:
                            accumulated_text.append(chunk.content)

                        # Capture final data
                        if chunk.type == "complete" and chunk.data:
                            final_data = chunk.data

                    # After streaming completes, save history
                    if accumulated_text:
                        reply_text = "".join(accumulated_text)
                        reply_text = normalize_visible_reply(reply_text)

                        # Update history
                        hist.append({"u": text, "a": reply_text})
                        save_history(conv_id, hist, user_id=uid)

                        # Log completion
                        target = final_data.get("target", "unknown") if final_data else "unknown"
                        book_ids = final_data.get("book_ids", []) if final_data else []

                        chatbot_logger.info(
                            "stream_request_end",
                            extra={
                                "extra": {
                                    "conv_id": conv_id,
                                    "uid": uid,
                                    "target": target,
                                    "num_books": len(book_ids),
                                    "reply_len": len(reply_text),
                                    "ids": book_ids[:12] if book_ids else [],
                                }
                            },
                        )
                        CHAT_REQUESTS.labels(target=target).inc()
                        CHAT_LATENCY.labels(target=target).observe(time.time() - start_time)
                        if book_ids:
                            BOOK_IMPRESSION_TOTAL.labels(
                                source="chatbot", mode="chatbot"
                            ).inc(len(book_ids))

                except Exception as e:
                    conductor_span.record_exception(e)
                    conductor_span.set_status(Status(StatusCode.ERROR))
                    logger.error(f"Streaming error: {e}", exc_info=True)
                    error_chunk = {
                        "type": "complete",
                        "data": {
                            "target": "error",
                            "success": False,
                            "text": "I encountered an error processing your request. Please try again.",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
        finally:
            try:
                otel_context.detach(ctx_token)
            except Exception:
                pass

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
