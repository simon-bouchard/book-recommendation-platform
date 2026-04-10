# main.py
"""
FastAPI application entry point for the bookrec web service.

Stateless application layer: owns no ML artifacts. All model operations
are delegated to model servers via HTTP clients managed by the client registry.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.telemetry.tracing import setup_tracing, shutdown_tracing
from metrics import apply_metrics
from models.client._exceptions import ModelServerError, ModelServerUnavailableError
from models.client.registry import (
    close_all,
    get_als_client,
    get_embedder_client,
    get_metadata_client,
    get_semantic_client,
    get_similarity_client,
)
from routes.api import router
from routes.auth import get_current_user
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.models import router as models_router

load_dotenv()

logger = logging.getLogger(__name__)

SECURE_MODE = os.getenv("SECURE_MODE", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize tracing and client connection pools on startup; drain them on shutdown.

    Tracing is set up first so that any spans created during client initialization
    are captured. No artifacts are loaded here — the model servers own all ML state.
    Clients warm their connection pools lazily on first request, so startup is fast.
    """
    setup_tracing(app)

    from app.database import close_aiomysql_pool, init_aiomysql_pool

    await init_aiomysql_pool()
    logger.info("aiomysql pool initialized")

    get_embedder_client()
    get_similarity_client()
    get_als_client()
    get_metadata_client()
    get_semantic_client()
    logger.info("Model server clients initialized")

    yield

    await close_all()
    logger.info("Model server clients closed")

    await close_aiomysql_pool()
    logger.info("aiomysql pool closed")

    shutdown_tracing()


app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)


# ===========================================================================
# Security / rate limiting
# ===========================================================================


def _no_limit():
    return None


if SECURE_MODE:
    from security_settings import apply_security, health_dependency

    apply_security(app)
    limiter_dep = health_dependency()
else:
    limiter_dep = _no_limit


# ===========================================================================
# Metrics
# ===========================================================================

apply_metrics(app)


# ===========================================================================
# Exception handlers
# ===========================================================================


@app.exception_handler(ModelServerUnavailableError)
async def model_server_unavailable_handler(
    request: Request, exc: ModelServerUnavailableError
) -> JSONResponse:
    """
    Translate a downstream model server connectivity failure into HTTP 503.

    This keeps 503 semantics distinct from 500 (application bugs) and allows
    load balancers and clients to apply appropriate retry logic.
    """
    logger.error("Model server unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Model server unavailable"})


@app.exception_handler(ModelServerError)
async def model_server_error_handler(request: Request, exc: ModelServerError) -> JSONResponse:
    """
    Translate any other model server error (e.g. bad request) into HTTP 503.

    ModelServerRequestError indicates a programming error in the client, not a
    transient fault, but it still means the request cannot be fulfilled.
    """
    logger.error("Model server error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Model server error"})


# ===========================================================================
# Health probes
# ===========================================================================


@app.get("/health/live")
def health_live():
    """
    Liveness probe.

    Returns 200 as long as the process is running. Container runtimes use
    this to decide whether to restart the container — it must never call
    downstream services, since a slow or failed model server should not
    trigger a process restart.
    """
    return {"status": "ok"}


@app.get("/health/ready", dependencies=[Depends(limiter_dep)])
async def health_ready():
    """
    Readiness probe.

    Calls each model server's /health endpoint concurrently. Returns 200 only
    when all four servers respond with a non-error status. Returns 503 with a
    per-server breakdown when any server is unreachable or not yet initialized.

    Load balancers use this to gate traffic: an instance reporting not-ready
    should not receive user requests, but the process should not be restarted.
    """
    checks = {
        "embedder": get_embedder_client(),
        "similarity": get_similarity_client(),
        "als": get_als_client(),
        "metadata": get_metadata_client(),
        "semantic": get_semantic_client(),
    }

    async def probe(name: str, client) -> tuple[str, bool, str]:
        try:
            await client._get("/health")
            return name, True, "ok"
        except ModelServerUnavailableError as exc:
            return name, False, str(exc)
        except Exception as exc:
            return name, False, str(exc)

    results = await asyncio.gather(*[probe(name, client) for name, client in checks.items()])

    statuses = {name: {"ok": ok, "detail": detail} for name, ok, detail in results}
    all_ready = all(r[1] for r in results)

    if not all_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "servers": statuses},
        )

    return {"status": "ok", "servers": statuses}


# ===========================================================================
# Static assets, middleware, templates
# ===========================================================================

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

templates = Jinja2Templates(directory="templates")
templates.env.globals["now"] = datetime.utcnow


# ===========================================================================
# Routers
# ===========================================================================

app.include_router(router)
app.include_router(chat_router)
app.include_router(auth_router)
app.include_router(models_router)


# ===========================================================================
# Pages
# ===========================================================================


@app.get("/", response_class=HTMLResponse)
def root(request: Request, current_user=Depends(get_current_user)):
    return templates.TemplateResponse(
        "home_shell.html", {"request": request, "logged_in": bool(current_user)}
    )
