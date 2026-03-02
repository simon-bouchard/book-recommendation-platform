# model_servers/als/main.py
"""
ALS recommendation model server.

Owns raw (non-normalized) user and book ALS factors. Normalization would change
the semantics of the dot product — factor magnitudes encode predicted interaction
strength and must be preserved. The similarity server holds a separate normalized
copy for cosine similarity; the two representations cannot be shared.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from model_servers._shared.contracts import (
    AlsRecsRequest,
    AlsRecsResponse,
    HasAlsUserRequest,
    HasAlsUserResponse,
    HealthResponse,
    ScoredItem,
)
from models.core.reload_poller import ModelReloadPoller
from models.infrastructure.als_model import ALSModel

logger = logging.getLogger(__name__)

_SERVER_NAME = "als"


def _get_artifact_version() -> str:
    """
    Read the current artifact version string from the version pointer file.

    Returns 'unknown' if the pointer file is absent, which prevents a missing
    file from blocking the health check.
    """
    pointer_path = os.environ.get("MODEL_VERSION_POINTER", "")
    if pointer_path and os.path.exists(pointer_path):
        try:
            with open(pointer_path) as f:
                return f.read().strip()
        except OSError:
            pass
    return "unknown"


def _load_artifacts() -> None:
    """
    Force-initialize the ALSModel singleton.

    ALSModel() with no arguments triggers disk loading of raw user and book
    factors via load_als_factors(normalized=False). Subsequent calls within
    this process return the cached instance at no cost.
    """
    logger.info("Loading ALS server artifacts...")
    start = time.monotonic()
    ALSModel()
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("ALS server artifacts loaded in %.0fms", elapsed_ms)


def _reload_artifacts() -> None:
    """
    Clear the ALSModel singleton and reload from disk.

    Called by the reload poller when a new ALS training signal is detected.
    The loader cache is also cleared so the new factor files are read rather
    than the old cached arrays.
    """
    from models.data.loaders import clear_cache

    logger.info("Reloading ALS server artifacts...")
    start = time.monotonic()

    ALSModel.reset()
    clear_cache()
    ALSModel()

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("ALS server artifacts reloaded in %.0fms", elapsed_ms)


class _ALSReloadPoller(ModelReloadPoller):
    """Reload poller bound to the ALS server's artifact set."""

    async def _reload_models(self) -> None:
        _reload_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts and start reload poller on startup; stop on shutdown."""
    _load_artifacts()

    poller = _ALSReloadPoller()
    await poller.start()

    yield

    await poller.stop()


app = FastAPI(
    title="ALS Recommendation Model Server",
    description=(
        "Collaborative filtering recommendations via raw ALS dot product. "
        "Factor magnitudes carry semantic meaning and are never normalized."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ===========================================================================
# Health
# ===========================================================================


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Liveness and readiness check.

    Returns 503 if the ALSModel singleton has not been initialized, covering
    the startup window and any failed reload.
    """
    if ALSModel._instance is None:
        raise HTTPException(status_code=503, detail="ALS server not initialized")

    return HealthResponse(
        status="ok",
        server=_SERVER_NAME,
        artifact_version=_get_artifact_version(),
    )


# ===========================================================================
# User ALS membership
# ===========================================================================


@app.post("/has_als_user", response_model=HasAlsUserResponse)
def has_als_user(request: HasAlsUserRequest) -> HasAlsUserResponse:
    """
    Check whether a user has ALS factors (warm/cold gate).

    Cold users (absent from the ALS model) should be routed to the subject
    recommendation pipeline. Warm users may use either pipeline.
    """
    return HasAlsUserResponse(
        user_id=request.user_id,
        is_warm=ALSModel().has_user(request.user_id),
    )


# ===========================================================================
# ALS recommendations
# ===========================================================================


@app.post("/als_recs", response_model=AlsRecsResponse)
def als_recs(request: AlsRecsRequest) -> AlsRecsResponse:
    """
    Generate top-k recommendations via raw dot product: book_factors @ user_vector.

    Returns an empty result list for cold users rather than raising an error,
    allowing the application layer to fall back gracefully without special-casing
    HTTP error codes.
    """
    try:
        item_ids, scores = ALSModel().score(user_id=request.user_id, k=request.k)
        return AlsRecsResponse(
            results=[
                ScoredItem(item_idx=int(iid), score=float(s)) for iid, s in zip(item_ids, scores)
            ]
        )
    except Exception as e:
        logger.error("als_recs failed for user %d: %s", request.user_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="ALS recommendation failed")
