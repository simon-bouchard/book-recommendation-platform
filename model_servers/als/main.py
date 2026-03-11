# model_servers/als/main.py
"""
ALS recommendation model server.

Owns raw (non-normalized) user and book ALS factors. Normalization would change
the semantics of the dot product — factor magnitudes encode predicted interaction
strength and must be preserved. The similarity server holds a separate normalized
copy for cosine similarity; the two representations cannot be shared.
"""

import logging
import time

from fastapi import FastAPI, HTTPException

from model_servers._shared.contracts import (
    AlsRecsRequest,
    AlsRecsResponse,
    HasAlsUserRequest,
    HasAlsUserResponse,
    HealthResponse,
    ScoredItem,
)
from model_servers._shared.server_utils import get_artifact_version, make_lifespan
from models.infrastructure.als_model import ALSModel

logger = logging.getLogger(__name__)

_SERVER_NAME = "als"


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


lifespan = make_lifespan(_load_artifacts)

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
        artifact_version=get_artifact_version(),
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
