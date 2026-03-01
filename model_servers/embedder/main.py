# model_servers/embedder/main.py
"""
Embedder model server.

Owns the PyTorch attention model weights and exposes a single embed operation.
Consumers: similarity server (for subject_recs), application layer (for candidate generation).
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from model_servers._shared.contracts import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
)
from models.core.reload_poller import ModelReloadPoller
from models.infrastructure.subject_embedder import SubjectEmbedder

logger = logging.getLogger(__name__)

_SERVER_NAME = "embedder"
_startup_time: float = 0.0


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
    Force-initialize the SubjectEmbedder singleton.

    Calling SubjectEmbedder() with no arguments triggers the singleton's
    lazy loader, which reads the attention strategy from ATTN_STRATEGY and
    loads the corresponding .pth file from disk. Subsequent calls within
    this process return the cached instance at no cost.
    """
    logger.info("Loading embedder artifacts...")
    start = time.monotonic()
    SubjectEmbedder()
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("Embedder artifacts loaded in %.0fms", elapsed_ms)


def _reload_artifacts() -> None:
    """
    Clear the SubjectEmbedder singleton and reload from disk.

    Called by the reload poller when a new signal timestamp is detected.
    The loaders cache for the attention strategy is also cleared so the
    new .pth file is read rather than the old cached instance.
    """
    from models.data.loaders import clear_cache

    logger.info("Reloading embedder artifacts...")
    start = time.monotonic()

    SubjectEmbedder.reset()
    clear_cache()
    SubjectEmbedder()

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("Embedder artifacts reloaded in %.0fms", elapsed_ms)


class _EmbedderReloadPoller(ModelReloadPoller):
    """Reload poller bound to the embedder's artifact set."""

    async def _reload_models(self) -> None:
        _reload_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts and start the reload poller on startup; stop poller on shutdown."""
    global _startup_time

    _load_artifacts()
    _startup_time = time.monotonic()

    poller = _EmbedderReloadPoller()
    await poller.start()

    yield

    await poller.stop()


app = FastAPI(
    title="Embedder Model Server",
    description="Computes normalized subject embeddings from subject index lists.",
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

    Returns 503 if the embedder singleton has not been initialized, which
    covers the brief window between container start and artifact load
    completion as well as any failed reload.
    """
    embedder = SubjectEmbedder._instance
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialized")

    return HealthResponse(
        status="ok",
        server=_SERVER_NAME,
        artifact_version=_get_artifact_version(),
    )


# ===========================================================================
# Embed
# ===========================================================================


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    """
    Compute a normalized embedding vector for a list of subject indices.

    The output vector is L2-normalized and ready for cosine similarity
    computations via matrix multiplication on the similarity server.
    """
    embedder = SubjectEmbedder._instance
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialized")

    try:
        vector = embedder.embed(request.subject_indices)
        return EmbedResponse(vector=vector.tolist())

    except Exception as e:
        logger.error("Embed failed for subjects %s: %s", request.subject_indices, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding computation failed")
