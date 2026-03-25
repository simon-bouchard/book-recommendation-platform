# model_servers/embedder/main.py
"""
Embedder model server.

Owns the PyTorch attention model weights and exposes a single embed operation.
Consumers: similarity server (for subject_recs), application layer (for candidate generation).
"""

import logging
import time

from fastapi import FastAPI, HTTPException, Response

from model_servers._shared.contracts import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
)
from model_servers._shared.server_utils import get_artifact_version, make_lifespan
from models.infrastructure.subject_embedder import SubjectEmbedder

logger = logging.getLogger(__name__)

_SERVER_NAME = "embedder"


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


lifespan = make_lifespan(_load_artifacts)

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
    if SubjectEmbedder._instance is None:
        raise HTTPException(status_code=503, detail="Embedder not initialized")

    return HealthResponse(
        status="ok",
        server=_SERVER_NAME,
        artifact_version=get_artifact_version(),
    )


# ===========================================================================
# Embed
# ===========================================================================


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest, response: Response) -> EmbedResponse:
    """
    Compute a normalized embedding vector for a list of subject indices.

    The output vector is L2-normalized and ready for cosine similarity
    computations via matrix multiplication on the similarity server.
    """
    embedder = SubjectEmbedder._instance
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialized")

    try:
        t0 = time.perf_counter()
        vector = embedder.embed(request.subject_indices)
        response.headers["X-Compute-Ms"] = f"{(time.perf_counter() - t0) * 1000:.3f}"
        return EmbedResponse(vector=vector.tolist())

    except Exception as e:
        logger.error("Embed failed for subjects %s: %s", request.subject_indices, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding computation failed")
