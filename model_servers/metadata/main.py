# model_servers/metadata/main.py
"""
Metadata model server.

Owns the book metadata lookup dict and Bayesian popularity scores. Provides
book enrichment and popular books retrieval.

All enrichment logic lives in models/infrastructure/metadata_enrichment.py.
This file is purely wiring: lifespan, app instantiation, and endpoint handlers.

Enrich and popular endpoints return raw Response objects with pre-serialized
JSON bodies, bypassing FastAPI's response_model serialization entirely.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from model_servers._shared.contracts import (
    EnrichRequest,
    HealthResponse,
    PopularRequest,
)
from models.core.reload_poller import ModelReloadPoller
from models.data.loaders import load_book_meta
from models.infrastructure.metadata_enrichment import build_lookup, enrich_items
from models.infrastructure.popularity_scorer import PopularityScorer

logger = logging.getLogger(__name__)

_SERVER_NAME = "metadata"

_book_lookup: Optional[dict[int, str]] = None


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
    Build the book lookup dict and initialize the PopularityScorer singleton.

    PopularityScorer is initialized after the lookup dict so that the loader
    cache warmed by load_book_meta can be reused.
    """
    global _book_lookup

    logger.info("Loading metadata server artifacts...")
    start = time.monotonic()

    _book_lookup = build_lookup(load_book_meta(use_cache=True))
    PopularityScorer()

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "Metadata server artifacts loaded in %.0fms — %d books indexed",
        elapsed_ms,
        len(_book_lookup),
    )


def _reload_artifacts() -> None:
    """Clear all owned artifacts and rebuild from disk."""
    from models.data.loaders import clear_cache

    global _book_lookup

    logger.info("Reloading metadata server artifacts...")
    start = time.monotonic()

    PopularityScorer.reset()
    clear_cache()

    _book_lookup = build_lookup(load_book_meta(use_cache=True))
    PopularityScorer()

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("Metadata server artifacts reloaded in %.0fms", elapsed_ms)


class _MetadataReloadPoller(ModelReloadPoller):
    """Reload poller bound to the metadata server's artifact set."""

    async def _reload_models(self) -> None:
        _reload_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts and start reload poller on startup; stop on shutdown."""
    _load_artifacts()

    poller = _MetadataReloadPoller()
    await poller.start()

    yield

    await poller.stop()


app = FastAPI(
    title="Metadata Model Server",
    description="Book metadata enrichment and Bayesian popularity retrieval.",
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

    Both the lookup dict and PopularityScorer must be initialized. The scorer
    is checked last since it is initialized after the lookup, making its
    presence a proxy for full readiness.
    """
    if _book_lookup is None or PopularityScorer._instance is None:
        raise HTTPException(status_code=503, detail="Metadata server not initialized")

    return HealthResponse(
        status="ok",
        server=_SERVER_NAME,
        artifact_version=_get_artifact_version(),
    )


# ===========================================================================
# Enrich
# ===========================================================================


@app.post("/enrich")
def enrich(request: EnrichRequest) -> Response:
    if _book_lookup is None:
        raise HTTPException(status_code=503, detail="Metadata server not initialized")

    content = enrich_items(_book_lookup, request.item_indices)

    return Response(content=content, media_type="application/json")


@app.post("/popular")
def popular(request: PopularRequest) -> Response:
    """
    Retrieve the top-k books ranked by precomputed Bayesian score.

    Returns a raw JSON response body assembled from pre-serialized per-book
    strings. Books absent from the metadata lookup are silently skipped.
    """
    if _book_lookup is None or PopularityScorer._instance is None:
        raise HTTPException(status_code=503, detail="Metadata server not initialized")

    try:
        item_ids, _ = PopularityScorer().top_k(k=request.k)
    except Exception as e:
        logger.error("popular failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Popular books retrieval failed")

    return Response(
        content=enrich_items(_book_lookup, item_ids.tolist()),
        media_type="application/json",
    )
