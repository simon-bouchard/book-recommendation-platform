# model_servers/metadata/main.py
"""
Metadata model server.

Owns the book metadata DataFrame and Bayesian popularity scores. Provides book
enrichment and popular books retrieval. Subject search is not implemented here
and will be added as a separate service when a sentence-transformer-based
approach replaces the TF-IDF design.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from model_servers._shared.contracts import (
    BookMeta,
    EnrichRequest,
    EnrichResponse,
    HealthResponse,
    PopularRequest,
    PopularResponse,
)
from models.core.reload_poller import ModelReloadPoller
from models.data.loaders import load_book_meta
from models.infrastructure.popularity_scorer import PopularityScorer

logger = logging.getLogger(__name__)

_SERVER_NAME = "metadata"

_book_meta: Optional[pd.DataFrame] = None


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
    Initialize the book metadata DataFrame and PopularityScorer singleton.

    PopularityScorer is initialized second so that the loader cache warmed by
    load_book_meta can be reused if the scorer needs overlapping data.
    """
    global _book_meta

    logger.info("Loading metadata server artifacts...")
    start = time.monotonic()

    _book_meta = load_book_meta(use_cache=True)
    PopularityScorer()

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("Metadata server artifacts loaded in %.0fms", elapsed_ms)


def _reload_artifacts() -> None:
    """Clear all owned artifacts and reload from disk."""
    from models.data.loaders import clear_cache

    global _book_meta

    logger.info("Reloading metadata server artifacts...")
    start = time.monotonic()

    PopularityScorer.reset()
    clear_cache()

    _book_meta = load_book_meta(use_cache=True)
    PopularityScorer()

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("Metadata server artifacts reloaded in %.0fms", elapsed_ms)


def _row_to_book_meta(item_idx: int, row: pd.Series) -> BookMeta:
    """
    Convert a book metadata DataFrame row to a BookMeta contract object.

    All optional fields are guarded against NaN and empty strings, both of
    which can appear in the raw DataFrame for books with incomplete records.
    """

    def _opt_str(key: str) -> Optional[str]:
        val = row.get(key)
        return str(val) if val and not (isinstance(val, float) and np.isnan(val)) else None

    def _opt_int(key: str) -> Optional[int]:
        val = row.get(key)
        try:
            return int(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None
        except (ValueError, TypeError):
            return None

    def _opt_float(key: str) -> Optional[float]:
        val = row.get(key)
        try:
            f = float(val)
            return f if np.isfinite(f) else None
        except (ValueError, TypeError):
            return None

    return BookMeta(
        item_idx=item_idx,
        title=str(row["title"]),
        author=_opt_str("author"),
        year=_opt_int("year"),
        isbn=_opt_str("isbn"),
        cover_id=_opt_str("cover_id"),
        avg_rating=_opt_float("book_avg_rating"),
        num_ratings=int(row["book_num_ratings"]) if "book_num_ratings" in row else 0,
        bayes_score=_opt_float("bayes"),
    )


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

    Both the metadata DataFrame and PopularityScorer must be initialized.
    PopularityScorer is checked via its singleton since it is the last artifact
    loaded, making its presence a proxy for full readiness.
    """
    if _book_meta is None or PopularityScorer._instance is None:
        raise HTTPException(status_code=503, detail="Metadata server not initialized")

    return HealthResponse(
        status="ok",
        server=_SERVER_NAME,
        artifact_version=_get_artifact_version(),
    )


# ===========================================================================
# Enrich
# ===========================================================================


@app.post("/enrich", response_model=EnrichResponse)
def enrich(request: EnrichRequest) -> EnrichResponse:
    """
    Fetch full book metadata for a list of item indices.

    Items absent from the metadata store are silently omitted. The response
    list may therefore be shorter than the request list.
    """
    if _book_meta is None:
        raise HTTPException(status_code=503, detail="Metadata server not initialized")

    books = []
    for item_idx in request.item_indices:
        if item_idx not in _book_meta.index:
            continue
        try:
            books.append(_row_to_book_meta(item_idx, _book_meta.loc[item_idx]))
        except Exception as e:
            logger.warning("Failed to enrich item %d: %s", item_idx, e)

    return EnrichResponse(books=books)


# ===========================================================================
# Popular
# ===========================================================================


@app.post("/popular", response_model=PopularResponse)
def popular(request: PopularRequest) -> PopularResponse:
    """
    Retrieve the top-k books ranked by precomputed Bayesian score.

    Books in the popularity ranking that are absent from the metadata store
    are silently skipped, so the response may contain fewer than k items.
    """
    if _book_meta is None or PopularityScorer._instance is None:
        raise HTTPException(status_code=503, detail="Metadata server not initialized")

    try:
        item_ids, _ = PopularityScorer().top_k(k=request.k)
    except Exception as e:
        logger.error("popular failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Popular books retrieval failed")

    books = []
    for item_idx in item_ids.tolist():
        if item_idx not in _book_meta.index:
            continue
        try:
            books.append(_row_to_book_meta(item_idx, _book_meta.loc[item_idx]))
        except Exception as e:
            logger.warning("Failed to enrich popular item %d: %s", item_idx, e)

    return PopularResponse(books=books)
