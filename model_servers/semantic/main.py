# model_servers/semantic/main.py
"""
Semantic search model server.

Owns the FAISS dense vector index over book descriptions and the
sentence-transformers model used to embed free-text queries. Returns
ranked (item_idx, score) pairs only — enrichment with book metadata
is handled by the application service layer.
"""

import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

from model_servers._shared.contracts import (
    HealthResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    ScoredItem,
)
from model_servers._shared.server_utils import get_artifact_version, make_lifespan
from models.infrastructure.semantic_searcher import SemanticSearcher

logger = logging.getLogger(__name__)

_SERVER_NAME = "semantic"

_DEFAULT_INDEX_DIR = "models/artifacts/semantic_indexes/enriched_v2"
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_searcher: Optional[SemanticSearcher] = None


def _load_artifacts() -> None:
    global _searcher
    logger.info("Loading semantic search artifacts...")
    start = time.monotonic()

    index_dir = os.environ.get("SEMANTIC_INDEX_DIR", _DEFAULT_INDEX_DIR)
    model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL", _DEFAULT_MODEL)

    st_model = SentenceTransformer(model_name)

    def _embed(texts: list[str]):
        return st_model.encode(texts, convert_to_numpy=True).astype("float32")

    _searcher = SemanticSearcher(dir_path=index_dir, embedder=_embed)

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("Semantic search artifacts loaded in %.0fms", elapsed_ms)


lifespan = make_lifespan(_load_artifacts)

app = FastAPI(
    title="Semantic Search Model Server",
    description=(
        "Dense vector search over book descriptions using sentence-transformers and FAISS. "
        "Returns ranked item indices only; metadata enrichment is handled by the app layer."
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

    Returns 503 until _load_artifacts() has completed successfully,
    preventing traffic from reaching the server before the FAISS index
    and sentence-transformers model are fully loaded.
    """
    if _searcher is None:
        raise HTTPException(status_code=503, detail="Semantic search not initialized")

    return HealthResponse(
        status="ok",
        server=_SERVER_NAME,
        artifact_version=get_artifact_version(),
    )


# ===========================================================================
# Semantic search
# ===========================================================================


@app.post("/semantic_search", response_model=SemanticSearchResponse)
def semantic_search(request: SemanticSearchRequest) -> SemanticSearchResponse:
    """
    Embed a free-text query and return the nearest neighbours from the FAISS index.

    Scores are negated inner-product distances from the index — higher is better.
    The result list is ordered highest score first and contains at most top_k items.
    """
    if _searcher is None:
        raise HTTPException(status_code=503, detail="Semantic search not initialized")

    try:
        raw = _searcher.search(query=request.query, top_k=request.top_k)
        return SemanticSearchResponse(
            results=[ScoredItem(item_idx=r["item_idx"], score=r["score"]) for r in raw]
        )
    except Exception as e:
        logger.error("semantic_search failed for query %r: %s", request.query, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Semantic search failed")
