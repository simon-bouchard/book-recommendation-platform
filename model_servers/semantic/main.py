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
    ScoredItem,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SubjectMatch,
    SubjectSearchRequest,
    SubjectSearchResponse,
)
from model_servers._shared.server_utils import get_artifact_version, make_lifespan
from models.infrastructure.semantic_searcher import SemanticSearcher
from models.infrastructure.subject_searcher import SubjectSearcher

logger = logging.getLogger(__name__)

_SERVER_NAME = "semantic"

_DEFAULT_INDEX_DIR = "models/artifacts/semantic_indexes/enriched_v2"
_DEFAULT_SUBJECT_INDEX_DIR = "models/artifacts/semantic_indexes/subjects_v1"
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_searcher: Optional[SemanticSearcher] = None
_subject_searcher: Optional[SubjectSearcher] = None


def _load_artifacts() -> None:
    global _searcher, _subject_searcher
    logger.info("Loading semantic search artifacts...")
    start = time.monotonic()

    index_dir = os.environ.get("SEMANTIC_INDEX_DIR", _DEFAULT_INDEX_DIR)
    subject_index_dir = os.environ.get("SUBJECT_INDEX_DIR", _DEFAULT_SUBJECT_INDEX_DIR)
    model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL", _DEFAULT_MODEL)

    st_model = SentenceTransformer(model_name)

    def _embed(texts: list[str]):
        return st_model.encode(texts, convert_to_numpy=True).astype("float32")

    _searcher = SemanticSearcher(dir_path=index_dir, embedder=_embed)

    try:
        _subject_searcher = SubjectSearcher(dir_path=subject_index_dir, embedder=_embed)
        logger.info("Subject search index loaded from %s", subject_index_dir)
    except FileNotFoundError:
        logger.warning(
            "Subject index not found at %s — /subject_search will return 503. "
            "Run build_subject_index.py to create it.",
            subject_index_dir,
        )

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


# ===========================================================================
# Subject search
# ===========================================================================


@app.post("/subject_search", response_model=SubjectSearchResponse)
def subject_search(request: SubjectSearchRequest) -> SubjectSearchResponse:
    """
    Embed a free-text phrase and return the nearest subject names from the index.

    Scores are cosine similarities in [-1, 1]; higher is more similar.
    Call once per phrase — grouping across multiple phrases is the caller's responsibility.
    """
    if _subject_searcher is None:
        raise HTTPException(status_code=503, detail="Subject search not initialized")

    try:
        raw = _subject_searcher.search(phrase=request.phrase, top_k=request.top_k)
        return SubjectSearchResponse(
            matches=[
                SubjectMatch(
                    phrase=request.phrase,
                    subject_idx=r["subject_idx"],
                    subject_name=r["subject_name"],
                    count=r["count"],
                    score=r["score"],
                )
                for r in raw
            ]
        )
    except Exception as e:
        logger.error("subject_search failed for phrase %r: %s", request.phrase, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Subject search failed")
