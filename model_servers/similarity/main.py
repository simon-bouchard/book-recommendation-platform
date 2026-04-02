# model_servers/similarity/main.py
"""
Unified similarity model server.

Owns normalized subject embeddings, normalized ALS factors, both FAISS indices,
the subject/ALS alignment map, and Bayesian scores. All four operations share
this process because hybrid_sim requires both embedding matrices simultaneously
— splitting them would force data duplication or a degraded two-list merge.
"""

import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse

from model_servers._shared.contracts import (
    AlsSimRequest,
    HasBookAlsRequest,
    HasBookAlsResponse,
    HealthResponse,
    HybridSimRequest,
    SubjectRecsRequest,
    SubjectSimRequest,
)
from model_servers._shared.server_utils import get_artifact_version, make_lifespan
from models.infrastructure.hybrid_scorer import HybridScorer
from models.infrastructure.similarity_indices import (
    get_als_similarity_index,
    get_subject_similarity_index,
)
from models.infrastructure.subject_scorer import SubjectScorer

logger = logging.getLogger(__name__)

_SERVER_NAME = "similarity"


def _load_artifacts() -> None:
    """
    Initialize all artifacts owned by this server.

    FAISS indices are built first because their factory functions pull the
    embedding arrays into the loader cache as a side effect. HybridScorer
    and SubjectScorer then build from those cached arrays, avoiding redundant
    disk reads.
    """
    logger.info("Loading similarity server artifacts...")
    start = time.monotonic()

    get_subject_similarity_index()
    get_als_similarity_index()
    HybridScorer()
    SubjectScorer()

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("Similarity server artifacts loaded in %.0fms", elapsed_ms)


lifespan = make_lifespan(_load_artifacts)

app = FastAPI(
    title="Similarity Model Server",
    description=(
        "Subject similarity, ALS similarity, hybrid similarity, and subject-based recommendations."
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

    HybridScorer is the last artifact initialized on startup, so its
    presence confirms all other artifacts are also ready.
    """
    if HybridScorer._instance is None:
        raise HTTPException(status_code=503, detail="Similarity server not initialized")

    return HealthResponse(
        status="ok",
        server=_SERVER_NAME,
        artifact_version=get_artifact_version(),
    )


# ===========================================================================
# Book ALS membership
# ===========================================================================


@app.post("/has_book_als", response_model=HasBookAlsResponse)
async def has_book_als(request: HasBookAlsRequest) -> HasBookAlsResponse:
    """
    Check whether a book has a normalized ALS factor in this server's index.

    Uses the ALS similarity index's full item set, which covers all books that
    have ALS factors regardless of the rating-count threshold applied to the
    candidate pool. A book absent here will contribute zero ALS signal to
    hybrid_sim and will return empty results from als_sim.
    """
    return HasBookAlsResponse(
        item_idx=request.item_idx,
        has_als=get_als_similarity_index().has_item(request.item_idx),
    )


# ===========================================================================
# Subject similarity
# ===========================================================================


@app.post("/subject_sim")
async def subject_sim(request: SubjectSimRequest) -> ORJSONResponse:
    """
    HNSW nearest-neighbour lookup in the subject embedding space.

    Any item_idx can be queried. No rating threshold on the candidate pool.
    Returns empty results if item_idx has no subject embedding.
    """
    try:
        t0 = time.perf_counter()
        scores, item_ids = get_subject_similarity_index().search(
            query_item_id=request.item_idx, k=request.k, exclude_query=True
        )
        compute_ms = (time.perf_counter() - t0) * 1000
        resp = ORJSONResponse(
            {"results": [{"item_idx": int(iid), "score": float(s)} for iid, s in zip(item_ids, scores)]}
        )
        resp.headers["X-Compute-Ms"] = f"{compute_ms:.3f}"
        return resp
    except Exception as e:
        logger.error("subject_sim failed for item %d: %s", request.item_idx, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Subject similarity search failed")


# ===========================================================================
# ALS similarity
# ===========================================================================


@app.post("/als_sim")
async def als_sim(request: AlsSimRequest) -> ORJSONResponse:
    """
    HNSW nearest-neighbour lookup in the ALS factor space.

    Candidate pool is restricted to books with 10+ ratings, enforced at
    index build time. Returns empty results if item_idx has no ALS factors.
    """
    try:
        t0 = time.perf_counter()
        scores, item_ids = get_als_similarity_index().search(
            query_item_id=request.item_idx, k=request.k, exclude_query=True
        )
        compute_ms = (time.perf_counter() - t0) * 1000
        resp = ORJSONResponse(
            {"results": [{"item_idx": int(iid), "score": float(s)} for iid, s in zip(item_ids, scores)]}
        )
        resp.headers["X-Compute-Ms"] = f"{compute_ms:.3f}"
        return resp
    except Exception as e:
        logger.error("als_sim failed for item %d: %s", request.item_idx, e, exc_info=True)
        raise HTTPException(status_code=500, detail="ALS similarity search failed")


# ===========================================================================
# Hybrid similarity
# ===========================================================================


@app.post("/hybrid_sim")
async def hybrid_sim(request: HybridSimRequest) -> ORJSONResponse:
    """
    Single-pass joint matmul over subject and ALS embedding matrices.

    Score = (1 - alpha) * subject_cosine + alpha * als_cosine.
    Returns empty results if item_idx has no ALS embedding or no subject embedding.
    """
    if not get_als_similarity_index().has_item(request.item_idx):
        return ORJSONResponse({"results": []})

    try:
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        item_ids, scores = await loop.run_in_executor(
            None,
            lambda: HybridScorer().score(
                item_idx=request.item_idx,
                k=request.k,
                alpha=request.alpha,
            ),
        )
        compute_ms = (time.perf_counter() - t0) * 1000
        resp = ORJSONResponse(
            {"results": [{"item_idx": int(iid), "score": float(s)} for iid, s in zip(item_ids, scores)]}
        )
        resp.headers["X-Compute-Ms"] = f"{compute_ms:.3f}"
        return resp
    except Exception as e:
        logger.error("hybrid_sim failed for item %d: %s", request.item_idx, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Hybrid similarity search failed")


# ===========================================================================
# Subject recommendations
# ===========================================================================


@app.post("/subject_recs")
async def subject_recs(request: SubjectRecsRequest) -> ORJSONResponse:
    """
    Joint scoring over all book subject embeddings and Bayesian popularity scores.

    Score = alpha * normalize(cosine_scores) + (1 - alpha) * normalize(bayesian_scores).
    The user_vector must be L2-normalized (as returned by the embedder server).
    """
    try:
        import numpy as np

        user_vec = np.array(request.user_vector, dtype="float32")
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        item_ids, scores = await loop.run_in_executor(
            None,
            lambda: SubjectScorer().score(
                user_vector=user_vec,
                k=request.k,
                alpha=request.alpha,
            ),
        )
        compute_ms = (time.perf_counter() - t0) * 1000
        resp = ORJSONResponse(
            {"results": [{"item_idx": int(iid), "score": float(s)} for iid, s in zip(item_ids, scores)]}
        )
        resp.headers["X-Compute-Ms"] = f"{compute_ms:.3f}"
        return resp
    except Exception as e:
        logger.error("subject_recs failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Subject recommendations failed")
