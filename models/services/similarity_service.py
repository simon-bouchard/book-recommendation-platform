# models/services/similarity_service.py
"""
Book similarity service using shared FAISS indices for optimal performance.
Hybrid mode uses FAISS for initial retrieval then blends scores for accuracy.
"""

import logging
from typing import Dict, List

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from models.client.registry import get_metadata_client, get_similarity_client

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


class SimilarityService:
    """
    Book similarity service using singleton FAISS indices.

    All subject operations share the same FAISS index for memory efficiency.
    Hybrid mode uses FAISS for fast initial retrieval then exact scoring for accuracy.

    Trace structure per request:
        similarity.service
        ├── similarity.retrieve
        │   └── POST /subject_sim | /als_sim | /hybrid_sim  (httpx auto-instrumented)
        └── similarity.enrich
            ├── POST /enrich  (httpx auto-instrumented)
            └── metadata.build_index
    """

    def __init__(self):
        """Initialize similarity service."""

    async def get_similar(
        self,
        item_idx: int,
        mode: str = "subject",
        k: int = 200,
        alpha: float = 0.6,
    ) -> List[Dict]:
        """Find similar books with mode-specific retrieval and metadata enrichment."""
        with tracer.start_as_current_span("similarity.service") as span:
            span.set_attribute("item.idx", item_idx)
            span.set_attribute("similarity.mode", mode)
            span.set_attribute("similarity.k", k)
            if mode == "hybrid":
                span.set_attribute("similarity.alpha", alpha)

            logger.info(
                "Similarity search started",
                extra={"item_idx": item_idx, "mode": mode, "k": k},
            )

            try:
                resp = await self._retrieve(item_idx, mode, k, alpha)
                span.set_attribute("similarity.retrieved_count", len(resp.results))

                result = await self._enrich(resp.results)
                span.set_attribute("similarity.result_count", len(result))

                logger.info(
                    "Similarity search completed",
                    extra={"item_idx": item_idx, "mode": mode, "count": len(result)},
                )

                return result

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                logger.error(
                    "Similarity search failed",
                    extra={"item_idx": item_idx, "mode": mode, "error": str(exc)},
                    exc_info=True,
                )
                raise

    async def _retrieve(self, item_idx: int, mode: str, k: int, alpha: float):
        """
        Dispatch to the similarity server under a named span.

        The httpx auto-instrumentor attaches the POST to the similarity server
        as a child of this span, giving the waterfall:
        retrieve -> POST /subject_sim (or /als_sim or /hybrid_sim).
        """
        with tracer.start_as_current_span("similarity.retrieve") as span:
            span.set_attribute("similarity.mode", mode)
            try:
                if mode == "subject":
                    return await get_similarity_client().subject_sim(item_idx, k)
                elif mode == "als":
                    return await get_similarity_client().als_sim(item_idx, k)
                elif mode == "hybrid":
                    return await get_similarity_client().hybrid_sim(item_idx, k, alpha)
                else:
                    raise ValueError(f"Unknown similarity mode: {mode!r}")
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def _enrich(self, results) -> List[Dict]:
        """
        Enrich similarity results with book metadata under a named span.

        get_metadata_client().enrich() returns a dict[int, dict] keyed by
        item_idx. Fields are accessed directly from the raw dict, avoiding
        any Pydantic object construction on the hot path.

        The httpx auto-instrumentor attaches POST /enrich as a child of this
        span so enrichment latency is cleanly separated from retrieval latency
        in the Jaeger waterfall.
        """
        with tracer.start_as_current_span("similarity.enrich") as span:
            span.set_attribute("enrich.input_count", len(results))
            try:
                meta = await get_metadata_client().enrich([r.item_idx for r in results])

                enriched = [
                    {
                        "item_idx": book["item_idx"],
                        "title": book["title"],
                        "author": book["author"],
                        "year": book["year"],
                        "isbn": book["isbn"],
                        "cover_id": book["cover_id"],
                        "score": r.score,
                    }
                    for r in results
                    if (book := meta.get(r.item_idx)) is not None
                ]

                span.set_attribute("enrich.output_count", len(enriched))
                span.set_attribute("enrich.missing_count", len(results) - len(enriched))
                return enriched

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
