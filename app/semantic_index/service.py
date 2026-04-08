# app/semantic_index/service.py
"""
Semantic search service.

Orchestrates the semantic search model server and the metadata model server:
the semantic server returns ranked (item_idx, score) pairs, and the metadata
server enriches them with title, author, year, and other book fields.
"""

import logging

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from models.client.registry import get_metadata_client, get_semantic_client

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class SemanticSearchService:
    """
    Stateless service for semantic book search with metadata enrichment.

    Trace structure per call:
        semantic.search
        ├── POST /semantic_search  (httpx auto-instrumented)
        └── semantic.enrich
            ├── POST /enrich       (httpx auto-instrumented)
            └── metadata.build_index
    """

    async def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search books by semantic similarity to a free-text query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with item_idx, title, author, year, and score,
            ordered by descending similarity score. Books absent from the
            metadata server are silently dropped.
        """
        with tracer.start_as_current_span("semantic.search") as span:
            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)

            try:
                response = await get_semantic_client().semantic_search(query=query, top_k=top_k)

                if not response.results:
                    span.set_attribute("result_count", 0)
                    return []

                item_ids = [r.item_idx for r in response.results]
                meta = await self._enrich(item_ids)

                results = []
                for item in response.results:
                    book = meta.get(item.item_idx)
                    if book is None:
                        continue
                    results.append(
                        {
                            "item_idx": item.item_idx,
                            "title": book["title"],
                            "author": book.get("author"),
                            "year": book.get("year"),
                            "num_ratings": book.get("num_ratings", 0),
                            "cover_id": book.get("cover_id"),
                            "score": item.score,
                        }
                    )

                span.set_attribute("result_count", len(results))
                return results

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def _enrich(self, item_ids: list[int]) -> dict[int, dict]:
        with tracer.start_as_current_span("semantic.enrich") as span:
            span.set_attribute("enrich.input_count", len(item_ids))
            try:
                meta = await get_metadata_client().enrich(item_ids)
                span.set_attribute("enrich.output_count", len(meta))
                return meta
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
