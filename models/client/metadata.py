# models/client/metadata.py
"""
HTTP client for the metadata model server.
"""

from __future__ import annotations

import os
import time

import httpx
from opentelemetry import trace

from model_servers._shared.contracts import (
    PopularRequest,
    PopularResponse,
    EnrichRequest,
)
from models.client._base import BaseModelServerClient, _DEFAULT_TIMEOUT

_DEFAULT_URL = "http://metadata:8004"

tracer = trace.get_tracer(__name__)


class MetadataClient(BaseModelServerClient):
    """
    Client for the metadata model server.

    Provides book metadata enrichment and Bayesian popularity retrieval.

    enrich() returns a plain dict[int, dict] keyed by item_idx rather than
    constructing Pydantic BookMeta objects. Callers access fields via dict
    lookup, which eliminates 200+ model_construct() calls per request —
    the dominant CPU cost on the recommendation hot path.

    Example:
        client = MetadataClient.from_env()
        meta = await client.enrich([101, 202, 303])
        title = meta[101]["title"]
    """

    _SERVER_NAME = "metadata"

    def __init__(
        self, base_url: str = _DEFAULT_URL, timeout: httpx.Timeout = _DEFAULT_TIMEOUT
    ) -> None:
        super().__init__(base_url, timeout)

    @classmethod
    def from_env(cls) -> "MetadataClient":
        """Instantiate using METADATA_URL env var, falling back to the default."""
        return cls(base_url=os.environ.get("METADATA_URL", _DEFAULT_URL))

    async def enrich(self, item_indices: list[int]) -> dict[int, dict]:
        """
        Enrich a list of item indices with book metadata.

        Returns a dict mapping item_idx -> raw metadata dict. Skips all Pydantic
        object construction — the response bytes from orjson are used directly.

        Args:
            item_indices: List of item indices to enrich.

        Returns:
            Mapping of item_idx to metadata dict for each returned book.
            Books absent from the server response are simply not present in
            the mapping; callers should guard with .get() or `in` checks.
        """
        body = EnrichRequest(item_indices=item_indices)
        data = await self._post("/enrich", body)

        with tracer.start_as_current_span("metadata.build_index") as span:
            span.set_attribute("item_count", len(item_indices))

            cpu_start = time.process_time()
            wall_start = time.perf_counter()

            index = {b["item_idx"]: b for b in data["books"]}

            cpu_ms = (time.process_time() - cpu_start) * 1000
            wall_ms = (time.perf_counter() - wall_start) * 1000

            span.set_attribute("books_returned", len(index))
            span.set_attribute("timing.wall_ms", round(wall_ms, 3))
            span.set_attribute("timing.cpu_ms", round(cpu_ms, 3))
            span.set_attribute("timing.offcpu_ms", round(wall_ms - cpu_ms, 3))

            return index

    async def popular(self, k: int = 100) -> PopularResponse:
        """
        Retrieve the top-k books ranked by precomputed Bayesian score.

        Args:
            k: Number of books to return.

        Returns:
            PopularResponse with books ordered by Bayesian score descending.
        """
        body = PopularRequest(k=k)
        data = await self._post("/popular", body)

        with tracer.start_as_current_span("metadata.model_validate") as span:
            span.set_attribute("model", "PopularResponse")
            span.set_attribute("k", k)
            return PopularResponse.model_validate(data)
