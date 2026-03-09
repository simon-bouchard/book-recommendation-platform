# models/client/metadata.py
"""
HTTP client for the metadata model server.
"""

from __future__ import annotations

import os

import httpx

from model_servers._shared.contracts import (
    BookMeta,
    EnrichRequest,
    EnrichResponse,
    PopularRequest,
    PopularResponse,
)
from models.client._base import BaseModelServerClient, _DEFAULT_TIMEOUT

_DEFAULT_URL = "http://metadata:8004"


class MetadataClient(BaseModelServerClient):
    """
    Client for the metadata model server.

    Provides book metadata enrichment and Bayesian popularity retrieval.

    Example:
        client = MetadataClient.from_env()
        enriched = await client.enrich([101, 202, 303])
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

    async def enrich(self, item_indices: list[int]) -> EnrichResponse:
        body = EnrichRequest(item_indices=item_indices)
        data = await self._post("/enrich", body)
        return EnrichResponse.model_construct(
            books=[BookMeta.model_construct(**b) for b in data["books"]]
        )

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
        return PopularResponse.model_validate(data)
