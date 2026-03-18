# models/client/semantic.py
"""
HTTP client for the semantic search model server.
"""

from __future__ import annotations

import os

import httpx

from model_servers._shared.contracts import (
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from models.client._base import BaseModelServerClient, _DEFAULT_TIMEOUT

_DEFAULT_URL = "http://localhost:8005"


class SemanticClient(BaseModelServerClient):
    """
    Client for the semantic search model server.

    Wraps the /semantic_search endpoint which performs dense vector search
    over book descriptions and returns ranked (item_idx, score) pairs.
    Enrichment with book metadata is handled by the application service layer.

    Example:
        client = SemanticClient.from_env()
        response = await client.semantic_search(query="cozy mysteries in the countryside", top_k=50)
    """

    _SERVER_NAME = "semantic"

    def __init__(
        self, base_url: str = _DEFAULT_URL, timeout: httpx.Timeout = _DEFAULT_TIMEOUT
    ) -> None:
        super().__init__(base_url, timeout)

    @classmethod
    def from_env(cls) -> "SemanticClient":
        """Instantiate using SEMANTIC_URL env var, falling back to the default."""
        return cls(base_url=os.environ.get("SEMANTIC_URL", _DEFAULT_URL))

    async def semantic_search(self, query: str, top_k: int = 10) -> SemanticSearchResponse:
        """
        Embed a free-text query and return the nearest neighbours from the FAISS index.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            SemanticSearchResponse with ordered (item_idx, score) pairs.
        """
        body = SemanticSearchRequest(query=query, top_k=top_k)
        data = await self._post("/semantic_search", body)
        return SemanticSearchResponse.model_validate(data)
