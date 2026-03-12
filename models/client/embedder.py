# models/client/embedder.py
"""
HTTP client for the embedder model server.
"""

from __future__ import annotations

import os

from model_servers._shared.contracts import EmbedRequest, EmbedResponse
from models.client._base import BaseModelServerClient, _DEFAULT_TIMEOUT
import httpx

_DEFAULT_URL = "http://embedder:8001"


class EmbedderClient(BaseModelServerClient):
    """
    Client for the embedder model server.

    Computes normalized subject embeddings from subject index lists.

    Example:
        client = EmbedderClient.from_env()
        response = await client.embed([12, 47, 103])
    """

    _SERVER_NAME = "embedder"

    def __init__(
        self, base_url: str = _DEFAULT_URL, timeout: httpx.Timeout = _DEFAULT_TIMEOUT
    ) -> None:
        super().__init__(base_url, timeout)

    @classmethod
    def from_env(cls) -> "EmbedderClient":
        """Instantiate using EMBEDDER_URL env var, falling back to the default."""
        return cls(base_url=os.environ.get("EMBEDDER_URL", _DEFAULT_URL))

    async def embed(self, subject_indices: list[int]) -> EmbedResponse:
        """
        Compute a normalized embedding vector for a list of subject indices.

        Args:
            subject_indices: Non-empty list of subject indices representing
                             user preferences.

        Returns:
            EmbedResponse containing the L2-normalized embedding vector.

        Raises:
            ModelServerUnavailableError: If the embedder server is unreachable.
            ModelServerRequestError: If the request is malformed.
        """
        body = EmbedRequest(subject_indices=subject_indices)
        data = await self._post("/embed", body)
        return EmbedResponse.model_validate(data)
