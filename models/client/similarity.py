# models/client/similarity.py
"""
HTTP client for the unified similarity model server.
"""

from __future__ import annotations

import os

import httpx

from model_servers._shared.contracts import (
    AlsSimRequest,
    HasBookAlsRequest,
    HasBookAlsResponse,
    HybridSimRequest,
    SimResponse,
    SubjectRecsRequest,
    SubjectRecsResponse,
    SubjectSimRequest,
)
from models.client._base import _DEFAULT_TIMEOUT, BaseModelServerClient

_DEFAULT_URL = "http://similarity:8002"


class SimilarityClient(BaseModelServerClient):
    """
    Client for the unified similarity model server.

    Covers subject similarity, ALS similarity, hybrid similarity,
    subject-based recommendations, and book ALS membership checks.
    All operations are served from a single container because hybrid_sim
    requires both embedding matrices in the same process.

    Example:
        client = SimilarityClient.from_env()
        response = await client.hybrid_sim(item_idx=42, k=100, alpha=0.6)
    """

    _SERVER_NAME = "similarity"

    def __init__(
        self, base_url: str = _DEFAULT_URL, timeout: httpx.Timeout = _DEFAULT_TIMEOUT
    ) -> None:
        super().__init__(base_url, timeout)

    @classmethod
    def from_env(cls) -> "SimilarityClient":
        """Instantiate using SIMILARITY_URL env var, falling back to the default."""
        return cls(base_url=os.environ.get("SIMILARITY_URL", _DEFAULT_URL))

    async def has_book_als(self, item_idx: int) -> HasBookAlsResponse:
        """
        Check whether a book has a normalized ALS factor in the similarity server.

        Args:
            item_idx: Book item index to check.

        Returns:
            HasBookAlsResponse with has_als flag.
        """
        body = HasBookAlsRequest(item_idx=item_idx)
        data = await self._post("/has_book_als", body)
        return HasBookAlsResponse.model_validate(data)

    async def subject_sim(self, item_idx: int, k: int = 200) -> SimResponse:
        """
        HNSW nearest-neighbour lookup in the subject embedding space.

        Args:
            item_idx: Query book index.
            k: Number of results to return.

        Returns:
            SimResponse with ordered similar items.
        """
        body = SubjectSimRequest(item_idx=item_idx, k=k)
        data = await self._post("/subject_sim", body)
        return SimResponse.model_validate(data)

    async def als_sim(self, item_idx: int, k: int = 200) -> SimResponse:
        """
        HNSW nearest-neighbour lookup in the ALS factor space.

        Args:
            item_idx: Query book index.
            k: Number of results to return.

        Returns:
            SimResponse with ordered similar items.
        """
        body = AlsSimRequest(item_idx=item_idx, k=k)
        data = await self._post("/als_sim", body)
        return SimResponse.model_validate(data)

    async def hybrid_sim(self, item_idx: int, k: int = 200, alpha: float = 0.6) -> SimResponse:
        """
        Single-pass joint matmul over subject and ALS embedding matrices.

        Args:
            item_idx: Query book index.
            k: Number of results to return.
            alpha: ALS weight; (1 - alpha) is subject weight.

        Returns:
            SimResponse with ordered similar items.
        """
        body = HybridSimRequest(item_idx=item_idx, k=k, alpha=alpha)
        data = await self._post("/hybrid_sim", body)
        return SimResponse.model_validate(data)

    async def subject_recs(
        self,
        user_vector: list[float],
        k: int = 200,
        alpha: float = 0.6,
    ) -> SubjectRecsResponse:
        """
        Joint scoring over all book subject embeddings and Bayesian popularity scores.

        Args:
            user_vector: L2-normalized user embedding from the embedder server.
            k: Number of results to return.
            alpha: Subject similarity weight; (1 - alpha) is popularity weight.

        Returns:
            SubjectRecsResponse with ordered recommended items.
        """
        body = SubjectRecsRequest(user_vector=user_vector, k=k, alpha=alpha)
        data = await self._post("/subject_recs", body)
        return SubjectRecsResponse.model_validate(data)
