# models/client/als.py
"""
HTTP client for the ALS recommendation model server.
"""

from __future__ import annotations

import os

import httpx

from model_servers._shared.contracts import (
    AlsRecsRequest,
    AlsRecsResponse,
    HasAlsUserRequest,
    HasAlsUserResponse,
)
from models.client._base import BaseModelServerClient, _DEFAULT_TIMEOUT

_DEFAULT_URL = "http://als:8003"


class AlsClient(BaseModelServerClient):
    """
    Client for the ALS recommendation model server.

    Exposes warm/cold user gating and collaborative filtering recommendations
    via raw dot product over non-normalized ALS factors.

    Example:
        client = AlsClient.from_env()
        if (await client.has_als_user(user_id)).is_warm:
            recs = await client.als_recs(user_id, k=200)
    """

    _SERVER_NAME = "als"

    def __init__(
        self, base_url: str = _DEFAULT_URL, timeout: httpx.Timeout = _DEFAULT_TIMEOUT
    ) -> None:
        super().__init__(base_url, timeout)

    @classmethod
    def from_env(cls) -> "AlsClient":
        """Instantiate using ALS_URL env var, falling back to the default."""
        return cls(base_url=os.environ.get("ALS_URL", _DEFAULT_URL))

    async def has_als_user(self, user_id: int) -> HasAlsUserResponse:
        """
        Check whether a user has ALS factors (warm/cold gate).

        Args:
            user_id: User ID to check.

        Returns:
            HasAlsUserResponse with is_warm flag.
        """
        body = HasAlsUserRequest(user_id=user_id)
        data = await self._post("/has_als_user", body)
        return HasAlsUserResponse.model_validate(data)

    async def als_recs(self, user_id: int, k: int = 200) -> AlsRecsResponse:
        """
        Generate top-k recommendations via raw ALS dot product.

        Returns an empty results list for cold users rather than raising.
        Callers should gate on has_als_user before calling this if they
        need to distinguish cold users from users with no good recommendations.

        Args:
            user_id: User to generate recommendations for.
            k: Number of recommendations to return.

        Returns:
            AlsRecsResponse with ordered recommended items.
        """
        body = AlsRecsRequest(user_id=user_id, k=k)
        data = await self._post("/als_recs", body)
        return AlsRecsResponse.model_validate(data)
