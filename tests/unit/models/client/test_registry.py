# tests/unit/models/client/test_registry.py
"""
Unit tests for the client registry.

The registry manages four module-level singletons. Tests verify lazy
initialization, from_env() delegation, identity stability across repeated
calls, and that close_all() drains every client and resets globals to None.

Registry globals are reset around every test by the autouse reset_registry
fixture in conftest.py, so each test starts from a clean slate.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import models.client.registry as registry
from models.client.als import AlsClient
from models.client.embedder import EmbedderClient
from models.client.metadata import MetadataClient
from models.client.similarity import SimilarityClient

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Lazy initialization
# ---------------------------------------------------------------------------


class TestLazyInitialization:
    def test_get_embedder_client_returns_embedder_instance(self) -> None:
        client = registry.get_embedder_client()

        assert isinstance(client, EmbedderClient)

    def test_get_similarity_client_returns_similarity_instance(self) -> None:
        client = registry.get_similarity_client()

        assert isinstance(client, SimilarityClient)

    def test_get_als_client_returns_als_instance(self) -> None:
        client = registry.get_als_client()

        assert isinstance(client, AlsClient)

    def test_get_metadata_client_returns_metadata_instance(self) -> None:
        client = registry.get_metadata_client()

        assert isinstance(client, MetadataClient)

    def test_get_embedder_client_returns_same_instance_on_repeated_calls(self) -> None:
        first = registry.get_embedder_client()
        second = registry.get_embedder_client()

        assert first is second

    def test_get_similarity_client_returns_same_instance_on_repeated_calls(self) -> None:
        first = registry.get_similarity_client()
        second = registry.get_similarity_client()

        assert first is second

    def test_get_als_client_returns_same_instance_on_repeated_calls(self) -> None:
        first = registry.get_als_client()
        second = registry.get_als_client()

        assert first is second

    def test_get_metadata_client_returns_same_instance_on_repeated_calls(self) -> None:
        first = registry.get_metadata_client()
        second = registry.get_metadata_client()

        assert first is second


# ---------------------------------------------------------------------------
# from_env delegation
# ---------------------------------------------------------------------------


class TestFromEnvDelegation:
    def test_embedder_client_uses_embedder_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDER_URL", "http://embedder-from-env:9001")

        client = registry.get_embedder_client()

        assert client._base_url == "http://embedder-from-env:9001"

    def test_similarity_client_uses_similarity_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIMILARITY_URL", "http://sim-from-env:9002")

        client = registry.get_similarity_client()

        assert client._base_url == "http://sim-from-env:9002"

    def test_als_client_uses_als_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALS_URL", "http://als-from-env:9003")

        client = registry.get_als_client()

        assert client._base_url == "http://als-from-env:9003"

    def test_metadata_client_uses_metadata_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("METADATA_URL", "http://meta-from-env:9004")

        client = registry.get_metadata_client()

        assert client._base_url == "http://meta-from-env:9004"


# ---------------------------------------------------------------------------
# close_all
# ---------------------------------------------------------------------------


class TestCloseAll:
    async def test_close_all_calls_aclose_on_every_client(self) -> None:
        embedder_mock = MagicMock(spec=EmbedderClient)
        embedder_mock.aclose = AsyncMock()
        similarity_mock = MagicMock(spec=SimilarityClient)
        similarity_mock.aclose = AsyncMock()
        als_mock = MagicMock(spec=AlsClient)
        als_mock.aclose = AsyncMock()
        metadata_mock = MagicMock(spec=MetadataClient)
        metadata_mock.aclose = AsyncMock()

        registry._embedder = embedder_mock
        registry._similarity = similarity_mock
        registry._als = als_mock
        registry._metadata = metadata_mock

        await registry.close_all()

        embedder_mock.aclose.assert_awaited_once()
        similarity_mock.aclose.assert_awaited_once()
        als_mock.aclose.assert_awaited_once()
        metadata_mock.aclose.assert_awaited_once()

    async def test_close_all_resets_all_globals_to_none(self) -> None:
        registry._embedder = MagicMock(spec=EmbedderClient, aclose=AsyncMock())
        registry._similarity = MagicMock(spec=SimilarityClient, aclose=AsyncMock())
        registry._als = MagicMock(spec=AlsClient, aclose=AsyncMock())
        registry._metadata = MagicMock(spec=MetadataClient, aclose=AsyncMock())

        await registry.close_all()

        assert registry._embedder is None
        assert registry._similarity is None
        assert registry._als is None
        assert registry._metadata is None

    async def test_close_all_with_no_clients_initialized_is_safe(self) -> None:
        await registry.close_all()

        assert registry._embedder is None
        assert registry._similarity is None
        assert registry._als is None
        assert registry._metadata is None

    async def test_close_all_with_partial_initialization_closes_active_clients(
        self,
    ) -> None:
        als_mock = MagicMock(spec=AlsClient)
        als_mock.aclose = AsyncMock()
        registry._als = als_mock

        await registry.close_all()

        als_mock.aclose.assert_awaited_once()
        assert registry._als is None

    async def test_close_all_continues_after_aclose_exception(self) -> None:
        embedder_mock = MagicMock(spec=EmbedderClient)
        embedder_mock.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
        metadata_mock = MagicMock(spec=MetadataClient)
        metadata_mock.aclose = AsyncMock()

        registry._embedder = embedder_mock
        registry._metadata = metadata_mock

        await registry.close_all()

        metadata_mock.aclose.assert_awaited_once()
        assert registry._embedder is None
        assert registry._metadata is None
