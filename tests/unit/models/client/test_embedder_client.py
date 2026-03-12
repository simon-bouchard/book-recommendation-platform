# tests/unit/models/client/test_embedder_client.py
"""
Unit tests for EmbedderClient.

The embedder has a single endpoint. Error paths are verified here against
/embed, confirming the full base class error translation applies.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from models.client._exceptions import ModelServerRequestError, ModelServerUnavailableError
from models.client.embedder import EmbedderClient
from tests.unit.models.client.conftest import (
    TEST_BASE_URL,
    mock_post_200,
    mock_post_422,
    mock_post_500,
    mock_post_503,
    mock_post_connect_error,
    mock_post_remote_protocol_error,
    mock_post_timeout,
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def client() -> EmbedderClient:
    return EmbedderClient(base_url=TEST_BASE_URL)


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_uses_embedder_url_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDER_URL", "http://custom-embedder:9001")

        client = EmbedderClient.from_env()

        assert client._base_url == "http://custom-embedder:9001"

    def test_falls_back_to_default_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EMBEDDER_URL", raising=False)

        client = EmbedderClient.from_env()

        assert "8001" in client._base_url


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


class TestEmbed:
    async def test_vector_is_parsed(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/embed",
            {"vector": [0.1, 0.2, 0.3, 0.4]},
        )

        result = await client.embed([5, 12, 23])

        assert result.vector == pytest.approx([0.1, 0.2, 0.3, 0.4])

    async def test_vector_length_is_preserved(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        dim = 64
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/embed",
            {"vector": [0.0] * dim},
        )

        result = await client.embed(list(range(10)))

        assert len(result.vector) == dim

    async def test_request_body_contains_subject_indices(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/embed").mock(
            return_value=httpx.Response(200, json={"vector": [0.5, 0.5]})
        )

        await client.embed([5, 12, 23])

        sent = json.loads(route.calls[0].request.content)
        assert sent["subject_indices"] == [5, 12, 23]

    async def test_single_subject_index_is_sent(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/embed").mock(
            return_value=httpx.Response(200, json={"vector": [1.0]})
        )

        await client.embed([7])

        sent = json.loads(route.calls[0].request.content)
        assert sent["subject_indices"] == [7]

    async def test_content_type_header_is_json(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/embed").mock(
            return_value=httpx.Response(200, json={"vector": [0.5]})
        )

        await client.embed([1])

        assert route.calls[0].request.headers["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestEmbedderClientErrors:
    async def test_5xx_raises_unavailable(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_500(respx_mock, f"{TEST_BASE_URL}/embed")

        with pytest.raises(ModelServerUnavailableError):
            await client.embed([1, 2, 3])

    async def test_503_raises_unavailable(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_503(respx_mock, f"{TEST_BASE_URL}/embed")

        with pytest.raises(ModelServerUnavailableError):
            await client.embed([1, 2, 3])

    async def test_4xx_raises_request_error(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_422(respx_mock, f"{TEST_BASE_URL}/embed")

        with pytest.raises(ModelServerRequestError):
            await client.embed([1, 2, 3])

    async def test_connect_error_raises_unavailable(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_connect_error(respx_mock, f"{TEST_BASE_URL}/embed")

        with pytest.raises(ModelServerUnavailableError):
            await client.embed([1, 2, 3])

    async def test_timeout_raises_unavailable(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_timeout(respx_mock, f"{TEST_BASE_URL}/embed")

        with pytest.raises(ModelServerUnavailableError):
            await client.embed([1, 2, 3])

    async def test_remote_protocol_error_raises_unavailable(
        self, client: EmbedderClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_remote_protocol_error(respx_mock, f"{TEST_BASE_URL}/embed")

        with pytest.raises(ModelServerUnavailableError):
            await client.embed([1, 2, 3])
