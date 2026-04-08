# tests/unit/models/client/test_als_client.py
"""
Unit tests for AlsClient.

Each test instantiates the client directly with TEST_BASE_URL rather than
using from_env(), so no environment is required. respx_mock intercepts all
httpx calls within the test and resets between tests automatically.

Error path tests are grouped in TestAlsClientErrors and cover every distinct
exception branch in BaseModelServerClient._post. These are tested once here
on has_als_user rather than duplicated across every method, since the error
handling is shared infrastructure in the base class.
"""

from __future__ import annotations

import pytest
import respx

from models.client._exceptions import ModelServerRequestError, ModelServerUnavailableError
from models.client.als import AlsClient
from tests.unit.models.client.conftest import (
    TEST_BASE_URL,
    mock_post_200,
    mock_post_422,
    mock_post_500,
    mock_post_503,
    mock_post_connect_error,
    mock_post_remote_protocol_error,
    mock_post_timeout,
    scored_items_json,
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def client() -> AlsClient:
    return AlsClient(base_url=TEST_BASE_URL)


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_uses_als_url_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALS_URL", "http://custom-als:9000")

        client = AlsClient.from_env()

        assert client._base_url == "http://custom-als:9000"

    def test_falls_back_to_default_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALS_URL", raising=False)

        client = AlsClient.from_env()

        assert "8003" in client._base_url


# ---------------------------------------------------------------------------
# has_als_user
# ---------------------------------------------------------------------------


class TestHasAlsUser:
    async def test_warm_user_response_is_parsed(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/has_als_user",
            {"user_id": 42, "is_warm": True},
        )

        result = await client.has_als_user(user_id=42)

        assert result.user_id == 42
        assert result.is_warm is True

    async def test_cold_user_response_is_parsed(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/has_als_user",
            {"user_id": 99, "is_warm": False},
        )

        result = await client.has_als_user(user_id=99)

        assert result.is_warm is False

    async def test_request_body_contains_user_id(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/has_als_user").mock(
            return_value=__import__("httpx").Response(200, json={"user_id": 7, "is_warm": True})
        )

        await client.has_als_user(user_id=7)

        assert route.called
        import json

        sent = json.loads(route.calls[0].request.content)
        assert sent["user_id"] == 7


# ---------------------------------------------------------------------------
# als_recs
# ---------------------------------------------------------------------------


class TestAlsRecs:
    async def test_results_are_parsed(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/als_recs",
            scored_items_json([(10, 0.9), (20, 0.7)]),
        )

        result = await client.als_recs(user_id=1, k=2)

        assert len(result.results) == 2
        assert result.results[0].item_idx == 10
        assert result.results[0].score == pytest.approx(0.9)
        assert result.results[1].item_idx == 20

    async def test_empty_results_are_parsed(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/als_recs",
            scored_items_json([]),
        )

        result = await client.als_recs(user_id=999)

        assert result.results == []

    async def test_request_body_contains_k(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        import json

        import httpx as _httpx

        route = respx_mock.post(f"{TEST_BASE_URL}/als_recs").mock(
            return_value=_httpx.Response(200, json=scored_items_json([]))
        )

        await client.als_recs(user_id=5, k=50)

        sent = json.loads(route.calls[0].request.content)
        assert sent["user_id"] == 5
        assert sent["k"] == 50

    async def test_default_k_is_200(self, client: AlsClient, respx_mock: respx.MockRouter) -> None:
        import json

        import httpx as _httpx

        route = respx_mock.post(f"{TEST_BASE_URL}/als_recs").mock(
            return_value=_httpx.Response(200, json=scored_items_json([]))
        )

        await client.als_recs(user_id=1)

        sent = json.loads(route.calls[0].request.content)
        assert sent["k"] == 200


# ---------------------------------------------------------------------------
# Error paths (BaseModelServerClient._post — tested once per client)
# ---------------------------------------------------------------------------


class TestAlsClientErrors:
    async def test_5xx_raises_unavailable(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_500(respx_mock, f"{TEST_BASE_URL}/has_als_user")

        with pytest.raises(ModelServerUnavailableError):
            await client.has_als_user(user_id=1)

    async def test_503_raises_unavailable(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_503(respx_mock, f"{TEST_BASE_URL}/has_als_user")

        with pytest.raises(ModelServerUnavailableError):
            await client.has_als_user(user_id=1)

    async def test_4xx_raises_request_error(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_422(respx_mock, f"{TEST_BASE_URL}/has_als_user")

        with pytest.raises(ModelServerRequestError):
            await client.has_als_user(user_id=1)

    async def test_connect_error_raises_unavailable(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_connect_error(respx_mock, f"{TEST_BASE_URL}/has_als_user")

        with pytest.raises(ModelServerUnavailableError):
            await client.has_als_user(user_id=1)

    async def test_timeout_raises_unavailable(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_timeout(respx_mock, f"{TEST_BASE_URL}/has_als_user")

        with pytest.raises(ModelServerUnavailableError):
            await client.has_als_user(user_id=1)

    async def test_remote_protocol_error_raises_unavailable(
        self, client: AlsClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_remote_protocol_error(respx_mock, f"{TEST_BASE_URL}/has_als_user")

        with pytest.raises(ModelServerUnavailableError):
            await client.has_als_user(user_id=1)
