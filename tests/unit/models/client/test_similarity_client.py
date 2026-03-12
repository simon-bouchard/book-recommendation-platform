# tests/unit/models/client/test_similarity_client.py
"""
Unit tests for SimilarityClient.

Covers all five endpoints. Error paths are tested once on subject_sim since
all methods share the same _post base implementation.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from models.client._exceptions import ModelServerRequestError, ModelServerUnavailableError
from models.client.similarity import SimilarityClient
from tests.unit.models.client.conftest import (
    TEST_BASE_URL,
    mock_post_200,
    mock_post_422,
    mock_post_500,
    mock_post_connect_error,
    mock_post_remote_protocol_error,
    mock_post_timeout,
    scored_items_json,
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def client() -> SimilarityClient:
    return SimilarityClient(base_url=TEST_BASE_URL)


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_uses_similarity_url_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIMILARITY_URL", "http://custom-sim:9002")

        client = SimilarityClient.from_env()

        assert client._base_url == "http://custom-sim:9002"

    def test_falls_back_to_default_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIMILARITY_URL", raising=False)

        client = SimilarityClient.from_env()

        assert "8002" in client._base_url


# ---------------------------------------------------------------------------
# has_book_als
# ---------------------------------------------------------------------------


class TestHasBookAls:
    async def test_book_with_als_is_parsed(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/has_book_als",
            {"item_idx": 42, "has_als": True},
        )

        result = await client.has_book_als(item_idx=42)

        assert result.item_idx == 42
        assert result.has_als is True

    async def test_book_without_als_is_parsed(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/has_book_als",
            {"item_idx": 99, "has_als": False},
        )

        result = await client.has_book_als(item_idx=99)

        assert result.has_als is False

    async def test_request_body_contains_item_idx(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/has_book_als").mock(
            return_value=httpx.Response(200, json={"item_idx": 7, "has_als": False})
        )

        await client.has_book_als(item_idx=7)

        sent = json.loads(route.calls[0].request.content)
        assert sent["item_idx"] == 7


# ---------------------------------------------------------------------------
# subject_sim
# ---------------------------------------------------------------------------


class TestSubjectSim:
    async def test_results_are_parsed(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/subject_sim",
            scored_items_json([(10, 0.9), (20, 0.7)]),
        )

        result = await client.subject_sim(item_idx=1, k=2)

        assert len(result.results) == 2
        assert result.results[0].item_idx == 10
        assert result.results[0].score == pytest.approx(0.9)

    async def test_empty_results_are_parsed(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/subject_sim",
            scored_items_json([]),
        )

        result = await client.subject_sim(item_idx=999)

        assert result.results == []

    async def test_request_body_contains_item_idx_and_k(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/subject_sim").mock(
            return_value=httpx.Response(200, json=scored_items_json([]))
        )

        await client.subject_sim(item_idx=7, k=50)

        sent = json.loads(route.calls[0].request.content)
        assert sent["item_idx"] == 7
        assert sent["k"] == 50

    async def test_default_k_is_200(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/subject_sim").mock(
            return_value=httpx.Response(200, json=scored_items_json([]))
        )

        await client.subject_sim(item_idx=1)

        sent = json.loads(route.calls[0].request.content)
        assert sent["k"] == 200


# ---------------------------------------------------------------------------
# als_sim
# ---------------------------------------------------------------------------


class TestAlsSim:
    async def test_results_are_parsed(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/als_sim",
            scored_items_json([(5, 0.8), (15, 0.6)]),
        )

        result = await client.als_sim(item_idx=1, k=2)

        assert len(result.results) == 2
        assert result.results[0].item_idx == 5
        assert result.results[0].score == pytest.approx(0.8)

    async def test_request_body_contains_item_idx_and_k(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/als_sim").mock(
            return_value=httpx.Response(200, json=scored_items_json([]))
        )

        await client.als_sim(item_idx=3, k=75)

        sent = json.loads(route.calls[0].request.content)
        assert sent["item_idx"] == 3
        assert sent["k"] == 75

    async def test_default_k_is_200(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/als_sim").mock(
            return_value=httpx.Response(200, json=scored_items_json([]))
        )

        await client.als_sim(item_idx=1)

        sent = json.loads(route.calls[0].request.content)
        assert sent["k"] == 200


# ---------------------------------------------------------------------------
# hybrid_sim
# ---------------------------------------------------------------------------


class TestHybridSim:
    async def test_results_are_parsed(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/hybrid_sim",
            scored_items_json([(10, 0.85), (20, 0.6)]),
        )

        result = await client.hybrid_sim(item_idx=1, k=2)

        assert len(result.results) == 2
        assert result.results[0].item_idx == 10
        assert result.results[0].score == pytest.approx(0.85)

    async def test_request_body_contains_alpha(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/hybrid_sim").mock(
            return_value=httpx.Response(200, json=scored_items_json([]))
        )

        await client.hybrid_sim(item_idx=1, k=10, alpha=0.3)

        sent = json.loads(route.calls[0].request.content)
        assert sent["alpha"] == pytest.approx(0.3)

    async def test_defaults_k_200_and_alpha_06(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/hybrid_sim").mock(
            return_value=httpx.Response(200, json=scored_items_json([]))
        )

        await client.hybrid_sim(item_idx=1)

        sent = json.loads(route.calls[0].request.content)
        assert sent["k"] == 200
        assert sent["alpha"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# subject_recs
# ---------------------------------------------------------------------------


class TestSubjectRecs:
    async def test_results_are_parsed(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/subject_recs",
            scored_items_json([(10, 0.9), (20, 0.7), (30, 0.5)]),
        )

        result = await client.subject_recs(user_vector=[0.1, 0.2, 0.3], k=3)

        assert len(result.results) == 3
        assert result.results[0].item_idx == 10
        assert result.results[0].score == pytest.approx(0.9)

    async def test_request_body_contains_user_vector(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/subject_recs").mock(
            return_value=httpx.Response(200, json=scored_items_json([]))
        )

        await client.subject_recs(user_vector=[0.1, 0.2, 0.3])

        sent = json.loads(route.calls[0].request.content)
        assert sent["user_vector"] == pytest.approx([0.1, 0.2, 0.3])

    async def test_defaults_k_200_and_alpha_06(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/subject_recs").mock(
            return_value=httpx.Response(200, json=scored_items_json([]))
        )

        await client.subject_recs(user_vector=[0.5, 0.5])

        sent = json.loads(route.calls[0].request.content)
        assert sent["k"] == 200
        assert sent["alpha"] == pytest.approx(0.6)

    async def test_empty_results_are_parsed(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(
            respx_mock,
            f"{TEST_BASE_URL}/subject_recs",
            scored_items_json([]),
        )

        result = await client.subject_recs(user_vector=[0.0, 0.0])

        assert result.results == []


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestSimilarityClientErrors:
    async def test_5xx_raises_unavailable(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_500(respx_mock, f"{TEST_BASE_URL}/subject_sim")

        with pytest.raises(ModelServerUnavailableError):
            await client.subject_sim(item_idx=1)

    async def test_4xx_raises_request_error(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_422(respx_mock, f"{TEST_BASE_URL}/subject_sim")

        with pytest.raises(ModelServerRequestError):
            await client.subject_sim(item_idx=1)

    async def test_connect_error_raises_unavailable(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_connect_error(respx_mock, f"{TEST_BASE_URL}/subject_sim")

        with pytest.raises(ModelServerUnavailableError):
            await client.subject_sim(item_idx=1)

    async def test_timeout_raises_unavailable(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_timeout(respx_mock, f"{TEST_BASE_URL}/subject_sim")

        with pytest.raises(ModelServerUnavailableError):
            await client.subject_sim(item_idx=1)

    async def test_remote_protocol_error_raises_unavailable(
        self, client: SimilarityClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_remote_protocol_error(respx_mock, f"{TEST_BASE_URL}/subject_sim")

        with pytest.raises(ModelServerUnavailableError):
            await client.subject_sim(item_idx=1)
