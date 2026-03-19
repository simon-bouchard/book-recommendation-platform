# tests/unit/models/client/test_metadata_client.py
"""
Unit tests for MetadataClient.

The metadata client is the only one that returns BookMeta objects rather than
ScoredItems. book_meta_json and books_json from conftest build the response
payloads; assertions check both envelope structure and individual field values.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from models.client._exceptions import ModelServerRequestError, ModelServerUnavailableError
from models.client.metadata import MetadataClient
from tests.unit.models.client.conftest import (
    TEST_BASE_URL,
    book_meta_json,
    books_json,
    mock_post_200,
    mock_post_422,
    mock_post_500,
    mock_post_connect_error,
    mock_post_remote_protocol_error,
    mock_post_timeout,
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def client() -> MetadataClient:
    return MetadataClient(base_url=TEST_BASE_URL)


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_uses_metadata_url_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("METADATA_URL", "http://custom-meta:9004")

        client = MetadataClient.from_env()

        assert client._base_url == "http://custom-meta:9004"

    def test_falls_back_to_default_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("METADATA_URL", raising=False)

        client = MetadataClient.from_env()

        assert "8004" in client._base_url


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------


class TestEnrich:
    async def test_books_are_parsed(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        payload = books_json(
            [
                book_meta_json(101, "Book Alpha", author="Author One", num_ratings=100),
                book_meta_json(202, "Book Beta"),
            ]
        )
        mock_post_200(respx_mock, f"{TEST_BASE_URL}/enrich", payload)

        result = await client.enrich([101, 202])

        assert len(result) == 2
        assert result[101]["item_idx"] == 101
        assert result[101]["title"] == "Book Alpha"
        assert result[101]["author"] == "Author One"
        assert result[101]["num_ratings"] == 100
        assert result[202]["item_idx"] == 202

    async def test_empty_books_list_is_parsed(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(respx_mock, f"{TEST_BASE_URL}/enrich", books_json([]))

        result = await client.enrich([9999])

        assert result == {}

    async def test_optional_fields_none_are_preserved(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        payload = books_json(
            [book_meta_json(101, "Sparse Book", author=None, year=None, isbn=None)]
        )
        mock_post_200(respx_mock, f"{TEST_BASE_URL}/enrich", payload)

        result = await client.enrich([101])

        book = result[101]
        assert book["author"] is None
        assert book["year"] is None
        assert book["isbn"] is None

    async def test_request_body_contains_item_indices(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/enrich").mock(
            return_value=httpx.Response(200, json=books_json([]))
        )

        await client.enrich([101, 202, 303])

        sent = json.loads(route.calls[0].request.content)
        assert sent["item_indices"] == [101, 202, 303]

    async def test_single_item_index_is_sent(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/enrich").mock(
            return_value=httpx.Response(200, json=books_json([book_meta_json(42, "Single")]))
        )

        result = await client.enrich([42])

        sent = json.loads(route.calls[0].request.content)
        assert sent["item_indices"] == [42]
        assert len(result) == 1


# ---------------------------------------------------------------------------
# popular
# ---------------------------------------------------------------------------


class TestPopular:
    async def test_books_are_parsed(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        payload = books_json(
            [
                book_meta_json(1, "Top Book", bayes_score=0.95, num_ratings=500),
                book_meta_json(2, "Second Book", bayes_score=0.88, num_ratings=300),
            ]
        )
        mock_post_200(respx_mock, f"{TEST_BASE_URL}/popular", payload)

        result = await client.popular(k=2)

        assert len(result.books) == 2
        assert result.books[0].item_idx == 1
        assert result.books[0].title == "Top Book"
        assert result.books[0].bayes_score == pytest.approx(0.95)

    async def test_empty_response_is_parsed(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_200(respx_mock, f"{TEST_BASE_URL}/popular", books_json([]))

        result = await client.popular(k=10)

        assert result.books == []

    async def test_request_body_contains_k(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/popular").mock(
            return_value=httpx.Response(200, json=books_json([]))
        )

        await client.popular(k=25)

        sent = json.loads(route.calls[0].request.content)
        assert sent["k"] == 25

    async def test_default_k_is_100(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{TEST_BASE_URL}/popular").mock(
            return_value=httpx.Response(200, json=books_json([]))
        )

        await client.popular()

        sent = json.loads(route.calls[0].request.content)
        assert sent["k"] == 100


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestMetadataClientErrors:
    async def test_5xx_raises_unavailable(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_500(respx_mock, f"{TEST_BASE_URL}/enrich")

        with pytest.raises(ModelServerUnavailableError):
            await client.enrich([1])

    async def test_4xx_raises_request_error(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_422(respx_mock, f"{TEST_BASE_URL}/enrich")

        with pytest.raises(ModelServerRequestError):
            await client.enrich([1])

    async def test_connect_error_raises_unavailable(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_connect_error(respx_mock, f"{TEST_BASE_URL}/enrich")

        with pytest.raises(ModelServerUnavailableError):
            await client.enrich([1])

    async def test_timeout_raises_unavailable(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_timeout(respx_mock, f"{TEST_BASE_URL}/enrich")

        with pytest.raises(ModelServerUnavailableError):
            await client.enrich([1])

    async def test_remote_protocol_error_raises_unavailable(
        self, client: MetadataClient, respx_mock: respx.MockRouter
    ) -> None:
        mock_post_remote_protocol_error(respx_mock, f"{TEST_BASE_URL}/enrich")

        with pytest.raises(ModelServerUnavailableError):
            await client.enrich([1])
