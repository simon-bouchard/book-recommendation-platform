# tests/unit/models/model_servers/test_semantic_server.py
"""
Unit tests for the semantic search model server.

Strategy overview
-----------------
The semantic server holds its artifact in a module-level ``_searcher``
variable rather than a class singleton. It is patched directly via
``monkeypatch.setattr`` against ``model_servers.semantic.main._searcher``.

The ``mock_searcher`` fixture installs a MagicMock and the ``autouse``
``reset_searcher`` fixture guarantees the module variable is None before
and after every test, regardless of whether monkeypatch was used.

``SemanticSearcher.search`` returns a plain list of dicts
``[{"item_idx": int, "score": float}, ...]``; the server wraps these into
``ScoredItem`` objects. Mock return values are set accordingly.
"""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

import model_servers.semantic.main as server_module
from model_servers.semantic.main import app
from tests.unit.models.model_servers.conftest import (
    assert_health_ok,
    assert_scored_items,
    make_test_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_searcher() -> Generator[None, None, None]:
    """Ensure _searcher is None before and after every test."""
    server_module._searcher = None
    yield
    server_module._searcher = None


@pytest.fixture()
def mock_searcher(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a MagicMock as the active _searcher."""
    mock = MagicMock()
    monkeypatch.setattr(server_module, "_searcher", mock)
    return mock


@pytest.fixture()
def client() -> TestClient:
    return make_test_client(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ok_when_searcher_initialized(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        response = client.get("/health")

        assert_health_ok(response, "semantic")

    def test_503_when_searcher_is_none(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 503

    def test_artifact_version_is_unknown_without_env_var(
        self,
        client: TestClient,
        mock_searcher: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MODEL_VERSION_POINTER", raising=False)

        response = client.get("/health")

        assert response.json()["artifact_version"] == "unknown"

    def test_artifact_version_read_from_pointer_file(
        self,
        client: TestClient,
        mock_searcher: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        version_pointer_file: tuple,
    ) -> None:
        pointer, version = version_pointer_file
        monkeypatch.setenv("MODEL_VERSION_POINTER", str(pointer))

        response = client.get("/health")

        assert response.json()["artifact_version"] == version


# ---------------------------------------------------------------------------
# POST /semantic_search
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    def test_returns_ordered_results(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        mock_searcher.search.return_value = [
            {"item_idx": 10, "score": 0.9},
            {"item_idx": 20, "score": 0.7},
            {"item_idx": 30, "score": 0.5},
        ]

        response = client.post(
            "/semantic_search", json={"query": "cozy mysteries", "top_k": 3}
        )

        assert response.status_code == 200
        assert_scored_items(response.json()["results"], [(10, 0.9), (20, 0.7), (30, 0.5)])

    def test_search_called_with_correct_args(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        mock_searcher.search.return_value = []

        client.post("/semantic_search", json={"query": "fantasy epics", "top_k": 25})

        mock_searcher.search.assert_called_once_with(query="fantasy epics", top_k=25)

    def test_default_top_k_is_10(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        mock_searcher.search.return_value = []

        client.post("/semantic_search", json={"query": "romance"})

        mock_searcher.search.assert_called_once_with(query="romance", top_k=10)

    def test_empty_results_for_no_matches(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        mock_searcher.search.return_value = []

        response = client.post(
            "/semantic_search", json={"query": "very obscure topic", "top_k": 5}
        )

        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_search_exception_returns_500(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        mock_searcher.search.side_effect = RuntimeError("FAISS error")

        response = client.post(
            "/semantic_search", json={"query": "science fiction", "top_k": 10}
        )

        assert response.status_code == 500

    def test_503_when_searcher_is_none(self, client: TestClient) -> None:
        response = client.post(
            "/semantic_search", json={"query": "science fiction", "top_k": 10}
        )

        assert response.status_code == 503

    def test_missing_query_returns_422(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        response = client.post("/semantic_search", json={"top_k": 10})

        assert response.status_code == 422

    def test_empty_query_returns_422(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        response = client.post("/semantic_search", json={"query": "", "top_k": 10})

        assert response.status_code == 422

    def test_query_too_long_returns_422(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        response = client.post(
            "/semantic_search", json={"query": "x" * 501, "top_k": 10}
        )

        assert response.status_code == 422

    def test_top_k_below_minimum_returns_422(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        response = client.post(
            "/semantic_search", json={"query": "mystery", "top_k": 0}
        )

        assert response.status_code == 422

    def test_top_k_above_maximum_returns_422(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        response = client.post(
            "/semantic_search", json={"query": "mystery", "top_k": 501}
        )

        assert response.status_code == 422

    def test_top_k_at_maximum_boundary_is_valid(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        mock_searcher.search.return_value = []

        response = client.post(
            "/semantic_search", json={"query": "mystery", "top_k": 500}
        )

        assert response.status_code == 200

    def test_top_k_at_minimum_boundary_is_valid(
        self, client: TestClient, mock_searcher: MagicMock
    ) -> None:
        mock_searcher.search.return_value = []

        response = client.post(
            "/semantic_search", json={"query": "mystery", "top_k": 1}
        )

        assert response.status_code == 200
