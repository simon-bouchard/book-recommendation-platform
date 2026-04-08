# tests/unit/models/model_servers/test_embedder_server.py
"""
Unit tests for the embedder model server.

Strategy: TestClient is constructed without entering its context manager, so
the FastAPI lifespan never runs and no artifact loading occurs. The embedder
endpoint reads SubjectEmbedder._instance directly, so patching _instance is
the only setup required — no constructor interception needed.
"""

from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from starlette.testclient import TestClient

from model_servers.embedder.main import app
from models.infrastructure.subject_embedder import SubjectEmbedder
from tests.unit.models.model_servers.conftest import (
    assert_health_ok,
    make_test_client,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None, None, None]:
    """Clear SubjectEmbedder._instance before and after every test in this module."""
    SubjectEmbedder._instance = None
    yield
    SubjectEmbedder._instance = None


@pytest.fixture()
def mock_embedder(install_mock_singleton) -> MagicMock:
    """Install a MagicMock as the active SubjectEmbedder singleton."""
    return install_mock_singleton(SubjectEmbedder)


@pytest.fixture()
def client() -> TestClient:
    return make_test_client(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ok_when_singleton_initialized(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        response = client.get("/health")

        assert_health_ok(response, "embedder")

    def test_503_when_singleton_is_none(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 503

    def test_artifact_version_is_unknown_without_env_var(
        self, client: TestClient, mock_embedder: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODEL_VERSION_POINTER", raising=False)

        response = client.get("/health")

        assert response.json()["artifact_version"] == "unknown"

    def test_artifact_version_read_from_pointer_file(
        self,
        client: TestClient,
        mock_embedder: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        version_pointer_file: tuple,
    ) -> None:
        pointer, version = version_pointer_file
        monkeypatch.setenv("MODEL_VERSION_POINTER", str(pointer))

        response = client.get("/health")

        assert response.json()["artifact_version"] == version


# ---------------------------------------------------------------------------
# POST /embed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_returns_vector_for_valid_subjects(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        mock_embedder.embed.return_value = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        response = client.post("/embed", json={"subject_indices": [5, 12, 23]})

        assert response.status_code == 200
        assert response.json()["vector"] == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-6)

    def test_embed_called_with_subject_indices(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        mock_embedder.embed.return_value = np.array([0.5, 0.5], dtype=np.float32)

        client.post("/embed", json={"subject_indices": [5, 12, 23]})

        mock_embedder.embed.assert_called_once_with([5, 12, 23])

    def test_single_subject_index_is_valid(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        mock_embedder.embed.return_value = np.array([0.5, 0.5], dtype=np.float32)

        response = client.post("/embed", json={"subject_indices": [7]})

        assert response.status_code == 200

    def test_large_subject_list_is_valid(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        mock_embedder.embed.return_value = np.zeros(64, dtype=np.float32)

        response = client.post("/embed", json={"subject_indices": list(range(100))})

        assert response.status_code == 200

    def test_vector_length_matches_model_output(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        mock_embedder.embed.return_value = np.ones(64, dtype=np.float32)

        response = client.post("/embed", json={"subject_indices": [1, 2, 3]})

        assert len(response.json()["vector"]) == 64

    def test_response_vector_contains_floats(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        mock_embedder.embed.return_value = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        response = client.post("/embed", json={"subject_indices": [1]})

        assert all(isinstance(v, float) for v in response.json()["vector"])

    def test_empty_subject_indices_returns_422(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        response = client.post("/embed", json={"subject_indices": []})

        assert response.status_code == 422

    def test_missing_subject_indices_field_returns_422(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        response = client.post("/embed", json={})

        assert response.status_code == 422

    def test_non_integer_subject_indices_returns_422(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        response = client.post("/embed", json={"subject_indices": ["a", "b"]})

        assert response.status_code == 422

    def test_503_when_singleton_is_none(self, client: TestClient) -> None:
        response = client.post("/embed", json={"subject_indices": [1, 2, 3]})

        assert response.status_code == 503

    def test_embed_exception_returns_500(
        self, client: TestClient, mock_embedder: MagicMock
    ) -> None:
        mock_embedder.embed.side_effect = RuntimeError("forward pass failed")

        response = client.post("/embed", json={"subject_indices": [1, 2, 3]})

        assert response.status_code == 500
