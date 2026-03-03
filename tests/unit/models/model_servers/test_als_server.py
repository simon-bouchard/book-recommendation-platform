# tests/unit/models/model_servers/test_als_server.py
"""
Unit tests for the ALS recommendation model server.

Strategy: TestClient is constructed without entering its context manager, so
the FastAPI lifespan never runs and no artifact loading occurs. ALSModel._instance
is patched with a MagicMock via the conftest ``install_mock_singleton`` factory.

Because the mock is not an ALSModel instance, Python's data model bypasses
__new__ and __init__ when ALSModel() is called inside an endpoint — the mock
is returned as-is, and we configure its return values per test.
"""

from typing import Generator
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from model_servers.als.main import app
from models.infrastructure.als_model import ALSModel
from tests.unit.models.model_servers.conftest import (
    assert_health_ok,
    assert_scored_items,
    make_scored_arrays,
    make_test_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None, None, None]:
    """Clear ALSModel._instance before and after every test in this module."""
    ALSModel._instance = None
    yield
    ALSModel._instance = None


@pytest.fixture()
def mock_model(install_mock_singleton) -> MagicMock:
    """Install a MagicMock as the active ALSModel singleton."""
    return install_mock_singleton(ALSModel)


@pytest.fixture()
def client() -> TestClient:
    return make_test_client(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ok_when_singleton_initialized(self, client: TestClient, mock_model: MagicMock) -> None:
        response = client.get("/health")

        assert_health_ok(response, "als")

    def test_503_when_singleton_is_none(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 503

    def test_artifact_version_is_unknown_without_env_var(
        self, client: TestClient, mock_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODEL_VERSION_POINTER", raising=False)

        response = client.get("/health")

        assert response.json()["artifact_version"] == "unknown"

    def test_artifact_version_read_from_pointer_file(
        self,
        client: TestClient,
        mock_model: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        version_pointer_file: tuple,
    ) -> None:
        pointer, version = version_pointer_file
        monkeypatch.setenv("MODEL_VERSION_POINTER", str(pointer))

        response = client.get("/health")

        assert response.json()["artifact_version"] == version


# ---------------------------------------------------------------------------
# POST /has_als_user
# ---------------------------------------------------------------------------


class TestHasAlsUser:
    def test_warm_user_returns_is_warm_true(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        mock_model.has_user.return_value = True

        response = client.post("/has_als_user", json={"user_id": 42})

        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == 42
        assert body["is_warm"] is True
        mock_model.has_user.assert_called_once_with(42)

    def test_cold_user_returns_is_warm_false(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        mock_model.has_user.return_value = False

        response = client.post("/has_als_user", json={"user_id": 99})

        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == 99
        assert body["is_warm"] is False

    def test_user_id_zero_is_valid(self, client: TestClient, mock_model: MagicMock) -> None:
        mock_model.has_user.return_value = False

        response = client.post("/has_als_user", json={"user_id": 0})

        assert response.status_code == 200

    def test_missing_user_id_returns_422(self, client: TestClient, mock_model: MagicMock) -> None:
        response = client.post("/has_als_user", json={})

        assert response.status_code == 422

    def test_non_integer_user_id_returns_422(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        response = client.post("/has_als_user", json={"user_id": "not-an-int"})

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /als_recs
# ---------------------------------------------------------------------------


class TestAlsRecs:
    def test_warm_user_returns_ordered_results(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        mock_model.score.return_value = make_scored_arrays([10, 20, 30], [0.9, 0.7, 0.5])

        response = client.post("/als_recs", json={"user_id": 1, "k": 3})

        assert response.status_code == 200
        assert_scored_items(response.json()["results"], [(10, 0.9), (20, 0.7), (30, 0.5)])

    def test_score_called_with_correct_args(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        mock_model.score.return_value = make_scored_arrays([5], [0.8])

        client.post("/als_recs", json={"user_id": 7, "k": 50})

        mock_model.score.assert_called_once_with(user_id=7, k=50)

    def test_default_k_is_200(self, client: TestClient, mock_model: MagicMock) -> None:
        mock_model.score.return_value = make_scored_arrays([], [])

        client.post("/als_recs", json={"user_id": 1})

        mock_model.score.assert_called_once_with(user_id=1, k=200)

    def test_cold_user_returns_empty_results(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        mock_model.score.return_value = make_scored_arrays([], [])

        response = client.post("/als_recs", json={"user_id": 999, "k": 10})

        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_score_exception_returns_500(self, client: TestClient, mock_model: MagicMock) -> None:
        mock_model.score.side_effect = RuntimeError("matmul failed")

        response = client.post("/als_recs", json={"user_id": 1, "k": 10})

        assert response.status_code == 500

    def test_k_below_minimum_returns_422(self, client: TestClient, mock_model: MagicMock) -> None:
        response = client.post("/als_recs", json={"user_id": 1, "k": 0})

        assert response.status_code == 422

    def test_k_above_maximum_returns_422(self, client: TestClient, mock_model: MagicMock) -> None:
        response = client.post("/als_recs", json={"user_id": 1, "k": 1001})

        assert response.status_code == 422

    def test_missing_user_id_returns_422(self, client: TestClient, mock_model: MagicMock) -> None:
        response = client.post("/als_recs", json={"k": 10})

        assert response.status_code == 422

    def test_single_result_is_correctly_shaped(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        mock_model.score.return_value = make_scored_arrays([42], [1.0])

        response = client.post("/als_recs", json={"user_id": 1, "k": 1})

        assert_scored_items(response.json()["results"], [(42, 1.0)])
