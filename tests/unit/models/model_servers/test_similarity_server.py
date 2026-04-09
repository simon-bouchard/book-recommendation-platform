# tests/unit/models/model_servers/test_similarity_server.py
"""
Unit tests for the unified similarity model server.

Strategy overview
-----------------
The similarity server owns four distinct artifact types, each requiring a
different patching approach:

- HybridScorer, SubjectScorer: standard singleton pattern — patch via
  ``_instance`` using ``install_mock_singleton``.
- get_subject_similarity_index, get_als_similarity_index: module-level
  factory functions imported into the server module — patched with
  ``monkeypatch.setattr`` against ``model_servers.similarity.main``.

Health check: only HybridScorer._instance is inspected (it is the last
artifact initialized, so its presence is a proxy for full readiness).

Return-order note: SimilarityIndex.search returns ``(scores, item_ids)``
while HybridScorer.score and SubjectScorer.score return ``(item_ids, scores)``.
Mock return values are set accordingly in each test class.
"""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from starlette.testclient import TestClient

from model_servers.similarity.main import app
from models.infrastructure.hybrid_scorer import HybridScorer
from models.infrastructure.subject_scorer import SubjectScorer
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
    """Clear both HybridScorer and SubjectScorer singletons around every test."""
    HybridScorer._instance = None
    SubjectScorer._instance = None
    yield
    HybridScorer._instance = None
    SubjectScorer._instance = None


@pytest.fixture()
def mock_hybrid_scorer(install_mock_singleton) -> MagicMock:
    """Install a MagicMock as the active HybridScorer singleton."""
    return install_mock_singleton(HybridScorer)


@pytest.fixture()
def mock_subject_scorer(install_mock_singleton) -> MagicMock:
    """Install a MagicMock as the active SubjectScorer singleton."""
    return install_mock_singleton(SubjectScorer)


@pytest.fixture()
def mock_subject_index(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Replace get_subject_similarity_index with a lambda returning a MagicMock.

    Patched on the server module so all calls within a test hit the mock.
    """
    mock_idx = MagicMock()
    monkeypatch.setattr(
        "model_servers.similarity.main.get_subject_similarity_index",
        lambda: mock_idx,
    )
    return mock_idx


@pytest.fixture()
def mock_als_index(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Replace get_als_similarity_index with a lambda returning a MagicMock.

    Patched on the server module so all calls within a test hit the mock.
    """
    mock_idx = MagicMock()
    monkeypatch.setattr(
        "model_servers.similarity.main.get_als_similarity_index",
        lambda: mock_idx,
    )
    return mock_idx


@pytest.fixture()
def client() -> TestClient:
    return make_test_client(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ok_when_hybrid_scorer_initialized(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        response = client.get("/health")

        assert_health_ok(response, "similarity")

    def test_503_when_singleton_is_none(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 503

    def test_artifact_version_is_unknown_without_env_var(
        self,
        client: TestClient,
        mock_hybrid_scorer: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MODEL_VERSION_POINTER", raising=False)

        response = client.get("/health")

        assert response.json()["artifact_version"] == "unknown"

    def test_artifact_version_read_from_pointer_file(
        self,
        client: TestClient,
        mock_hybrid_scorer: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        version_pointer_file: tuple,
    ) -> None:
        pointer, version = version_pointer_file
        monkeypatch.setenv("MODEL_VERSION_POINTER", str(pointer))

        response = client.get("/health")

        assert response.json()["artifact_version"] == version


# ---------------------------------------------------------------------------
# POST /has_book_als
# ---------------------------------------------------------------------------


class TestHasBookAls:
    def test_book_with_als_returns_has_als_true(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        mock_als_index.has_item.return_value = True

        response = client.post("/has_book_als", json={"item_idx": 42})

        assert response.status_code == 200
        body = response.json()
        assert body["item_idx"] == 42
        assert body["has_als"] is True
        mock_als_index.has_item.assert_called_once_with(42)

    def test_book_without_als_returns_has_als_false(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        mock_als_index.has_item.return_value = False

        response = client.post("/has_book_als", json={"item_idx": 99})

        assert response.status_code == 200
        assert response.json()["has_als"] is False

    def test_missing_item_idx_returns_422(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        response = client.post("/has_book_als", json={})

        assert response.status_code == 422

    def test_non_integer_item_idx_returns_422(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        response = client.post("/has_book_als", json={"item_idx": "bad"})

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /subject_sim
# ---------------------------------------------------------------------------


class TestSubjectSim:
    def test_returns_ordered_results(
        self, client: TestClient, mock_subject_index: MagicMock
    ) -> None:
        item_ids = np.array([10, 20, 30], dtype=np.int64)
        scores = np.array([0.9, 0.7, 0.5], dtype=np.float32)
        mock_subject_index.search.return_value = (scores, item_ids)

        response = client.post("/subject_sim", json={"item_idx": 1, "k": 3})

        assert response.status_code == 200
        assert_scored_items(response.json()["results"], [(10, 0.9), (20, 0.7), (30, 0.5)])

    def test_search_called_with_correct_args(
        self, client: TestClient, mock_subject_index: MagicMock
    ) -> None:
        mock_subject_index.search.return_value = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

        client.post("/subject_sim", json={"item_idx": 7, "k": 50})

        mock_subject_index.search.assert_called_once_with(query_item_id=7, k=50, exclude_query=True)

    def test_default_k_is_200(self, client: TestClient, mock_subject_index: MagicMock) -> None:
        mock_subject_index.search.return_value = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

        client.post("/subject_sim", json={"item_idx": 1})

        mock_subject_index.search.assert_called_once_with(
            query_item_id=1, k=200, exclude_query=True
        )

    def test_item_not_in_index_returns_empty_results(
        self, client: TestClient, mock_subject_index: MagicMock
    ) -> None:
        mock_subject_index.search.return_value = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

        response = client.post("/subject_sim", json={"item_idx": 999, "k": 10})

        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_search_exception_returns_500(
        self, client: TestClient, mock_subject_index: MagicMock
    ) -> None:
        mock_subject_index.search.side_effect = RuntimeError("FAISS error")

        response = client.post("/subject_sim", json={"item_idx": 1, "k": 10})

        assert response.status_code == 500

    def test_k_below_minimum_returns_422(
        self, client: TestClient, mock_subject_index: MagicMock
    ) -> None:
        response = client.post("/subject_sim", json={"item_idx": 1, "k": 0})

        assert response.status_code == 422

    def test_k_above_maximum_returns_422(
        self, client: TestClient, mock_subject_index: MagicMock
    ) -> None:
        response = client.post("/subject_sim", json={"item_idx": 1, "k": 1001})

        assert response.status_code == 422

    def test_missing_item_idx_returns_422(
        self, client: TestClient, mock_subject_index: MagicMock
    ) -> None:
        response = client.post("/subject_sim", json={"k": 10})

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /als_sim
# ---------------------------------------------------------------------------


class TestAlsSim:
    def test_returns_ordered_results(self, client: TestClient, mock_als_index: MagicMock) -> None:
        item_ids = np.array([5, 15], dtype=np.int64)
        scores = np.array([0.8, 0.6], dtype=np.float32)
        mock_als_index.search.return_value = (scores, item_ids)

        response = client.post("/als_sim", json={"item_idx": 1, "k": 2})

        assert response.status_code == 200
        assert_scored_items(response.json()["results"], [(5, 0.8), (15, 0.6)])

    def test_search_called_with_correct_args(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        mock_als_index.search.return_value = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

        client.post("/als_sim", json={"item_idx": 7, "k": 50})

        mock_als_index.search.assert_called_once_with(query_item_id=7, k=50, exclude_query=True)

    def test_default_k_is_200(self, client: TestClient, mock_als_index: MagicMock) -> None:
        mock_als_index.search.return_value = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

        client.post("/als_sim", json={"item_idx": 1})

        mock_als_index.search.assert_called_once_with(query_item_id=1, k=200, exclude_query=True)

    def test_item_not_in_index_returns_empty_results(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        mock_als_index.search.return_value = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

        response = client.post("/als_sim", json={"item_idx": 999, "k": 10})

        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_search_exception_returns_500(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        mock_als_index.search.side_effect = RuntimeError("FAISS error")

        response = client.post("/als_sim", json={"item_idx": 1, "k": 10})

        assert response.status_code == 500

    def test_k_below_minimum_returns_422(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        response = client.post("/als_sim", json={"item_idx": 1, "k": 0})

        assert response.status_code == 422

    def test_k_above_maximum_returns_422(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        response = client.post("/als_sim", json={"item_idx": 1, "k": 1001})

        assert response.status_code == 422

    def test_missing_item_idx_returns_422(
        self, client: TestClient, mock_als_index: MagicMock
    ) -> None:
        response = client.post("/als_sim", json={"k": 10})

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /hybrid_sim
# ---------------------------------------------------------------------------


class TestHybridSim:
    @pytest.fixture(autouse=True)
    def _mock_als_index(self, mock_als_index: MagicMock) -> None:
        """Ensure get_als_similarity_index() is mocked for all hybrid_sim tests."""
        mock_als_index.has_item.return_value = True

    def test_returns_ordered_results(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        mock_hybrid_scorer.score.return_value = make_scored_arrays([10, 20], [0.85, 0.6])

        response = client.post("/hybrid_sim", json={"item_idx": 1, "k": 2})

        assert response.status_code == 200
        assert_scored_items(response.json()["results"], [(10, 0.85), (20, 0.6)])

    def test_score_called_with_correct_args(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        mock_hybrid_scorer.score.return_value = make_scored_arrays([], [])

        client.post("/hybrid_sim", json={"item_idx": 7, "k": 50, "alpha": 0.4})

        mock_hybrid_scorer.score.assert_called_once_with(item_idx=7, k=50, alpha=0.4)

    def test_default_k_and_alpha(self, client: TestClient, mock_hybrid_scorer: MagicMock) -> None:
        mock_hybrid_scorer.score.return_value = make_scored_arrays([], [])

        client.post("/hybrid_sim", json={"item_idx": 1})

        mock_hybrid_scorer.score.assert_called_once_with(item_idx=1, k=200, alpha=0.6)

    def test_item_not_in_index_returns_empty_results(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        mock_hybrid_scorer.score.return_value = make_scored_arrays([], [])

        response = client.post("/hybrid_sim", json={"item_idx": 999, "k": 10})

        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_alpha_boundary_zero_is_valid(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        mock_hybrid_scorer.score.return_value = make_scored_arrays([], [])

        response = client.post("/hybrid_sim", json={"item_idx": 1, "alpha": 0.0})

        assert response.status_code == 200

    def test_alpha_boundary_one_is_valid(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        mock_hybrid_scorer.score.return_value = make_scored_arrays([], [])

        response = client.post("/hybrid_sim", json={"item_idx": 1, "alpha": 1.0})

        assert response.status_code == 200

    def test_alpha_above_maximum_returns_422(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        response = client.post("/hybrid_sim", json={"item_idx": 1, "alpha": 1.1})

        assert response.status_code == 422

    def test_alpha_below_minimum_returns_422(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        response = client.post("/hybrid_sim", json={"item_idx": 1, "alpha": -0.1})

        assert response.status_code == 422

    def test_k_below_minimum_returns_422(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        response = client.post("/hybrid_sim", json={"item_idx": 1, "k": 0})

        assert response.status_code == 422

    def test_k_above_maximum_returns_422(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        response = client.post("/hybrid_sim", json={"item_idx": 1, "k": 1001})

        assert response.status_code == 422

    def test_score_exception_returns_500(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        mock_hybrid_scorer.score.side_effect = RuntimeError("matmul failed")

        response = client.post("/hybrid_sim", json={"item_idx": 1, "k": 10})

        assert response.status_code == 500

    def test_missing_item_idx_returns_422(
        self, client: TestClient, mock_hybrid_scorer: MagicMock
    ) -> None:
        response = client.post("/hybrid_sim", json={"k": 10})

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /subject_recs
# ---------------------------------------------------------------------------


class TestSubjectRecs:
    def test_returns_ordered_results(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        mock_subject_scorer.score.return_value = make_scored_arrays([10, 20, 30], [0.9, 0.7, 0.5])

        response = client.post(
            "/subject_recs",
            json={"user_vector": [0.1, 0.2, 0.3], "k": 3},
        )

        assert response.status_code == 200
        assert_scored_items(response.json()["results"], [(10, 0.9), (20, 0.7), (30, 0.5)])

    def test_score_called_with_correct_k_and_alpha(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        mock_subject_scorer.score.return_value = make_scored_arrays([], [])

        client.post(
            "/subject_recs",
            json={"user_vector": [0.5, 0.5], "k": 50, "alpha": 0.8},
        )

        call_kwargs = mock_subject_scorer.score.call_args.kwargs
        assert call_kwargs["k"] == 50
        assert call_kwargs["alpha"] == pytest.approx(0.8)

    def test_user_vector_passed_as_numpy_float32(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        mock_subject_scorer.score.return_value = make_scored_arrays([], [])

        client.post("/subject_recs", json={"user_vector": [0.1, 0.2, 0.3]})

        call_kwargs = mock_subject_scorer.score.call_args.kwargs
        vec = call_kwargs["user_vector"]
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        np.testing.assert_allclose(vec, [0.1, 0.2, 0.3], atol=1e-6)

    def test_default_k_and_alpha(self, client: TestClient, mock_subject_scorer: MagicMock) -> None:
        mock_subject_scorer.score.return_value = make_scored_arrays([], [])

        client.post("/subject_recs", json={"user_vector": [0.5, 0.5]})

        call_kwargs = mock_subject_scorer.score.call_args.kwargs
        assert call_kwargs["k"] == 200
        assert call_kwargs["alpha"] == pytest.approx(0.6)

    def test_empty_results_returned_for_zero_score_vector(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        mock_subject_scorer.score.return_value = make_scored_arrays([], [])

        response = client.post("/subject_recs", json={"user_vector": [0.0, 0.0, 0.0]})

        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_empty_user_vector_returns_422(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        response = client.post("/subject_recs", json={"user_vector": []})

        assert response.status_code == 422

    def test_missing_user_vector_returns_422(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        response = client.post("/subject_recs", json={"k": 10})

        assert response.status_code == 422

    def test_alpha_above_maximum_returns_422(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        response = client.post(
            "/subject_recs",
            json={"user_vector": [0.5, 0.5], "alpha": 1.1},
        )

        assert response.status_code == 422

    def test_alpha_below_minimum_returns_422(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        response = client.post(
            "/subject_recs",
            json={"user_vector": [0.5, 0.5], "alpha": -0.1},
        )

        assert response.status_code == 422

    def test_k_above_maximum_returns_422(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        response = client.post(
            "/subject_recs",
            json={"user_vector": [0.5, 0.5], "k": 1001},
        )

        assert response.status_code == 422

    def test_score_exception_returns_500(
        self, client: TestClient, mock_subject_scorer: MagicMock
    ) -> None:
        mock_subject_scorer.score.side_effect = RuntimeError("matmul failed")

        response = client.post("/subject_recs", json={"user_vector": [0.1, 0.2, 0.3]})

        assert response.status_code == 500
