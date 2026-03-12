# tests/unit/models/model_servers/test_metadata_server.py
"""
Unit tests for the metadata model server.

Strategy overview
-----------------
The metadata server owns two distinct artifacts that require different patching:

- _book_lookup: a module-level Optional[dict[int, str]] — set via direct
  assignment on the server module. The autouse reset_singleton fixture restores
  it to None after every test.
- PopularityScorer: standard singleton — patched via install_mock_singleton.

Health check requires BOTH artifacts to be present; each can independently
cause a 503 response.

_row_to_book_meta is a private conversion function with non-trivial NaN
handling and type coercion. It lives in metadata_enrichment and is tested
directly in TestRowToBookMeta before the endpoint tests so that endpoint
assertions can rely on it being correct.
"""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

import model_servers.metadata.main as metadata_main
from model_servers.metadata.main import app
from models.infrastructure.metadata_enrichment import _row_to_book_meta, build_lookup
from models.infrastructure.popularity_scorer import PopularityScorer
from tests.unit.models.model_servers.conftest import (
    assert_health_ok,
    make_test_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None, None, None]:
    """
    Clear PopularityScorer._instance and _book_lookup around every test.

    Direct assignment is safe here because the autouse teardown runs after
    every test regardless of pass/fail, ensuring no state leaks between tests.
    """
    PopularityScorer._instance = None
    metadata_main._book_lookup = None
    yield
    PopularityScorer._instance = None
    metadata_main._book_lookup = None


@pytest.fixture()
def mock_popularity_scorer(install_mock_singleton) -> MagicMock:
    """Install a MagicMock as the active PopularityScorer singleton."""
    return install_mock_singleton(PopularityScorer)


@pytest.fixture()
def book_meta_df() -> pd.DataFrame:
    """
    Minimal book metadata DataFrame covering both complete and sparse records.

    Item 101 has all fields populated. Item 202 has all optional fields absent
    (NaN) to exercise the NaN-guard paths in _row_to_book_meta.
    """
    return pd.DataFrame(
        {
            "title": ["Book Alpha", "Book Beta"],
            "author": ["Author One", np.nan],
            "year": [2001.0, np.nan],
            "isbn": ["978-0-06-112008-4", np.nan],
            "cover_id": ["OL1234M", np.nan],
            "book_avg_rating": [4.2, np.nan],
            "book_num_ratings": [100, 50],
            "bayes": [0.85, np.nan],
        },
        index=[101, 202],
    )


@pytest.fixture()
def with_book_meta(book_meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build and install the pre-serialized lookup dict from book_meta_df.

    Assigns the result of build_lookup to metadata_main._book_lookup so that
    endpoint handlers see the same dict structure they would at runtime.
    """
    metadata_main._book_lookup = build_lookup(book_meta_df)
    return book_meta_df


@pytest.fixture()
def client() -> TestClient:
    return make_test_client(app)


# ---------------------------------------------------------------------------
# _row_to_book_meta (private conversion function)
# ---------------------------------------------------------------------------


class TestRowToBookMeta:
    def test_full_row_maps_all_fields(self, book_meta_df: pd.DataFrame) -> None:
        result = _row_to_book_meta(101, book_meta_df.loc[101])

        assert result.item_idx == 101
        assert result.title == "Book Alpha"
        assert result.author == "Author One"
        assert result.year == 2001
        assert result.isbn == "978-0-06-112008-4"
        assert result.cover_id == "OL1234M"
        assert result.avg_rating == pytest.approx(4.2)
        assert result.num_ratings == 100
        assert result.bayes_score == pytest.approx(0.85)

    def test_nan_author_maps_to_none(self, book_meta_df: pd.DataFrame) -> None:
        result = _row_to_book_meta(202, book_meta_df.loc[202])

        assert result.author is None

    def test_nan_year_maps_to_none(self, book_meta_df: pd.DataFrame) -> None:
        result = _row_to_book_meta(202, book_meta_df.loc[202])

        assert result.year is None

    def test_nan_isbn_maps_to_none(self, book_meta_df: pd.DataFrame) -> None:
        result = _row_to_book_meta(202, book_meta_df.loc[202])

        assert result.isbn is None

    def test_nan_cover_id_maps_to_none(self, book_meta_df: pd.DataFrame) -> None:
        result = _row_to_book_meta(202, book_meta_df.loc[202])

        assert result.cover_id is None

    def test_nan_avg_rating_maps_to_none(self, book_meta_df: pd.DataFrame) -> None:
        result = _row_to_book_meta(202, book_meta_df.loc[202])

        assert result.avg_rating is None

    def test_nan_bayes_score_maps_to_none(self, book_meta_df: pd.DataFrame) -> None:
        result = _row_to_book_meta(202, book_meta_df.loc[202])

        assert result.bayes_score is None

    def test_num_ratings_present_in_sparse_row(self, book_meta_df: pd.DataFrame) -> None:
        result = _row_to_book_meta(202, book_meta_df.loc[202])

        assert result.num_ratings == 50

    def test_missing_num_ratings_column_defaults_to_zero(self) -> None:
        row = pd.Series({"title": "No Ratings Book"})

        result = _row_to_book_meta(999, row)

        assert result.num_ratings == 0

    def test_empty_string_author_maps_to_none(self) -> None:
        row = pd.Series({"title": "Test", "author": "", "book_num_ratings": 0})

        result = _row_to_book_meta(1, row)

        assert result.author is None

    def test_inf_avg_rating_maps_to_none(self) -> None:
        row = pd.Series({"title": "Test", "book_avg_rating": float("inf"), "book_num_ratings": 0})

        result = _row_to_book_meta(1, row)

        assert result.avg_rating is None

    def test_item_idx_is_preserved(self, book_meta_df: pd.DataFrame) -> None:
        result = _row_to_book_meta(101, book_meta_df.loc[101])

        assert result.item_idx == 101


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ok_when_both_artifacts_initialized(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.get("/health")

        assert_health_ok(response, "metadata")

    def test_503_when_book_lookup_is_none(
        self, client: TestClient, mock_popularity_scorer: MagicMock
    ) -> None:
        response = client.get("/health")

        assert response.status_code == 503

    def test_503_when_popularity_scorer_is_none(
        self, client: TestClient, with_book_meta: pd.DataFrame
    ) -> None:
        response = client.get("/health")

        assert response.status_code == 503

    def test_artifact_version_is_unknown_without_env_var(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MODEL_VERSION_POINTER", raising=False)

        response = client.get("/health")

        assert response.json()["artifact_version"] == "unknown"

    def test_artifact_version_read_from_pointer_file(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        version_pointer_file: tuple,
    ) -> None:
        pointer, version = version_pointer_file
        monkeypatch.setenv("MODEL_VERSION_POINTER", str(pointer))

        response = client.get("/health")

        assert response.json()["artifact_version"] == version


# ---------------------------------------------------------------------------
# POST /enrich
# ---------------------------------------------------------------------------


class TestEnrich:
    def test_returns_metadata_for_known_items(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.post("/enrich", json={"item_indices": [101, 202]})

        assert response.status_code == 200
        books = response.json()["books"]
        assert len(books) == 2
        assert books[0]["item_idx"] == 101
        assert books[0]["title"] == "Book Alpha"
        assert books[1]["item_idx"] == 202
        assert books[1]["title"] == "Book Beta"

    def test_missing_items_are_silently_omitted(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.post("/enrich", json={"item_indices": [101, 9999]})

        assert response.status_code == 200
        books = response.json()["books"]
        assert len(books) == 1
        assert books[0]["item_idx"] == 101

    def test_all_items_missing_returns_empty_list(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.post("/enrich", json={"item_indices": [8888, 9999]})

        assert response.status_code == 200
        assert response.json()["books"] == []

    def test_optional_fields_are_none_for_sparse_record(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.post("/enrich", json={"item_indices": [202]})

        book = response.json()["books"][0]
        assert book["author"] is None
        assert book["year"] is None
        assert book["isbn"] is None

    def test_empty_item_indices_returns_422(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.post("/enrich", json={"item_indices": []})

        assert response.status_code == 422

    def test_item_indices_above_maximum_returns_422(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.post("/enrich", json={"item_indices": list(range(2001))})

        assert response.status_code == 422

    def test_missing_item_indices_field_returns_422(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.post("/enrich", json={})

        assert response.status_code == 422

    def test_503_when_book_lookup_is_none(
        self, client: TestClient, mock_popularity_scorer: MagicMock
    ) -> None:
        response = client.post("/enrich", json={"item_indices": [101]})

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# POST /popular
# ---------------------------------------------------------------------------


class TestPopular:
    def test_returns_top_k_books(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        mock_popularity_scorer.top_k.return_value = (
            np.array([101, 202], dtype=np.int64),
            np.array([0.9, 0.8], dtype=np.float32),
        )

        response = client.post("/popular", json={"k": 2})

        assert response.status_code == 200
        books = response.json()["books"]
        assert len(books) == 2
        assert books[0]["item_idx"] == 101
        assert books[1]["item_idx"] == 202

    def test_top_k_called_with_correct_k(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        mock_popularity_scorer.top_k.return_value = (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

        client.post("/popular", json={"k": 25})

        mock_popularity_scorer.top_k.assert_called_once_with(k=25)

    def test_default_k_is_100(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        mock_popularity_scorer.top_k.return_value = (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

        client.post("/popular", json={})

        mock_popularity_scorer.top_k.assert_called_once_with(k=100)

    def test_books_absent_from_meta_are_silently_skipped(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        mock_popularity_scorer.top_k.return_value = (
            np.array([101, 9999], dtype=np.int64),
            np.array([0.9, 0.8], dtype=np.float32),
        )

        response = client.post("/popular", json={"k": 2})

        assert response.status_code == 200
        books = response.json()["books"]
        assert len(books) == 1
        assert books[0]["item_idx"] == 101

    def test_top_k_exception_returns_500(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        mock_popularity_scorer.top_k.side_effect = RuntimeError("scores corrupted")

        response = client.post("/popular", json={"k": 10})

        assert response.status_code == 500

    def test_k_below_minimum_returns_422(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.post("/popular", json={"k": 0})

        assert response.status_code == 422

    def test_k_above_maximum_returns_422(
        self,
        client: TestClient,
        with_book_meta: pd.DataFrame,
        mock_popularity_scorer: MagicMock,
    ) -> None:
        response = client.post("/popular", json={"k": 1001})

        assert response.status_code == 422

    def test_503_when_book_lookup_is_none(
        self, client: TestClient, mock_popularity_scorer: MagicMock
    ) -> None:
        response = client.post("/popular", json={"k": 10})

        assert response.status_code == 503

    def test_503_when_popularity_scorer_is_none(
        self, client: TestClient, with_book_meta: pd.DataFrame
    ) -> None:
        response = client.post("/popular", json={"k": 10})

        assert response.status_code == 503
