# tests/unit/models/services/test_similarity_service.py
"""
Unit tests for SimilarityService.

Architecture under test:
  - get_similar() is async and a pure router — no local indices or data loaders.
  - subject/als/hybrid modes map directly to get_similarity_client() methods.
  - Results are enriched via get_metadata_client().enrich().
  - Invalid mode raises ValueError immediately (before any client call).
  - All filtering, blending, and quality thresholds are the model server's
    responsibility; the service does not re-implement them.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model_servers._shared.contracts import ScoredItem, SimResponse
from models.services.similarity_service import SimilarityService

_SVC = "models.services.similarity_service"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_response(*pairs: tuple) -> SimResponse:
    """Build a SimResponse from (item_idx, score) pairs."""
    return SimResponse(results=[ScoredItem(item_idx=idx, score=score) for idx, score in pairs])


def _enrich_response(*item_ids: int) -> dict:
    """Build a dict[int, dict] matching what MetadataClient.enrich() returns."""
    return {
        idx: {
            "item_idx": idx,
            "title": f"Book {idx}",
            "author": f"Author {idx}",
            "year": 2020 + (idx % 10),
            "isbn": f"ISBN-{idx}",
            "cover_id": f"cover-{idx}",
            "avg_rating": 4.0,
            "num_ratings": 50,
        }
        for idx in item_ids
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sim_client():
    """
    Patch get_similarity_client with an AsyncMock whose methods return
    non-empty default responses so result-formatting tests work out of the box.
    """
    with patch(f"{_SVC}.get_similarity_client") as mock_get:
        client = AsyncMock()
        client.subject_sim.return_value = _sim_response((1001, 0.9), (1002, 0.8))
        client.als_sim.return_value = _sim_response((1003, 0.85), (1004, 0.75))
        client.hybrid_sim.return_value = _sim_response((1001, 0.88), (1003, 0.77))
        mock_get.return_value = client
        yield client


@pytest.fixture
def mock_meta_client():
    """
    Patch get_metadata_client with an AsyncMock whose enrich method returns
    metadata for the default sim-client item ids.
    """
    with patch(f"{_SVC}.get_metadata_client") as mock_get:
        client = AsyncMock()
        client.enrich.return_value = _enrich_response(1001, 1002, 1003, 1004)
        mock_get.return_value = client
        yield client


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Service is stateless — no pre-loaded data."""

    def test_constructs_without_errors(self):
        assert SimilarityService() is not None

    def test_holds_no_instance_state(self):
        service = SimilarityService()
        assert not hasattr(service, "_book_meta")
        assert not hasattr(service, "_subject_index")
        assert not hasattr(service, "_als_index")
        assert not hasattr(service, "_hybrid_data")
        assert not hasattr(service, "ALS_MIN_RATINGS")
        assert not hasattr(service, "HYBRID_MIN_RATINGS")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouting:
    """get_similar routes each mode to exactly the right client method."""

    @pytest.mark.asyncio
    async def test_subject_mode_calls_subject_sim(self, mock_sim_client, mock_meta_client):
        await SimilarityService().get_similar(item_idx=1000, mode="subject", k=10)

        mock_sim_client.subject_sim.assert_called_once_with(1000, 10)
        mock_sim_client.als_sim.assert_not_called()
        mock_sim_client.hybrid_sim.assert_not_called()

    @pytest.mark.asyncio
    async def test_als_mode_calls_als_sim(self, mock_sim_client, mock_meta_client):
        await SimilarityService().get_similar(item_idx=1000, mode="als", k=10)

        mock_sim_client.als_sim.assert_called_once_with(1000, 10)
        mock_sim_client.subject_sim.assert_not_called()
        mock_sim_client.hybrid_sim.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_mode_calls_hybrid_sim(self, mock_sim_client, mock_meta_client):
        await SimilarityService().get_similar(item_idx=1000, mode="hybrid", k=10, alpha=0.7)

        mock_sim_client.hybrid_sim.assert_called_once_with(1000, 10, 0.7)
        mock_sim_client.subject_sim.assert_not_called()
        mock_sim_client.als_sim.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_default_alpha_is_0_6(self, mock_sim_client, mock_meta_client):
        await SimilarityService().get_similar(item_idx=1000, mode="hybrid", k=5)

        _, called_k, called_alpha = mock_sim_client.hybrid_sim.call_args.args
        assert called_alpha == 0.6

    @pytest.mark.asyncio
    async def test_k_forwarded_to_similarity_client(self, mock_sim_client, mock_meta_client):
        await SimilarityService().get_similar(item_idx=1000, mode="subject", k=42)

        _, called_k = mock_sim_client.subject_sim.call_args.args
        assert called_k == 42

    @pytest.mark.asyncio
    async def test_invalid_mode_raises_value_error_before_any_client_call(self, mock_sim_client):
        """ValueError is raised synchronously inside the async body before any I/O."""
        with pytest.raises(ValueError, match="Unknown similarity mode"):
            await SimilarityService().get_similar(item_idx=1000, mode="unknown", k=10)

        mock_sim_client.subject_sim.assert_not_called()
        mock_sim_client.als_sim.assert_not_called()
        mock_sim_client.hybrid_sim.assert_not_called()


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


class TestEnrichment:
    """Results from the similarity client are enriched through the metadata client."""

    @pytest.mark.asyncio
    async def test_enrich_called_with_sim_response_item_indices(
        self, mock_sim_client, mock_meta_client
    ):
        mock_sim_client.subject_sim.return_value = _sim_response((1010, 0.9), (1011, 0.8))

        await SimilarityService().get_similar(item_idx=1000, mode="subject", k=5)

        mock_meta_client.enrich.assert_called_once_with([1010, 1011])

    @pytest.mark.asyncio
    async def test_empty_sim_response_returns_empty_list(self, mock_sim_client, mock_meta_client):
        """Empty similarity results produce empty output; enrich may still be called."""
        mock_sim_client.subject_sim.return_value = SimResponse(results=[])
        mock_meta_client.enrich.return_value = {}

        results = await SimilarityService().get_similar(item_idx=1000, mode="subject", k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_books_absent_from_enrich_response_are_excluded(
        self, mock_sim_client, mock_meta_client
    ):
        """Items the metadata server omits are silently dropped from results."""
        mock_sim_client.subject_sim.return_value = _sim_response((1001, 0.9), (9999, 0.8))
        mock_meta_client.enrich.return_value = _enrich_response(1001)  # 9999 not returned

        results = await SimilarityService().get_similar(item_idx=1000, mode="subject", k=5)

        assert len(results) == 1
        assert results[0]["item_idx"] == 1001


# ---------------------------------------------------------------------------
# Result format
# ---------------------------------------------------------------------------


class TestResultFormat:
    """Each result dict must carry the correct fields and types."""

    @pytest.mark.asyncio
    async def test_results_are_list_of_dicts(self, mock_sim_client, mock_meta_client):
        results = await SimilarityService().get_similar(item_idx=1000, mode="subject", k=5)

        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    @pytest.mark.asyncio
    async def test_required_fields_present(self, mock_sim_client, mock_meta_client):
        results = await SimilarityService().get_similar(item_idx=1000, mode="subject", k=5)

        for r in results:
            assert "item_idx" in r
            assert "title" in r
            assert "score" in r

    @pytest.mark.asyncio
    async def test_optional_metadata_fields_present(self, mock_sim_client, mock_meta_client):
        results = await SimilarityService().get_similar(item_idx=1000, mode="subject", k=5)

        for r in results:
            assert "author" in r
            assert "year" in r
            assert "isbn" in r
            assert "cover_id" in r

    @pytest.mark.asyncio
    async def test_score_comes_from_similarity_response_not_metadata(
        self, mock_sim_client, mock_meta_client
    ):
        """Score is the sim server value; metadata server does not supply it."""
        mock_sim_client.subject_sim.return_value = _sim_response((1001, 0.95))
        mock_meta_client.enrich.return_value = _enrich_response(1001)

        results = await SimilarityService().get_similar(item_idx=1000, mode="subject", k=5)

        assert results[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_field_types_are_correct(self, mock_sim_client, mock_meta_client):
        mock_sim_client.subject_sim.return_value = _sim_response((1001, 0.9))
        mock_meta_client.enrich.return_value = _enrich_response(1001)

        results = await SimilarityService().get_similar(item_idx=1000, mode="subject", k=5)

        r = results[0]
        assert isinstance(r["item_idx"], int)
        assert isinstance(r["title"], str)
        assert isinstance(r["score"], float)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    """get_similar emits structured log events at start, completion, and on error."""

    @pytest.mark.asyncio
    async def test_logs_search_started(self, mock_sim_client, mock_meta_client):
        with patch(f"{_SVC}.logger") as mock_logger:
            await SimilarityService().get_similar(item_idx=1000, mode="subject", k=10)

        started = [c for c in mock_logger.info.call_args_list if "started" in str(c).lower()]
        assert len(started) == 1
        call_str = str(started[0])
        assert "Similarity search started" in call_str
        assert "item_idx" in call_str
        assert "mode" in call_str

    @pytest.mark.asyncio
    async def test_logs_search_completed_with_metrics(self, mock_sim_client, mock_meta_client):
        with patch(f"{_SVC}.logger") as mock_logger:
            await SimilarityService().get_similar(item_idx=1000, mode="subject", k=10)

        completed = [c for c in mock_logger.info.call_args_list if "completed" in str(c).lower()]
        assert len(completed) == 1
        call_str = str(completed[0])
        assert "Similarity search completed" in call_str
        assert "count" in call_str

    @pytest.mark.asyncio
    async def test_logs_error_and_reraises(self, mock_meta_client):
        with patch(f"{_SVC}.get_similarity_client") as mock_get:
            client = AsyncMock()
            client.subject_sim.side_effect = RuntimeError("server down")
            mock_get.return_value = client

            with patch(f"{_SVC}.logger") as mock_logger:
                with pytest.raises(RuntimeError):
                    await SimilarityService().get_similar(item_idx=1000, mode="subject", k=10)

        assert mock_logger.error.called
        call_str = str(mock_logger.error.call_args)
        assert "Similarity search failed" in call_str
        assert "item_idx" in call_str
        assert "mode" in call_str
