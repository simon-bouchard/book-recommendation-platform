# tests/unit/routes/models/test_similarity_endpoint.py
"""
Unit tests for GET /book/{item_idx}/similar endpoint.
Tests request validation, ALS availability check, service integration,
and response formatting.
"""

import pytest
from unittest.mock import AsyncMock, Mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_als_available(monkeypatch) -> None:
    """
    Patch get_similarity_client so has_book_als returns has_als=True.

    The route calls await get_similarity_client().has_book_als(item_idx) for
    als and hybrid modes. We replace the client returned by the registry with
    a mock whose has_book_als is an AsyncMock, matching the real async method.
    """
    response = Mock()
    response.has_als = True

    client = Mock()
    client.has_book_als = AsyncMock(return_value=response)

    monkeypatch.setattr("routes.models.get_similarity_client", lambda: client)


def _patch_als_unavailable(monkeypatch) -> None:
    """
    Patch get_similarity_client so has_book_als returns has_als=False.

    Simulates a book that has no ALS factors, which should trigger a 422.
    """
    response = Mock()
    response.has_als = False

    client = Mock()
    client.has_book_als = AsyncMock(return_value=response)

    monkeypatch.setattr("routes.models.get_similarity_client", lambda: client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimilarityEndpointBasics:
    """Happy-path smoke tests."""

    def test_returns_200_with_valid_item_idx(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Should return 200 and a list of similar books for a valid item_idx."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2  # From sample_similar_books fixture

    def test_calls_service_with_correct_parameters(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Should forward item_idx, mode, alpha, and top_k to the service."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        test_client.get("/book/1234/similar?mode=subject&alpha=0.7&top_k=50")

        assert mock_similarity_service.get_similar.called
        call_kwargs = mock_similarity_service.get_similar.call_args[1]

        assert call_kwargs["item_idx"] == 1234
        assert call_kwargs["mode"] == "subject"
        assert call_kwargs["alpha"] == pytest.approx(0.7)
        assert call_kwargs["k"] == 50


class TestSimilarityEndpointModes:
    """Test mode-specific behaviour."""

    def test_subject_mode_calls_service_directly(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Subject mode requires no ALS availability check."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar?mode=subject")

        assert response.status_code == 200
        call_kwargs = mock_similarity_service.get_similar.call_args[1]
        assert call_kwargs["mode"] == "subject"

    def test_als_mode_checks_availability_before_calling_service(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """ALS mode should confirm the book has ALS factors before calling the service."""
        _patch_als_available(monkeypatch)
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar?mode=als")

        assert response.status_code == 200
        call_kwargs = mock_similarity_service.get_similar.call_args[1]
        assert call_kwargs["mode"] == "als"

    def test_als_mode_returns_422_when_book_has_no_als_data(self, test_client, monkeypatch):
        """Should return 422 when the requested book has no ALS factors (empty results)."""
        from unittest.mock import AsyncMock, Mock
        mock_service = Mock()
        mock_service.get_similar = AsyncMock(return_value=[])
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_service)

        response = test_client.get("/book/9999/similar?mode=als")

        assert response.status_code == 422
        assert (
            "Behavioral similarity unavailable" in response.json()["detail"]
            or "als factors" in response.json()["detail"].lower()
        )

    def test_hybrid_mode_calls_service_with_alpha(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Hybrid mode should pass alpha to the service after confirming ALS availability."""
        _patch_als_available(monkeypatch)
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar?mode=hybrid&alpha=0.3")

        assert response.status_code == 200
        call_kwargs = mock_similarity_service.get_similar.call_args[1]
        assert call_kwargs["mode"] == "hybrid"
        assert call_kwargs["alpha"] == pytest.approx(0.3)


class TestSimilarityEndpointResponseFormat:
    """Test response structure for backward compatibility."""

    def test_response_is_flat_list(self, test_client, mock_similarity_service, monkeypatch):
        """Response should be a flat list, not nested under a key."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar")
        data = response.json()

        assert isinstance(data, list)

    def test_response_includes_all_required_fields(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Each result should include item_idx, title, score, author, year, isbn, cover_id."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar")
        first = response.json()[0]

        assert "item_idx" in first
        assert "title" in first
        assert "score" in first
        assert "author" in first
        assert "year" in first
        assert "isbn" in first
        assert "cover_id" in first

    def test_response_matches_service_output_exactly(
        self, test_client, mock_similarity_service, sample_similar_books, monkeypatch
    ):
        """Endpoint should return service results without transformation."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar")
        data = response.json()

        assert data[0]["item_idx"] == sample_similar_books[0]["item_idx"]
        assert data[0]["title"] == sample_similar_books[0]["title"]
        assert data[0]["score"] == pytest.approx(sample_similar_books[0]["score"])
        assert data[1]["item_idx"] == sample_similar_books[1]["item_idx"]


class TestSimilarityEndpointParameterValidation:
    """Test query parameter validation and default values."""

    def test_default_parameters_when_not_provided(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Should use mode=subject, alpha=0.6, top_k=200 when params are omitted."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        test_client.get("/book/1234/similar")

        call_kwargs = mock_similarity_service.get_similar.call_args[1]
        assert call_kwargs["mode"] == "subject"
        assert call_kwargs["alpha"] == pytest.approx(0.6)
        assert call_kwargs["k"] == 200

    def test_rejects_invalid_mode(self, test_client, monkeypatch):
        """Should return 422 for a mode not in (subject, als, hybrid)."""
        response = test_client.get("/book/1234/similar?mode=invalid")

        assert response.status_code == 422

    def test_rejects_alpha_above_one(self, test_client, monkeypatch):
        """Should return 422 for alpha > 1."""
        response = test_client.get("/book/1234/similar?alpha=1.5")

        assert response.status_code == 422

    def test_rejects_alpha_below_zero(self, test_client, monkeypatch):
        """Should return 422 for alpha < 0."""
        response = test_client.get("/book/1234/similar?alpha=-0.1")

        assert response.status_code == 422

    def test_rejects_top_k_out_of_range(self, test_client, monkeypatch):
        """Should return 422 for top_k above the 500 ceiling."""
        response = test_client.get("/book/1234/similar?top_k=501")

        assert response.status_code == 422

    def test_accepts_valid_parameter_combinations(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Should accept boundary and mid-range parameter values without error."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        for params in [
            "mode=subject&alpha=0.0&top_k=1",
            "mode=subject&alpha=1.0&top_k=500",
            "mode=subject&alpha=0.5&top_k=100",
        ]:
            response = test_client.get(f"/book/1234/similar?{params}")
            assert response.status_code == 200, f"Failed for params: {params}"


class TestSimilarityEndpointEdgeCases:
    """Test edge cases in item_idx values and result sizes."""

    def test_handles_item_idx_zero(self, test_client, mock_similarity_service, monkeypatch):
        """item_idx=0 is a valid integer path parameter and should return 200."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/0/similar")

        assert response.status_code == 200
        call_kwargs = mock_similarity_service.get_similar.call_args[1]
        assert call_kwargs["item_idx"] == 0

    def test_handles_large_item_idx(self, test_client, mock_similarity_service, monkeypatch):
        """Large item_idx values should be accepted without validation errors."""
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/9999999/similar")

        assert response.status_code == 200
        call_kwargs = mock_similarity_service.get_similar.call_args[1]
        assert call_kwargs["item_idx"] == 9999999

    def test_returns_empty_list_with_200_when_service_returns_no_results(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Should return 200 with an empty list when the service finds no similar books."""
        mock_similarity_service.get_similar = AsyncMock(return_value=[])
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar")

        assert response.status_code == 200
        assert response.json() == []


class TestSimilarityEndpointErrorHandling:
    """Test error propagation from the service layer."""

    def test_returns_500_when_service_raises_unexpected_exception(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Should return 500 if the service raises an unexpected exception."""
        mock_similarity_service.get_similar = AsyncMock(side_effect=RuntimeError("Index crashed"))
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar")

        assert response.status_code == 500

    def test_returns_422_when_service_raises_value_error(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Should return 422 if the service raises a ValueError."""
        mock_similarity_service.get_similar = AsyncMock(side_effect=ValueError("Unknown mode"))
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        response = test_client.get("/book/1234/similar")

        assert response.status_code == 422
        assert "Unknown mode" in response.json()["detail"]


class TestSimilarityEndpointIntegration:
    """Test parameter pass-through and cross-mode consistency."""

    def test_alpha_is_forwarded_for_hybrid_mode(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """Alpha value should be passed through to the service for hybrid mode."""
        _patch_als_available(monkeypatch)
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        test_client.get("/book/1234/similar?mode=hybrid&alpha=0.25")

        call_kwargs = mock_similarity_service.get_similar.call_args[1]
        assert call_kwargs["alpha"] == pytest.approx(0.25)

    def test_alpha_is_forwarded_for_all_modes(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """
        Alpha is forwarded to the service for all modes.
        The service decides whether to use it; the endpoint does not filter it out.
        """
        _patch_als_available(monkeypatch)
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        for mode in ("subject", "als", "hybrid"):
            mock_similarity_service.get_similar.reset_mock()
            response = test_client.get(f"/book/1234/similar?mode={mode}&alpha=0.4")

            assert response.status_code == 200
            call_kwargs = mock_similarity_service.get_similar.call_args[1]
            assert call_kwargs["alpha"] == pytest.approx(0.4), f"alpha not forwarded for {mode}"

    def test_top_k_is_forwarded_as_k_for_all_modes(
        self, test_client, mock_similarity_service, monkeypatch
    ):
        """top_k query param should be passed as k= to get_similar for every mode."""
        _patch_als_available(monkeypatch)
        monkeypatch.setattr("routes.models.SimilarityService", lambda: mock_similarity_service)

        for mode in ("subject", "als", "hybrid"):
            mock_similarity_service.get_similar.reset_mock()
            response = test_client.get(f"/book/1234/similar?mode={mode}&top_k=15")

            assert response.status_code == 200
            call_kwargs = mock_similarity_service.get_similar.call_args[1]
            assert call_kwargs["k"] == 15, f"k not forwarded correctly for {mode}"
