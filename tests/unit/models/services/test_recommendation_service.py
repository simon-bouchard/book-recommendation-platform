# tests/unit/models/services/test_recommendation_service.py
"""
Unit tests for RecommendationService.

Architecture under test:
  - recommend() and _enrich_candidates() are async.
  - Strategy selection awaits user.is_warm() (async method backed by ALS client)
    and reads user.has_preferences (pure computation on fav_subjects).
  - Enrichment uses get_metadata_client().enrich() — no local file loaders.
  - RecommendationPipeline is constructed per-request with singleton
    generators/filter/ranker passed as keyword arguments.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.services.recommendation_service import RecommendationService
from models.domain.user import User
from models.domain.config import RecommendationConfig
from models.domain.recommendation import Candidate, RecommendedBook
from models.core.constants import PAD_IDX
from model_servers._shared.contracts import BookMeta, EnrichResponse

_SVC = "models.services.recommendation_service"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enrich_response(*item_ids: int) -> EnrichResponse:
    """Build an EnrichResponse with minimal but complete metadata."""
    return EnrichResponse(
        books=[
            BookMeta(
                item_idx=idx,
                title=f"Book {idx}",
                author=f"Author {idx}",
                year=2020,
                isbn=f"ISBN-{idx}",
                cover_id=f"cover-{idx}",
                avg_rating=4.0,
                num_ratings=100,
            )
            for idx in item_ids
        ]
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def warm_user():
    """User whose is_warm() method will be patched to return True in tests."""
    return User(user_id=123, fav_subjects=[5, 12, 23])


@pytest.fixture
def cold_user_with_prefs():
    """User whose is_warm() method will be patched to return False; has real subjects."""
    return User(user_id=456, fav_subjects=[5, 12, 23])


@pytest.fixture
def cold_user_no_prefs():
    """User whose is_warm() method will be patched to return False; has only PAD_IDX."""
    return User(user_id=789, fav_subjects=[PAD_IDX])


@pytest.fixture
def mock_db():
    return Mock()


@pytest.fixture
def patched_meta_client():
    """Patch get_metadata_client; default enrich returns an empty response."""
    with patch(f"{_SVC}.get_metadata_client") as mock_get:
        client = AsyncMock()
        client.enrich.return_value = EnrichResponse(books=[])
        mock_get.return_value = client
        yield client


@pytest.fixture
def patched_pipeline():
    """
    Patch RecommendationPipeline at the service module level.

    Yields (MockClass, mock_instance) so tests can:
      - Inspect mock_cls.call_args.kwargs to verify constructor arguments.
      - Set mock_instance.recommend.return_value to control pipeline output.
    """
    with patch(f"{_SVC}.RecommendationPipeline") as mock_cls:
        instance = Mock()
        instance.recommend = AsyncMock(return_value=[])
        mock_cls.return_value = instance
        yield mock_cls, instance


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Service is a lightweight, stateless object."""

    def test_constructs_without_errors(self):
        assert RecommendationService() is not None

    def test_holds_no_preloaded_data(self):
        """No book metadata or model artifacts are loaded at construction time."""
        service = RecommendationService()
        assert not hasattr(service, "_book_meta")
        assert not hasattr(service, "_model")


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


class TestStrategySelection:
    """_build_pipeline selects the correct primary generator for each scenario."""

    @pytest.mark.asyncio
    async def test_auto_warm_user_uses_als_generator(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        mock_cls, _ = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
            await RecommendationService().recommend(
                warm_user, RecommendationConfig.default(), mock_db
            )

        assert mock_cls.call_args.kwargs["generator"].__class__.__name__ == "ALSBasedGenerator"

    @pytest.mark.asyncio
    async def test_auto_cold_with_prefs_uses_joint_subject_generator(
        self, cold_user_with_prefs, mock_db, patched_meta_client, patched_pipeline
    ):
        mock_cls, _ = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=False)):
            await RecommendationService().recommend(
                cold_user_with_prefs, RecommendationConfig.default(), mock_db
            )

        assert mock_cls.call_args.kwargs["generator"].__class__.__name__ == "JointSubjectGenerator"

    @pytest.mark.asyncio
    async def test_auto_cold_no_prefs_uses_popularity_generator(
        self, cold_user_no_prefs, mock_db, patched_meta_client, patched_pipeline
    ):
        mock_cls, _ = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=False)):
            await RecommendationService().recommend(
                cold_user_no_prefs, RecommendationConfig.default(), mock_db
            )

        assert (
            mock_cls.call_args.kwargs["generator"].__class__.__name__ == "PopularityBasedGenerator"
        )

    @pytest.mark.asyncio
    async def test_behavioral_mode_forces_als_regardless_of_warmth(
        self, cold_user_with_prefs, mock_db, patched_meta_client, patched_pipeline
    ):
        """mode='behavioral' bypasses the warm/cold gate and always uses ALS."""
        mock_cls, _ = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=False)):
            await RecommendationService().recommend(
                cold_user_with_prefs, RecommendationConfig(k=20, mode="behavioral"), mock_db
            )

        assert mock_cls.call_args.kwargs["generator"].__class__.__name__ == "ALSBasedGenerator"

    @pytest.mark.asyncio
    async def test_subject_mode_forces_joint_subject_regardless_of_warmth(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        """mode='subject' bypasses the warm/cold gate and always uses JointSubjectGenerator."""
        mock_cls, _ = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
            await RecommendationService().recommend(
                warm_user, RecommendationConfig(k=20, mode="subject"), mock_db
            )

        assert mock_cls.call_args.kwargs["generator"].__class__.__name__ == "JointSubjectGenerator"

    @pytest.mark.asyncio
    async def test_warm_user_gets_popularity_fallback(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        mock_cls, _ = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
            await RecommendationService().recommend(
                warm_user, RecommendationConfig.default(), mock_db
            )

        fallback = mock_cls.call_args.kwargs["fallback_generator"]
        assert fallback.__class__.__name__ == "PopularityBasedGenerator"

    @pytest.mark.asyncio
    async def test_cold_no_prefs_has_no_fallback(
        self, cold_user_no_prefs, mock_db, patched_meta_client, patched_pipeline
    ):
        """Popularity is the primary generator here; there is nothing to fall back to."""
        mock_cls, _ = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=False)):
            await RecommendationService().recommend(
                cold_user_no_prefs, RecommendationConfig.default(), mock_db
            )

        assert mock_cls.call_args.kwargs["fallback_generator"] is None

    @pytest.mark.asyncio
    async def test_pipeline_always_uses_read_books_filter(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        mock_cls, _ = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
            await RecommendationService().recommend(
                warm_user, RecommendationConfig.default(), mock_db
            )

        assert mock_cls.call_args.kwargs["filter"].__class__.__name__ == "ReadBooksFilter"

    @pytest.mark.asyncio
    async def test_pipeline_always_uses_noop_ranker(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        mock_cls, _ = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
            await RecommendationService().recommend(
                warm_user, RecommendationConfig.default(), mock_db
            )

        assert mock_cls.call_args.kwargs["ranker"].__class__.__name__ == "NoOpRanker"


# ---------------------------------------------------------------------------
# Pipeline call contract
# ---------------------------------------------------------------------------


class TestPipelineCallContract:
    """recommend() must forward user, k, and db to pipeline.recommend()."""

    @pytest.mark.asyncio
    async def test_passes_user_as_first_arg(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        _, mock_instance = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
            await RecommendationService().recommend(
                warm_user, RecommendationConfig.default(), mock_db
            )

        # pipeline.recommend(user, k, db)
        assert mock_instance.recommend.call_args.args[0] is warm_user

    @pytest.mark.asyncio
    async def test_passes_config_k_as_second_arg(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        _, mock_instance = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
            await RecommendationService().recommend(
                warm_user, RecommendationConfig(k=42, mode="auto"), mock_db
            )

        assert mock_instance.recommend.call_args.args[1] == 42

    @pytest.mark.asyncio
    async def test_passes_db_as_third_arg(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        _, mock_instance = patched_pipeline
        with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
            await RecommendationService().recommend(
                warm_user, RecommendationConfig.default(), mock_db
            )

        assert mock_instance.recommend.call_args.args[2] is mock_db


# ---------------------------------------------------------------------------
# Candidate enrichment
# ---------------------------------------------------------------------------


class TestCandidateEnrichment:
    """_enrich_candidates converts Candidates to RecommendedBooks via the metadata client."""

    @pytest.mark.asyncio
    async def test_returns_recommended_book_instances(self, patched_meta_client):
        patched_meta_client.enrich.return_value = _enrich_response(1000)
        results = await RecommendationService()._enrich_candidates(
            [Candidate(item_idx=1000, score=0.9, source="test")]
        )

        assert len(results) == 1
        assert isinstance(results[0], RecommendedBook)

    @pytest.mark.asyncio
    async def test_maps_all_metadata_fields(self, patched_meta_client):
        patched_meta_client.enrich.return_value = EnrichResponse(
            books=[
                BookMeta(
                    item_idx=1000,
                    title="My Book",
                    author="Jane Doe",
                    year=2021,
                    isbn="978-0",
                    cover_id="cov42",
                    avg_rating=3.7,
                    num_ratings=55,
                )
            ]
        )
        results = await RecommendationService()._enrich_candidates(
            [Candidate(item_idx=1000, score=0.9, source="test")]
        )

        rec = results[0]
        assert rec.item_idx == 1000
        assert rec.score == 0.9
        assert rec.title == "My Book"
        assert rec.author == "Jane Doe"
        assert rec.year == 2021
        assert rec.isbn == "978-0"
        assert rec.cover_id == "cov42"
        assert rec.avg_rating == 3.7
        assert rec.num_ratings == 55

    @pytest.mark.asyncio
    async def test_skips_candidates_absent_from_enrich_response(self, patched_meta_client):
        patched_meta_client.enrich.return_value = _enrich_response(1000)
        candidates = [
            Candidate(item_idx=1000, score=0.9, source="test"),
            Candidate(item_idx=9999, score=0.8, source="test"),  # absent from response
        ]

        results = await RecommendationService()._enrich_candidates(candidates)

        assert len(results) == 1
        assert results[0].item_idx == 1000

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self, patched_meta_client):
        """Empty candidate list produces empty output; enrich may still be called."""
        patched_meta_client.enrich.return_value = EnrichResponse(books=[])

        results = await RecommendationService()._enrich_candidates([])

        assert results == []

    @pytest.mark.asyncio
    async def test_passes_all_item_indices_to_enrich(self, patched_meta_client):
        patched_meta_client.enrich.return_value = EnrichResponse(books=[])
        await RecommendationService()._enrich_candidates(
            [
                Candidate(item_idx=1000, score=0.9, source="test"),
                Candidate(item_idx=1001, score=0.8, source="test"),
                Candidate(item_idx=1002, score=0.7, source="test"),
            ]
        )

        patched_meta_client.enrich.assert_called_once_with([1000, 1001, 1002])


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    """Full path: user + config → pipeline → enrich → RecommendedBook list."""

    @pytest.mark.asyncio
    async def test_complete_flow(self, warm_user, mock_db, patched_pipeline):
        _, mock_instance = patched_pipeline
        mock_instance.recommend.return_value = [
            Candidate(item_idx=1000, score=0.9, source="als"),
            Candidate(item_idx=1001, score=0.8, source="als"),
        ]

        with patch(f"{_SVC}.get_metadata_client") as mock_get_meta:
            meta_client = AsyncMock()
            meta_client.enrich.return_value = _enrich_response(1000, 1001)
            mock_get_meta.return_value = meta_client

            with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
                results = await RecommendationService().recommend(
                    warm_user, RecommendationConfig(k=10, mode="auto"), mock_db
                )

        assert len(results) == 2
        assert all(isinstance(r, RecommendedBook) for r in results)
        assert results[0].item_idx == 1000
        assert results[0].score == 0.9
        assert results[0].title == "Book 1000"

    @pytest.mark.asyncio
    async def test_empty_pipeline_output_returns_empty_list(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        _, mock_instance = patched_pipeline
        mock_instance.recommend.return_value = []

        with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
            results = await RecommendationService().recommend(
                warm_user, RecommendationConfig.default(), mock_db
            )

        assert results == []


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    """recommend() emits structured log events for observability."""

    @pytest.mark.asyncio
    async def test_logs_recommendation_started(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        with patch(f"{_SVC}.logger") as mock_logger:
            with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
                await RecommendationService().recommend(
                    warm_user, RecommendationConfig.default(), mock_db
                )

        mock_logger.info.assert_any_call(
            "Recommendation started",
            extra={
                "user_id": 123,
                "mode": "auto",
                "is_warm": True,
                "has_preferences": True,  # warm_user has fav_subjects [5, 12, 23]
                "k": 200,
            },
        )

    @pytest.mark.asyncio
    async def test_logs_recommendation_completed_with_metrics(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        with patch(f"{_SVC}.logger") as mock_logger:
            with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
                await RecommendationService().recommend(
                    warm_user, RecommendationConfig.default(), mock_db
                )

        completed = [
            c for c in mock_logger.info.call_args_list if "Recommendation completed" in str(c)
        ]
        assert len(completed) == 1
        extra = completed[0].kwargs["extra"]
        assert extra["user_id"] == 123
        assert "count" in extra
        assert "latency_ms" in extra

    @pytest.mark.asyncio
    async def test_logs_error_and_reraises(
        self, warm_user, mock_db, patched_meta_client, patched_pipeline
    ):
        _, mock_instance = patched_pipeline
        mock_instance.recommend.side_effect = RuntimeError("model server unavailable")

        with patch(f"{_SVC}.logger") as mock_logger:
            with patch.object(User, "is_warm", new=AsyncMock(return_value=True)):
                with pytest.raises(RuntimeError):
                    await RecommendationService().recommend(
                        warm_user, RecommendationConfig.default(), mock_db
                    )

        mock_logger.error.assert_called_once()
        call = mock_logger.error.call_args
        assert "Recommendation failed" in call.args[0]
        assert call.kwargs["extra"]["user_id"] == 123
        assert "model server unavailable" in call.kwargs["extra"]["error"]
        assert call.kwargs["exc_info"] is True


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """RecommendationConfig enforces its own invariants independently of the service."""

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            RecommendationConfig(k=0, mode="auto")

    def test_k_above_maximum_raises(self):
        with pytest.raises(ValueError):
            RecommendationConfig(k=501, mode="auto")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            RecommendationConfig(k=10, mode="invalid")

    def test_default_factory_produces_k200_auto(self):
        cfg = RecommendationConfig.default()
        assert cfg.k == 200
        assert cfg.mode == "auto"
