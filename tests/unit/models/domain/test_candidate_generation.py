# tests/unit/models/domain/test_candidate_generation.py
"""
Unit tests for candidate generation strategies.

Generators delegate all computation to HTTP model server clients. Tests patch
the registry getter functions so no real servers or infrastructure is needed.
All generate() methods are async; tests use pytest.mark.asyncio.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.core.constants import PAD_IDX
from models.domain.candidate_generation import (
    ALSBasedGenerator,
    CandidateGenerator,
    JointSubjectGenerator,
    PopularityBasedGenerator,
)
from models.domain.recommendation import Candidate
from models.domain.user import User
from model_servers._shared.contracts import (
    AlsRecsResponse,
    BookMeta,
    EmbedResponse,
    PopularResponse,
    ScoredItem,
    SubjectRecsResponse,
)

_GEN_PATH = "models.domain.candidate_generation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scored_items(n: int, id_offset: int, base_score: float = 0.9) -> list[ScoredItem]:
    """Produce n ScoredItems with strictly descending scores."""
    return [
        ScoredItem(item_idx=id_offset + i, score=round(base_score - i * 0.01, 4)) for i in range(n)
    ]


def _book_metas(n: int, id_offset: int, base_score: float = 0.9) -> list[BookMeta]:
    """Produce n BookMeta objects with strictly descending bayes_scores."""
    return [
        BookMeta(
            item_idx=id_offset + i, title=f"Book {i}", bayes_score=round(base_score - i * 0.01, 4)
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def warm_user():
    return User(user_id=123, fav_subjects=[5, 12, 23])


@pytest.fixture
def cold_user():
    """User with no subject preferences (PAD_IDX only)."""
    return User(user_id=456, fav_subjects=[PAD_IDX])


# ---------------------------------------------------------------------------
# JointSubjectGenerator
# ---------------------------------------------------------------------------


class TestJointSubjectGenerator:
    """Test JointSubjectGenerator: embeds subjects then calls subject_recs."""

    @pytest.fixture
    def mock_embedder_client(self):
        with patch(f"{_GEN_PATH}.get_embedder_client") as mock_get:
            client = AsyncMock()
            client.embed.return_value = EmbedResponse(vector=[0.1] * 16)
            mock_get.return_value = client
            yield client

    @pytest.fixture
    def mock_similarity_client(self):
        with patch(f"{_GEN_PATH}.get_similarity_client") as mock_get:
            client = AsyncMock()
            client.subject_recs.return_value = SubjectRecsResponse(
                results=_scored_items(50, id_offset=1000)
            )
            mock_get.return_value = client
            yield client

    def test_implements_candidate_generator_interface(self):
        gen = JointSubjectGenerator()
        assert isinstance(gen, CandidateGenerator)
        assert hasattr(gen, "generate") and hasattr(gen, "name")

    def test_name_is_joint_subject(self):
        assert JointSubjectGenerator().name == "joint_subject"

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError):
            JointSubjectGenerator(alpha=1.5)
        with pytest.raises(ValueError):
            JointSubjectGenerator(alpha=-0.1)

    @pytest.mark.asyncio
    async def test_returns_list_of_candidates(
        self, warm_user, mock_embedder_client, mock_similarity_client
    ):
        candidates = await JointSubjectGenerator().generate(warm_user, k=10)
        assert isinstance(candidates, list)
        assert all(isinstance(c, Candidate) for c in candidates)

    @pytest.mark.asyncio
    async def test_respects_k(self, warm_user, mock_embedder_client, mock_similarity_client):
        mock_similarity_client.subject_recs.return_value = SubjectRecsResponse(
            results=_scored_items(10, id_offset=1000)
        )
        candidates = await JointSubjectGenerator().generate(warm_user, k=10)
        assert len(candidates) == 10

    @pytest.mark.asyncio
    async def test_source_tagged_correctly(
        self, warm_user, mock_embedder_client, mock_similarity_client
    ):
        candidates = await JointSubjectGenerator().generate(warm_user, k=5)
        assert all(c.source == "joint_subject" for c in candidates)

    @pytest.mark.asyncio
    async def test_calls_embedder_with_user_subjects(
        self, warm_user, mock_embedder_client, mock_similarity_client
    ):
        await JointSubjectGenerator().generate(warm_user, k=5)
        mock_embedder_client.embed.assert_called_once_with(warm_user.fav_subjects)

    @pytest.mark.asyncio
    async def test_passes_embed_vector_to_subject_recs(
        self, warm_user, mock_embedder_client, mock_similarity_client
    ):
        gen = JointSubjectGenerator(alpha=0.7)
        await gen.generate(warm_user, k=5)
        call_kwargs = mock_similarity_client.subject_recs.call_args
        assert call_kwargs.kwargs.get("alpha") == 0.7 or call_kwargs.args[1] == 0.7

    @pytest.mark.asyncio
    async def test_no_preferences_returns_empty_without_calling_clients(
        self, cold_user, mock_embedder_client, mock_similarity_client
    ):
        candidates = await JointSubjectGenerator().generate(cold_user, k=10)
        assert candidates == []
        mock_embedder_client.embed.assert_not_called()
        mock_similarity_client.subject_recs.assert_not_called()

    @pytest.mark.asyncio
    async def test_k_zero_returns_empty(
        self, warm_user, mock_embedder_client, mock_similarity_client
    ):
        candidates = await JointSubjectGenerator().generate(warm_user, k=0)
        assert candidates == []

    @pytest.mark.asyncio
    async def test_scores_are_non_negative(
        self, warm_user, mock_embedder_client, mock_similarity_client
    ):
        candidates = await JointSubjectGenerator().generate(warm_user, k=10)
        assert all(c.score >= 0 for c in candidates)


# ---------------------------------------------------------------------------
# ALSBasedGenerator
# ---------------------------------------------------------------------------


class TestALSBasedGenerator:
    """Test ALSBasedGenerator: calls als_recs on the ALS model server."""

    @pytest.fixture
    def mock_als_client(self):
        with patch(f"{_GEN_PATH}.get_als_client") as mock_get:
            client = AsyncMock()
            client.als_recs.return_value = AlsRecsResponse(
                results=_scored_items(50, id_offset=2000)
            )
            mock_get.return_value = client
            yield client

    def test_implements_candidate_generator_interface(self):
        gen = ALSBasedGenerator()
        assert isinstance(gen, CandidateGenerator)
        assert hasattr(gen, "generate") and hasattr(gen, "name")

    def test_name_is_als(self):
        assert ALSBasedGenerator().name == "als"

    @pytest.mark.asyncio
    async def test_returns_list_of_candidates(self, warm_user, mock_als_client):
        candidates = await ALSBasedGenerator().generate(warm_user, k=10)
        assert isinstance(candidates, list)
        assert all(isinstance(c, Candidate) for c in candidates)

    @pytest.mark.asyncio
    async def test_respects_k(self, warm_user, mock_als_client):
        mock_als_client.als_recs.return_value = AlsRecsResponse(
            results=_scored_items(15, id_offset=2000)
        )
        candidates = await ALSBasedGenerator().generate(warm_user, k=15)
        assert len(candidates) == 15

    @pytest.mark.asyncio
    async def test_source_tagged_correctly(self, warm_user, mock_als_client):
        candidates = await ALSBasedGenerator().generate(warm_user, k=5)
        assert all(c.source == "als" for c in candidates)

    @pytest.mark.asyncio
    async def test_calls_als_recs_with_user_id_and_k(self, warm_user, mock_als_client):
        await ALSBasedGenerator().generate(warm_user, k=20)
        mock_als_client.als_recs.assert_called_once_with(warm_user.user_id, k=20)

    @pytest.mark.asyncio
    async def test_cold_user_returns_empty_when_server_returns_empty(
        self, cold_user, mock_als_client
    ):
        """Cold users have no ALS factors; the server returns empty results."""
        mock_als_client.als_recs.return_value = AlsRecsResponse(results=[])
        candidates = await ALSBasedGenerator().generate(cold_user, k=10)
        assert candidates == []

    @pytest.mark.asyncio
    async def test_k_zero_returns_empty_without_calling_client(self, warm_user, mock_als_client):
        candidates = await ALSBasedGenerator().generate(warm_user, k=0)
        assert candidates == []
        mock_als_client.als_recs.assert_not_called()

    @pytest.mark.asyncio
    async def test_scores_are_non_negative(self, warm_user, mock_als_client):
        candidates = await ALSBasedGenerator().generate(warm_user, k=10)
        assert all(c.score >= 0 for c in candidates)

    @pytest.mark.asyncio
    async def test_is_deterministic(self, warm_user, mock_als_client):
        gen = ALSBasedGenerator()
        r1 = await gen.generate(warm_user, k=10)
        r2 = await gen.generate(warm_user, k=10)
        assert [c.item_idx for c in r1] == [c.item_idx for c in r2]


# ---------------------------------------------------------------------------
# PopularityBasedGenerator
# ---------------------------------------------------------------------------


class TestPopularityBasedGenerator:
    """Test PopularityBasedGenerator: calls popular() on the metadata server."""

    @pytest.fixture
    def mock_metadata_client(self):
        with patch(f"{_GEN_PATH}.get_metadata_client") as mock_get:
            client = AsyncMock()
            client.popular.return_value = PopularResponse(books=_book_metas(50, id_offset=3000))
            mock_get.return_value = client
            yield client

    def test_implements_candidate_generator_interface(self):
        gen = PopularityBasedGenerator()
        assert isinstance(gen, CandidateGenerator)
        assert hasattr(gen, "generate") and hasattr(gen, "name")

    def test_name_is_popularity(self):
        assert PopularityBasedGenerator().name == "popularity"

    @pytest.mark.asyncio
    async def test_returns_list_of_candidates(self, warm_user, mock_metadata_client):
        candidates = await PopularityBasedGenerator().generate(warm_user, k=10)
        assert isinstance(candidates, list)
        assert all(isinstance(c, Candidate) for c in candidates)

    @pytest.mark.asyncio
    async def test_respects_k(self, warm_user, mock_metadata_client):
        mock_metadata_client.popular.return_value = PopularResponse(
            books=_book_metas(10, id_offset=3000)
        )
        candidates = await PopularityBasedGenerator().generate(warm_user, k=10)
        assert len(candidates) == 10

    @pytest.mark.asyncio
    async def test_source_tagged_correctly(self, warm_user, mock_metadata_client):
        candidates = await PopularityBasedGenerator().generate(warm_user, k=5)
        assert all(c.source == "popularity" for c in candidates)

    @pytest.mark.asyncio
    async def test_calls_popular_with_k(self, warm_user, mock_metadata_client):
        await PopularityBasedGenerator().generate(warm_user, k=25)
        mock_metadata_client.popular.assert_called_once_with(k=25)

    @pytest.mark.asyncio
    async def test_succeeds_for_user_without_preferences(self, cold_user, mock_metadata_client):
        """Popularity is user-agnostic and should work for any user."""
        candidates = await PopularityBasedGenerator().generate(cold_user, k=10)
        assert len(candidates) > 0

    @pytest.mark.asyncio
    async def test_k_zero_returns_empty_without_calling_client(
        self, warm_user, mock_metadata_client
    ):
        candidates = await PopularityBasedGenerator().generate(warm_user, k=0)
        assert candidates == []
        mock_metadata_client.popular.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_bayes_score_defaults_to_zero(self, warm_user, mock_metadata_client):
        """Books missing bayes_score should produce a 0.0 candidate score."""
        mock_metadata_client.popular.return_value = PopularResponse(
            books=[BookMeta(item_idx=9999, title="No Score Book", bayes_score=None)]
        )
        candidates = await PopularityBasedGenerator().generate(warm_user, k=5)
        assert len(candidates) == 1
        assert candidates[0].score == 0.0
