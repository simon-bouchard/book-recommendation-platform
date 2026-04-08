# tests/unit/models/domain/test_pipeline.py
"""
Unit tests for recommendation pipeline.
Tests pipeline orchestration with mocked components.

Filter and slicing to k are no longer done by the pipeline — they are handled
by RecommendationService._filter_and_enrich() after the pipeline returns all
ranked candidates (buffer_k = k + 50).
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.domain.pipeline import RecommendationPipeline
from models.domain.rankers import NoOpRanker
from models.domain.recommendation import Candidate
from models.domain.user import User


@pytest.fixture
def mock_user():
    """Create a mock user."""
    return User(
        user_id=123,
        fav_subjects=[5, 12, 23],
        country="US",
        age=25,
    )


@pytest.fixture
def mock_user_no_preferences():
    """Create a mock user without preferences."""
    return User(
        user_id=456,
        fav_subjects=[0],  # PAD_IDX
        country="US",
        age=30,
    )


@pytest.fixture
def sample_candidates():
    """Create sample candidates."""
    return [
        Candidate(item_idx=100, score=0.9, source="primary"),
        Candidate(item_idx=101, score=0.8, source="primary"),
        Candidate(item_idx=102, score=0.7, source="primary"),
        Candidate(item_idx=103, score=0.6, source="primary"),
        Candidate(item_idx=104, score=0.5, source="primary"),
    ]


@pytest.fixture
def fallback_candidates():
    """Create fallback candidates."""
    return [
        Candidate(item_idx=200, score=0.85, source="fallback"),
        Candidate(item_idx=201, score=0.75, source="fallback"),
        Candidate(item_idx=202, score=0.65, source="fallback"),
    ]


@pytest.fixture
def mock_generator():
    """Create mock primary generator."""
    generator = Mock()
    generator.name = "primary_generator"
    generator.generate = AsyncMock()
    return generator


@pytest.fixture
def mock_fallback_generator():
    """Create mock fallback generator."""
    generator = Mock()
    generator.name = "fallback_generator"
    generator.generate = AsyncMock()
    return generator


@pytest.fixture
def mock_ranker():
    """Create mock ranker."""
    ranker = Mock()
    return ranker


class TestPipelineInitialization:
    """Test RecommendationPipeline initialization."""

    def test_stores_components(self, mock_generator, mock_fallback_generator, mock_ranker):
        """Should store all pipeline components."""
        pipeline = RecommendationPipeline(
            generator=mock_generator,
            fallback_generator=mock_fallback_generator,
            ranker=mock_ranker,
        )

        assert pipeline.generator is mock_generator
        assert pipeline.fallback_generator is mock_fallback_generator
        assert pipeline.ranker is mock_ranker

    def test_accepts_none_for_optional_components(self, mock_generator):
        """Should accept None for fallback and ranker (providing defaults)."""
        pipeline = RecommendationPipeline(
            generator=mock_generator,
            fallback_generator=None,
            ranker=None,
        )

        assert pipeline.generator is mock_generator
        assert pipeline.fallback_generator is None
        assert isinstance(pipeline.ranker, NoOpRanker)


class TestBasicFlow:
    """Test basic pipeline flow with all components."""

    @pytest.mark.asyncio
    async def test_calls_generator_with_correct_params(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        sample_candidates,
    ):
        """Should call generator with user and buffer_k (k + 50)."""
        mock_generator.generate.return_value = sample_candidates
        mock_ranker.rank.return_value = sample_candidates

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        await pipeline.recommend(mock_user, k=10)

        # buffer_k = k + 50 = 60
        mock_generator.generate.assert_called_once_with(mock_user, 60)

    @pytest.mark.asyncio
    async def test_applies_ranker_to_candidates(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        sample_candidates,
    ):
        """Should apply ranker to generated candidates."""
        mock_generator.generate.return_value = sample_candidates
        mock_ranker.rank.return_value = sample_candidates

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        await pipeline.recommend(mock_user, k=10)

        mock_ranker.rank.assert_called_once_with(sample_candidates, mock_user)

    @pytest.mark.asyncio
    async def test_returns_all_ranked_candidates_unsliced(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        sample_candidates,
    ):
        """Pipeline returns all ranked candidates without slicing to k."""
        mock_generator.generate.return_value = sample_candidates
        mock_ranker.rank.return_value = sample_candidates

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        result = await pipeline.recommend(mock_user, k=3)

        # All 5 candidates returned — slicing to k is the caller's responsibility
        assert len(result) == len(sample_candidates)

    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        sample_candidates,
    ):
        """Should execute complete pipeline in correct order and return all ranked candidates."""
        mock_generator.generate.return_value = sample_candidates
        ranked = list(reversed(sample_candidates))
        mock_ranker.rank.return_value = ranked

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        result = await pipeline.recommend(mock_user, k=2)

        assert mock_generator.generate.called
        assert mock_ranker.rank.called

        # Returns all ranked candidates — caller (RecommendationService) slices to k
        assert result == ranked


class TestFallbackBehavior:
    """Test fallback generator activation."""

    @pytest.mark.asyncio
    async def test_uses_fallback_when_primary_returns_empty(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        fallback_candidates,
    ):
        """Should use fallback when primary returns empty."""
        mock_generator.generate.return_value = []
        mock_fallback_generator.generate.return_value = fallback_candidates
        mock_ranker.rank.return_value = fallback_candidates

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        result = await pipeline.recommend(mock_user, k=10)

        mock_fallback_generator.generate.assert_called_once()
        assert len(result) > 0
        assert all(c.source == "fallback" for c in result)

    @pytest.mark.asyncio
    async def test_does_not_use_fallback_when_primary_succeeds(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        sample_candidates,
    ):
        """Should not call fallback when primary returns results."""
        mock_generator.generate.return_value = sample_candidates
        mock_ranker.rank.return_value = sample_candidates

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        await pipeline.recommend(mock_user, k=10)

        mock_fallback_generator.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_when_fallback_also_fails(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
    ):
        """Should return empty when both primary and fallback fail."""
        mock_generator.generate.return_value = []
        mock_fallback_generator.generate.return_value = []

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        result = await pipeline.recommend(mock_user, k=10)

        assert result == []


class TestOptionalComponents:
    """Test pipeline with optional components set to None."""

    @pytest.mark.asyncio
    async def test_works_without_fallback_generator(
        self, mock_generator, mock_ranker, mock_user, sample_candidates
    ):
        """Should work when fallback_generator is None."""
        mock_generator.generate.return_value = sample_candidates
        mock_ranker.rank.return_value = sample_candidates

        pipeline = RecommendationPipeline(
            generator=mock_generator,
            fallback_generator=None,
            ranker=mock_ranker,
        )

        result = await pipeline.recommend(mock_user, k=10)

        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_works_without_ranker(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_user,
        sample_candidates,
    ):
        """Should work when ranker is None (preserves generator order)."""
        mock_generator.generate.return_value = sample_candidates

        pipeline = RecommendationPipeline(
            generator=mock_generator,
            fallback_generator=mock_fallback_generator,
            ranker=None,
        )

        result = await pipeline.recommend(mock_user, k=10)

        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_minimal_pipeline_with_only_generator(self, mock_generator, mock_user):
        """Should work with only generator (all other components None)."""
        mock_generator.generate.return_value = [
            Candidate(100, 0.9, "test"),
            Candidate(101, 0.8, "test"),
            Candidate(102, 0.7, "test"),
        ]

        pipeline = RecommendationPipeline(
            generator=mock_generator,
            fallback_generator=None,
            ranker=None,
        )

        result = await pipeline.recommend(mock_user, k=2)

        # Pipeline returns all 3 ranked candidates — caller slices to k=2
        assert len(result) == 3
        assert result[0].item_idx == 100
        assert result[1].item_idx == 101


class TestKHandling:
    """Test handling of k parameter."""

    @pytest.mark.asyncio
    async def test_passes_buffer_k_to_generator(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        sample_candidates,
    ):
        """Generator should be called with buffer_k = k + 50, not k."""
        mock_generator.generate.return_value = sample_candidates
        mock_ranker.rank.return_value = sample_candidates

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        await pipeline.recommend(mock_user, k=3)

        mock_generator.generate.assert_called_once_with(mock_user, 53)  # 3 + 50

    @pytest.mark.asyncio
    async def test_handles_k_larger_than_candidates(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        sample_candidates,
    ):
        """Should handle k larger than available candidates."""
        mock_generator.generate.return_value = sample_candidates
        mock_ranker.rank.return_value = sample_candidates

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        result = await pipeline.recommend(mock_user, k=100)

        assert len(result) == len(sample_candidates)

    @pytest.mark.asyncio
    async def test_handles_k_zero(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        sample_candidates,
    ):
        """Should handle k=0 gracefully."""
        mock_generator.generate.return_value = sample_candidates
        mock_ranker.rank.return_value = sample_candidates

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        result = await pipeline.recommend(mock_user, k=0)

        assert result == []


class TestMultipleFallbackAttempts:
    """Test pipeline behavior with multiple fallback scenarios."""

    @pytest.mark.asyncio
    async def test_fallback_candidates_ranked(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        fallback_candidates,
    ):
        """Fallback candidates should be ranked."""
        mock_generator.generate.return_value = []
        mock_fallback_generator.generate.return_value = fallback_candidates

        ranked = list(reversed(fallback_candidates))
        mock_ranker.rank.return_value = ranked

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        result = await pipeline.recommend(mock_user, k=10)

        assert result == ranked


class TestPipelineWithRealComponents:
    """Test pipeline with real (simple) component implementations."""

    @pytest.mark.asyncio
    async def test_pipeline_with_noop_ranker(self, mock_user):
        """Test pipeline with NoOpRanker preserves generator order."""
        mock_gen = Mock()
        mock_gen.generate = AsyncMock()
        mock_gen.generate.return_value = [
            Candidate(100, 0.9, "test"),
            Candidate(101, 0.8, "test"),
            Candidate(102, 0.7, "test"),
        ]

        pipeline = RecommendationPipeline(
            generator=mock_gen,
            fallback_generator=None,
            ranker=NoOpRanker(),
        )

        result = await pipeline.recommend(mock_user, k=2)

        # All 3 returned — slicing to k=2 is done by the service
        assert len(result) == 3
        assert result[0].item_idx == 100
        assert result[1].item_idx == 101

    @pytest.mark.asyncio
    async def test_pipeline_with_score_ranker(self, mock_user):
        """Test pipeline with ScoreRanker."""
        from models.domain.rankers import ScoreRanker

        mock_gen = Mock()
        mock_gen.generate = AsyncMock()
        mock_gen.generate.return_value = [
            Candidate(100, 0.5, "test"),
            Candidate(101, 0.9, "test"),
            Candidate(102, 0.7, "test"),
        ]

        pipeline = RecommendationPipeline(
            generator=mock_gen,
            fallback_generator=None,
            ranker=ScoreRanker(),
        )

        result = await pipeline.recommend(mock_user, k=3)

        assert result[0].item_idx == 101  # 0.9
        assert result[1].item_idx == 102  # 0.7
        assert result[2].item_idx == 100  # 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_all_components_returning_empty(
        self, mock_generator, mock_fallback_generator, mock_ranker, mock_user
    ):
        """Should handle case where everything returns empty."""
        mock_generator.generate.return_value = []
        mock_fallback_generator.generate.return_value = []

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        result = await pipeline.recommend(mock_user, k=10)

        assert result == []

    @pytest.mark.asyncio
    async def test_preserves_candidate_attributes_through_pipeline(
        self,
        mock_generator,
        mock_fallback_generator,
        mock_ranker,
        mock_user,
        sample_candidates,
    ):
        """Candidate attributes should be preserved through pipeline."""
        mock_generator.generate.return_value = sample_candidates
        mock_ranker.rank.return_value = sample_candidates

        pipeline = RecommendationPipeline(mock_generator, mock_fallback_generator, mock_ranker)

        result = await pipeline.recommend(mock_user, k=10)

        for candidate in result:
            assert hasattr(candidate, "item_idx")
            assert hasattr(candidate, "score")
            assert hasattr(candidate, "source")
