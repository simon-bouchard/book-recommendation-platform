# tests/unit/models/domain/test_candidate_generation.py
"""
Unit tests for candidate generation components.
Tests all generator types with mocked infrastructure dependencies.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.domain.candidate_generation import (
    CandidateGenerator,
    SubjectBasedGenerator,
    ALSBasedGenerator,
    BayesianPopularityGenerator,
    HybridGenerator,
)
from models.domain.recommendation import Candidate
from models.domain.user import User
from models.core.constants import PAD_IDX


@pytest.fixture
def mock_user():
    """Create a mock user with preferences."""
    return User(
        user_id=123,
        fav_subjects=[5, 12, 23],
        country="US",
        age=25,
    )


@pytest.fixture
def mock_user_no_preferences():
    """Create a mock user without preferences (PAD_IDX only)."""
    return User(
        user_id=456,
        fav_subjects=[PAD_IDX],
        country="US",
        age=30,
    )


@pytest.fixture
def mock_embedder():
    """Create mock SubjectEmbedder that returns proper numpy arrays."""
    embedder = Mock()

    def embed_side_effect(subjects):
        # Return deterministic 1D embedding based on sum of subjects
        val = sum(subjects) if subjects else 0
        return np.full(16, float(val) / 100.0, dtype=np.float32)

    embedder.embed.side_effect = embed_side_effect
    return embedder


@pytest.fixture
def mock_similarity_index():
    """Create mock SimilarityIndex with embeddings_full and ids_full attributes."""
    index = Mock()

    # Mock the embeddings_full attribute (100 books, 16 dimensions)
    index.embeddings_full = np.random.randn(100, 16).astype(np.float32)

    # Mock the ids_full attribute
    index.ids_full = np.arange(1000, 1100, dtype=np.int64)

    return index


@pytest.fixture
def mock_als_model():
    """Create mock ALSModel with properly configured attributes."""
    model = Mock()

    # Basic checks
    model.has_user.return_value = True

    # Mock factors - 100 books, 16 dimensions
    model.book_factors = np.random.randn(100, 16).astype(np.float32)

    # Mock book_row_to_id mapping
    model.book_row_to_id = {i: 2000 + i for i in range(100)}

    def get_user_factors_side_effect(user_id):
        return np.random.randn(16).astype(np.float32)

    model.get_user_factors.side_effect = get_user_factors_side_effect

    return model


class TestSubjectBasedGenerator:
    """Test SubjectBasedGenerator with mocked infrastructure."""

    def test_implements_candidate_generator_interface(self, mock_embedder, mock_similarity_index):
        """Should implement CandidateGenerator abstract class."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        assert isinstance(generator, CandidateGenerator)
        assert hasattr(generator, "generate")
        assert hasattr(generator, "name")

    def test_name_property_returns_string(self, mock_embedder, mock_similarity_index):
        """Name property should return descriptive string."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        assert isinstance(generator.name, str)
        assert generator.name == "subject"

    def test_generate_returns_list_of_candidates(
        self, mock_embedder, mock_similarity_index, mock_user
    ):
        """Should return list of Candidate objects."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        candidates = generator.generate(mock_user, k=10)

        assert isinstance(candidates, list)
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_generate_returns_empty_for_user_without_preferences(
        self, mock_embedder, mock_similarity_index, mock_user_no_preferences
    ):
        """Should return empty list for users with no preferences."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        candidates = generator.generate(mock_user_no_preferences, k=10)

        assert candidates == []
        mock_embedder.embed.assert_not_called()

    def test_generate_calls_embedder_with_user_subjects(
        self, mock_embedder, mock_similarity_index, mock_user
    ):
        """Should call embedder with user's favorite subjects."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        generator.generate(mock_user, k=10)

        mock_embedder.embed.assert_called_once_with(mock_user.fav_subjects)

    def test_generate_calls_similarity_search_with_embedding(
        self, mock_embedder, mock_similarity_index, mock_user
    ):
        """Should compute similarities using user embedding."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        candidates = generator.generate(mock_user, k=10)

        # Should have called embedder
        assert mock_embedder.embed.called
        # Should use index.embeddings_full for computation
        assert len(candidates) > 0

    def test_generate_respects_k_parameter(self, mock_embedder, mock_similarity_index, mock_user):
        """Should return exactly k candidates."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        candidates = generator.generate(mock_user, k=20)

        assert len(candidates) == 20

    def test_candidates_have_correct_source(self, mock_embedder, mock_similarity_index, mock_user):
        """Candidates should be tagged with correct source."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        candidates = generator.generate(mock_user, k=10)

        assert all(c.source == "subject" for c in candidates)

    def test_candidates_have_scores(self, mock_embedder, mock_similarity_index, mock_user):
        """Candidates should include similarity scores."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        candidates = generator.generate(mock_user, k=10)

        assert all(isinstance(c.score, float) for c in candidates)
        assert all(c.score >= 0 for c in candidates)

    def test_candidates_sorted_by_score_descending(
        self, mock_embedder, mock_similarity_index, mock_user
    ):
        """Candidates should be sorted by score (highest first)."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        candidates = generator.generate(mock_user, k=10)

        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)


class TestALSBasedGenerator:
    """Test ALSBasedGenerator with mocked ALS model."""

    def test_implements_candidate_generator_interface(self, mock_als_model):
        """Should implement CandidateGenerator abstract class."""
        generator = ALSBasedGenerator(mock_als_model)

        assert isinstance(generator, CandidateGenerator)
        assert hasattr(generator, "generate")
        assert hasattr(generator, "name")

    def test_name_property_returns_string(self, mock_als_model):
        """Name property should return descriptive string."""
        generator = ALSBasedGenerator(mock_als_model)

        assert isinstance(generator.name, str)
        assert generator.name == "als"

    def test_generate_returns_list_of_candidates(self, mock_als_model, mock_user):
        """Should return list of Candidate objects."""
        generator = ALSBasedGenerator(mock_als_model)

        candidates = generator.generate(mock_user, k=10)

        assert isinstance(candidates, list)
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_generate_returns_empty_for_cold_user(self, mock_als_model, mock_user):
        """Should return empty list for users not in ALS model."""
        mock_als_model.has_user.return_value = False

        generator = ALSBasedGenerator(mock_als_model)
        candidates = generator.generate(mock_user, k=10)

        assert candidates == []

    def test_generate_checks_if_user_exists(self, mock_als_model, mock_user):
        """Should check if user exists in ALS model."""
        generator = ALSBasedGenerator(mock_als_model)

        generator.generate(mock_user, k=10)

        mock_als_model.has_user.assert_called_once_with(mock_user.user_id)

    def test_generate_computes_scores_from_factors(self, mock_als_model, mock_user):
        """Should compute scores using matrix multiplication."""
        generator = ALSBasedGenerator(mock_als_model)

        candidates = generator.generate(mock_user, k=10)

        # Should call get_user_factors
        mock_als_model.get_user_factors.assert_called_once_with(mock_user.user_id)
        # Should generate candidates
        assert len(candidates) == 10

    def test_candidates_have_correct_source(self, mock_als_model, mock_user):
        """Candidates should be tagged with correct source."""
        generator = ALSBasedGenerator(mock_als_model)

        candidates = generator.generate(mock_user, k=10)

        assert all(c.source == "als" for c in candidates)

    def test_candidates_sorted_by_score_descending(self, mock_als_model, mock_user):
        """Candidates should be sorted by score (highest first)."""
        generator = ALSBasedGenerator(mock_als_model)

        candidates = generator.generate(mock_user, k=10)

        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_generate_respects_k_parameter(self, mock_als_model, mock_user):
        """Should return exactly k candidates."""
        generator = ALSBasedGenerator(mock_als_model)

        candidates = generator.generate(mock_user, k=15)

        assert len(candidates) == 15


class TestBayesianPopularityGenerator:
    """Test BayesianPopularityGenerator with mocked data."""

    def test_implements_candidate_generator_interface(self, monkeypatch):
        """Should implement CandidateGenerator abstract class."""
        # Mock the loaders
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        generator = BayesianPopularityGenerator()

        assert isinstance(generator, CandidateGenerator)
        assert hasattr(generator, "generate")
        assert hasattr(generator, "name")

    def test_name_property_returns_string(self, monkeypatch):
        """Name property should return descriptive string."""
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        generator = BayesianPopularityGenerator()

        assert isinstance(generator.name, str)
        assert generator.name == "popularity"

    def test_generate_returns_list_of_candidates(self, mock_user, monkeypatch):
        """Should return list of Candidate objects."""
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        generator = BayesianPopularityGenerator()
        candidates = generator.generate(mock_user, k=10)

        assert isinstance(candidates, list)
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_generate_always_succeeds(self, mock_user_no_preferences, monkeypatch):
        """Should succeed even for users without preferences."""
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        generator = BayesianPopularityGenerator()
        candidates = generator.generate(mock_user_no_preferences, k=10)

        assert len(candidates) > 0

    def test_candidates_sorted_by_bayesian_score(self, mock_user, monkeypatch):
        """Candidates should be sorted by Bayesian score (highest first)."""
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        generator = BayesianPopularityGenerator()
        candidates = generator.generate(mock_user, k=10)

        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_candidates_have_correct_source(self, mock_user, monkeypatch):
        """Candidates should be tagged with correct source."""
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        generator = BayesianPopularityGenerator()
        candidates = generator.generate(mock_user, k=10)

        assert all(c.source == "popularity" for c in candidates)

    def test_generate_respects_k_parameter(self, mock_user, monkeypatch):
        """Should return at most k candidates."""
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        generator = BayesianPopularityGenerator()
        candidates = generator.generate(mock_user, k=5)

        assert len(candidates) <= 5

    def test_generate_handles_k_larger_than_catalog(self, mock_user, monkeypatch):
        """Should handle k larger than number of books."""
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        generator = BayesianPopularityGenerator()
        k_large = 150  # More than the 100 books we have
        candidates = generator.generate(mock_user, k=k_large)

        assert len(candidates) == 100


class TestHybridGenerator:
    """Test HybridGenerator with multiple source generators."""

    @pytest.fixture
    def mock_generators(self, mock_embedder, mock_similarity_index, monkeypatch):
        """Create two mock generators for hybrid blending."""
        # Mock the loaders for BayesianPopularityGenerator
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        gen1 = SubjectBasedGenerator(mock_embedder, mock_similarity_index)
        gen2 = BayesianPopularityGenerator()
        return gen1, gen2

    def test_implements_candidate_generator_interface(self, mock_generators):
        """Should implement CandidateGenerator abstract class."""
        gen1, gen2 = mock_generators
        generator = HybridGenerator([(gen1, 0.6), (gen2, 0.4)])

        assert isinstance(generator, CandidateGenerator)

    def test_name_property_returns_string(self, mock_generators):
        """Name property should return descriptive string."""
        gen1, gen2 = mock_generators
        generator = HybridGenerator([(gen1, 0.6), (gen2, 0.4)])

        assert isinstance(generator.name, str)
        assert "hybrid" in generator.name.lower()

    def test_generate_returns_list_of_candidates(self, mock_generators, mock_user):
        """Should return list of Candidate objects."""
        gen1, gen2 = mock_generators
        generator = HybridGenerator([(gen1, 0.6), (gen2, 0.4)])

        candidates = generator.generate(mock_user, k=20)

        assert isinstance(candidates, list)
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_generate_calls_all_generators(self, mock_user):
        """Should call all component generators."""
        mock_gen1 = Mock(spec=CandidateGenerator)
        mock_gen1.generate.return_value = [
            Candidate(1, 0.8, "gen1"),
            Candidate(2, 0.7, "gen1"),
        ]
        mock_gen1.name = "gen1"

        mock_gen2 = Mock(spec=CandidateGenerator)
        mock_gen2.generate.return_value = [
            Candidate(3, 0.9, "gen2"),
            Candidate(4, 0.6, "gen2"),
        ]
        mock_gen2.name = "gen2"

        generator = HybridGenerator([(mock_gen1, 0.5), (mock_gen2, 0.5)])
        generator.generate(mock_user, k=20)

        mock_gen1.generate.assert_called_once()
        mock_gen2.generate.assert_called_once()

    def test_generate_deduplicates_candidates(self, mock_user):
        """Should deduplicate candidates from different sources."""
        # Both generators return same book
        mock_gen1 = Mock(spec=CandidateGenerator)
        mock_gen1.generate.return_value = [Candidate(100, 0.8, "gen1")]
        mock_gen1.name = "gen1"

        mock_gen2 = Mock(spec=CandidateGenerator)
        mock_gen2.generate.return_value = [Candidate(100, 0.9, "gen2")]
        mock_gen2.name = "gen2"

        generator = HybridGenerator([(mock_gen1, 0.5), (mock_gen2, 0.5)])
        candidates = generator.generate(mock_user, k=20)

        # Should only have one instance of book 100
        assert len(candidates) == 1
        assert candidates[0].item_idx == 100

    def test_deduplication_keeps_highest_blended_score(self, mock_user):
        """When deduplicating, should keep candidate with highest blended score."""
        mock_gen1 = Mock(spec=CandidateGenerator)
        mock_gen1.generate.return_value = [Candidate(100, 0.8, "gen1")]
        mock_gen1.name = "gen1"

        mock_gen2 = Mock(spec=CandidateGenerator)
        mock_gen2.generate.return_value = [Candidate(100, 0.6, "gen2")]
        mock_gen2.name = "gen2"

        # Weight gen1 more heavily
        generator = HybridGenerator([(mock_gen1, 0.8), (mock_gen2, 0.2)])
        candidates = generator.generate(mock_user, k=20)

        assert len(candidates) == 1
        # Source should indicate hybrid blending (set ordering may vary)
        assert "hybrid" in candidates[0].source
        assert "gen1" in candidates[0].source
        assert "gen2" in candidates[0].source

    def test_candidates_sorted_by_blended_score(self, mock_generators, mock_user):
        """Final candidates should be sorted by blended score."""
        gen1, gen2 = mock_generators
        generator = HybridGenerator([(gen1, 0.6), (gen2, 0.4)])

        candidates = generator.generate(mock_user, k=20)

        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_generate_respects_k_parameter(self, mock_generators, mock_user):
        """Should return at most k candidates."""
        gen1, gen2 = mock_generators
        generator = HybridGenerator([(gen1, 0.6), (gen2, 0.4)])

        candidates = generator.generate(mock_user, k=15)

        assert len(candidates) <= 15

    def test_weights_must_be_positive(self, monkeypatch):
        """Generator weights should be positive."""
        # Mock loaders for BayesianPopularityGenerator
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        gen1 = Mock(spec=CandidateGenerator)
        gen2 = BayesianPopularityGenerator()

        with pytest.raises(ValueError, match="must be positive"):
            HybridGenerator([(gen1, 0.0), (gen2, 0.5)])

        with pytest.raises(ValueError, match="must be positive"):
            HybridGenerator([(gen1, -0.5), (gen2, 0.5)])

    def test_must_have_at_least_one_generator(self):
        """Should require at least one generator."""
        with pytest.raises(ValueError, match="at least one"):
            HybridGenerator([])


class TestEdgeCases:
    """Test edge cases across all generators."""

    def test_subject_generator_handles_k_zero(
        self, mock_embedder, mock_similarity_index, mock_user
    ):
        """Should handle k=0 gracefully."""
        generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

        candidates = generator.generate(mock_user, k=0)

        assert candidates == []

    def test_als_generator_handles_k_zero(self, mock_als_model, mock_user):
        """Should handle k=0 gracefully."""
        generator = ALSBasedGenerator(mock_als_model)

        candidates = generator.generate(mock_user, k=0)

        assert candidates == []

    def test_popularity_generator_handles_k_zero(self, mock_user, monkeypatch):
        """Should handle k=0 gracefully."""
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_bayesian_scores",
            lambda **kwargs: np.linspace(0.9, 0.1, 100),
        )
        monkeypatch.setattr(
            "models.domain.candidate_generation.load_book_subject_embeddings",
            lambda **kwargs: (None, list(range(3000, 3100))),
        )

        generator = BayesianPopularityGenerator()
        candidates = generator.generate(mock_user, k=0)

        assert candidates == []

    def test_hybrid_generator_handles_all_empty_sources(self, mock_user_no_preferences):
        """Should handle case where all generators return empty."""
        mock_gen1 = Mock(spec=CandidateGenerator)
        mock_gen1.generate.return_value = []
        mock_gen1.name = "empty1"

        mock_gen2 = Mock(spec=CandidateGenerator)
        mock_gen2.generate.return_value = []
        mock_gen2.name = "empty2"

        generator = HybridGenerator([(mock_gen1, 0.5), (mock_gen2, 0.5)])
        candidates = generator.generate(mock_user_no_preferences, k=20)

        assert candidates == []

    def test_hybrid_generator_handles_single_source(
        self, mock_embedder, mock_similarity_index, mock_user
    ):
        """Should work with single generator (degenerates to that generator)."""
        gen = SubjectBasedGenerator(mock_embedder, mock_similarity_index)
        generator = HybridGenerator([(gen, 1.0)])

        candidates = generator.generate(mock_user, k=10)

        assert len(candidates) > 0
        assert all(c.source.startswith("hybrid") for c in candidates)
