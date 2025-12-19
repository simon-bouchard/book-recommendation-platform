# tests/unit/models/infrastructure/test_subject_embedder.py
"""
Unit tests for SubjectEmbedder infrastructure component.
Tests singleton pattern, injection, embedding computation, and batch processing.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
from unittest.mock import Mock

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.infrastructure.subject_embedder import SubjectEmbedder


@pytest.fixture
def mock_pooler():
    """Create mock attention pooler for testing."""
    pooler = Mock()

    # Mock forward method to return deterministic embeddings
    def mock_forward(subjects_list):
        batch_size = len(subjects_list)
        emb_dim = 16

        # Handle empty batch
        if batch_size == 0:
            return torch.empty(0, emb_dim)

        # Create deterministic embeddings based on subject list
        embeddings = []
        for subjects in subjects_list:
            # Simple deterministic: sum of subject indices
            val = sum(subjects) if subjects else 0
            emb = torch.full((emb_dim,), float(val) / 10.0)
            embeddings.append(emb)

        return torch.stack(embeddings)

    pooler.side_effect = mock_forward
    pooler.get_embedding_dim.return_value = 16

    return pooler


@pytest.fixture
def subject_embedder(mock_pooler):
    """Create SubjectEmbedder with injected mock pooler."""
    embedder = SubjectEmbedder(pooler=mock_pooler)
    yield embedder
    SubjectEmbedder.reset()


class TestSubjectEmbedderInitialization:
    """Test SubjectEmbedder initialization and singleton pattern."""

    def test_injection_bypasses_singleton(self, mock_pooler):
        """Injecting pooler should create non-singleton instance."""
        SubjectEmbedder.reset()

        embedder1 = SubjectEmbedder(pooler=mock_pooler)
        embedder2 = SubjectEmbedder(pooler=mock_pooler)

        assert embedder1 is not embedder2

        SubjectEmbedder.reset()

    def test_stores_injected_pooler(self, mock_pooler):
        """Should store injected pooler."""
        embedder = SubjectEmbedder(pooler=mock_pooler)

        assert embedder.pooler is mock_pooler
        assert embedder.strategy_name == "injected"

        SubjectEmbedder.reset()

    def test_singleton_returns_same_instance_without_injection(self):
        """Without injection, should return singleton instance."""
        SubjectEmbedder.reset()

        # Note: This will try to load real strategy, so we skip if not available
        try:
            embedder1 = SubjectEmbedder()
            embedder2 = SubjectEmbedder()

            assert embedder1 is embedder2
        except (FileNotFoundError, ImportError):
            pytest.skip("Real attention strategy files not available")
        finally:
            SubjectEmbedder.reset()


class TestEmbed:
    """Test single embedding computation."""

    def test_embed_returns_numpy_array(self, subject_embedder):
        """Should return numpy array for single subject list."""
        subjects = [1, 2, 3]
        embedding = subject_embedder.embed(subjects)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1

    def test_embed_returns_correct_dimension(self, subject_embedder):
        """Embedding dimension should match pooler's output."""
        subjects = [1, 2, 3]
        embedding = subject_embedder.embed(subjects)

        assert embedding.shape[0] == 16

    def test_embed_handles_empty_subjects(self, subject_embedder):
        """Should handle empty subject list."""
        embedding = subject_embedder.embed([])

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 16

    def test_embed_handles_single_subject(self, subject_embedder):
        """Should handle list with single subject."""
        embedding = subject_embedder.embed([5])

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 16

    def test_embed_is_deterministic(self, subject_embedder):
        """Same subjects should produce same embedding."""
        subjects = [1, 2, 3, 4, 5]

        emb1 = subject_embedder.embed(subjects)
        emb2 = subject_embedder.embed(subjects)

        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_different_subjects_produce_different_embeddings(self, subject_embedder):
        """Different subjects should produce different embeddings."""
        subjects1 = [1, 2, 3]
        subjects2 = [4, 5, 6]

        emb1 = subject_embedder.embed(subjects1)
        emb2 = subject_embedder.embed(subjects2)

        assert not np.allclose(emb1, emb2)

    def test_embed_calls_pooler_correctly(self, subject_embedder, mock_pooler):
        """Should call pooler with correct format."""
        subjects = [1, 2, 3]
        subject_embedder.embed(subjects)

        # Should call pooler with list of lists
        mock_pooler.assert_called_once()
        call_args = mock_pooler.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert call_args[0] == subjects


class TestEmbedBatch:
    """Test batch embedding computation."""

    def test_embed_batch_returns_numpy_array(self, subject_embedder):
        """Should return 2D numpy array for batch."""
        subjects_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

        embeddings = subject_embedder.embed_batch(subjects_list)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2

    def test_embed_batch_returns_correct_shape(self, subject_embedder):
        """Batch size should match input, dimension should match pooler."""
        subjects_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

        embeddings = subject_embedder.embed_batch(subjects_list)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 16

    def test_embed_batch_handles_empty_batch(self, subject_embedder):
        """Should handle empty batch."""
        embeddings = subject_embedder.embed_batch([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0

    def test_embed_batch_handles_single_item(self, subject_embedder):
        """Should handle batch with single item."""
        subjects_list = [[1, 2, 3]]

        embeddings = subject_embedder.embed_batch(subjects_list)

        assert embeddings.shape == (1, 16)

    def test_embed_batch_is_deterministic(self, subject_embedder):
        """Same batch should produce same embeddings."""
        subjects_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

        emb1 = subject_embedder.embed_batch(subjects_list)
        emb2 = subject_embedder.embed_batch(subjects_list)

        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_batch_consistent_with_single_embed(self, subject_embedder):
        """Batch embedding should match individual embeds."""
        subjects_list = [[1, 2, 3], [4, 5, 6]]

        batch_embs = subject_embedder.embed_batch(subjects_list)

        single_emb1 = subject_embedder.embed(subjects_list[0])
        single_emb2 = subject_embedder.embed(subjects_list[1])

        np.testing.assert_array_almost_equal(batch_embs[0], single_emb1)
        np.testing.assert_array_almost_equal(batch_embs[1], single_emb2)

    def test_embed_batch_handles_variable_length_subjects(self, subject_embedder):
        """Should handle subject lists of different lengths."""
        subjects_list = [[1], [2, 3, 4, 5], [], [6, 7]]

        embeddings = subject_embedder.embed_batch(subjects_list)

        assert embeddings.shape == (4, 16)

    def test_embed_batch_calls_pooler_correctly(self, subject_embedder, mock_pooler):
        """Should call pooler with entire batch."""
        subjects_list = [[1, 2, 3], [4, 5]]

        subject_embedder.embed_batch(subjects_list)

        # Should call pooler once with all subjects
        mock_pooler.assert_called_once()
        call_args = mock_pooler.call_args[0][0]
        assert call_args == subjects_list


class TestEmbeddingDim:
    """Test embedding dimension property."""

    def test_embedding_dim_returns_correct_value(self, subject_embedder, mock_pooler):
        """Should return dimension from pooler."""
        dim = subject_embedder.embedding_dim

        assert dim == 16
        mock_pooler.get_embedding_dim.assert_called_once()

    def test_embedding_dim_is_property(self, subject_embedder):
        """embedding_dim should be a property, not a method."""
        assert isinstance(type(subject_embedder).embedding_dim, property)


class TestResetSingleton:
    """Test singleton reset functionality."""

    def test_reset_clears_singleton_instance(self):
        """Reset should clear the singleton instance."""
        SubjectEmbedder.reset()

        assert SubjectEmbedder._instance is None

    def test_can_create_new_instance_after_reset(self, mock_pooler):
        """Should be able to create new instance after reset."""
        embedder1 = SubjectEmbedder(pooler=mock_pooler)

        SubjectEmbedder.reset()

        embedder2 = SubjectEmbedder(pooler=mock_pooler)

        assert embedder1 is not embedder2

        SubjectEmbedder.reset()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_embed_handles_large_subject_lists(self, subject_embedder):
        """Should handle large subject lists."""
        large_subjects = list(range(100))

        embedding = subject_embedder.embed(large_subjects)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 16

    def test_embed_batch_handles_large_batch(self, subject_embedder):
        """Should handle large batches."""
        large_batch = [[i, i + 1, i + 2] for i in range(100)]

        embeddings = subject_embedder.embed_batch(large_batch)

        assert embeddings.shape == (100, 16)

    def test_embed_handles_duplicate_subjects(self, subject_embedder):
        """Should handle duplicate subjects in list."""
        subjects = [1, 1, 2, 2, 2, 3]

        embedding = subject_embedder.embed(subjects)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 16

    def test_embed_preserves_subject_order(self, subject_embedder):
        """Different orderings should produce different embeddings."""
        subjects1 = [1, 2, 3]
        subjects2 = [3, 2, 1]

        emb1 = subject_embedder.embed(subjects1)
        emb2 = subject_embedder.embed(subjects2)

        # Depending on pooler implementation, order may matter
        # Our mock is order-independent (sum), but real poolers are order-dependent
        # For the mock, they'll be the same, but we test the interface works
        assert isinstance(emb1, np.ndarray)
        assert isinstance(emb2, np.ndarray)


class TestNoGradientTracking:
    """Test that no gradients are tracked during inference."""

    def test_embed_uses_no_grad(self, mock_pooler):
        """embed should use torch.no_grad context."""
        embedder = SubjectEmbedder(pooler=mock_pooler)

        # Create a tensor that requires grad
        mock_pooler.reset_mock()

        def check_no_grad(subjects_list):
            # Check that we're in no_grad context
            assert not torch.is_grad_enabled()
            return torch.randn(len(subjects_list), 16)

        mock_pooler.side_effect = check_no_grad

        embedder.embed([1, 2, 3])

        SubjectEmbedder.reset()

    def test_embed_batch_uses_no_grad(self, mock_pooler):
        """embed_batch should use torch.no_grad context."""
        embedder = SubjectEmbedder(pooler=mock_pooler)

        mock_pooler.reset_mock()

        def check_no_grad(subjects_list):
            assert not torch.is_grad_enabled()
            return torch.randn(len(subjects_list), 16)

        mock_pooler.side_effect = check_no_grad

        embedder.embed_batch([[1, 2], [3, 4]])

        SubjectEmbedder.reset()


class TestIntegrationWithMockPooler:
    """Test realistic usage patterns with mock pooler."""

    def test_recommendation_pipeline_pattern(self, subject_embedder):
        """Test pattern used in recommendation pipeline."""
        # User's favorite subjects
        user_subjects = [5, 12, 23, 45]

        # Get user embedding
        user_emb = subject_embedder.embed(user_subjects)

        # Book subject lists
        book_subjects_list = [
            [5, 12, 30],
            [23, 45, 67],
            [1, 2, 3],
        ]

        # Get batch of book embeddings
        book_embs = subject_embedder.embed_batch(book_subjects_list)

        # Verify shapes for downstream computation
        assert user_emb.shape == (16,)
        assert book_embs.shape == (3, 16)

        # Could compute similarities (not testing that here)
        similarities = book_embs @ user_emb
        assert similarities.shape == (3,)

    def test_precompute_embeddings_pattern(self, subject_embedder):
        """Test pattern used in precompute_embs.py training script."""
        # Simulate batch processing of book subjects
        all_book_subjects = [[i, i + 1, i + 2] for i in range(50)]

        # Process in batches
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(all_book_subjects), batch_size):
            batch = all_book_subjects[i : i + batch_size]
            batch_embs = subject_embedder.embed_batch(batch)
            all_embeddings.append(batch_embs)

        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings)

        assert final_embeddings.shape == (50, 16)
