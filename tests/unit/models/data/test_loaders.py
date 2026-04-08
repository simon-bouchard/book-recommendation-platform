# tests/unit/models/data/test_loaders.py
"""
Unit tests for models.data.loaders module.
Tests pure utility functions that require no filesystem or external dependencies.

Filesystem-dependent loader tests (load_book_subject_embeddings, load_als_factors,
etc.) live in tests/integration/models/data/test_loaders.py since they require
real artifact files on disk.
"""

import numpy as np
import pytest

from models.data.loaders import get_item_idx_to_row, load_attention_strategy, normalize_embeddings


class TestHelperFunctions:
    """Test utility helper functions."""

    def test_normalize_embeddings_produces_unit_vectors(self):
        """normalize_embeddings should produce L2-normalized vectors."""
        raw_vectors = np.random.randn(10, 64).astype(np.float32)
        normalized = normalize_embeddings(raw_vectors)

        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_normalize_embeddings_returns_float32(self):
        """Normalized embeddings should be float32."""
        raw_vectors = np.random.randn(10, 64).astype(np.float64)
        normalized = normalize_embeddings(raw_vectors)

        assert normalized.dtype == np.float32

    def test_normalize_embeddings_handles_zero_vectors(self):
        """Should handle zero vectors without crashing."""
        vectors = np.zeros((5, 64))
        normalized = normalize_embeddings(vectors)

        assert not np.any(np.isnan(normalized))

    def test_get_item_idx_to_row_creates_correct_mapping(self):
        """Should create correct item_idx -> row mapping."""
        item_ids = [100, 200, 300, 400]
        mapping = get_item_idx_to_row(item_ids)

        assert mapping[100] == 0
        assert mapping[200] == 1
        assert mapping[300] == 2
        assert mapping[400] == 3

    def test_get_item_idx_to_row_returns_dict(self):
        """Should return dictionary."""
        item_ids = [1, 2, 3]
        mapping = get_item_idx_to_row(item_ids)

        assert isinstance(mapping, dict)
        assert len(mapping) == 3


class TestAttentionStrategyValidation:
    """Test attention strategy name validation (no filesystem access)."""

    def test_load_attention_strategy_invalid_name_raises_error(self):
        """Should raise ValueError for invalid strategy name before hitting disk."""
        with pytest.raises(ValueError, match="Unknown attention strategy"):
            load_attention_strategy(strategy="invalid_strategy", use_cache=False)
