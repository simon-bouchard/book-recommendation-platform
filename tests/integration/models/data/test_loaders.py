# tests/integration/models/data/test_loaders.py
"""
Integration tests for models.data.loaders module.
Tests all loader functions with real artifact files.

Requires the training pipeline to have run and produced a valid active artifact
version. Tests are skipped automatically when artifacts are absent (see conftest.py).

Pure utility function tests live in tests/unit/models/data/test_loaders.py.
"""

import numpy as np
import pandas as pd
import pytest

from models.data.loaders import (
    load_als_factors,
    load_attention_strategy,
    load_bayesian_scores,
    load_book_meta,
    load_book_subject_embeddings,
    load_user_meta,
)
from models.core.paths import PATHS


class TestBookSubjectEmbeddings:
    """Test loading book subject embeddings."""

    def test_load_book_subject_embeddings_returns_correct_types(self):
        """Loader should return numpy array and list of ints."""
        embeddings, ids = load_book_subject_embeddings(use_cache=False)

        assert isinstance(embeddings, np.ndarray)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids[:10])

    def test_embeddings_and_ids_have_same_length(self):
        """Number of embeddings should match number of IDs."""
        embeddings, ids = load_book_subject_embeddings(use_cache=False)

        assert len(embeddings) == len(ids)

    def test_embeddings_have_expected_shape(self):
        """Embeddings should be 2D array with reasonable dimensions."""
        embeddings, _ = load_book_subject_embeddings(use_cache=False)

        assert embeddings.ndim == 2
        assert embeddings.shape[0] > 0
        assert embeddings.shape[1] > 0

    def test_normalized_embeddings_are_unit_vectors(self):
        """Normalized embeddings should have L2 norm of ~1."""
        embeddings, _ = load_book_subject_embeddings(normalized=True, use_cache=False)

        norms = np.linalg.norm(embeddings[:10], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_normalized_and_raw_have_same_shape(self):
        """Normalized and raw embeddings should have same shape."""
        raw_embs, raw_ids = load_book_subject_embeddings(normalized=False, use_cache=False)
        norm_embs, norm_ids = load_book_subject_embeddings(normalized=True, use_cache=False)

        assert raw_embs.shape == norm_embs.shape
        assert raw_ids == norm_ids

    def test_cache_disabled_returns_new_objects(self):
        """Calls with cache=False should load fresh data."""
        embs1, _ = load_book_subject_embeddings(use_cache=False)
        embs2, _ = load_book_subject_embeddings(use_cache=False)

        assert embs1 is not embs2
        assert np.array_equal(embs1, embs2)


class TestALSEmbeddings:
    """Test loading ALS collaborative filtering factors."""

    def test_load_als_factors(self):
        """Should return user factors, book factors, and two mappings."""
        result = load_als_factors(use_cache=False)

        assert len(result) == 4
        user_factors, book_factors, user_map, book_map = result

        assert isinstance(user_factors, np.ndarray)
        assert isinstance(book_factors, np.ndarray)
        assert isinstance(user_map, dict)
        assert isinstance(book_map, dict)

    def test_als_factors_are_2d_arrays(self):
        """User and book factors should be 2D arrays."""
        user_factors, book_factors, _, _ = load_als_factors(use_cache=False)

        assert user_factors.ndim == 2
        assert book_factors.ndim == 2

    def test_als_factors_have_same_embedding_dimension(self):
        """User and book factors should have same number of latent dimensions."""
        user_factors, book_factors, _, _ = load_als_factors(use_cache=False)

        assert user_factors.shape[1] == book_factors.shape[1]

    def test_als_mappings_are_consistent(self):
        """Mapping sizes should match factor array sizes."""
        user_factors, book_factors, user_map, book_map = load_als_factors(use_cache=False)

        assert len(user_map) == user_factors.shape[0]
        assert len(book_map) == book_factors.shape[0]

    def test_normalized_als_factors_are_unit_vectors(self):
        """Normalized ALS factors should have L2 norm of ~1."""
        user_factors, book_factors, _, _ = load_als_factors(normalized=True, use_cache=False)

        user_norms = np.linalg.norm(user_factors[:10], axis=1)
        assert np.allclose(user_norms, 1.0, atol=1e-5)

        book_norms = np.linalg.norm(book_factors[:10], axis=1)
        assert np.allclose(book_norms, 1.0, atol=1e-5)


class TestBayesianScores:
    """Test loading precomputed Bayesian scores."""

    def test_load_bayesian_scores_returns_array(self):
        """Should return numpy array of scores."""
        scores = load_bayesian_scores(use_cache=False)

        assert isinstance(scores, np.ndarray)

    def test_bayesian_scores_are_1d(self):
        """Scores should be 1D array."""
        scores = load_bayesian_scores(use_cache=False)

        assert scores.ndim == 1

    def test_bayesian_scores_have_no_nan_or_inf(self):
        """Scores should be cleaned of NaN and inf values."""
        scores = load_bayesian_scores(use_cache=False)

        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_bayesian_scores_align_with_book_ids(self):
        """Scores array should have same length as book IDs."""
        scores = load_bayesian_scores(use_cache=False)
        _, book_ids = load_book_subject_embeddings(use_cache=False)

        assert len(scores) == len(book_ids)


class TestMetadata:
    """Test loading book and user metadata."""

    def test_load_book_meta_returns_dataframe(self):
        """Should return pandas DataFrame."""
        book_meta = load_book_meta(use_cache=False)

        assert isinstance(book_meta, pd.DataFrame)

    def test_book_meta_indexed_by_item_idx(self):
        """Book metadata should be indexed by item_idx."""
        book_meta = load_book_meta(use_cache=False)

        assert book_meta.index.name == "item_idx" or "item_idx" in str(book_meta.index.dtype)

    def test_book_meta_has_bayesian_scores(self):
        """Book metadata should include bayes column."""
        book_meta = load_book_meta(use_cache=False)

        assert "bayes" in book_meta.columns

    def test_book_meta_has_expected_columns(self):
        """Book metadata should have standard book attributes."""
        book_meta = load_book_meta(use_cache=False)

        expected_cols = ["title", "author", "year"]
        for col in expected_cols:
            assert col in book_meta.columns, f"Missing column: {col}"

    def test_load_user_meta_returns_dataframe(self):
        """Should return pandas DataFrame."""
        user_meta = load_user_meta(use_cache=False)

        assert isinstance(user_meta, pd.DataFrame)

    def test_user_meta_indexed_by_user_id(self):
        """User metadata should be indexed by user_id."""
        user_meta = load_user_meta(use_cache=False)

        assert user_meta.index.name == "user_id" or "user_id" in str(user_meta.index.dtype)


class TestAttentionStrategy:
    """Test loading attention pooling strategies."""

    @pytest.mark.parametrize("strategy", ["scalar", "perdim", "selfattn", "selfattn_perdim"])
    def test_load_attention_strategy_succeeds(self, strategy):
        """Should load each attention strategy variant."""
        path = PATHS.get_attention_path(strategy)
        if not path.exists():
            pytest.skip(f"Attention model not found: {strategy}")

        strat_obj = load_attention_strategy(strategy=strategy, use_cache=False)

        assert strat_obj is not None
        assert hasattr(strat_obj, "forward") or callable(strat_obj)
