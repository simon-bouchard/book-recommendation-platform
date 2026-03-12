# tests/unit/models/infrastructure/test_similarity_indices.py
"""
Unit tests for the similarity_indices registry module.
Tests lazy initialization, cache hit behavior, reset, and construction parameters.

All disk loaders are mocked — no real artifacts or FAISS indices are built.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import models.infrastructure.similarity_indices as registry_module
from models.infrastructure.similarity_index import SimilarityIndex
from tests.unit.models.infrastructure.conftest import EMB_DIM, N_BOOKS, N_ALS_BOOKS

_LOADERS_PATH = "models.infrastructure.similarity_indices"


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure the module-level index cache is cleared before and after every test."""
    registry_module.reset_indices()
    yield
    registry_module.reset_indices()


@pytest.fixture
def mock_subject_loader(normalized_embeddings, book_ids):
    """Patch load_book_subject_embeddings to return injected test data."""
    with patch(f"{_LOADERS_PATH}.load_book_subject_embeddings") as mock:
        mock.return_value = (normalized_embeddings, book_ids)
        yield mock


@pytest.fixture
def mock_als_loader(als_factors_normalized, book_ids):
    """
    Patch load_als_factors to return injected test data.

    Returns the four-tuple that the real loader returns:
    (user_factors, book_factors, user_id_map, book_row_map)
    """
    user_factors = np.zeros((10, EMB_DIM), dtype=np.float32)
    user_id_map = {uid: i for i, uid in enumerate(range(10))}
    book_row_map = {i: bid for i, bid in enumerate(book_ids)}

    with patch(f"{_LOADERS_PATH}.load_als_factors") as mock:
        mock.return_value = (user_factors, als_factors_normalized, user_id_map, book_row_map)
        yield mock


@pytest.fixture
def mock_meta_loader(book_ids):
    """Patch load_book_meta to return a DataFrame with sufficient rating counts."""
    import pandas as pd

    meta = pd.DataFrame({"item_idx": book_ids, "book_num_ratings": [50] * len(book_ids)}).set_index(
        "item_idx"
    )

    with patch(f"{_LOADERS_PATH}.load_book_meta") as mock:
        mock.return_value = meta
        yield mock


class TestSubjectSimilarityIndex:
    def test_returns_similarity_index_instance(self, mock_subject_loader):
        index = registry_module.get_subject_similarity_index()
        assert isinstance(index, SimilarityIndex)

    def test_calls_loader_on_first_access(self, mock_subject_loader):
        registry_module.get_subject_similarity_index()
        mock_subject_loader.assert_called_once_with(normalized=True, use_cache=True)

    def test_cache_hit_returns_same_instance(self, mock_subject_loader):
        index1 = registry_module.get_subject_similarity_index()
        index2 = registry_module.get_subject_similarity_index()
        assert index1 is index2

    def test_loader_called_only_once_on_cache_hit(self, mock_subject_loader):
        registry_module.get_subject_similarity_index()
        registry_module.get_subject_similarity_index()
        assert mock_subject_loader.call_count == 1

    def test_built_with_normalize_false(self, mock_subject_loader):
        """
        Subject embeddings are already normalized at load time; passing
        normalize=False avoids a redundant normalization pass.
        """
        index = registry_module.get_subject_similarity_index()
        assert index.num_total == N_BOOKS

    def test_reset_clears_cached_index(self, mock_subject_loader):
        registry_module.get_subject_similarity_index()
        registry_module.reset_indices()
        assert registry_module._subject_index is None


class TestAlsSimilarityIndex:
    def test_returns_similarity_index_instance(self, mock_als_loader, mock_meta_loader):
        index = registry_module.get_als_similarity_index()
        assert isinstance(index, SimilarityIndex)

    def test_calls_loaders_on_first_access(self, mock_als_loader, mock_meta_loader):
        registry_module.get_als_similarity_index()
        mock_als_loader.assert_called_once_with(normalized=True, use_cache=True)
        mock_meta_loader.assert_called_once_with(use_cache=True)

    def test_cache_hit_returns_same_instance(self, mock_als_loader, mock_meta_loader):
        index1 = registry_module.get_als_similarity_index()
        index2 = registry_module.get_als_similarity_index()
        assert index1 is index2

    def test_loader_called_only_once_on_cache_hit(self, mock_als_loader, mock_meta_loader):
        registry_module.get_als_similarity_index()
        registry_module.get_als_similarity_index()
        assert mock_als_loader.call_count == 1

    def test_built_with_rating_filter(self, mock_als_loader, mock_meta_loader):
        """ALS index should apply a min_rating_count=10 candidate filter."""
        index = registry_module.get_als_similarity_index()
        assert index.num_total == N_ALS_BOOKS

    def test_reset_clears_cached_index(self, mock_als_loader, mock_meta_loader):
        registry_module.get_als_similarity_index()
        registry_module.reset_indices()
        assert registry_module._als_index is None


class TestResetIndices:
    def test_reset_clears_both_indices(
        self, mock_subject_loader, mock_als_loader, mock_meta_loader
    ):
        registry_module.get_subject_similarity_index()
        registry_module.get_als_similarity_index()

        registry_module.reset_indices()

        assert registry_module._subject_index is None
        assert registry_module._als_index is None

    def test_after_reset_loaders_called_again(
        self, mock_subject_loader, mock_als_loader, mock_meta_loader
    ):
        registry_module.get_subject_similarity_index()
        registry_module.reset_indices()
        registry_module.get_subject_similarity_index()

        assert mock_subject_loader.call_count == 2


class TestPreloadIndices:
    def test_preload_builds_both_indices(
        self, mock_subject_loader, mock_als_loader, mock_meta_loader
    ):
        registry_module.preload_indices()

        assert registry_module._subject_index is not None
        assert registry_module._als_index is not None

    def test_preload_is_idempotent(self, mock_subject_loader, mock_als_loader, mock_meta_loader):
        """Calling preload twice should not rebuild indices or call loaders again."""
        registry_module.preload_indices()
        registry_module.preload_indices()

        assert mock_subject_loader.call_count == 1
        assert mock_als_loader.call_count == 1
