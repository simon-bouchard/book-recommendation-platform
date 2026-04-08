# tests/unit/models/infrastructure/test_similarity_indices.py
"""
Unit tests for the similarity_indices registry module.
Tests lazy initialization, cache hit behavior, reset, and construction parameters.

SimilarityIndex.load is mocked — no real artifact files or FAISS index builds occur.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import models.infrastructure.similarity_indices as registry_module
from models.infrastructure.similarity_index import SimilarityIndex
from tests.unit.models.infrastructure.conftest import N_ALS_BOOKS, N_BOOKS

_REGISTRY_PATH = "models.infrastructure.similarity_indices"


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure the module-level index cache is cleared before and after every test."""
    registry_module.reset_indices()
    yield
    registry_module.reset_indices()


@pytest.fixture
def subject_index(normalized_embeddings, book_ids):
    """Real SimilarityIndex built from test subject embeddings."""
    return SimilarityIndex(
        embeddings=normalized_embeddings,
        ids=book_ids,
        normalize=False,
    )


@pytest.fixture
def als_index(als_factors_normalized, book_ids):
    """Real SimilarityIndex built from test ALS factors (N_ALS_BOOKS items)."""
    return SimilarityIndex(
        embeddings=als_factors_normalized,
        ids=book_ids[:N_ALS_BOOKS],
        normalize=False,
    )


@pytest.fixture
def mock_index_load(subject_index, als_index):
    """
    Patch SimilarityIndex.load to return pre-built test indices.

    Routes to subject_index or als_index based on whether "subject" appears
    in the path, mirroring the real path layout (similarity/subject vs similarity/als).
    """

    def _load(path):
        return subject_index if "subject" in str(path) else als_index

    with patch(f"{_REGISTRY_PATH}.SimilarityIndex.load", side_effect=_load) as mock:
        yield mock


class TestSubjectSimilarityIndex:
    def test_returns_similarity_index_instance(self, mock_index_load):
        index = registry_module.get_subject_similarity_index()
        assert isinstance(index, SimilarityIndex)

    def test_loads_from_disk_on_first_access(self, mock_index_load):
        registry_module.get_subject_similarity_index()
        assert mock_index_load.call_count == 1

    def test_cache_hit_returns_same_instance(self, mock_index_load):
        index1 = registry_module.get_subject_similarity_index()
        index2 = registry_module.get_subject_similarity_index()
        assert index1 is index2

    def test_load_called_only_once_on_cache_hit(self, mock_index_load):
        registry_module.get_subject_similarity_index()
        registry_module.get_subject_similarity_index()
        assert mock_index_load.call_count == 1

    def test_index_covers_all_books(self, mock_index_load):
        index = registry_module.get_subject_similarity_index()
        assert index.num_total == N_BOOKS

    def test_reset_clears_cached_index(self, mock_index_load):
        registry_module.get_subject_similarity_index()
        registry_module.reset_indices()
        assert registry_module._subject_index is None


class TestAlsSimilarityIndex:
    def test_returns_similarity_index_instance(self, mock_index_load):
        index = registry_module.get_als_similarity_index()
        assert isinstance(index, SimilarityIndex)

    def test_loads_from_disk_on_first_access(self, mock_index_load):
        registry_module.get_als_similarity_index()
        assert mock_index_load.call_count == 1

    def test_cache_hit_returns_same_instance(self, mock_index_load):
        index1 = registry_module.get_als_similarity_index()
        index2 = registry_module.get_als_similarity_index()
        assert index1 is index2

    def test_load_called_only_once_on_cache_hit(self, mock_index_load):
        registry_module.get_als_similarity_index()
        registry_module.get_als_similarity_index()
        assert mock_index_load.call_count == 1

    def test_index_covers_als_books(self, mock_index_load):
        index = registry_module.get_als_similarity_index()
        assert index.num_total == N_ALS_BOOKS

    def test_reset_clears_cached_index(self, mock_index_load):
        registry_module.get_als_similarity_index()
        registry_module.reset_indices()
        assert registry_module._als_index is None


class TestResetIndices:
    def test_reset_clears_both_indices(self, mock_index_load):
        registry_module.get_subject_similarity_index()
        registry_module.get_als_similarity_index()

        registry_module.reset_indices()

        assert registry_module._subject_index is None
        assert registry_module._als_index is None

    def test_after_reset_load_called_again(self, mock_index_load):
        registry_module.get_subject_similarity_index()
        registry_module.reset_indices()
        registry_module.get_subject_similarity_index()

        assert mock_index_load.call_count == 2


class TestPreloadIndices:
    def test_preload_builds_both_indices(self, mock_index_load):
        registry_module.preload_indices()

        assert registry_module._subject_index is not None
        assert registry_module._als_index is not None

    def test_preload_is_idempotent(self, mock_index_load):
        """Calling preload twice should not reload indices."""
        registry_module.preload_indices()
        registry_module.preload_indices()

        assert mock_index_load.call_count == 2  # one per index, not per preload call
