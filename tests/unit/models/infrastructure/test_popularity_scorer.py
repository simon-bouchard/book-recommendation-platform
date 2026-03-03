# tests/unit/models/infrastructure/test_popularity_scorer.py
"""
Unit tests for PopularityScorer infrastructure component.
Tests singleton pattern, top-k retrieval, sort order, and boundary conditions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.infrastructure.popularity_scorer import PopularityScorer
from tests.unit.models.infrastructure.conftest import N_BOOKS


@pytest.fixture
def scorer(book_ids, bayesian_scores_raw):
    instance = PopularityScorer(bayesian_scores=bayesian_scores_raw, book_ids=book_ids)
    yield instance
    PopularityScorer.reset()


class TestSingletonPattern:
    def test_injection_bypasses_singleton(self, book_ids, bayesian_scores_raw):
        PopularityScorer.reset()
        s1 = PopularityScorer(bayesian_scores=bayesian_scores_raw, book_ids=book_ids)
        s2 = PopularityScorer(bayesian_scores=bayesian_scores_raw, book_ids=book_ids)
        assert s1 is not s2
        PopularityScorer.reset()

    def test_reset_clears_instance(self):
        PopularityScorer.reset()
        assert PopularityScorer._instance is None

    def test_can_create_new_instance_after_reset(self, book_ids, bayesian_scores_raw):
        s1 = PopularityScorer(bayesian_scores=bayesian_scores_raw, book_ids=book_ids)
        PopularityScorer.reset()
        s2 = PopularityScorer(bayesian_scores=bayesian_scores_raw, book_ids=book_ids)
        assert s1 is not s2
        PopularityScorer.reset()


class TestTopK:
    def test_returns_tuple_of_numpy_arrays(self, scorer):
        item_ids, scores = scorer.top_k(k=10)
        assert isinstance(item_ids, np.ndarray)
        assert isinstance(scores, np.ndarray)

    def test_returns_correct_dtypes(self, scorer):
        item_ids, scores = scorer.top_k(k=10)
        assert item_ids.dtype in (np.int32, np.int64)
        assert scores.dtype in (np.float32, np.float64)

    def test_respects_k(self, scorer):
        item_ids, scores = scorer.top_k(k=10)
        assert len(item_ids) == 10
        assert len(scores) == 10

    def test_results_sorted_descending(self, scorer):
        _, scores = scorer.top_k(k=20)
        assert np.all(scores[:-1] >= scores[1:])

    def test_results_are_from_book_ids(self, scorer, book_ids):
        item_ids, _ = scorer.top_k(k=20)
        book_id_set = set(book_ids)
        assert all(int(iid) in book_id_set for iid in item_ids)

    def test_no_duplicates_in_results(self, scorer):
        item_ids, _ = scorer.top_k(k=30)
        assert len(item_ids) == len(set(item_ids.tolist()))

    def test_top_result_has_highest_score(self, scorer, book_ids, bayesian_scores_raw):
        item_ids, scores = scorer.top_k(k=1)
        expected_row = int(np.argmax(bayesian_scores_raw))
        expected_id = book_ids[expected_row]
        assert int(item_ids[0]) == expected_id

    def test_k_larger_than_catalog_returns_all(self, scorer):
        item_ids, scores = scorer.top_k(k=N_BOOKS + 100)
        assert len(item_ids) == N_BOOKS

    def test_k_zero_returns_empty_arrays(self, scorer):
        item_ids, scores = scorer.top_k(k=0)
        assert len(item_ids) == 0
        assert len(scores) == 0

    def test_is_deterministic(self, scorer):
        ids1, scores1 = scorer.top_k(k=20)
        ids2, scores2 = scorer.top_k(k=20)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_array_equal(scores1, scores2)
