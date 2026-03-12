# tests/unit/models/infrastructure/test_subject_scorer.py
"""
Unit tests for SubjectScorer infrastructure component.
Tests singleton pattern, blending math, alpha extremes, and edge cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.infrastructure.subject_scorer import SubjectScorer
from tests.unit.models.infrastructure.conftest import EMB_DIM, N_BOOKS


@pytest.fixture
def scorer(normalized_embeddings, book_ids, bayesian_scores_norm):
    instance = SubjectScorer(
        embeddings=normalized_embeddings,
        book_ids=book_ids,
        bayesian_scores_norm=bayesian_scores_norm,
    )
    yield instance
    SubjectScorer.reset()


@pytest.fixture
def user_vector(normalized_embeddings) -> np.ndarray:
    """L2-normalized query vector aligned with the first book's embedding."""
    return normalized_embeddings[0].copy()


class TestSingletonPattern:
    def test_injection_bypasses_singleton(
        self, normalized_embeddings, book_ids, bayesian_scores_norm
    ):
        SubjectScorer.reset()
        s1 = SubjectScorer(
            embeddings=normalized_embeddings,
            book_ids=book_ids,
            bayesian_scores_norm=bayesian_scores_norm,
        )
        s2 = SubjectScorer(
            embeddings=normalized_embeddings,
            book_ids=book_ids,
            bayesian_scores_norm=bayesian_scores_norm,
        )
        assert s1 is not s2
        SubjectScorer.reset()

    def test_reset_clears_instance(self):
        SubjectScorer.reset()
        assert SubjectScorer._instance is None

    def test_can_create_new_instance_after_reset(
        self, normalized_embeddings, book_ids, bayesian_scores_norm
    ):
        s1 = SubjectScorer(
            embeddings=normalized_embeddings,
            book_ids=book_ids,
            bayesian_scores_norm=bayesian_scores_norm,
        )
        SubjectScorer.reset()
        s2 = SubjectScorer(
            embeddings=normalized_embeddings,
            book_ids=book_ids,
            bayesian_scores_norm=bayesian_scores_norm,
        )
        assert s1 is not s2
        SubjectScorer.reset()


class TestScore:
    def test_returns_tuple_of_numpy_arrays(self, scorer, user_vector):
        item_ids, scores = scorer.score(user_vector=user_vector, k=10, alpha=0.6)
        assert isinstance(item_ids, np.ndarray)
        assert isinstance(scores, np.ndarray)

    def test_returns_matching_lengths(self, scorer, user_vector):
        item_ids, scores = scorer.score(user_vector=user_vector, k=10, alpha=0.6)
        assert len(item_ids) == len(scores)

    def test_respects_k(self, scorer, user_vector):
        item_ids, _ = scorer.score(user_vector=user_vector, k=10, alpha=0.6)
        assert len(item_ids) == 10

    def test_results_sorted_descending(self, scorer, user_vector):
        _, scores = scorer.score(user_vector=user_vector, k=20, alpha=0.6)
        assert np.all(scores[:-1] >= scores[1:])

    def test_results_from_book_ids(self, scorer, user_vector, book_ids):
        item_ids, _ = scorer.score(user_vector=user_vector, k=20, alpha=0.6)
        book_id_set = set(book_ids)
        assert all(int(iid) in book_id_set for iid in item_ids)

    def test_k_larger_than_catalog_returns_all(self, scorer, user_vector):
        item_ids, _ = scorer.score(user_vector=user_vector, k=N_BOOKS + 100, alpha=0.6)
        assert len(item_ids) == N_BOOKS

    def test_is_deterministic(self, scorer, user_vector):
        ids1, scores1 = scorer.score(user_vector=user_vector, k=20, alpha=0.6)
        ids2, scores2 = scorer.score(user_vector=user_vector, k=20, alpha=0.6)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_array_equal(scores1, scores2)


class TestBlendingMath:
    def test_alpha_one_ranks_by_subject_similarity(
        self, normalized_embeddings, book_ids, bayesian_scores_norm
    ):
        """
        At alpha=1.0 blended scores equal normalized cosine scores, so the
        top result must be the book most similar to the query vector.
        """
        scorer = SubjectScorer(
            embeddings=normalized_embeddings,
            book_ids=book_ids,
            bayesian_scores_norm=bayesian_scores_norm,
        )
        query = normalized_embeddings[5].copy()
        item_ids, _ = scorer.score(user_vector=query, k=5, alpha=1.0)
        assert int(item_ids[0]) == book_ids[5]
        SubjectScorer.reset()

    def test_alpha_zero_ranks_by_popularity(
        self, normalized_embeddings, book_ids, bayesian_scores_norm
    ):
        """
        At alpha=0.0 blended scores equal normalized bayesian scores, so the
        top result must be the most popular book.
        """
        scorer = SubjectScorer(
            embeddings=normalized_embeddings,
            book_ids=book_ids,
            bayesian_scores_norm=bayesian_scores_norm,
        )
        query = normalized_embeddings[0].copy()
        item_ids, _ = scorer.score(user_vector=query, k=1, alpha=0.0)
        expected_row = int(np.argmax(bayesian_scores_norm))
        assert int(item_ids[0]) == book_ids[expected_row]
        SubjectScorer.reset()

    def test_flat_cosine_scores_do_not_crash(self, book_ids, bayesian_scores_norm):
        """
        When all cosine scores are identical (hi == lo), the normalizer
        should return ones rather than dividing by zero.
        """
        n = len(book_ids)
        uniform_embs = np.ones((n, EMB_DIM), dtype=np.float32)
        uniform_embs /= np.linalg.norm(uniform_embs[0])
        query = uniform_embs[0].copy()

        scorer = SubjectScorer(
            embeddings=uniform_embs,
            book_ids=book_ids,
            bayesian_scores_norm=bayesian_scores_norm,
        )
        item_ids, scores = scorer.score(user_vector=query, k=10, alpha=0.5)
        assert not np.any(np.isnan(scores))
        assert len(item_ids) == 10
        SubjectScorer.reset()

    def test_blended_scores_within_valid_range(self, scorer, user_vector):
        """Blended scores are always in [0, 1] since both components are normalized."""
        _, scores = scorer.score(user_vector=user_vector, k=N_BOOKS, alpha=0.6)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0 + 1e-5)
