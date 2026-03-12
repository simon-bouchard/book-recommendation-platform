# tests/unit/models/infrastructure/test_hybrid_scorer.py
"""
Unit tests for HybridScorer infrastructure component.
Tests singleton pattern, alignment map behavior, alpha blending, candidate
filtering, and query exclusion.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.infrastructure.hybrid_scorer import HybridScorer
from tests.unit.models.infrastructure.conftest import N_ALS_BOOKS, N_BOOKS


@pytest.fixture
def scorer(
    normalized_embeddings, book_ids, als_factors_normalized, als_row_for_subject, rating_counts
):
    instance = HybridScorer(
        subject_embeddings=normalized_embeddings,
        subject_ids=book_ids,
        als_factors=als_factors_normalized,
        als_row_for_subject=als_row_for_subject,
        rating_counts=rating_counts,
    )
    yield instance
    HybridScorer.reset()


@pytest.fixture
def als_item_id(book_ids) -> int:
    """ID of a book that has an ALS factor (first N_ALS_BOOKS)."""
    return book_ids[0]


@pytest.fixture
def non_als_item_id(book_ids) -> int:
    """ID of a book that does NOT have an ALS factor (last 15 books)."""
    return book_ids[N_ALS_BOOKS]


class TestSingletonPattern:
    def test_injection_bypasses_singleton(
        self,
        normalized_embeddings,
        book_ids,
        als_factors_normalized,
        als_row_for_subject,
        rating_counts,
    ):
        HybridScorer.reset()
        s1 = HybridScorer(
            subject_embeddings=normalized_embeddings,
            subject_ids=book_ids,
            als_factors=als_factors_normalized,
            als_row_for_subject=als_row_for_subject,
            rating_counts=rating_counts,
        )
        s2 = HybridScorer(
            subject_embeddings=normalized_embeddings,
            subject_ids=book_ids,
            als_factors=als_factors_normalized,
            als_row_for_subject=als_row_for_subject,
            rating_counts=rating_counts,
        )
        assert s1 is not s2
        HybridScorer.reset()

    def test_reset_clears_instance(self):
        HybridScorer.reset()
        assert HybridScorer._instance is None


class TestHasItem:
    def test_returns_true_for_item_in_subject_pool(self, scorer, als_item_id):
        assert scorer.has_item(als_item_id) is True

    def test_returns_true_for_item_without_als(self, scorer, non_als_item_id):
        """A book without ALS factors can still be queried via its subject embedding."""
        assert scorer.has_item(non_als_item_id) is True

    def test_returns_false_for_unknown_item(self, scorer):
        assert scorer.has_item(-99999) is False


class TestScore:
    def test_returns_tuple_of_numpy_arrays(self, scorer, als_item_id):
        item_ids, scores = scorer.score(item_idx=als_item_id, k=10, alpha=0.6)
        assert isinstance(item_ids, np.ndarray)
        assert isinstance(scores, np.ndarray)

    def test_returns_matching_lengths(self, scorer, als_item_id):
        item_ids, scores = scorer.score(item_idx=als_item_id, k=10, alpha=0.6)
        assert len(item_ids) == len(scores)

    def test_respects_k(self, scorer, als_item_id):
        item_ids, _ = scorer.score(item_idx=als_item_id, k=10, alpha=0.6)
        assert len(item_ids) <= 10

    def test_results_sorted_descending(self, scorer, als_item_id):
        _, scores = scorer.score(item_idx=als_item_id, k=15, alpha=0.6, filter_candidates=False)
        assert np.all(scores[:-1] >= scores[1:])

    def test_query_item_excluded_from_results(self, scorer, als_item_id):
        item_ids, _ = scorer.score(item_idx=als_item_id, k=20, alpha=0.6, filter_candidates=False)
        assert als_item_id not in item_ids.tolist()

    def test_unknown_item_returns_empty_arrays(self, scorer):
        item_ids, scores = scorer.score(item_idx=-99999, k=10, alpha=0.6)
        assert len(item_ids) == 0
        assert len(scores) == 0

    def test_is_deterministic(self, scorer, als_item_id):
        ids1, scores1 = scorer.score(item_idx=als_item_id, k=10, alpha=0.6)
        ids2, scores2 = scorer.score(item_idx=als_item_id, k=10, alpha=0.6)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_array_equal(scores1, scores2)


class TestAlignmentMap:
    def test_non_als_query_does_not_crash(self, scorer, non_als_item_id):
        """
        Querying a book with no ALS factor should fall back to pure subject
        similarity without raising. The query has als_row == -1.
        """
        item_ids, scores = scorer.score(
            item_idx=non_als_item_id, k=10, alpha=0.6, filter_candidates=False
        )
        assert len(item_ids) > 0
        assert not np.any(np.isnan(scores))

    def test_alpha_zero_uses_only_subject_scores(
        self,
        normalized_embeddings,
        book_ids,
        als_factors_normalized,
        als_row_for_subject,
        rating_counts,
    ):
        """
        At alpha=0.0 the ALS component is weighted to zero. Scores from a book
        with and without ALS factors should be equal when their subject embeddings
        are identical, because the ALS contribution is multiplied by zero.
        """
        # Make book 0 and book N_ALS_BOOKS have the same subject embedding
        embs = normalized_embeddings.copy()
        embs[N_ALS_BOOKS] = embs[0].copy()

        scorer_a = HybridScorer(
            subject_embeddings=embs,
            subject_ids=book_ids,
            als_factors=als_factors_normalized,
            als_row_for_subject=als_row_for_subject,
            rating_counts=rating_counts,
        )
        # Query with a third book so neither 0 nor N_ALS_BOOKS is the query
        query_id = book_ids[1]
        ids, scores = scorer_a.score(
            item_idx=query_id, k=N_BOOKS, alpha=0.0, filter_candidates=False
        )

        score_map = {int(iid): float(s) for iid, s in zip(ids, scores)}
        assert score_map[book_ids[0]] == pytest.approx(score_map[book_ids[N_ALS_BOOKS]], abs=1e-5)
        HybridScorer.reset()

    def test_alpha_one_uses_only_als_scores(self, scorer, als_item_id, book_ids):
        """
        At alpha=1.0 only ALS cosine scores contribute. Books without ALS
        factors receive a score of zero for the ALS component.
        """
        ids, scores = scorer.score(
            item_idx=als_item_id, k=N_BOOKS, alpha=1.0, filter_candidates=False
        )
        score_map = {int(iid): float(s) for iid, s in zip(ids, scores)}

        non_als_ids = set(book_ids[N_ALS_BOOKS:])
        for non_als_id in non_als_ids:
            if non_als_id in score_map:
                assert score_map[non_als_id] == pytest.approx(0.0, abs=1e-5)


class TestCandidateFiltering:
    def test_filter_candidates_restricts_to_high_rated_books(
        self, scorer, als_item_id, book_ids, rating_counts
    ):
        """
        With filter_candidates=True, all returned books must meet the
        HYBRID_MIN_RATINGS threshold. The conftest sets first 25 books
        above threshold, last 25 below.
        """
        threshold = HybridScorer.HYBRID_MIN_RATINGS
        ids, _ = scorer.score(item_idx=als_item_id, k=N_BOOKS, alpha=0.6, filter_candidates=True)

        id_to_row = {bid: i for i, bid in enumerate(book_ids)}
        for iid in ids:
            row = id_to_row[int(iid)]
            assert rating_counts[row] >= threshold

    def test_filter_candidates_false_returns_all_books(self, scorer, als_item_id):
        """Disabling the filter should allow low-rated books to appear in results."""
        ids_filtered, _ = scorer.score(
            item_idx=als_item_id, k=N_BOOKS, alpha=0.6, filter_candidates=True
        )
        ids_unfiltered, _ = scorer.score(
            item_idx=als_item_id, k=N_BOOKS, alpha=0.6, filter_candidates=False
        )
        assert len(ids_unfiltered) > len(ids_filtered)

    def test_custom_min_rating_count_overrides_default(
        self, scorer, als_item_id, book_ids, rating_counts
    ):
        """Explicit min_rating_count should override HYBRID_MIN_RATINGS."""
        custom_threshold = 20
        ids, _ = scorer.score(
            item_idx=als_item_id,
            k=N_BOOKS,
            alpha=0.6,
            filter_candidates=True,
            min_rating_count=custom_threshold,
        )
        id_to_row = {bid: i for i, bid in enumerate(book_ids)}
        for iid in ids:
            row = id_to_row[int(iid)]
            assert rating_counts[row] >= custom_threshold
