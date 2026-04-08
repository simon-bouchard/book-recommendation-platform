# tests/unit/models/infrastructure/test_similarity_index.py
"""
Unit tests for SimilarityIndex infrastructure component.
Tests two-pool filtering system, FAISS search, and candidate filtering.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.infrastructure.similarity_index import SimilarityIndex


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    np.random.seed(42)

    n_items = 50
    dim = 16

    embeddings = np.random.randn(n_items, dim).astype(np.float32)
    ids = list(range(1000, 1000 + n_items))

    return embeddings, ids


@pytest.fixture
def mock_metadata():
    """Create mock metadata with rating counts."""
    n_items = 50
    ids = list(range(1000, 1000 + n_items))

    rating_counts = np.random.randint(0, 100, size=n_items)

    df = pd.DataFrame(
        {
            "item_idx": ids,
            "book_num_ratings": rating_counts,
            "title": [f"Book {i}" for i in ids],
        }
    )
    df = df.set_index("item_idx")

    return df


class TestSimilarityIndexInitialization:
    """Test SimilarityIndex initialization and validation."""

    def test_creates_index_without_filtering(self, mock_embeddings):
        """Should create index with all items as candidates."""
        embeddings, ids = mock_embeddings

        index = SimilarityIndex(embeddings, ids, normalize=True, candidate_mask=None)

        assert index.num_total == len(ids)
        assert index.num_candidates == len(ids)

    def test_creates_index_with_candidate_filtering(self, mock_embeddings):
        """Should create index with filtered candidate pool."""
        embeddings, ids = mock_embeddings

        candidate_mask = np.array([i % 2 == 0 for i in range(len(ids))])

        index = SimilarityIndex(embeddings, ids, normalize=True, candidate_mask=candidate_mask)

        assert index.num_total == len(ids)
        assert index.num_candidates == candidate_mask.sum()

    def test_raises_error_on_length_mismatch(self):
        """Should raise ValueError if embeddings and IDs have different lengths."""
        embeddings = np.random.randn(50, 16)
        ids = list(range(40))

        with pytest.raises(ValueError, match="length mismatch"):
            SimilarityIndex(embeddings, ids)

    def test_normalizes_embeddings_when_requested(self, mock_embeddings):
        """Should L2-normalize embeddings when normalize=True."""
        embeddings, ids = mock_embeddings

        index = SimilarityIndex(embeddings, ids, normalize=True)

        norms = np.linalg.norm(index.embeddings_full, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_keeps_raw_embeddings_when_not_normalized(self, mock_embeddings):
        """Should keep raw embeddings when normalize=False."""
        embeddings, ids = mock_embeddings

        index = SimilarityIndex(embeddings, ids, normalize=False)

        np.testing.assert_array_almost_equal(index.embeddings_full, embeddings.astype(np.float32))

    def test_handles_zero_vectors_in_normalization(self):
        """Should handle zero vectors without crashing."""
        embeddings = np.zeros((10, 16))
        ids = list(range(10))

        index = SimilarityIndex(embeddings, ids, normalize=True)

        assert not np.any(np.isnan(index.embeddings_full))


class TestTwoPoolSystem:
    """Test two-pool query vs candidate system."""

    def test_can_query_any_item_even_if_filtered(self, mock_embeddings):
        """Should allow querying items not in candidate pool."""
        embeddings, ids = mock_embeddings

        candidate_mask = np.zeros(len(ids), dtype=bool)
        candidate_mask[:25] = True

        index = SimilarityIndex(embeddings, ids, normalize=True, candidate_mask=candidate_mask)

        query_id = ids[40]
        scores, results = index.search(query_id, k=10)

        assert len(results) > 0
        assert all(result_id in ids[:25] for result_id in results)

    def test_results_only_from_candidate_pool(self, mock_embeddings):
        """Search results should only include candidates."""
        embeddings, ids = mock_embeddings

        candidate_mask = np.array([i < 25 for i in range(len(ids))])

        index = SimilarityIndex(embeddings, ids, normalize=True, candidate_mask=candidate_mask)

        query_id = ids[0]
        scores, results = index.search(query_id, k=20)

        candidate_ids = set(ids[:25])
        assert all(result_id in candidate_ids for result_id in results)

    def test_has_item_checks_full_pool(self, mock_embeddings):
        """has_item should check full pool, not just candidates."""
        embeddings, ids = mock_embeddings

        candidate_mask = np.zeros(len(ids), dtype=bool)
        candidate_mask[:25] = True

        index = SimilarityIndex(embeddings, ids, normalize=True, candidate_mask=candidate_mask)

        assert index.has_item(ids[0]) is True
        assert index.has_item(ids[40]) is True
        assert index.has_item(-99999) is False

    def test_is_candidate_checks_candidate_pool(self, mock_embeddings):
        """is_candidate should check filtered candidate pool."""
        embeddings, ids = mock_embeddings

        candidate_mask = np.zeros(len(ids), dtype=bool)
        candidate_mask[:25] = True

        index = SimilarityIndex(embeddings, ids, normalize=True, candidate_mask=candidate_mask)

        assert index.is_candidate(ids[0]) is True
        assert index.is_candidate(ids[40]) is False


class TestSearch:
    """Test similarity search functionality."""

    def test_search_returns_correct_types(self, mock_embeddings):
        """Should return numpy arrays of scores and IDs."""
        embeddings, ids = mock_embeddings
        index = SimilarityIndex(embeddings, ids, normalize=True)

        scores, result_ids = index.search(ids[0], k=10)

        assert isinstance(scores, np.ndarray)
        assert isinstance(result_ids, np.ndarray)
        assert scores.dtype in [np.float32, np.float64]
        assert result_ids.dtype in [np.int32, np.int64]

    def test_search_respects_k_parameter(self, mock_embeddings):
        """Should return at most k results."""
        embeddings, ids = mock_embeddings
        index = SimilarityIndex(embeddings, ids, normalize=True)

        scores_5, results_5 = index.search(ids[0], k=5)
        scores_20, results_20 = index.search(ids[0], k=20)

        assert len(results_5) <= 5
        assert len(results_20) <= 20

    def test_search_excludes_query_by_default(self, mock_embeddings):
        """Should exclude query item from results by default."""
        embeddings, ids = mock_embeddings
        index = SimilarityIndex(embeddings, ids, normalize=True)

        query_id = ids[0]
        scores, results = index.search(query_id, k=10, exclude_query=True)

        assert query_id not in results

    def test_search_includes_query_when_requested(self, mock_embeddings):
        """Should include query item when exclude_query=False."""
        embeddings, ids = mock_embeddings
        index = SimilarityIndex(embeddings, ids, normalize=True)

        query_id = ids[0]
        scores, results = index.search(query_id, k=10, exclude_query=False)

        assert query_id in results

    def test_search_returns_highest_scoring_items(self, mock_embeddings):
        """Results should be sorted by similarity score descending."""
        embeddings, ids = mock_embeddings
        index = SimilarityIndex(embeddings, ids, normalize=True)

        scores, results = index.search(ids[0], k=10)

        assert np.all(scores[:-1] >= scores[1:])

    def test_search_returns_empty_for_missing_query(self, mock_embeddings):
        """Should return empty arrays for non-existent query."""
        embeddings, ids = mock_embeddings
        index = SimilarityIndex(embeddings, ids, normalize=True)

        scores, results = index.search(-99999, k=10)

        assert len(scores) == 0
        assert len(results) == 0

    def test_search_handles_k_larger_than_candidates(self, mock_embeddings):
        """Should handle k larger than candidate pool size."""
        embeddings, ids = mock_embeddings
        index = SimilarityIndex(embeddings, ids, normalize=True)

        scores, results = index.search(ids[0], k=1000)

        assert len(results) <= index.num_candidates

    def test_search_is_deterministic(self, mock_embeddings):
        """Same query should return same results."""
        embeddings, ids = mock_embeddings
        index = SimilarityIndex(embeddings, ids, normalize=True)

        scores1, results1 = index.search(ids[0], k=10)
        scores2, results2 = index.search(ids[0], k=10)

        np.testing.assert_array_equal(results1, results2)
        np.testing.assert_allclose(scores1, scores2)


class TestCreateFilteredIndex:
    """Test factory method for creating filtered indices."""

    def test_creates_unfiltered_index_with_zero_threshold(self, mock_embeddings, mock_metadata):
        """Should create unfiltered index when min_rating_count=0."""
        embeddings, ids = mock_embeddings

        index = SimilarityIndex.create_filtered_index(
            embeddings, ids, mock_metadata, min_rating_count=0
        )

        assert index.num_total == len(ids)
        assert index.num_candidates == len(ids)

    def test_filters_by_rating_count(self, mock_embeddings, mock_metadata):
        """Should filter candidates by rating threshold."""
        embeddings, ids = mock_embeddings

        threshold = 50
        expected_candidates = (mock_metadata["book_num_ratings"] >= threshold).sum()

        index = SimilarityIndex.create_filtered_index(
            embeddings, ids, mock_metadata, min_rating_count=threshold
        )

        assert index.num_candidates == expected_candidates
        assert index.num_total == len(ids)

    def test_filtered_items_meet_threshold(self, mock_embeddings, mock_metadata):
        """Candidate items should all meet rating threshold."""
        embeddings, ids = mock_embeddings

        threshold = 50
        index = SimilarityIndex.create_filtered_index(
            embeddings, ids, mock_metadata, min_rating_count=threshold
        )

        for candidate_id in index.candidate_ids:
            rating_count = mock_metadata.loc[candidate_id, "book_num_ratings"]
            assert rating_count >= threshold

    def test_handles_missing_metadata_gracefully(self, mock_embeddings):
        """Should handle items missing from metadata."""
        embeddings, ids = mock_embeddings

        partial_metadata = pd.DataFrame(
            {
                "item_idx": ids[:25],
                "book_num_ratings": [10] * 25,
            }
        ).set_index("item_idx")

        index = SimilarityIndex.create_filtered_index(
            embeddings, ids, partial_metadata, min_rating_count=5
        )

        assert index.num_total == len(ids)

    def test_normalizes_when_requested(self, mock_embeddings, mock_metadata):
        """Should normalize embeddings when normalize=True."""
        embeddings, ids = mock_embeddings

        index = SimilarityIndex.create_filtered_index(
            embeddings, ids, mock_metadata, min_rating_count=0, normalize=True
        )

        norms = np.linalg.norm(index.embeddings_full, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestProperties:
    """Test index property accessors."""

    def test_num_total_returns_full_pool_size(self, mock_embeddings):
        """Should return total number of items."""
        embeddings, ids = mock_embeddings

        candidate_mask = np.array([i < 25 for i in range(len(ids))])
        index = SimilarityIndex(embeddings, ids, candidate_mask=candidate_mask)

        assert index.num_total == len(ids)

    def test_num_candidates_returns_filtered_pool_size(self, mock_embeddings):
        """Should return number of candidates."""
        embeddings, ids = mock_embeddings

        candidate_mask = np.array([i < 25 for i in range(len(ids))])
        index = SimilarityIndex(embeddings, ids, candidate_mask=candidate_mask)

        assert index.num_candidates == 25


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_candidate_pool_returns_empty_results(self, mock_embeddings):
        """Should handle empty candidate pool gracefully."""
        embeddings, ids = mock_embeddings

        candidate_mask = np.zeros(len(ids), dtype=bool)
        index = SimilarityIndex(embeddings, ids, candidate_mask=candidate_mask)

        scores, results = index.search(ids[0], k=10)

        assert len(scores) == 0
        assert len(results) == 0

    def test_single_candidate_index(self, mock_embeddings):
        """Should handle index with single candidate."""
        embeddings, ids = mock_embeddings

        candidate_mask = np.zeros(len(ids), dtype=bool)
        candidate_mask[0] = True

        index = SimilarityIndex(embeddings, ids, candidate_mask=candidate_mask)

        scores, results = index.search(ids[1], k=5)

        assert len(results) == 1
        assert results[0] == ids[0]

    def test_search_with_k_zero(self, mock_embeddings):
        """Should handle k=0 gracefully."""
        embeddings, ids = mock_embeddings
        index = SimilarityIndex(embeddings, ids)

        scores, results = index.search(ids[0], k=0)

        assert len(results) == 0

    def test_identical_embeddings_have_equal_scores(self):
        """Items with identical embeddings should have same similarity."""
        embeddings = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        ids = [100, 200, 300]

        index = SimilarityIndex(embeddings, ids, normalize=True)

        scores, results = index.search(100, k=3, exclude_query=False)

        assert scores[0] == pytest.approx(scores[1], abs=1e-5)


class TestRealWorldScenario:
    """Test realistic usage scenarios."""

    def test_als_mode_filtering_scenario(self, mock_embeddings, mock_metadata):
        """Simulate ALS similarity with 10+ rating filter."""
        embeddings, ids = mock_embeddings

        mock_metadata.loc[:, "book_num_ratings"] = np.random.randint(0, 100, size=len(ids))

        index = SimilarityIndex.create_filtered_index(
            embeddings, ids, mock_metadata, min_rating_count=10, normalize=True
        )

        low_rated_book = mock_metadata[mock_metadata["book_num_ratings"] < 10].index[0]

        scores, results = index.search(low_rated_book, k=20)

        for result_id in results:
            assert mock_metadata.loc[result_id, "book_num_ratings"] >= 10

    def test_hybrid_mode_filtering_scenario(self, mock_embeddings, mock_metadata):
        """Simulate hybrid similarity with 5+ rating filter."""
        embeddings, ids = mock_embeddings

        mock_metadata.loc[:, "book_num_ratings"] = np.random.randint(0, 100, size=len(ids))

        index = SimilarityIndex.create_filtered_index(
            embeddings, ids, mock_metadata, min_rating_count=5, normalize=True
        )

        query_book = ids[0]
        scores, results = index.search(query_book, k=20)

        for result_id in results:
            assert mock_metadata.loc[result_id, "book_num_ratings"] >= 5

    def test_subject_mode_no_filtering_scenario(self, mock_embeddings, mock_metadata):
        """Simulate subject similarity with no filtering."""
        embeddings, ids = mock_embeddings

        index = SimilarityIndex.create_filtered_index(
            embeddings, ids, mock_metadata, min_rating_count=0, normalize=True
        )

        assert index.num_candidates == index.num_total

        scores, results = index.search(ids[0], k=20)
        assert len(results) > 0
