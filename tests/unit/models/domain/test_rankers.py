# tests/unit/models/domain/test_rankers.py
"""
Unit tests for recommendation rankers.
Tests ranking logic with various candidate orderings.
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.domain.rankers import Ranker, NoOpRanker, ScoreRanker
from models.domain.recommendation import Candidate
from models.domain.user import User


@pytest.fixture
def mock_user():
    """Create a mock user."""
    return User(
        user_id=123,
        fav_subjects=[5, 12, 23],
        country="US",
        age=25,
    )


@pytest.fixture
def sorted_candidates():
    """Create candidates already sorted by score (descending)."""
    return [
        Candidate(item_idx=100, score=0.9, source="test"),
        Candidate(item_idx=101, score=0.8, source="test"),
        Candidate(item_idx=102, score=0.7, source="test"),
        Candidate(item_idx=103, score=0.6, source="test"),
        Candidate(item_idx=104, score=0.5, source="test"),
    ]


@pytest.fixture
def unsorted_candidates():
    """Create candidates in random order."""
    return [
        Candidate(item_idx=102, score=0.7, source="test"),
        Candidate(item_idx=100, score=0.9, source="test"),
        Candidate(item_idx=104, score=0.5, source="test"),
        Candidate(item_idx=101, score=0.8, source="test"),
        Candidate(item_idx=103, score=0.6, source="test"),
    ]


@pytest.fixture
def tied_score_candidates():
    """Create candidates with tied scores."""
    return [
        Candidate(item_idx=100, score=0.9, source="test"),
        Candidate(item_idx=101, score=0.8, source="test"),
        Candidate(item_idx=102, score=0.8, source="test"),  # Tied with 101
        Candidate(item_idx=103, score=0.8, source="test"),  # Tied with 101, 102
        Candidate(item_idx=104, score=0.5, source="test"),
    ]


class TestNoOpRanker:
    """Test NoOpRanker preserves original ordering."""

    def test_implements_ranker_protocol(self):
        """Should implement Ranker protocol."""
        ranker = NoOpRanker()

        assert hasattr(ranker, "rank")
        assert callable(ranker.rank)

    def test_preserves_original_order(self, sorted_candidates, mock_user):
        """Should return candidates in original order."""
        ranker = NoOpRanker()
        ranked = ranker.rank(sorted_candidates, mock_user)

        assert [c.item_idx for c in ranked] == [c.item_idx for c in sorted_candidates]

    def test_returns_same_list_object(self, sorted_candidates, mock_user):
        """Should return the same list object (no copy)."""
        ranker = NoOpRanker()
        ranked = ranker.rank(sorted_candidates, mock_user)

        assert ranked is sorted_candidates

    def test_does_not_modify_candidates(self, sorted_candidates, mock_user):
        """Should not modify candidate attributes."""
        original_scores = [c.score for c in sorted_candidates]

        ranker = NoOpRanker()
        ranked = ranker.rank(sorted_candidates, mock_user)

        assert [c.score for c in ranked] == original_scores

    def test_handles_empty_list(self, mock_user):
        """Should handle empty candidate list."""
        ranker = NoOpRanker()
        ranked = ranker.rank([], mock_user)

        assert ranked == []

    def test_handles_single_candidate(self, mock_user):
        """Should handle single candidate."""
        candidates = [Candidate(item_idx=100, score=0.9, source="test")]

        ranker = NoOpRanker()
        ranked = ranker.rank(candidates, mock_user)

        assert len(ranked) == 1
        assert ranked[0].item_idx == 100


class TestScoreRanker:
    """Test ScoreRanker sorts by score descending."""

    def test_implements_ranker_protocol(self):
        """Should implement Ranker protocol."""
        ranker = ScoreRanker()

        assert hasattr(ranker, "rank")
        assert callable(ranker.rank)

    def test_sorts_by_score_descending(self, unsorted_candidates, mock_user):
        """Should sort candidates by score (highest first)."""
        ranker = ScoreRanker()
        ranked = ranker.rank(unsorted_candidates, mock_user)

        scores = [c.score for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_correct_order_after_sorting(self, unsorted_candidates, mock_user):
        """Should produce correct item order after sorting."""
        ranker = ScoreRanker()
        ranked = ranker.rank(unsorted_candidates, mock_user)

        # Expected order: 100 (0.9), 101 (0.8), 102 (0.7), 103 (0.6), 104 (0.5)
        assert [c.item_idx for c in ranked] == [100, 101, 102, 103, 104]

    def test_preserves_already_sorted_order(self, sorted_candidates, mock_user):
        """Should preserve order if already sorted."""
        ranker = ScoreRanker()
        ranked = ranker.rank(sorted_candidates, mock_user)

        assert [c.item_idx for c in ranked] == [c.item_idx for c in sorted_candidates]

    def test_stable_sort_with_tied_scores(self, mock_user):
        """Should maintain relative order for tied scores (stable sort)."""
        # Create candidates with specific order and tied scores
        candidates = [
            Candidate(item_idx=100, score=0.8, source="test"),  # First with 0.8
            Candidate(item_idx=101, score=0.8, source="test"),  # Second with 0.8
            Candidate(item_idx=102, score=0.8, source="test"),  # Third with 0.8
        ]

        ranker = ScoreRanker()
        ranked = ranker.rank(candidates, mock_user)

        # Stable sort should preserve original relative order for ties
        assert [c.item_idx for c in ranked] == [100, 101, 102]

    def test_handles_empty_list(self, mock_user):
        """Should handle empty candidate list."""
        ranker = ScoreRanker()
        ranked = ranker.rank([], mock_user)

        assert ranked == []

    def test_handles_single_candidate(self, mock_user):
        """Should handle single candidate."""
        candidates = [Candidate(item_idx=100, score=0.9, source="test")]

        ranker = ScoreRanker()
        ranked = ranker.rank(candidates, mock_user)

        assert len(ranked) == 1
        assert ranked[0].item_idx == 100

    def test_does_not_modify_original_list(self, unsorted_candidates, mock_user):
        """Should not modify the original candidate list."""
        original_order = [c.item_idx for c in unsorted_candidates]

        ranker = ScoreRanker()
        ranked = ranker.rank(unsorted_candidates, mock_user)

        # Original list should be unchanged
        assert [c.item_idx for c in unsorted_candidates] == original_order

        # But ranked list should be different
        assert [c.item_idx for c in ranked] != original_order

    def test_returns_new_list(self, sorted_candidates, mock_user):
        """Should return a new list, not the same object."""
        ranker = ScoreRanker()
        ranked = ranker.rank(sorted_candidates, mock_user)

        assert ranked is not sorted_candidates


class TestRankerComparison:
    """Compare behavior of different rankers."""

    def test_noop_vs_score_ranker_with_sorted_input(self, sorted_candidates, mock_user):
        """NoOp and Score ranker should produce same result for sorted input."""
        noop_ranker = NoOpRanker()
        score_ranker = ScoreRanker()

        noop_result = noop_ranker.rank(sorted_candidates, mock_user)
        score_result = score_ranker.rank(sorted_candidates, mock_user)

        # Same order
        assert [c.item_idx for c in noop_result] == [c.item_idx for c in score_result]

    def test_noop_vs_score_ranker_with_unsorted_input(self, unsorted_candidates, mock_user):
        """NoOp and Score ranker produce different results for unsorted input."""
        noop_ranker = NoOpRanker()
        score_ranker = ScoreRanker()

        noop_result = noop_ranker.rank(unsorted_candidates, mock_user)
        score_result = score_ranker.rank(unsorted_candidates, mock_user)

        # Different order
        assert [c.item_idx for c in noop_result] != [c.item_idx for c in score_result]

        # NoOp preserves original order
        assert [c.item_idx for c in noop_result] == [c.item_idx for c in unsorted_candidates]

        # ScoreRanker sorts by score
        assert [c.item_idx for c in score_result] == [100, 101, 102, 103, 104]


class TestEdgeCases:
    """Test edge cases for rankers."""

    def test_candidate_rejects_negative_scores(self, mock_user):
        """Candidate model should reject negative scores."""
        # Candidate enforces non-negative scores in __post_init__
        with pytest.raises(ValueError, match="Score must be non-negative"):
            Candidate(item_idx=100, score=-0.5, source="test")

    def test_score_ranker_handles_zero_scores(self, mock_user):
        """Should handle candidates with zero scores."""
        candidates = [
            Candidate(item_idx=100, score=0.0, source="test"),
            Candidate(item_idx=101, score=0.5, source="test"),
            Candidate(item_idx=102, score=0.0, source="test"),
        ]

        ranker = ScoreRanker()
        ranked = ranker.rank(candidates, mock_user)

        # 0.5 first, then two zeros (stable sort preserves order)
        assert ranked[0].item_idx == 101
        assert {ranked[1].item_idx, ranked[2].item_idx} == {100, 102}

    def test_score_ranker_handles_identical_scores(self, mock_user):
        """Should handle all candidates with identical scores."""
        candidates = [
            Candidate(item_idx=100, score=0.7, source="test"),
            Candidate(item_idx=101, score=0.7, source="test"),
            Candidate(item_idx=102, score=0.7, source="test"),
        ]

        ranker = ScoreRanker()
        ranked = ranker.rank(candidates, mock_user)

        # Stable sort should preserve original order
        assert [c.item_idx for c in ranked] == [100, 101, 102]

    def test_score_ranker_handles_very_large_scores(self, mock_user):
        """Should handle very large score values."""
        candidates = [
            Candidate(item_idx=100, score=1e6, source="test"),
            Candidate(item_idx=101, score=1e9, source="test"),
            Candidate(item_idx=102, score=1e3, source="test"),
        ]

        ranker = ScoreRanker()
        ranked = ranker.rank(candidates, mock_user)

        assert [c.item_idx for c in ranked] == [101, 100, 102]

    def test_score_ranker_handles_very_small_scores(self, mock_user):
        """Should handle very small score values."""
        candidates = [
            Candidate(item_idx=100, score=1e-6, source="test"),
            Candidate(item_idx=101, score=1e-9, source="test"),
            Candidate(item_idx=102, score=1e-3, source="test"),
        ]

        ranker = ScoreRanker()
        ranked = ranker.rank(candidates, mock_user)

        assert [c.item_idx for c in ranked] == [102, 100, 101]

    def test_rankers_preserve_candidate_sources(self, unsorted_candidates, mock_user):
        """Rankers should preserve source attribution."""
        noop_ranker = NoOpRanker()
        score_ranker = ScoreRanker()

        noop_result = noop_ranker.rank(unsorted_candidates, mock_user)
        score_result = score_ranker.rank(unsorted_candidates, mock_user)

        # All should still have source="test"
        assert all(c.source == "test" for c in noop_result)
        assert all(c.source == "test" for c in score_result)

    def test_rankers_preserve_all_candidate_attributes(self, mock_user):
        """Rankers should preserve all candidate attributes."""
        candidates = [
            Candidate(item_idx=100, score=0.9, source="source_a"),
            Candidate(item_idx=101, score=0.7, source="source_b"),
            Candidate(item_idx=102, score=0.8, source="source_c"),
        ]

        score_ranker = ScoreRanker()
        ranked = score_ranker.rank(candidates, mock_user)

        # Check all attributes preserved
        assert ranked[0].item_idx == 100
        assert ranked[0].score == 0.9
        assert ranked[0].source == "source_a"

        assert ranked[1].item_idx == 102
        assert ranked[1].score == 0.8
        assert ranked[1].source == "source_c"

        assert ranked[2].item_idx == 101
        assert ranked[2].score == 0.7
        assert ranked[2].source == "source_b"
