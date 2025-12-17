# tests/unit/models/data/test_queries.py
"""
Unit tests for models.data.queries module.
Tests query functions and DataFrame operations with real data.
"""

import pytest
import pandas as pd
import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.data.queries import (
    get_candidate_book_df,
    compute_subject_overlap,
    decompose_embeddings,
    clean_row,
    get_user_num_ratings,
    add_book_embeddings,
)
from models.data.loaders import load_book_meta


class TestGetCandidateBookDF:
    """Test getting candidate book metadata."""

    def test_returns_dataframe(self):
        """Should return pandas DataFrame."""
        # Get some real book IDs
        book_meta = load_book_meta()
        candidate_ids = list(book_meta.index[:10])

        df = get_candidate_book_df(candidate_ids)

        assert isinstance(df, pd.DataFrame)

    def test_preserves_candidate_order(self):
        """Should return books in the order specified by candidate_ids."""
        book_meta = load_book_meta()

        # Deliberately scrambled order
        all_ids = list(book_meta.index[:20])
        candidate_ids = [all_ids[5], all_ids[2], all_ids[15], all_ids[0]]

        df = get_candidate_book_df(candidate_ids)

        # Check order is preserved
        assert list(df["item_idx"]) == candidate_ids

    def test_filters_nonexistent_books(self):
        """Should only return books that exist in metadata."""
        book_meta = load_book_meta()
        valid_ids = list(book_meta.index[:5])

        # Mix valid and invalid IDs
        candidate_ids = valid_ids + [-99999, -88888]

        df = get_candidate_book_df(candidate_ids)

        # Should only return valid books
        assert len(df) == len(valid_ids)
        assert all(idx in valid_ids for idx in df["item_idx"])

    def test_includes_item_idx_column(self):
        """Should include item_idx as a column."""
        book_meta = load_book_meta()
        candidate_ids = list(book_meta.index[:5])

        df = get_candidate_book_df(candidate_ids)

        assert "item_idx" in df.columns

    def test_handles_empty_candidate_list(self):
        """Should handle empty candidate list gracefully."""
        df = get_candidate_book_df([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestAddBookEmbeddings:
    """Test adding book embeddings to DataFrame."""

    def test_adds_embedding_columns(self):
        """Should add book_emb_0, book_emb_1, ... columns."""
        book_meta = load_book_meta()
        candidate_ids = list(book_meta.index[:5])
        df = get_candidate_book_df(candidate_ids)

        df_with_embs = add_book_embeddings(df)

        # Check embedding columns exist
        emb_cols = [col for col in df_with_embs.columns if col.startswith("book_emb_")]
        assert len(emb_cols) > 0

    def test_embedding_dimension_matches_loaded_embeddings(self):
        """Number of embedding columns should match embedding dimension."""
        from models.data.loaders import load_book_subject_embeddings

        embeddings, _ = load_book_subject_embeddings()
        expected_dim = embeddings.shape[1]

        book_meta = load_book_meta()
        candidate_ids = list(book_meta.index[:5])
        df = get_candidate_book_df(candidate_ids)

        df_with_embs = add_book_embeddings(df)

        emb_cols = [col for col in df_with_embs.columns if col.startswith("book_emb_")]
        assert len(emb_cols) == expected_dim

    def test_handles_books_without_embeddings(self):
        """Should handle books that don't have embeddings (use zero vector)."""
        # Create fake DataFrame with potentially missing book
        df = pd.DataFrame(
            {
                "item_idx": [-99999],  # Invalid book ID
                "title": ["Test Book"],
            }
        )

        df_with_embs = add_book_embeddings(df)

        # Should not crash, should add zero embeddings
        emb_cols = [col for col in df_with_embs.columns if col.startswith("book_emb_")]
        assert len(emb_cols) > 0


class TestComputeSubjectOverlap:
    """Test computing subject overlap between user and book."""

    def test_returns_integer(self):
        """Should return integer count."""
        overlap = compute_subject_overlap([1, 2, 3], [2, 3, 4])

        assert isinstance(overlap, int)

    def test_counts_overlapping_subjects(self):
        """Should count subjects that appear in both lists."""
        user_subjects = [1, 2, 3, 4]
        book_subjects = [3, 4, 5, 6]

        overlap = compute_subject_overlap(user_subjects, book_subjects)

        assert overlap == 2  # 3 and 4 overlap

    def test_handles_no_overlap(self):
        """Should return 0 when no overlap."""
        overlap = compute_subject_overlap([1, 2, 3], [4, 5, 6])

        assert overlap == 0

    def test_handles_complete_overlap(self):
        """Should count all when complete overlap."""
        subjects = [1, 2, 3]
        overlap = compute_subject_overlap(subjects, subjects)

        assert overlap == 3

    def test_handles_empty_lists(self):
        """Should return 0 for empty lists."""
        assert compute_subject_overlap([], [1, 2, 3]) == 0
        assert compute_subject_overlap([1, 2, 3], []) == 0
        assert compute_subject_overlap([], []) == 0

    def test_handles_duplicates_correctly(self):
        """Should count unique subjects only."""
        user_subjects = [1, 1, 2, 2, 3]
        book_subjects = [1, 2, 2, 4, 4]

        overlap = compute_subject_overlap(user_subjects, book_subjects)

        # Should be 2 (1 and 2), not 4 (counting duplicates)
        assert overlap == 2


class TestDecomposeEmbeddings:
    """Test decomposing embeddings into dictionary."""

    def test_returns_dict(self):
        """Should return dictionary."""
        tensor = torch.randn(1, 64)
        result = decompose_embeddings(tensor, prefix="test")

        assert isinstance(result, dict)

    def test_uses_correct_prefix(self):
        """Dictionary keys should use provided prefix."""
        tensor = torch.randn(1, 64)
        result = decompose_embeddings(tensor, prefix="user_emb")

        assert all(key.startswith("user_emb_") for key in result.keys())

    def test_has_correct_number_of_keys(self):
        """Should have one key per embedding dimension."""
        dim = 32
        tensor = torch.randn(1, dim)
        result = decompose_embeddings(tensor, prefix="test")

        assert len(result) == dim

    def test_keys_are_sequential(self):
        """Keys should be numbered sequentially from 0."""
        tensor = torch.randn(1, 10)
        result = decompose_embeddings(tensor, prefix="emb")

        expected_keys = [f"emb_{i}" for i in range(10)]
        assert sorted(result.keys()) == sorted(expected_keys)

    def test_values_are_floats(self):
        """Dictionary values should be float type."""
        tensor = torch.randn(1, 10)
        result = decompose_embeddings(tensor, prefix="test")

        assert all(isinstance(v, float) for v in result.values())

    def test_handles_multidimensional_tensor(self):
        """Should flatten multidimensional tensor."""
        tensor = torch.randn(2, 3, 4)  # 24 elements
        result = decompose_embeddings(tensor, prefix="test")

        assert len(result) == 24


class TestCleanRow:
    """Test cleaning dictionary rows of NaN/inf values."""

    def test_replaces_nan_with_none(self):
        """Should replace NaN with None."""
        row = {"score": float("nan"), "rating": 4.5}
        cleaned = clean_row(row)

        assert cleaned["score"] is None
        assert cleaned["rating"] == 4.5

    def test_replaces_inf_with_none(self):
        """Should replace inf with None."""
        row = {"score": float("inf"), "rating": 4.5}
        cleaned = clean_row(row)

        assert cleaned["score"] is None
        assert cleaned["rating"] == 4.5

    def test_replaces_neg_inf_with_none(self):
        """Should replace -inf with None."""
        row = {"score": float("-inf"), "rating": 4.5}
        cleaned = clean_row(row)

        assert cleaned["score"] is None
        assert cleaned["rating"] == 4.5

    def test_preserves_valid_floats(self):
        """Should keep valid float values."""
        row = {"a": 1.5, "b": -2.3, "c": 0.0}
        cleaned = clean_row(row)

        assert cleaned["a"] == 1.5
        assert cleaned["b"] == -2.3
        assert cleaned["c"] == 0.0

    def test_preserves_non_float_values(self):
        """Should keep non-float values unchanged."""
        row = {"str": "hello", "int": 42, "bool": True, "none": None}
        cleaned = clean_row(row)

        assert cleaned["str"] == "hello"
        assert cleaned["int"] == 42
        assert cleaned["bool"] is True
        assert cleaned["none"] is None

    def test_handles_empty_dict(self):
        """Should handle empty dictionary."""
        cleaned = clean_row({})

        assert cleaned == {}


class TestGetUserNumRatings:
    """Test getting user rating count from cached metadata."""

    def test_returns_integer(self):
        """Should return integer."""
        # Try with first user in metadata
        from models.data.loaders import load_user_meta

        user_meta = load_user_meta()

        if len(user_meta) > 0:
            user_id = user_meta.index[0]
            count = get_user_num_ratings(user_id)

            assert isinstance(count, int)

    def test_returns_zero_for_nonexistent_user(self):
        """Should return 0 for user not in metadata."""
        count = get_user_num_ratings(-99999)

        assert count == 0

    def test_returns_correct_count_for_known_user(self):
        """Should return actual rating count from metadata."""
        from models.data.loaders import load_user_meta

        user_meta = load_user_meta()

        if len(user_meta) > 0:
            user_id = user_meta.index[0]
            expected = user_meta.loc[user_id].get("user_num_ratings")

            if expected is not None:
                count = get_user_num_ratings(user_id)
                assert count == int(expected)
