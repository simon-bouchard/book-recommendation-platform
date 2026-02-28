# models/data/__init__.py
"""
Data loading layer providing functions to load model artifacts and query databases.
"""

from models.data.loaders import (
    load_book_subject_embeddings,
    load_als_factors,
    load_bayesian_scores,
    load_book_meta,
    load_user_meta,
    load_book_to_subjects,
    load_attention_strategy,
    normalize_embeddings,
    get_item_idx_to_row,
)

from models.data.queries import (
    get_read_books,
    get_candidate_book_df,
    filter_read_books,
    add_book_embeddings,
    compute_subject_overlap,
    decompose_embeddings,
    clean_row,
)

__all__ = [
    # Loaders
    "load_book_subject_embeddings",
    "load_als_embeddings",
    "load_bayesian_scores",
    "load_book_meta",
    "load_user_meta",
    "load_book_to_subjects",
    "load_gbt_cold_model",
    "load_gbt_warm_model",
    "load_attention_strategy",
    "normalize_embeddings",
    "get_item_idx_to_row",
    # Queries
    "get_read_books",
    "get_candidate_book_df",
    "filter_read_books",
    "add_book_embeddings",
    "compute_subject_overlap",
    "decompose_embeddings",
    "clean_row",
]
