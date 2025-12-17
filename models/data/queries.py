# models/data/queries.py
"""
Database queries and DataFrame operations for recommendation pipeline.
Provides utilities for filtering, enriching, and transforming candidate data.
"""

from typing import Set, List, Dict
import numpy as np
import pandas as pd
import torch
from sqlalchemy.orm import Session

from models.core import PAD_IDX
from models.data.loaders import (
    load_book_subject_embeddings,
    load_book_meta,
    get_item_idx_to_row,
)

try:
    from app.table_models import Interaction
except ImportError:
    Interaction = None


def get_read_books(user_id: int, db: Session) -> Set[int]:
    """
    Get set of item_idx that a user has already interacted with.

    Args:
        user_id: User ID to query
        db: SQLAlchemy database session

    Returns:
        Set of item indices the user has read or rated
    """
    if Interaction is None:
        return set()

    return {
        row.item_idx
        for row in db.query(Interaction.item_idx).filter(Interaction.user_id == user_id).all()
    }


def get_candidate_book_df(candidate_ids: List[int]) -> pd.DataFrame:
    """
    Get DataFrame of book metadata for candidate items.
    Preserves order of candidate_ids in the output.

    Args:
        candidate_ids: List of item indices to retrieve

    Returns:
        DataFrame with book metadata, ordered by candidate_ids
    """
    book_meta = load_book_meta()

    # Select only candidates that exist in metadata
    df = book_meta.loc[book_meta.index.intersection(candidate_ids)].copy()

    # Add item_idx as column for convenience
    df["item_idx"] = df.index

    # Preserve original candidate order
    df["__sort"] = df["item_idx"].map({idx: i for i, idx in enumerate(candidate_ids)})
    df = df.sort_values("__sort").drop(columns="__sort")

    return df.reset_index(drop=True)


def filter_read_books(df: pd.DataFrame, user_id: int, db: Session) -> pd.DataFrame:
    """
    Remove books that the user has already read or rated.

    Args:
        df: DataFrame with 'item_idx' column
        user_id: User ID to check against
        db: SQLAlchemy database session

    Returns:
        DataFrame with read books filtered out
    """
    read_items = get_read_books(user_id, db)
    return df[~df["item_idx"].isin(read_items)]


def add_book_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add book subject embeddings as columns to DataFrame.

    Args:
        df: DataFrame with 'item_idx' column

    Returns:
        DataFrame with additional 'book_emb_0', 'book_emb_1', ... columns
    """
    embeddings, _ = load_book_subject_embeddings()
    idx_to_row = get_item_idx_to_row(
        load_book_subject_embeddings()[1]  # Get book_ids
    )

    dim = embeddings.shape[1]

    # Build embedding matrix for candidates
    emb_matrix = []
    for item_idx in df["item_idx"]:
        row_idx = idx_to_row.get(item_idx)
        if row_idx is not None:
            emb_matrix.append(embeddings[row_idx])
        else:
            emb_matrix.append(np.zeros(dim))

    emb_matrix = np.array(emb_matrix)

    # Create DataFrame with embedding columns
    emb_df = pd.DataFrame(emb_matrix, columns=[f"book_emb_{i}" for i in range(dim)])

    return pd.concat([df.reset_index(drop=True), emb_df], axis=1)


def compute_subject_overlap(user_subjects: List[int], book_subjects: List[int]) -> int:
    """
    Compute number of overlapping subjects between user and book.

    Args:
        user_subjects: List of user's favorite subject indices
        book_subjects: List of book's subject indices

    Returns:
        Count of overlapping subjects
    """
    return len(set(user_subjects) & set(book_subjects))


def decompose_embeddings(tensor: torch.Tensor, prefix: str) -> Dict[str, float]:
    """
    Decompose embedding tensor into dictionary of individual dimensions.

    Args:
        tensor: Tensor to decompose (will be flattened)
        prefix: Prefix for dictionary keys (e.g., 'user_emb', 'book_emb')

    Returns:
        Dictionary with keys '{prefix}_0', '{prefix}_1', ...
    """
    arr = tensor.detach().cpu().numpy().flatten()
    return {f"{prefix}_{i}": float(arr[i]) for i in range(len(arr))}


def clean_row(row: Dict) -> Dict:
    """
    Clean dictionary row by replacing NaN and inf values with None.

    Args:
        row: Dictionary representing a data row

    Returns:
        Cleaned dictionary with NaN/inf replaced by None
    """
    return {
        k: None
        if (isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")))
        else v
        for k, v in row.items()
    }


def get_user_num_ratings(user_id: int) -> int:
    """
    Get number of ratings for a user from cached metadata.

    Args:
        user_id: User ID to query

    Returns:
        Number of ratings, or 0 if user not found
    """
    try:
        from models.data.loaders import load_user_meta

        meta = load_user_meta()

        if user_id in meta.index:
            val = meta.loc[user_id].get("user_num_ratings")
            return int(val) if val is not None else 0
    except Exception:
        pass

    return 0
