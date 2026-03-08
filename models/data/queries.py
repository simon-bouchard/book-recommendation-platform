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
    Get the full set of item_idx that a user has already interacted with.

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


def get_read_books_for_candidates(user_id: int, candidate_ids: List[int], db: Session) -> Set[int]:
    """
    Get the subset of candidate_ids that the user has already read or rated.

    Scoped to the candidate list via an IN clause.  With the composite index
    on (user_id, item_idx), MySQL performs a targeted seek per candidate ID
    rather than scanning the user's full interaction history.  This is faster
    than get_read_books() for warm users because it reads fewer index entries:
    at most len(candidate_ids) seeks versus the user's full interaction count.

    The composite index also makes this a covering index query — item_idx is
    stored in the index, so MySQL never touches the table rows.

    Args:
        user_id: User ID to query
        candidate_ids: Candidate item indices to check membership for
        db: SQLAlchemy database session

    Returns:
        Subset of candidate_ids the user has already interacted with
    """
    if Interaction is None or not candidate_ids:
        return set()

    return {
        row.item_idx
        for row in db.query(Interaction.item_idx)
        .filter(Interaction.user_id == user_id, Interaction.item_idx.in_(candidate_ids))
        .all()
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

    df = book_meta.loc[book_meta.index.intersection(candidate_ids)].copy()
    df["item_idx"] = df.index

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

    Uses vectorized operations for optimal performance — significantly faster
    than row-by-row iteration for large DataFrames.

    Args:
        df: DataFrame with 'item_idx' column

    Returns:
        DataFrame with additional 'book_emb_0', 'book_emb_1', ... columns
    """
    embeddings, book_ids = load_book_subject_embeddings()
    idx_to_row = get_item_idx_to_row(book_ids)

    dim = embeddings.shape[1]

    item_indices = df["item_idx"].map(idx_to_row).fillna(-1).astype(int).values
    emb_matrix = np.zeros((len(df), dim), dtype=embeddings.dtype)
    valid_mask = item_indices >= 0

    if valid_mask.any():
        emb_matrix[valid_mask] = embeddings[item_indices[valid_mask]]

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


def has_book_subjects(item_idx: int) -> bool:
    """
    Check if a book has valid subject embeddings.

    Args:
        item_idx: Book item index to check

    Returns:
        True if book has subject embeddings
    """
    try:
        from models.data.loaders import load_book_subject_embeddings

        _, book_ids = load_book_subject_embeddings(use_cache=True)
        return int(item_idx) in book_ids
    except Exception:
        return False


def has_book_als(item_idx: int) -> bool:
    """
    Check if a book has ALS factors.

    Args:
        item_idx: Book item index to check

    Returns:
        True if book has ALS factors
    """
    try:
        from models.infrastructure.als_model import ALSModel

        return ALSModel().has_book(item_idx)
    except Exception:
        return False
