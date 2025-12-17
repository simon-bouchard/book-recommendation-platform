# models/data/loaders.py
"""
Functions to load model artifacts, embeddings, and trained models.
Replaces the monolithic ModelStore with focused, cacheable loading functions.
"""

import json
import pickle
from typing import Tuple, Optional, Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd

from models.core import PATHS, Config


# Module-level cache for large artifacts that shouldn't be reloaded
_CACHE: Dict[str, any] = {}


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings along the last axis.

    Args:
        embeddings: Array of shape (N, D) to normalize

    Returns:
        Normalized embeddings as float32 array
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return (embeddings / norms).astype(np.float32)


def get_item_idx_to_row(item_ids: List[int]) -> Dict[int, int]:
    """
    Create mapping from item_idx to row index in embedding matrix.

    Args:
        item_ids: List of item indices

    Returns:
        Dictionary mapping item_idx -> row index
    """
    return {idx: i for i, idx in enumerate(item_ids)}


def load_book_subject_embeddings(
    normalized: bool = False, use_cache: bool = True
) -> Tuple[np.ndarray, List[int]]:
    """
    Load attention-pooled book embeddings derived from subjects.

    Args:
        normalized: If True, return L2-normalized embeddings
        use_cache: If True, cache loaded embeddings in memory

    Returns:
        Tuple of (embeddings array, list of item_idx)

    Raises:
        FileNotFoundError: If embedding files don't exist
    """
    cache_key = "book_subject_embeddings"
    cache_key_norm = "book_subject_embeddings_normalized"

    if use_cache and cache_key in _CACHE:
        embeddings, ids = _CACHE[cache_key]
    else:
        embeddings = np.load(PATHS.book_subject_embeddings)
        with open(PATHS.book_subject_ids, "r") as f:
            ids = json.load(f)

        if len(embeddings) != len(ids):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(ids)} IDs")

        if use_cache:
            _CACHE[cache_key] = (embeddings, ids)

    if normalized:
        if use_cache and cache_key_norm in _CACHE:
            return _CACHE[cache_key_norm], ids

        norm_embeddings = normalize_embeddings(embeddings)
        if use_cache:
            _CACHE[cache_key_norm] = norm_embeddings
        return norm_embeddings, ids

    return embeddings, ids


def load_als_embeddings(
    normalized: bool = False, use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    """
    Load ALS collaborative filtering latent factors.

    Args:
        normalized: If True, return L2-normalized factors
        use_cache: If True, cache loaded factors in memory

    Returns:
        Tuple of:
            - user_factors: (n_users, n_factors) array
            - book_factors: (n_books, n_factors) array
            - user_id_to_row: Dict mapping user_id -> row index
            - book_row_to_item_idx: Dict mapping row index -> item_idx

    Raises:
        FileNotFoundError: If ALS files don't exist
    """
    cache_key = "als_embeddings"
    cache_key_norm = "als_embeddings_normalized"

    if use_cache and cache_key in _CACHE:
        user_factors, book_factors, user_map, book_map = _CACHE[cache_key]
    else:
        user_factors = np.load(PATHS.user_als_factors)
        book_factors = np.load(PATHS.book_als_factors)

        with open(PATHS.user_als_ids, "r") as f:
            user_ids = json.load(f)
        with open(PATHS.book_als_ids, "r") as f:
            book_ids = json.load(f)

        user_map = {uid: i for i, uid in enumerate(user_ids)}
        book_map = {i: item_idx for i, item_idx in enumerate(book_ids)}

        if use_cache:
            _CACHE[cache_key] = (user_factors, book_factors, user_map, book_map)

    if normalized:
        if use_cache and cache_key_norm in _CACHE:
            user_norm, book_norm = _CACHE[cache_key_norm]
            return user_norm, book_norm, user_map, book_map

        user_norm = normalize_embeddings(user_factors)
        book_norm = normalize_embeddings(book_factors)

        if use_cache:
            _CACHE[cache_key_norm] = (user_norm, book_norm)

        return user_norm, book_norm, user_map, book_map

    return user_factors, book_factors, user_map, book_map


def has_book_als(item_idx: int) -> bool:
    """
    Check if a book has ALS factors available.

    Args:
        item_idx: Book item index to check

    Returns:
        True if book has ALS factors, False otherwise
    """
    cache_key = "als_book_id_set"

    if cache_key not in _CACHE:
        with open(PATHS.book_als_ids, "r") as f:
            book_ids = json.load(f)
        _CACHE[cache_key] = set(int(i) for i in book_ids)

    return int(item_idx) in _CACHE[cache_key]


def load_bayesian_scores(use_cache: bool = True) -> np.ndarray:
    """
    Load precomputed Bayesian popularity scores.

    Args:
        use_cache: If True, cache loaded scores in memory

    Returns:
        Array of Bayesian scores aligned with book_subject_ids

    Raises:
        FileNotFoundError: If scores file doesn't exist
    """
    cache_key = "bayesian_scores"

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    scores = np.load(PATHS.bayesian_scores)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if use_cache:
        _CACHE[cache_key] = scores

    return scores


def load_book_meta(use_cache: bool = True) -> pd.DataFrame:
    """
    Load book metadata with precomputed Bayesian scores.

    Args:
        use_cache: If True, cache loaded metadata in memory

    Returns:
        DataFrame indexed by item_idx with book metadata and Bayesian scores

    Raises:
        FileNotFoundError: If training data doesn't exist
    """
    cache_key = "book_meta"

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    df = pd.read_pickle(PATHS.training_books).set_index("item_idx")

    # Add Bayesian scores aligned to item_idx
    bayesian_scores = load_bayesian_scores(use_cache=use_cache)
    _, book_ids = load_book_subject_embeddings(use_cache=use_cache)
    idx_to_row = get_item_idx_to_row(book_ids)

    df["bayes"] = df.index.to_series().map(
        lambda idx: float(bayesian_scores[idx_to_row.get(int(idx), -1)])
        if idx_to_row.get(int(idx)) is not None
        else float("-inf")
    )

    # Stable sort by bayes desc, title asc
    df = df.sort_values(["bayes", "title"], ascending=[False, True], kind="mergesort")

    if use_cache:
        _CACHE[cache_key] = df

    return df


def load_user_meta(use_cache: bool = True) -> pd.DataFrame:
    """
    Load user metadata.

    Args:
        use_cache: If True, cache loaded metadata in memory

    Returns:
        DataFrame indexed by user_id with user metadata

    Raises:
        FileNotFoundError: If training data doesn't exist
    """
    cache_key = "user_meta"

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    df = pd.read_pickle(PATHS.training_users).set_index("user_id")

    if use_cache:
        _CACHE[cache_key] = df

    return df


def load_book_to_subjects(use_cache: bool = True) -> Dict[int, List[int]]:
    """
    Load book-to-subjects mapping.

    Args:
        use_cache: If True, cache loaded mapping in memory

    Returns:
        Dictionary mapping item_idx -> list of subject_idx

    Raises:
        FileNotFoundError: If training data doesn't exist
    """
    cache_key = "book_to_subjects"

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    mapping = defaultdict(list)
    df = pd.read_pickle(PATHS.training_book_subjects)

    for row in df.itertuples(index=False):
        mapping[row.item_idx].append(row.subject_idx)

    if use_cache:
        _CACHE[cache_key] = dict(mapping)

    return mapping


def load_gbt_cold_model(use_cache: bool = True):
    """
    Load gradient boosted tree model for cold-start users.

    Args:
        use_cache: If True, cache loaded model in memory

    Returns:
        Trained LGBMRegressor model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    cache_key = "gbt_cold"

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    with open(PATHS.gbt_cold, "rb") as f:
        model = pickle.load(f)

    if use_cache:
        _CACHE[cache_key] = model

    return model


def load_gbt_warm_model(use_cache: bool = True):
    """
    Load gradient boosted tree model for warm-start users.

    Args:
        use_cache: If True, cache loaded model in memory

    Returns:
        Trained LGBMRegressor model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    cache_key = "gbt_warm"

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    with open(PATHS.gbt_warm, "rb") as f:
        model = pickle.load(f)

    if use_cache:
        _CACHE[cache_key] = model

    return model


def load_attention_strategy(strategy: Optional[str] = None, use_cache: bool = True):
    """
    Load attention pooling strategy from saved components.

    Args:
        strategy: One of 'scalar', 'perdim', 'selfattn', 'selfattn_perdim'.
                 If None, uses ATTN_STRATEGY environment variable.
        use_cache: If True, cache loaded strategy in memory

    Returns:
        Instantiated attention strategy ready for inference

    Raises:
        ValueError: If strategy is invalid
        FileNotFoundError: If strategy components file doesn't exist
    """
    if strategy is None:
        strategy = Config.get_attention_strategy()

    cache_key = f"attention_strategy_{strategy}"

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    # Import here to avoid circular dependency
    from models.subject_attention_strategy import STRATEGY_REGISTRY

    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown attention strategy: {strategy}. "
            f"Available: {', '.join(STRATEGY_REGISTRY.keys())}"
        )

    strategy_class = STRATEGY_REGISTRY[strategy]
    path = PATHS.get_attention_path(strategy)

    if not path.exists():
        raise FileNotFoundError(f"Attention strategy file not found: {path}")

    instance = strategy_class(path=str(path))

    if use_cache:
        _CACHE[cache_key] = instance

    return instance


def clear_cache():
    """Clear all cached artifacts to free memory."""
    _CACHE.clear()


def preload_all_artifacts():
    """
    Preload all artifacts into cache for faster subsequent access.
    Useful for warming up the application at startup.
    """
    load_book_subject_embeddings(normalized=False, use_cache=True)
    load_book_subject_embeddings(normalized=True, use_cache=True)
    load_als_embeddings(normalized=False, use_cache=True)
    load_als_embeddings(normalized=True, use_cache=True)
    load_bayesian_scores(use_cache=True)
    load_book_meta(use_cache=True)
    load_user_meta(use_cache=True)
    load_book_to_subjects(use_cache=True)
    load_gbt_warm_model(use_cache=True)
    load_attention_strategy(use_cache=True)
