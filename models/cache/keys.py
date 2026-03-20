# models/cache/keys.py
"""
Cache key builders for ML endpoints.
Provides type-safe, consistent key generation for similarity and recommendations.
"""

import hashlib
from typing import List


def hash_subjects(subject_idxs: List[int]) -> str:
    """
    Create stable hash of subject list for cache key.

    Sorts subjects before hashing so different orderings produce same key.
    This allows cache sharing between users with identical subject preferences.

    Args:
        subject_idxs: List of subject indices

    Returns:
        16-character hexadecimal hash

    Example:
        hash_subjects([5, 1, 12]) == hash_subjects([1, 5, 12])  # True
    """
    sorted_subjects = sorted(subject_idxs)
    subject_str = ",".join(map(str, sorted_subjects))
    return hashlib.md5(subject_str.encode()).hexdigest()[:16]


def popularity_key(k: int) -> str:
    """
    Build cache key for popularity requests.

    Args:
        k: Number of popular books requested

    Returns:
        Cache key string

    Example:
        popularity_key(100)
        # "ml:popular:100"
    """
    return f"ml:popular:{k}"


def similarity_key(item_idx: int, mode: str, k: int, alpha: float = None) -> str:
    """
    Build cache key for book similarity requests.

    Args:
        item_idx: Book item index
        mode: Similarity mode ("subject", "als", "hybrid")
        k: Number of similar books requested
        alpha: Blend weight for hybrid mode (ignored for other modes)

    Returns:
        Cache key string

    Examples:
        similarity_key(12345, "subject", 200)
        # "ml:sim:subject:12345:200"

        similarity_key(12345, "hybrid", 200, 0.6)
        # "ml:sim:hybrid:12345:200:0.6"
    """
    if mode == "hybrid" and alpha is not None:
        return f"ml:sim:hybrid:{item_idx}:{k}:{alpha}"
    else:
        return f"ml:sim:{mode}:{item_idx}:{k}"


def recommendation_key(
    mode: str,
    user_id: int = None,
    subject_idxs: List[int] = None,
    top_n: int = 200,
    w: float = None,
) -> str:
    """
    Build cache key for recommendation requests.

    Key structure depends on mode:
    - behavioral: Uses user_id (user-specific ALS recommendations)
    - subject: Uses subject_hash (shareable across users with same subjects)
    - auto: Uses user_id (decision logic is user-specific)

    Args:
        mode: Recommendation mode ("behavioral", "subject", "auto")
        user_id: User ID (required for behavioral and auto modes)
        subject_idxs: List of subject indices (required for subject mode)
        top_n: Number of recommendations requested
        w: Subject weight for hybrid blending (required for subject mode)

    Returns:
        Cache key string

    Examples:
        recommendation_key("behavioral", user_id=789, top_n=50)
        # "ml:rec:behavioral:789:50"

        recommendation_key("subject", subject_idxs=[1,5,12], top_n=50, w=0.6)
        # "ml:rec:subject:a3f2e1d9c4b8f7e6:50:0.6"

        recommendation_key("auto", user_id=789, top_n=50, w=0.6)
        # "ml:rec:auto:789:50:0.6"
    """
    if mode == "behavioral":
        if user_id is None:
            raise ValueError("user_id required for behavioral recommendations")
        return f"ml:rec:behavioral:{user_id}:{top_n}"

    elif mode == "subject":
        if subject_idxs is None:
            raise ValueError("subject_idxs required for subject recommendations")
        if w is None:
            raise ValueError("w (weight) required for subject recommendations")

        subjects_hash = hash_subjects(subject_idxs)
        return f"ml:rec:subject:{subjects_hash}:{top_n}:{w}"

    elif mode == "auto":
        if user_id is None:
            raise ValueError("user_id required for auto recommendations")
        if w is None:
            raise ValueError("w (weight) required for auto recommendations")
        return f"ml:rec:auto:{user_id}:{top_n}:{w}"

    else:
        raise ValueError(f"Unknown recommendation mode: {mode}")
