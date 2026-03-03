# tests/unit/models/infrastructure/conftest.py
"""
Shared fixtures for infrastructure layer unit tests.

Constants are module-level so tests can reference them directly without
fixture injection overhead. Fixtures provide numpy arrays and ID lists
that multiple test modules need, avoiding duplication across files.

Data layout:
    - 50 total books (book_ids 1000–1049)
    - First 35 have ALS factors; last 15 do not (als_row_for_subject == -1)
    - Rating counts: first 25 books have >= 10 ratings, last 25 have < 10
"""

import numpy as np
import pytest

N_BOOKS: int = 50
N_ALS_BOOKS: int = 35
EMB_DIM: int = 16
BOOK_ID_OFFSET: int = 1000


@pytest.fixture
def book_ids() -> list[int]:
    """50 sequential book item indices."""
    return list(range(BOOK_ID_OFFSET, BOOK_ID_OFFSET + N_BOOKS))


@pytest.fixture
def normalized_embeddings() -> np.ndarray:
    """
    L2-normalized subject embeddings of shape (N_BOOKS, EMB_DIM).

    Seeded so tests are deterministic. Row-normalized so dot products
    equal cosine similarities directly.
    """
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((N_BOOKS, EMB_DIM)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


@pytest.fixture
def als_factors_normalized() -> np.ndarray:
    """
    L2-normalized ALS factors for the first N_ALS_BOOKS books, shape (N_ALS_BOOKS, EMB_DIM).

    Uses a different seed from normalized_embeddings so subject and ALS
    spaces are not correlated — this makes alpha-blending tests meaningful.
    """
    rng = np.random.default_rng(99)
    factors = rng.standard_normal((N_ALS_BOOKS, EMB_DIM)).astype(np.float32)
    norms = np.linalg.norm(factors, axis=1, keepdims=True)
    return factors / norms


@pytest.fixture
def als_row_for_subject() -> np.ndarray:
    """
    Alignment map from subject row to ALS row, shape (N_BOOKS,).

    Books 0–34 (subject rows) map directly to ALS rows 0–34.
    Books 35–49 have no ALS factors and are assigned -1.
    """
    mapping = np.full(N_BOOKS, -1, dtype=np.int32)
    mapping[:N_ALS_BOOKS] = np.arange(N_ALS_BOOKS, dtype=np.int32)
    return mapping


@pytest.fixture
def bayesian_scores_raw() -> np.ndarray:
    """
    Raw Bayesian popularity scores of shape (N_BOOKS,).

    Values are in [0, 1) with deliberate spread so top-k retrieval
    tests have unambiguous expected orderings.
    """
    rng = np.random.default_rng(7)
    return rng.random(N_BOOKS).astype(np.float32)


@pytest.fixture
def bayesian_scores_norm(bayesian_scores_raw: np.ndarray) -> np.ndarray:
    """
    Min-max normalized Bayesian scores of shape (N_BOOKS,).

    Derived from bayesian_scores_raw so both fixtures are consistent
    when used together in SubjectScorer tests.
    """
    lo, hi = bayesian_scores_raw.min(), bayesian_scores_raw.max()
    return ((bayesian_scores_raw - lo) / (hi - lo)).astype(np.float32)


@pytest.fixture
def rating_counts() -> np.ndarray:
    """
    Integer rating counts of shape (N_BOOKS,).

    First 25 books: >= 10 ratings (qualify as candidates at threshold=10).
    Last 25 books:  < 10 ratings (filtered out at threshold=10).

    This split is deliberate so filtering tests have a known expected
    candidate count without needing to re-derive it from the data.
    """
    counts = np.zeros(N_BOOKS, dtype=np.int32)
    counts[:25] = np.arange(10, 35, dtype=np.int32)
    counts[25:] = np.zeros(N_BOOKS - 25, dtype=np.int32)
    return counts
