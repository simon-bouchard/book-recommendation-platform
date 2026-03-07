# tests/integration/model_servers/test_metadata.py
"""
Contract and latency tests for the metadata model server (default port 8004).

Key invariants under test:
  - enrich returns valid BookMeta for known item indices.
  - enrich silently omits item indices not present in the metadata store.
  - popular returns exactly k books sorted by Bayesian score descending.
  - All returned books carry a non-empty title.
"""

from __future__ import annotations

import pytest

from models.client.metadata import MetadataClient

from ._utils import TEST_BOOK_IDS, measure_latency

_SMALL_BATCH = TEST_BOOK_IDS[:5]
_LARGE_BATCH = TEST_BOOK_IDS
_UNKNOWN_ITEM_IDX = 999_999_999


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health_status_ok(metadata_client: MetadataClient):
    """Health endpoint returns status=ok and the correct server identifier."""
    response = await metadata_client._client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["server"] == "metadata"
    assert "artifact_version" in data
    assert data["artifact_version"]


# ---------------------------------------------------------------------------
# Contract: /enrich
# ---------------------------------------------------------------------------


async def test_enrich_returns_books_with_required_fields(metadata_client: MetadataClient):
    """enrich returns metadata for known books with required fields populated."""
    response = await metadata_client.enrich(_SMALL_BATCH)

    assert len(response.books) > 0
    for book in response.books:
        assert book.item_idx in _SMALL_BATCH
        assert isinstance(book.title, str)
        assert len(book.title) > 0


async def test_enrich_unknown_items_are_silently_omitted(metadata_client: MetadataClient):
    """Item indices absent from the metadata store are omitted, not errored."""
    response = await metadata_client.enrich(_SMALL_BATCH + [_UNKNOWN_ITEM_IDX])

    result_ids = {book.item_idx for book in response.books}
    assert _UNKNOWN_ITEM_IDX not in result_ids
    assert len(response.books) <= len(_SMALL_BATCH)


async def test_enrich_large_batch_succeeds(metadata_client: MetadataClient):
    """enrich handles the full test batch (20 books) without error."""
    response = await metadata_client.enrich(_LARGE_BATCH)

    assert isinstance(response.books, list)
    assert len(response.books) > 0


# ---------------------------------------------------------------------------
# Contract: /popular
# ---------------------------------------------------------------------------


async def test_popular_returns_exactly_k_books(metadata_client: MetadataClient):
    """popular(k) returns exactly k books when the catalog is large enough."""
    k = 10
    response = await metadata_client.popular(k=k)

    assert len(response.books) == k


async def test_popular_scores_are_descending(metadata_client: MetadataClient):
    """Popular books are sorted by Bayesian score descending."""
    response = await metadata_client.popular(k=20)

    scores = [b.bayes_score for b in response.books if b.bayes_score is not None]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Popular scores not sorted descending at index {i}: "
            f"{scores[i]:.4f} < {scores[i + 1]:.4f}"
        )


async def test_popular_all_books_have_titles(metadata_client: MetadataClient):
    """Every book returned by popular has a non-empty title."""
    response = await metadata_client.popular(k=10)

    for book in response.books:
        assert isinstance(book.title, str)
        assert len(book.title) > 0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


async def test_enrich_latency_small_batch(metadata_client: MetadataClient, performance_report):
    """enrich latency for a small batch (5 books)."""
    stats = await measure_latency(
        "metadata_enrich_5",
        lambda: metadata_client.enrich(_SMALL_BATCH),
    )
    performance_report["metadata_enrich_5"] = stats
    print(f"\n{stats}")


async def test_enrich_latency_large_batch(metadata_client: MetadataClient, performance_report):
    """enrich latency for the full test batch (20 books)."""
    stats = await measure_latency(
        "metadata_enrich_20",
        lambda: metadata_client.enrich(_LARGE_BATCH),
    )
    performance_report["metadata_enrich_20"] = stats
    print(f"\n{stats}")


async def test_popular_latency_k100(metadata_client: MetadataClient, performance_report):
    """popular latency at k=100."""
    stats = await measure_latency(
        "metadata_popular_k100",
        lambda: metadata_client.popular(k=100),
    )
    performance_report["metadata_popular_k100"] = stats
    print(f"\n{stats}")
