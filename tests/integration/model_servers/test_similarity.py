# tests/integration/model_servers/test_similarity.py
"""
Contract and latency tests for the unified similarity model server
(default port 8002).

Covers all five operations: subject_sim, als_sim, hybrid_sim, subject_recs,
and has_book_als. Tests that require ALS factors for the query book gate on
has_book_als and skip cleanly when the book has no ALS representation.
"""

from __future__ import annotations

import pytest

from models.client.embedder import EmbedderClient
from models.client.similarity import SimilarityClient

from ._utils import (
    TEST_BOOK_IDS,
    TEST_SUBJECT_INDICES,
    assert_scores_descending,
    measure_latency,
)

_QUERY_BOOK = TEST_BOOK_IDS[0]
_K = 50


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health_status_ok(similarity_client: SimilarityClient):
    """Health endpoint returns status=ok and the correct server identifier."""
    response = await similarity_client._client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["server"] == "similarity"
    assert "artifact_version" in data
    assert data["artifact_version"]


# ---------------------------------------------------------------------------
# Contract: /has_book_als
# ---------------------------------------------------------------------------


async def test_has_book_als_returns_correct_item_idx(similarity_client: SimilarityClient):
    """has_book_als echoes the requested item_idx and returns a boolean flag."""
    response = await similarity_client.has_book_als(_QUERY_BOOK)

    assert response.item_idx == _QUERY_BOOK
    assert isinstance(response.has_als, bool)


# ---------------------------------------------------------------------------
# Contract: /subject_sim
# ---------------------------------------------------------------------------


async def test_subject_sim_returns_ordered_results(similarity_client: SimilarityClient):
    """subject_sim returns a non-empty list of results bounded by k."""
    response = await similarity_client.subject_sim(_QUERY_BOOK, k=_K)

    assert 0 < len(response.results) <= _K


async def test_subject_sim_scores_are_descending(similarity_client: SimilarityClient):
    """subject_sim results are sorted by score descending."""
    response = await similarity_client.subject_sim(_QUERY_BOOK, k=_K)

    assert_scores_descending([item.score for item in response.results])


async def test_subject_sim_excludes_query_book(similarity_client: SimilarityClient):
    """The query book itself is not present in its own subject similarity results."""
    response = await similarity_client.subject_sim(_QUERY_BOOK, k=_K)

    assert _QUERY_BOOK not in {item.item_idx for item in response.results}


# ---------------------------------------------------------------------------
# Contract: /als_sim
# ---------------------------------------------------------------------------


async def test_als_sim_returns_ordered_results(similarity_client: SimilarityClient):
    """als_sim returns an ordered result list for a book with ALS factors."""
    has_als = await similarity_client.has_book_als(_QUERY_BOOK)
    if not has_als.has_als:
        pytest.skip(f"Book {_QUERY_BOOK} has no ALS factors")

    response = await similarity_client.als_sim(_QUERY_BOOK, k=_K)

    assert 0 < len(response.results) <= _K
    assert_scores_descending([item.score for item in response.results])


async def test_als_sim_excludes_query_book(similarity_client: SimilarityClient):
    """The query book itself is not present in its own ALS similarity results."""
    has_als = await similarity_client.has_book_als(_QUERY_BOOK)
    if not has_als.has_als:
        pytest.skip(f"Book {_QUERY_BOOK} has no ALS factors")

    response = await similarity_client.als_sim(_QUERY_BOOK, k=_K)

    assert _QUERY_BOOK not in {item.item_idx for item in response.results}


# ---------------------------------------------------------------------------
# Contract: /hybrid_sim
# ---------------------------------------------------------------------------


async def test_hybrid_sim_default_alpha_returns_results(similarity_client: SimilarityClient):
    """hybrid_sim with alpha=0.6 returns an ordered result list."""
    has_als = await similarity_client.has_book_als(_QUERY_BOOK)
    if not has_als.has_als:
        pytest.skip(f"Book {_QUERY_BOOK} has no ALS factors")

    response = await similarity_client.hybrid_sim(_QUERY_BOOK, k=_K, alpha=0.6)

    assert 0 < len(response.results) <= _K
    assert_scores_descending([item.score for item in response.results])


async def test_hybrid_sim_pure_subject_alpha(similarity_client: SimilarityClient):
    """alpha=0.0 (pure subject) still returns a valid ordered result list."""
    response = await similarity_client.hybrid_sim(_QUERY_BOOK, k=_K, alpha=0.0)

    assert len(response.results) > 0
    assert_scores_descending([item.score for item in response.results])


async def test_hybrid_sim_pure_als_alpha(similarity_client: SimilarityClient):
    """alpha=1.0 (pure ALS) still returns a valid ordered result list."""
    has_als = await similarity_client.has_book_als(_QUERY_BOOK)
    if not has_als.has_als:
        pytest.skip(f"Book {_QUERY_BOOK} has no ALS factors")

    response = await similarity_client.hybrid_sim(_QUERY_BOOK, k=_K, alpha=1.0)

    assert len(response.results) > 0
    assert_scores_descending([item.score for item in response.results])


# ---------------------------------------------------------------------------
# Contract: /subject_recs
# ---------------------------------------------------------------------------


async def test_subject_recs_returns_ordered_results(
    embedder_client: EmbedderClient,
    similarity_client: SimilarityClient,
):
    """subject_recs with a real user vector returns an ordered result list bounded by k."""
    embed_response = await embedder_client.embed(TEST_SUBJECT_INDICES)
    recs_response = await similarity_client.subject_recs(
        user_vector=embed_response.vector,
        k=_K,
        alpha=0.6,
    )

    assert 0 < len(recs_response.results) <= _K
    assert_scores_descending([item.score for item in recs_response.results])


async def test_subject_recs_k_is_respected(
    embedder_client: EmbedderClient,
    similarity_client: SimilarityClient,
):
    """subject_recs never returns more items than the requested k."""
    embed_response = await embedder_client.embed(TEST_SUBJECT_INDICES)

    for k in [10, 50, 200]:
        response = await similarity_client.subject_recs(
            user_vector=embed_response.vector,
            k=k,
            alpha=0.6,
        )
        assert len(response.results) <= k


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


async def test_subject_sim_latency_k200(similarity_client: SimilarityClient, performance_report):
    """subject_sim latency at k=200."""
    stats = await measure_latency(
        "similarity_subject_sim_k200",
        lambda: similarity_client.subject_sim(_QUERY_BOOK, k=200),
    )
    performance_report["similarity_subject_sim_k200"] = stats
    print(f"\n{stats}")


async def test_als_sim_latency_k200(similarity_client: SimilarityClient, performance_report):
    """als_sim latency at k=200."""
    has_als = await similarity_client.has_book_als(_QUERY_BOOK)
    if not has_als.has_als:
        pytest.skip(f"Book {_QUERY_BOOK} has no ALS factors")

    stats = await measure_latency(
        "similarity_als_sim_k200",
        lambda: similarity_client.als_sim(_QUERY_BOOK, k=200),
    )
    performance_report["similarity_als_sim_k200"] = stats
    print(f"\n{stats}")


async def test_hybrid_sim_latency_k200(similarity_client: SimilarityClient, performance_report):
    """hybrid_sim latency at k=200, alpha=0.6."""
    has_als = await similarity_client.has_book_als(_QUERY_BOOK)
    if not has_als.has_als:
        pytest.skip(f"Book {_QUERY_BOOK} has no ALS factors")

    stats = await measure_latency(
        "similarity_hybrid_sim_k200",
        lambda: similarity_client.hybrid_sim(_QUERY_BOOK, k=200, alpha=0.6),
    )
    performance_report["similarity_hybrid_sim_k200"] = stats
    print(f"\n{stats}")


async def test_subject_recs_latency_k200(
    embedder_client: EmbedderClient,
    similarity_client: SimilarityClient,
    performance_report,
):
    """subject_recs latency at k=200. The user vector is computed once outside the loop."""
    embed_response = await embedder_client.embed(TEST_SUBJECT_INDICES)
    vector = embed_response.vector

    stats = await measure_latency(
        "similarity_subject_recs_k200",
        lambda: similarity_client.subject_recs(user_vector=vector, k=200, alpha=0.6),
    )
    performance_report["similarity_subject_recs_k200"] = stats
    print(f"\n{stats}")
