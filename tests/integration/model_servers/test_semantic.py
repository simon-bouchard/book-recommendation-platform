# tests/integration/model_servers/test_semantic.py
"""
Contract and latency tests for the semantic search model server
(default port 8005).

Requires the semantic server to be running with real artifacts loaded.
Tests verify response shape, score ordering, result count bounds, and
steady-state latency at two pool sizes.
"""

from __future__ import annotations

from models.client.semantic import SemanticClient

from ._utils import assert_scores_descending, measure_latency

_QUERIES = [
    "cozy mysteries set in a small town",
    "epic fantasy with magic systems",
    "science fiction space exploration",
]
_QUERY = _QUERIES[0]
_K = 50


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health_status_ok(semantic_client: SemanticClient):
    """Health endpoint returns status=ok and the correct server identifier."""
    response = await semantic_client._client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["server"] == "semantic"
    assert "artifact_version" in data
    assert data["artifact_version"]


# ---------------------------------------------------------------------------
# Contract: /semantic_search
# ---------------------------------------------------------------------------


async def test_semantic_search_returns_results(semantic_client: SemanticClient):
    """semantic_search returns a non-empty result list bounded by top_k."""
    response = await semantic_client.semantic_search(query=_QUERY, top_k=_K)

    assert 0 < len(response.results) <= _K


async def test_semantic_search_scores_are_descending(semantic_client: SemanticClient):
    """Results are sorted by score descending."""
    response = await semantic_client.semantic_search(query=_QUERY, top_k=_K)

    assert_scores_descending([item.score for item in response.results])


async def test_semantic_search_result_fields(semantic_client: SemanticClient):
    """Each result has an integer item_idx and a float score."""
    response = await semantic_client.semantic_search(query=_QUERY, top_k=10)

    for item in response.results:
        assert isinstance(item.item_idx, int)
        assert isinstance(item.score, float)


async def test_semantic_search_top_k_is_respected(semantic_client: SemanticClient):
    """Never returns more items than the requested top_k."""
    for top_k in [1, 10, 50, 200]:
        response = await semantic_client.semantic_search(query=_QUERY, top_k=top_k)
        assert len(response.results) <= top_k


async def test_semantic_search_no_duplicate_item_ids(semantic_client: SemanticClient):
    """Results contain no duplicate item_idx values."""
    response = await semantic_client.semantic_search(query=_QUERY, top_k=_K)

    item_ids = [item.item_idx for item in response.results]
    assert len(item_ids) == len(set(item_ids))


async def test_semantic_search_different_queries_return_different_results(
    semantic_client: SemanticClient,
):
    """Different queries produce different top results."""
    response_a = await semantic_client.semantic_search(query=_QUERIES[0], top_k=10)
    response_b = await semantic_client.semantic_search(query=_QUERIES[1], top_k=10)

    top_ids_a = {item.item_idx for item in response_a.results}
    top_ids_b = {item.item_idx for item in response_b.results}
    assert top_ids_a != top_ids_b


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


async def test_semantic_search_latency_k50(semantic_client: SemanticClient, performance_report):
    """semantic_search latency at top_k=50."""
    stats = await measure_latency(
        "semantic_search_k50",
        lambda: semantic_client.semantic_search(query=_QUERY, top_k=50),
    )
    performance_report["semantic_search_k50"] = stats
    print(f"\n{stats}")


async def test_semantic_search_latency_k200(semantic_client: SemanticClient, performance_report):
    """semantic_search latency at top_k=200."""
    stats = await measure_latency(
        "semantic_search_k200",
        lambda: semantic_client.semantic_search(query=_QUERY, top_k=200),
    )
    performance_report["semantic_search_k200"] = stats
    print(f"\n{stats}")
