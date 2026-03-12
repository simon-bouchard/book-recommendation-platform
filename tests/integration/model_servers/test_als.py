# tests/integration/model_servers/test_als.py
"""
Contract and latency tests for the ALS recommendation model server
(default port 8003).

Key invariants under test:
  - Warm users are correctly identified as warm.
  - Cold users are correctly identified as cold.
  - Cold users receive an empty results list, not an error.
  - Results for warm users are sorted by score descending.
  - The number of returned results never exceeds the requested k.
"""

from __future__ import annotations

import pytest

from models.client.als import AlsClient

from ._utils import (
    COLD_USER_IDS_WITHOUT_SUBJECTS,
    WARM_USER_IDS,
    measure_latency,
)

_WARM_USER = WARM_USER_IDS[0]
_COLD_USER = COLD_USER_IDS_WITHOUT_SUBJECTS[0]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health_status_ok(als_client: AlsClient):
    """Health endpoint returns status=ok and the correct server identifier."""
    response = await als_client._client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["server"] == "als"
    assert "artifact_version" in data
    assert data["artifact_version"]


# ---------------------------------------------------------------------------
# Contract: /has_als_user
# ---------------------------------------------------------------------------


async def test_has_als_user_warm_user_is_warm(als_client: AlsClient):
    """A known warm user is reported as is_warm=True with the correct user_id."""
    response = await als_client.has_als_user(_WARM_USER)

    assert response.user_id == _WARM_USER
    assert response.is_warm is True


async def test_has_als_user_cold_user_is_not_warm(als_client: AlsClient):
    """A known cold user is reported as is_warm=False with the correct user_id."""
    response = await als_client.has_als_user(_COLD_USER)

    assert response.user_id == _COLD_USER
    assert response.is_warm is False


# ---------------------------------------------------------------------------
# Contract: /als_recs
# ---------------------------------------------------------------------------


async def test_als_recs_warm_user_returns_results(als_client: AlsClient):
    """Warm users receive a non-empty recommendation list bounded by k."""
    response = await als_client.als_recs(_WARM_USER, k=200)

    assert 0 < len(response.results) <= 200


async def test_als_recs_warm_user_scores_descending(als_client: AlsClient):
    """Recommendations for warm users are sorted by score descending."""
    response = await als_client.als_recs(_WARM_USER, k=200)

    scores = [item.score for item in response.results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Scores not sorted descending at index {i}: {scores[i]:.4f} < {scores[i + 1]:.4f}"
        )


async def test_als_recs_cold_user_returns_empty_list(als_client: AlsClient):
    """Cold users receive an empty results list rather than an error response."""
    response = await als_client.als_recs(_COLD_USER, k=200)

    assert isinstance(response.results, list)
    assert len(response.results) == 0


@pytest.mark.parametrize("k", [10, 50, 200])
async def test_als_recs_k_is_respected(als_client: AlsClient, k: int):
    """als_recs never returns more items than the requested k."""
    response = await als_client.als_recs(_WARM_USER, k=k)

    assert len(response.results) <= k


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


async def test_has_als_user_latency(als_client: AlsClient, performance_report):
    """has_als_user latency (warm user lookup)."""
    stats = await measure_latency(
        "als_has_als_user_warm",
        lambda: als_client.has_als_user(_WARM_USER),
    )
    performance_report["als_has_als_user_warm"] = stats
    print(f"\n{stats}")


async def test_als_recs_latency_k200(als_client: AlsClient, performance_report):
    """als_recs latency for a warm user at k=200."""
    stats = await measure_latency(
        "als_recs_warm_k200",
        lambda: als_client.als_recs(_WARM_USER, k=200),
    )
    performance_report["als_recs_warm_k200"] = stats
    print(f"\n{stats}")


@pytest.mark.parametrize("k", [50, 200, 500])
async def test_als_recs_latency_varying_k(als_client: AlsClient, performance_report, k: int):
    """als_recs latency across k values to characterize scaling behaviour."""
    stats = await measure_latency(
        f"als_recs_warm_k{k}",
        lambda: als_client.als_recs(_WARM_USER, k=k),
    )
    performance_report[f"als_recs_warm_k{k}"] = stats
    print(f"\n{stats}")
