# tests/integration/model_servers/test_embedder.py
"""
Contract and latency tests for the embedder model server (default port 8001).

Contract tests verify that the server adheres to the schemas defined in
contracts.py and that the mathematical properties of its outputs are correct.
Latency tests establish per-operation baselines at the model server level,
one layer below the application API latency suite.
"""

from __future__ import annotations

import pytest

from pydantic import ValidationError
from models.client.embedder import EmbedderClient

from ._utils import (
    TEST_SUBJECT_INDICES,
    assert_l2_normalized,
    measure_latency,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health_status_ok(embedder_client: EmbedderClient):
    """Health endpoint returns status=ok and the correct server identifier."""
    response = await embedder_client._client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["server"] == "embedder"
    assert "artifact_version" in data
    assert data["artifact_version"]


# ---------------------------------------------------------------------------
# Contract: /embed
# ---------------------------------------------------------------------------


async def test_embed_returns_l2_normalized_vector(embedder_client: EmbedderClient):
    """A valid embed request returns a non-empty L2-normalized vector."""
    response = await embedder_client.embed(TEST_SUBJECT_INDICES)

    assert len(response.vector) > 0
    assert_l2_normalized(response.vector)


async def test_embed_single_subject_index(embedder_client: EmbedderClient):
    """Embedding a single subject index returns a valid normalized vector."""
    response = await embedder_client.embed([TEST_SUBJECT_INDICES[0]])

    assert len(response.vector) > 0
    assert_l2_normalized(response.vector)


async def test_embed_dimension_is_consistent(embedder_client: EmbedderClient):
    """All embed calls return the same embedding dimension regardless of input size."""
    small = await embedder_client.embed(TEST_SUBJECT_INDICES[:1])
    large = await embedder_client.embed(TEST_SUBJECT_INDICES)

    assert len(small.vector) == len(large.vector)


async def test_embed_empty_list_is_rejected(embedder_client: EmbedderClient):
    """An empty subject index list is rejected by the Pydantic contract on the client side."""
    # EmbedRequest enforces min_length=1, so the client raises ValidationError
    # before issuing any HTTP request. This is intentional: the contract rejects
    # invalid input at the earliest possible point.
    with pytest.raises(ValidationError):
        await embedder_client.embed([])


# ---------------------------------------------------------------------------
# Latency: /embed
# ---------------------------------------------------------------------------


async def test_embed_latency_small(embedder_client: EmbedderClient, performance_report):
    """Embedding latency for a small subject list (2 subjects)."""
    stats = await measure_latency(
        "embedder_embed_small",
        lambda: embedder_client.embed(TEST_SUBJECT_INDICES[:2]),
    )
    performance_report["embedder_embed_small"] = stats
    print(f"\n{stats}")


async def test_embed_latency_medium(embedder_client: EmbedderClient, performance_report):
    """Embedding latency for a medium subject list (5 subjects)."""
    stats = await measure_latency(
        "embedder_embed_medium",
        lambda: embedder_client.embed(TEST_SUBJECT_INDICES),
    )
    performance_report["embedder_embed_medium"] = stats
    print(f"\n{stats}")
