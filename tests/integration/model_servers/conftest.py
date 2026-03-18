# tests/integration/model_servers/conftest.py
"""
Pytest fixtures for model server integration tests.

Shared utilities, test data constants, and assertion helpers live in _utils.py.
This file is restricted to pytest fixtures only, since conftest.py cannot be
imported as a regular module by test files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

import pytest

from models.client.als import AlsClient
from models.client.embedder import EmbedderClient
from models.client.metadata import MetadataClient
from models.client.semantic import SemanticClient
from models.client.similarity import SimilarityClient

from ._utils import MEASUREMENT_RUNS, WARMUP_RUNS, LatencyStats


load_dotenv()


# ---------------------------------------------------------------------------
# Session-scoped result accumulation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def performance_results():
    """
    Session-scoped accumulator for all model server latency results.

    Yields a dict that individual tests populate via the performance_report
    fixture. After the session ends, the accumulated stats are written to a
    timestamped JSON file under performance_baselines/.
    """
    results: Dict[str, LatencyStats] = {}
    yield results

    output_dir = Path(__file__).parent / "performance_baselines"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"model_servers_{timestamp}.json"

    export_data = {
        "timestamp": timestamp,
        "config": {"warmup_runs": WARMUP_RUNS, "measurement_runs": MEASUREMENT_RUNS},
        "results": {name: stats.get_stats() for name, stats in results.items()},
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"MODEL SERVER BASELINES SAVED TO: {output_file}")
    print(f"{'=' * 80}")


@pytest.fixture
def performance_report(performance_results):
    """
    Test-scoped fixture that collects results from a single test and merges
    them into the session-scoped accumulator on teardown.
    """
    report: Dict[str, LatencyStats] = {}
    yield report

    for name, stats in report.items():
        performance_results[name] = stats

    print("\n" + "-" * 60)
    for name, stats in report.items():
        s = stats.get_stats()
        print(f"{name}: {s.get('mean_ms', 0):.2f}ms mean")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Client fixtures
#
# Function-scoped (the default). httpx.AsyncClient binds to the running event
# loop on first use. With asyncio_mode = "auto", each async test runs in its
# own event loop, so any client created in a previous test's loop is unusable
# in the next. A fresh client per test sidesteps this entirely.
#
# The constructor is synchronous and cheap. There is nothing worth sharing
# across tests except the TCP connection, and httpx reconnects transparently.
# ---------------------------------------------------------------------------


@pytest.fixture
async def embedder_client():
    """Per-test EmbedderClient, created and closed within the test's own loop."""
    client = EmbedderClient.from_env()
    yield client
    await client.aclose()


@pytest.fixture
async def similarity_client():
    """Per-test SimilarityClient, created and closed within the test's own loop."""
    client = SimilarityClient.from_env()
    yield client
    await client.aclose()


@pytest.fixture
async def als_client():
    """Per-test AlsClient, created and closed within the test's own loop."""
    client = AlsClient.from_env()
    yield client
    await client.aclose()


@pytest.fixture
async def metadata_client():
    """Per-test MetadataClient, created and closed within the test's own loop."""
    client = MetadataClient.from_env()
    yield client
    await client.aclose()


@pytest.fixture
async def semantic_client():
    """Per-test SemanticClient, created and closed within the test's own loop."""
    client = SemanticClient.from_env()
    yield client
    await client.aclose()
