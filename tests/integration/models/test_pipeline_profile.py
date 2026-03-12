# tests/integration/models/test_pipeline_profile.py
"""
Flamegraph profiling for the recommendation and similarity pipeline using py-spy.

Attaches py-spy to the current process while firing the same request sequences
as the performance test suite. Each test run writes all profiles into a single
timestamped subdirectory under tests/integration/profiles/<YYYYMMDD_HHMMSS>/
so runs are never interleaved and are easy to locate and compare.

Open the .json files at https://speedscope.app — drag and drop, no install needed.

Usage:
    pytest tests/integration/models/test_pipeline_profile.py -v -s
    pytest tests/integration/models/test_pipeline_profile.py -v -s -k "warm"
    pytest tests/integration/models/test_pipeline_profile.py -v -s -k "similarity"

Requirements:
    pip install py-spy --break-system-packages
    sudo access required (py-spy attaches to a running process)
"""

import asyncio
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from tests.integration.models.test_models_performance import (
    COLD_WITH_SUBJECTS_USER_IDS,
    COLD_WITHOUT_SUBJECTS_USER_IDS,
    TEST_BOOK_IDS,
    WARM_USER_IDS,
    measure_endpoint_latency,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")

_PROFILES_DIR = Path(__file__).parent / "profiles"
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_RUN_DIR = _PROFILES_DIR / _TIMESTAMP

_PROFILE_WARMUP = 3
_PROFILE_REQUESTS = 20

# Resolve the full path under the current user's PATH so that sudo — which
# runs with a restricted PATH that excludes conda/venv bin directories —
# can still find the executable. None if py-spy is not installed.
_PYSPY_BIN: str | None = shutil.which("py-spy")


# ---------------------------------------------------------------------------
# Module-level py-spy availability check
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def require_pyspy():
    """
    Skip the entire module at collection time if py-spy is not on PATH.

    Checking here rather than inside the context manager means the skip
    reason is visible immediately and no warmup requests are wasted.
    """
    if _PYSPY_BIN is None:
        pytest.skip("py-spy not found — install with: pip install py-spy --break-system-packages")


# ---------------------------------------------------------------------------
# py-spy context manager
# ---------------------------------------------------------------------------


@contextmanager
def _pyspy_record(label: str, duration: int):
    """
    Context manager that runs py-spy record for the lifetime of the block.

    Attaches to os.getpid() — valid because ASGITransport executes the full
    application stack in the same process as pytest. All profiles from a
    single test session are written into the shared _RUN_DIR so they are
    grouped by run and easy to load together in Speedscope.

    Args:
        label: Short name used as the output filename stem (e.g. "warm_behavioral").
        duration: Maximum recording time in seconds. Set comfortably above
                  the expected total request duration so py-spy does not stop
                  early and truncate the profile.

    Yields:
        Path to the output .json file (exists after the context exits).
    """
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _RUN_DIR / f"{label}.json"

    cmd = [
        _PYSPY_BIN,
        "record",
        "--pid",
        str(os.getpid()),
        "--output",
        str(output_path),
        "--format",
        "speedscope",
        "--duration",
        str(duration),
        "--nonblocking",
    ]

    proc = subprocess.Popen(cmd)

    # Allow py-spy to fully initialize before the first request lands.
    time.sleep(1.5)

    try:
        yield output_path
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # SIGKILL prevents py-spy from writing the output file.
            # SIGINT triggers a clean shutdown that flushes the profile.
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        if output_path.exists():
            print(f"\n  Profile written -> {output_path}")
            print("  Open at https://speedscope.app")
        else:
            print(f"\n  Warning: profile file not written for label '{label}'")


def _print_stats(label: str, s: dict) -> None:
    print(
        f"\n  {label} — mean={s['mean_ms']:.1f}ms  p50={s['median_ms']:.1f}ms  p95={s['p95_ms']:.1f}ms"
    )


# ---------------------------------------------------------------------------
# Recommendation profile tests
# ---------------------------------------------------------------------------


async def test_profile_warm_behavioral(client: httpx.AsyncClient):
    """
    Flamegraph for the warm ALS (behavioral) recommendation path.

    Expected hot frames: ALSBasedGenerator.generate -> AlsClient._post,
    ReadBooksFilter.apply, MetadataClient._post. Any unexpectedly wide frame
    is the bottleneck. The UI always sends mode=behavioral for warm users so
    this is the highest-traffic production path.
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs configured")

    user_id = WARM_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "behavioral"}

    for _ in range(_PROFILE_WARMUP):
        await client.get(endpoint, params=params)

    with _pyspy_record("warm_behavioral", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = await measure_endpoint_latency(
            client, endpoint, params, warmup_runs=0, measurement_runs=_PROFILE_REQUESTS
        )

    _print_stats("warm_behavioral", stats.get_stats())


async def test_profile_warm_subject(client: httpx.AsyncClient):
    """
    Flamegraph for a warm user on the subject path.

    Comparing this profile against warm_behavioral isolates which cost is
    path-specific versus shared overhead (filter, enrich). Both profiles use
    the same user so any difference is attributable solely to the candidate
    generation strategy.
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs configured")

    user_id = WARM_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"}

    for _ in range(_PROFILE_WARMUP):
        await client.get(endpoint, params=params)

    with _pyspy_record("warm_subject", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = await measure_endpoint_latency(
            client, endpoint, params, warmup_runs=0, measurement_runs=_PROFILE_REQUESTS
        )

    _print_stats("warm_subject", stats.get_stats())


async def test_profile_cold_with_subjects(client: httpx.AsyncClient):
    """
    Flamegraph for the cold subject path (embed + subject_recs + enrich).

    The UI sends mode=subject for all cold users.
    """
    if not COLD_WITH_SUBJECTS_USER_IDS:
        pytest.skip("No cold users with subjects configured")

    user_id = COLD_WITH_SUBJECTS_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"}

    for _ in range(_PROFILE_WARMUP):
        await client.get(endpoint, params=params)

    with _pyspy_record("cold_with_subjects", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = await measure_endpoint_latency(
            client, endpoint, params, warmup_runs=0, measurement_runs=_PROFILE_REQUESTS
        )

    _print_stats("cold_with_subjects", stats.get_stats())


async def test_profile_cold_without_subjects(client: httpx.AsyncClient):
    """
    Flamegraph for the pure Bayesian fallback path.

    The UI sends mode=subject for cold users regardless of whether they have
    favorite subjects set. With no subject embeddings available, the pipeline
    falls back to Bayesian popularity — this profile isolates that fallback's
    cost.
    """
    if not COLD_WITHOUT_SUBJECTS_USER_IDS:
        pytest.skip("No cold users without subjects configured")

    user_id = COLD_WITHOUT_SUBJECTS_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"}

    for _ in range(_PROFILE_WARMUP):
        await client.get(endpoint, params=params)

    with _pyspy_record("cold_without_subjects", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = await measure_endpoint_latency(
            client, endpoint, params, warmup_runs=0, measurement_runs=_PROFILE_REQUESTS
        )

    _print_stats("cold_without_subjects", stats.get_stats())


# ---------------------------------------------------------------------------
# Similarity profile tests
# ---------------------------------------------------------------------------


async def test_profile_similarity_subject(client: httpx.AsyncClient):
    """
    Flamegraph for subject similarity search.

    Expected hot frames: SubjectSimilarityStrategy.get_similar_books ->
    FAISS index query -> metadata enrichment. This is the cheapest similarity
    path and sets the baseline floor for similarity latency.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    book_id = TEST_BOOK_IDS[0]
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": "subject", "top_k": 200}

    for _ in range(_PROFILE_WARMUP):
        await client.get(endpoint, params=params)

    with _pyspy_record("similarity_subject", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = await measure_endpoint_latency(
            client, endpoint, params, warmup_runs=0, measurement_runs=_PROFILE_REQUESTS
        )

    _print_stats("similarity_subject", stats.get_stats())


async def test_profile_similarity_als(client: httpx.AsyncClient):
    """
    Flamegraph for ALS-based item similarity search.

    Expected hot frames: AlsSimilarityStrategy -> AlsClient._post -> metadata
    enrichment. Comparing against similarity_subject reveals the cost
    attributable to ALS item-factor retrieval.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    book_id = TEST_BOOK_IDS[0]
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": "als", "top_k": 200}

    response = await client.get(endpoint, params=params)
    if response.status_code == 422:
        pytest.skip(f"Book {book_id} has no ALS data")

    for _ in range(_PROFILE_WARMUP):
        await client.get(endpoint, params=params)

    with _pyspy_record("similarity_als", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = await measure_endpoint_latency(
            client, endpoint, params, warmup_runs=0, measurement_runs=_PROFILE_REQUESTS
        )

    _print_stats("similarity_als", stats.get_stats())


async def test_profile_similarity_hybrid(client: httpx.AsyncClient):
    """
    Flamegraph for hybrid similarity (subject + ALS score fusion).

    Expected to show both FAISS and ALS frames plus the fusion step.
    Comparing against the subject and ALS profiles identifies whether the
    ~2x latency observed in performance tests is truly additive or whether
    there is additional overhead in the fusion path itself.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    book_id = TEST_BOOK_IDS[0]
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": "hybrid", "top_k": 200}

    response = await client.get(endpoint, params=params)
    if response.status_code == 422:
        pytest.skip(f"Book {book_id} has no ALS data, hybrid unavailable")

    for _ in range(_PROFILE_WARMUP):
        await client.get(endpoint, params=params)

    with _pyspy_record("similarity_hybrid", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = await measure_endpoint_latency(
            client, endpoint, params, warmup_runs=0, measurement_runs=_PROFILE_REQUESTS
        )

    _print_stats("similarity_hybrid", stats.get_stats())
