# tests/load/models/test_profile.py
"""
Flamegraph profiling for the recommendation and similarity pipelines under
real concurrent load against a live Gunicorn server.

py-spy is attached to an actual Gunicorn worker PID (discovered at runtime via
psutil) rather than the pytest process. This captures the full application call
graph including GIL contention around numpy/FAISS operations, inter-process
communication costs, and thread-pool dispatch overhead — none of which are
visible when profiling via ASGITransport in the integration test suite.

All profiles are written in Speedscope JSON format to a timestamped directory
under tests/load/models/profiles/<YYYYMMDD_HHMMSS>/. Open them at
https://speedscope.app — drag and drop, no install needed.

Usage:
    export PERF_TEST_BASE_URL=http://localhost:8000
    pytest tests/load/models/test_profile.py -v -s

    # Profile only warm user path:
    pytest tests/load/models/test_profile.py -v -s -k "warm_behavioral"

Requirements:
    pip install py-spy psutil --break-system-packages
    py-spy requires sudo on Linux: run pytest under sudo, or grant the
    necessary capabilities with:
        sudo setcap cap_sys_ptrace=eip $(which py-spy)
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
from typing import Generator
from urllib.parse import urlparse

import httpx
import pytest

from tests.load.models._constants import (
    COLD_WITHOUT_SUBJECTS_USER_IDS,
    COLD_WITH_SUBJECTS_USER_IDS,
    SUSTAINED_DURATION_S,
    SUSTAINED_WORKERS,
    TEST_BOOK_IDS,
    WARM_USER_IDS,
    WARMUP_RUNS,
)
from tests.load.models.test_load import (
    _make_recommend_factory,
    _make_similarity_factory,
    _run_sustained,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")

_PROFILES_DIR = Path(__file__).parent / "profiles"
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_RUN_DIR = _PROFILES_DIR / _TIMESTAMP

_PROFILE_WARMUP = 5
_PROFILE_WORKERS = SUSTAINED_WORKERS
_PROFILE_DURATION_S = SUSTAINED_DURATION_S

_PYSPY_BIN: str | None = shutil.which("py-spy")


# ---------------------------------------------------------------------------
# Module-level guards
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def require_pyspy() -> None:
    """Skip the entire module if py-spy is not on PATH."""
    if _PYSPY_BIN is None:
        pytest.skip("py-spy not found. Install with: pip install py-spy --break-system-packages")


# ---------------------------------------------------------------------------
# Worker PID discovery
# ---------------------------------------------------------------------------


def _find_gunicorn_worker_pid(base_url: str) -> int:
    """
    Locate a Gunicorn worker PID to attach py-spy to.

    Scans active network connections for the server port, identifies the
    Gunicorn master process, and returns the PID of its first worker child.
    Falls back to returning the master PID directly if no children are found
    (e.g. a single-process uvicorn dev server).

    Raises RuntimeError if no matching process is found, which surfaces a
    clear message rather than a cryptic py-spy failure.
    """
    import psutil

    port = urlparse(base_url).port or 80

    listening_pids: set[int] = set()
    for conn in psutil.net_connections(kind="inet"):
        if conn.status == "LISTEN" and conn.laddr.port == port and conn.pid is not None:
            listening_pids.add(conn.pid)

    for pid in listening_pids:
        try:
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline()).lower()
            if "gunicorn" not in cmdline and "uvicorn" not in cmdline:
                continue
            workers = proc.children(recursive=False)
            if workers:
                return workers[0].pid
            return pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    raise RuntimeError(
        f"No Gunicorn or Uvicorn process found listening on port {port}. "
        f"Ensure the server is running and PERF_TEST_BASE_URL ({base_url}) is correct."
    )


# ---------------------------------------------------------------------------
# py-spy context manager
# ---------------------------------------------------------------------------


@contextmanager
def _pyspy_record(
    label: str,
    worker_pid: int,
    duration: int,
) -> Generator[Path, None, None]:
    """
    Context manager that runs py-spy record for the lifetime of the block.

    Attaches to worker_pid — a real Gunicorn worker — rather than the pytest
    process. All profiles from a single test run are grouped under _RUN_DIR so
    they can be loaded together in Speedscope for side-by-side comparison.

    Args:
        label:      Filename stem for the output JSON (e.g. "warm_behavioral").
        worker_pid: PID of the Gunicorn worker to attach to.
        duration:   Maximum recording time in seconds. Should be comfortably
                    above the total request window so py-spy does not stop
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
        str(worker_pid),
        "--output",
        str(output_path),
        "--format",
        "speedscope",
        "--duration",
        str(duration),
        "--nonblocking",
    ]

    proc = subprocess.Popen(cmd)
    time.sleep(1.5)

    try:
        yield output_path
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
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


def _print_stats(label: str, stats: dict) -> None:
    print(
        f"\n  {label}: "
        f"mean={stats.get('mean_ms', 0):.1f}ms  "
        f"p95={stats.get('p95_ms', 0):.1f}ms  "
        f"rps={stats.get('throughput_rps', 0):.1f}  "
        f"failures={stats.get('failures', 0)}"
    )


# ---------------------------------------------------------------------------
# Profile tests — recommendations
# ---------------------------------------------------------------------------


async def test_profile_warm_behavioral(
    http_client: httpx.AsyncClient,
    base_url: str,
) -> None:
    """
    Flamegraph for the warm ALS behavioral recommendation path under load.

    Expected hot frames: ALSBasedGenerator.generate -> AlsClient._post,
    ReadBooksFilter.apply, MetadataClient._post. Any unexpectedly wide frame
    is a bottleneck candidate. This is the highest-traffic production path and
    should be the first profile examined after any change to the ALS pipeline.
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs configured")

    worker_pid = _find_gunicorn_worker_pid(base_url)
    factory = _make_recommend_factory(http_client, WARM_USER_IDS, "behavioral")

    for _ in range(_PROFILE_WARMUP):
        await factory()

    duration = _PROFILE_DURATION_S + 15
    with _pyspy_record("warm_behavioral", worker_pid, duration):
        stats = await _run_sustained(
            factory, _PROFILE_WORKERS, _PROFILE_DURATION_S, 0, "profile_warm_behavioral"
        )

    _print_stats("warm_behavioral", stats.get_stats())


async def test_profile_warm_subject(
    http_client: httpx.AsyncClient,
    base_url: str,
) -> None:
    """
    Flamegraph for warm users on the subject path under load.

    Comparing this profile against warm_behavioral isolates cost that is
    path-specific versus shared (filter, enrich). Both profiles use the same
    user IDs so differences are attributable solely to candidate generation.
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs configured")

    worker_pid = _find_gunicorn_worker_pid(base_url)
    factory = _make_recommend_factory(http_client, WARM_USER_IDS, "subject")

    for _ in range(_PROFILE_WARMUP):
        await factory()

    duration = _PROFILE_DURATION_S + 15
    with _pyspy_record("warm_subject", worker_pid, duration):
        stats = await _run_sustained(
            factory, _PROFILE_WORKERS, _PROFILE_DURATION_S, 0, "profile_warm_subject"
        )

    _print_stats("warm_subject", stats.get_stats())


async def test_profile_cold_with_subjects(
    http_client: httpx.AsyncClient,
    base_url: str,
) -> None:
    """
    Flamegraph for the cold subject path (embed + similarity + Bayesian blend).

    Expected hot frames: SubjectEmbedder.embed -> EmbedderClient._post,
    SimilarityIndex.query -> SimilarityClient._post. The UI sends mode=subject
    for all cold users; this is the most compute-intensive cold path.
    """
    if not COLD_WITH_SUBJECTS_USER_IDS:
        pytest.skip("No cold users with subjects configured")

    worker_pid = _find_gunicorn_worker_pid(base_url)
    factory = _make_recommend_factory(http_client, COLD_WITH_SUBJECTS_USER_IDS, "subject")

    for _ in range(_PROFILE_WARMUP):
        await factory()

    duration = _PROFILE_DURATION_S + 15
    with _pyspy_record("cold_with_subjects", worker_pid, duration):
        stats = await _run_sustained(
            factory, _PROFILE_WORKERS, _PROFILE_DURATION_S, 0, "profile_cold_with_subjects"
        )

    _print_stats("cold_with_subjects", stats.get_stats())


async def test_profile_cold_without_subjects(
    http_client: httpx.AsyncClient,
    base_url: str,
) -> None:
    """
    Flamegraph for the Bayesian popularity fallback path.

    With no subject embeddings to score against, the pipeline skips FAISS and
    returns pre-computed Bayesian scores directly. This profile sets the lower
    bound for cold-user latency and should show almost no ML frames.
    """
    if not COLD_WITHOUT_SUBJECTS_USER_IDS:
        pytest.skip("No cold users without subjects configured")

    worker_pid = _find_gunicorn_worker_pid(base_url)
    factory = _make_recommend_factory(http_client, COLD_WITHOUT_SUBJECTS_USER_IDS, "subject")

    for _ in range(_PROFILE_WARMUP):
        await factory()

    duration = _PROFILE_DURATION_S + 15
    with _pyspy_record("cold_without_subjects", worker_pid, duration):
        stats = await _run_sustained(
            factory, _PROFILE_WORKERS, _PROFILE_DURATION_S, 0, "profile_cold_without_subjects"
        )

    _print_stats("cold_without_subjects", stats.get_stats())


# ---------------------------------------------------------------------------
# Profile tests — similarity
# ---------------------------------------------------------------------------


async def test_profile_similarity_subject(
    http_client: httpx.AsyncClient,
    base_url: str,
) -> None:
    """
    Flamegraph for subject similarity search under load.

    Expected hot frames: FAISS index query, metadata enrichment. This is the
    cheapest similarity path and forms the baseline floor for similarity latency.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    worker_pid = _find_gunicorn_worker_pid(base_url)
    factory = _make_similarity_factory(http_client, "subject")

    for _ in range(_PROFILE_WARMUP):
        await factory()

    duration = _PROFILE_DURATION_S + 15
    with _pyspy_record("similarity_subject", worker_pid, duration):
        stats = await _run_sustained(
            factory, _PROFILE_WORKERS, _PROFILE_DURATION_S, 0, "profile_similarity_subject"
        )

    _print_stats("similarity_subject", stats.get_stats())


async def test_profile_similarity_als(
    http_client: httpx.AsyncClient,
    base_url: str,
) -> None:
    """
    Flamegraph for ALS item-factor similarity search under load.

    Expected hot frames: AlsSimilarityStrategy -> AlsClient._post -> metadata
    enrichment. Comparing against similarity_subject isolates the cost of ALS
    factor retrieval from the similarity server.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    worker_pid = _find_gunicorn_worker_pid(base_url)
    factory = _make_similarity_factory(http_client, "als")

    for _ in range(_PROFILE_WARMUP):
        await factory()

    duration = _PROFILE_DURATION_S + 15
    with _pyspy_record("similarity_als", worker_pid, duration):
        stats = await _run_sustained(
            factory, _PROFILE_WORKERS, _PROFILE_DURATION_S, 0, "profile_similarity_als"
        )

    _print_stats("similarity_als", stats.get_stats())


async def test_profile_similarity_hybrid(
    http_client: httpx.AsyncClient,
    base_url: str,
) -> None:
    """
    Flamegraph for hybrid similarity (FAISS + ALS score fusion) under load.

    Expected to show both FAISS and ALS frames plus the fusion step.
    A throughput_rps disproportionately lower than subject or ALS alone
    indicates that the fusion step itself carries meaningful overhead beyond
    simply running both retrievals sequentially.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    worker_pid = _find_gunicorn_worker_pid(base_url)
    factory = _make_similarity_factory(http_client, "hybrid")

    for _ in range(_PROFILE_WARMUP):
        await factory()

    duration = _PROFILE_DURATION_S + 15
    with _pyspy_record("similarity_hybrid", worker_pid, duration):
        stats = await _run_sustained(
            factory, _PROFILE_WORKERS, _PROFILE_DURATION_S, 0, "profile_similarity_hybrid"
        )

    _print_stats("similarity_hybrid", stats.get_stats())
