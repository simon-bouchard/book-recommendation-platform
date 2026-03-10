# tests/load/model_servers/test_profile.py
"""
Flamegraph profiling for model server workers under real concurrent load.

py-spy is attached directly to a model server Gunicorn worker PID (discovered
at runtime via psutil cmdline scanning) rather than the main application
process. The main app is pure async I/O — all CPU-bound work (numpy, FAISS,
ALS matrix operations) lives in the model server workers, which is where
profiling is meaningful.

Each test targets the model server whose worker does the actual compute for
that scenario:

  warm_behavioral       -> als        (ALS dot-product recommendation)
  warm_subject          -> similarity (FAISS subject_recs)
  cold_with_subjects    -> similarity (FAISS subject_recs + Bayesian blend)
  cold_without_subjects -> metadata   (Bayesian popularity lookup)
  similarity_subject    -> similarity (FAISS subject_sim)
  similarity_als        -> similarity (ALS item-factor similarity)
  similarity_hybrid     -> similarity (FAISS + ALS score fusion)

All profiles are written in Speedscope JSON format to a timestamped directory
under tests/load/model_servers/profiles/<YYYYMMDD_HHMMSS>/. Open them at
https://speedscope.app — drag and drop, no install needed.

Usage:
    export PERF_TEST_BASE_URL=http://localhost:8000
    pytest tests/load/model_servers/test_profile.py -v -s

    # Profile only one scenario:
    pytest tests/load/model_servers/test_profile.py -v -s -k "warm_behavioral"

Requirements:
    pip install py-spy psutil --break-system-packages
    py-spy requires sudo on Linux: run pytest under sudo, or grant the
    necessary capabilities with:
        sudo setcap cap_sys_ptrace=eip $(which py-spy)
"""

import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import httpx
import pytest

from tests.load.models._constants import (
    COLD_WITHOUT_SUBJECTS_USER_IDS,
    COLD_WITH_SUBJECTS_USER_IDS,
    SUSTAINED_DURATION_S,
    SUSTAINED_WORKERS,
    TEST_BOOK_IDS,
    WARM_USER_IDS,
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


def _find_model_server_worker_pid(server: str) -> int:
    """
    Locate a Gunicorn worker PID for the given model server.

    Scans all processes by cmdline looking for the pattern
    'model_servers.<server>.main:app'. Among matching processes, returns
    the PID of a worker child (a process whose parent also matches the
    pattern) rather than the master. Falls back to the master PID if no
    workers are found, which should not happen in normal operation.

    This approach is more reliable than scanning network connections because
    socket inheritance from forked children can cause unrelated processes to
    appear as listeners on the model server ports.

    Args:
        server: Model server name — one of: als, similarity, embedder, metadata.

    Raises:
        RuntimeError: If no matching Gunicorn process is found.
    """
    import psutil

    pattern = f"model_servers.{server}.main:app"

    master_pid: int | None = None
    worker_pid: int | None = None

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if pattern not in cmdline:
                continue
            parent = proc.parent()
            parent_cmdline = " ".join(parent.cmdline()) if parent else ""
            if pattern in parent_cmdline:
                # Parent also matches — this process is a worker child.
                worker_pid = proc.pid
                break
            else:
                master_pid = proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if worker_pid is not None:
        return worker_pid

    if master_pid is not None:
        try:
            children = psutil.Process(master_pid).children(recursive=False)
            if children:
                return children[0].pid
            return master_pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    raise RuntimeError(
        f"No Gunicorn worker found for model server '{server}'. "
        f"Ensure the model servers are running (docker compose up)."
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

    Attaches to the given worker PID for up to `duration` seconds. All
    profiles from a single pytest run are grouped under _RUN_DIR so they
    can be loaded together in Speedscope for side-by-side comparison.

    Args:
        label:      Filename stem for the output JSON (e.g. "warm_behavioral").
        worker_pid: PID of the model server Gunicorn worker to attach to.
        duration:   Maximum recording time in seconds. Should exceed the
                    total request window so py-spy does not stop early.

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
) -> None:
    """
    Flamegraph for the warm ALS behavioral recommendation path under load.

    Targets the ALS model server worker, where the dot-product recommendation
    computation runs. Expected hot frames: als_recs handler, numpy dot product,
    user/item factor lookups. This is the highest-traffic production path.
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs configured")

    worker_pid = _find_model_server_worker_pid("als")
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
) -> None:
    """
    Flamegraph for warm users on the subject path under load.

    Targets the similarity server worker, which handles FAISS subject_recs.
    Comparing against warm_behavioral isolates candidate generation cost:
    both paths share the same filter and enrich steps, so differences are
    attributable to FAISS retrieval versus ALS dot-product.
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs configured")

    worker_pid = _find_model_server_worker_pid("similarity")
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
) -> None:
    """
    Flamegraph for the cold subject path under load.

    Targets the similarity server worker. Expected hot frames: FAISS search,
    Bayesian score blending. The embedder call is not captured here — run
    test_profile_embedder separately if the embed step is suspected as a
    bottleneck.
    """
    if not COLD_WITH_SUBJECTS_USER_IDS:
        pytest.skip("No cold users with subjects configured")

    worker_pid = _find_model_server_worker_pid("similarity")
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
) -> None:
    """
    Flamegraph for the Bayesian popularity fallback path under load.

    Targets the metadata server worker. This path skips FAISS entirely and
    returns pre-computed Bayesian scores, so the flamegraph should show
    minimal compute — mostly serialization overhead. Sets the lower bound
    for achievable latency on any recommendation path.
    """
    if not COLD_WITHOUT_SUBJECTS_USER_IDS:
        pytest.skip("No cold users without subjects configured")

    worker_pid = _find_model_server_worker_pid("metadata")
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
) -> None:
    """
    Flamegraph for subject similarity search under load.

    Targets the similarity server worker. Expected hot frames: FAISS index
    query. This is the cheapest similarity path and forms the baseline floor
    — any unexpectedly wide frame here is a serialization or overhead issue,
    not a retrieval issue.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    worker_pid = _find_model_server_worker_pid("similarity")
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
) -> None:
    """
    Flamegraph for ALS item-factor similarity search under load.

    Targets the similarity server worker. Expected hot frames: ALS factor
    lookup, dot-product against item factor matrix. Comparing against
    similarity_subject isolates cost attributable to ALS factor retrieval
    versus FAISS index search.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    worker_pid = _find_model_server_worker_pid("similarity")
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
) -> None:
    """
    Flamegraph for hybrid similarity (FAISS + ALS score fusion) under load.

    Targets the similarity server worker. Expected hot frames: both FAISS
    search and ALS factor retrieval, plus the score fusion step. A fusion
    frame that is disproportionately wide relative to the two retrieval frames
    indicates overhead beyond simple additivity.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    worker_pid = _find_model_server_worker_pid("similarity")
    factory = _make_similarity_factory(http_client, "hybrid")

    for _ in range(_PROFILE_WARMUP):
        await factory()

    duration = _PROFILE_DURATION_S + 15
    with _pyspy_record("similarity_hybrid", worker_pid, duration):
        stats = await _run_sustained(
            factory, _PROFILE_WORKERS, _PROFILE_DURATION_S, 0, "profile_similarity_hybrid"
        )

    _print_stats("similarity_hybrid", stats.get_stats())
