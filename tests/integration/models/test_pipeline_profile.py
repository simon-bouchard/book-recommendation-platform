# tests/integration/models/test_pipeline_profile.py
"""
Flamegraph profiling for the recommendation pipeline using py-spy.

Attaches py-spy to the current process (TestClient runs in-process) while
firing the same request sequences as the performance test suite.  Outputs
one Speedscope JSON per scenario to tests/integration/profiles/.

Open the files at https://speedscope.app — drag and drop, no install needed.

Usage:
    pytest tests/integration/models/test_pipeline_profile.py -v -s
    pytest tests/integration/models/test_pipeline_profile.py -v -s -k "warm"

Requirements:
    pip install py-spy --break-system-packages
    sudo access required (py-spy attaches to a running process)
"""

import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.integration.models.test_models_performance import (
    COLD_WITH_SUBJECTS_USER_IDS,
    COLD_WITHOUT_SUBJECTS_USER_IDS,
    WARM_USER_IDS,
    measure_endpoint_latency,
)

_PROFILES_DIR = Path(__file__).parent / "profiles"
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

_PROFILE_WARMUP = 3
_PROFILE_REQUESTS = 20

# Resolve the full path under the current user's PATH so that sudo — which
# runs with a restricted PATH that excludes conda/venv bin directories —
# can still find the executable.  None if py-spy is not installed.
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

    Attaches to os.getpid() — valid because TestClient executes the full
    application stack in the same process as pytest.

    Args:
        label: Short name used in the output filename.
        duration: Maximum recording time in seconds.  Set comfortably above
                  the expected total request duration so py-spy does not stop
                  early and truncate the profile.

    Yields:
        Path to the output .json file (exists after the context exits).
    """
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _PROFILES_DIR / f"profile_{_TIMESTAMP}_{label}.json"

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
        import signal as _signal

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # SIGKILL prevents py-spy from writing the output file.
            # SIGINT triggers a clean shutdown that flushes the profile.
            proc.send_signal(_signal.SIGINT)
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


# ---------------------------------------------------------------------------
# Profile tests
# ---------------------------------------------------------------------------


def test_profile_warm_als(client: TestClient):
    """
    Flamegraph for the warm ALS recommendation path.

    Expected hot frames: ALSBasedGenerator.generate -> AlsClient._post,
    ReadBooksFilter.apply, MetadataClient._post.  Any unexpectedly wide
    frame is the bottleneck.
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs configured")

    user_id = WARM_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "behavioral"}

    for _ in range(_PROFILE_WARMUP):
        client.get(endpoint, params=params)

    with _pyspy_record("warm_als", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = measure_endpoint_latency(
            client,
            endpoint,
            params,
            warmup_runs=0,
            measurement_runs=_PROFILE_REQUESTS,
        )

    s = stats.get_stats()
    print(
        f"\n  warm_als — mean={s['mean_ms']:.1f}ms  "
        f"p50={s['median_ms']:.1f}ms  p95={s['p95_ms']:.1f}ms"
    )


def test_profile_warm_subject(client: TestClient):
    """
    Flamegraph for a warm user forced onto the subject path.

    Comparing this profile against warm_als isolates which cost is
    path-specific versus shared overhead (filter, enrich).
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs configured")

    user_id = WARM_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"}

    for _ in range(_PROFILE_WARMUP):
        client.get(endpoint, params=params)

    with _pyspy_record("warm_subject", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = measure_endpoint_latency(
            client,
            endpoint,
            params,
            warmup_runs=0,
            measurement_runs=_PROFILE_REQUESTS,
        )

    s = stats.get_stats()
    print(
        f"\n  warm_subject — mean={s['mean_ms']:.1f}ms  "
        f"p50={s['median_ms']:.1f}ms  p95={s['p95_ms']:.1f}ms"
    )


def test_profile_cold_with_subjects(client: TestClient):
    """Flamegraph for the cold subject path (embed + subject_recs + enrich)."""
    if not COLD_WITH_SUBJECTS_USER_IDS:
        pytest.skip("No cold users with subjects configured")

    user_id = COLD_WITH_SUBJECTS_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "auto"}

    for _ in range(_PROFILE_WARMUP):
        client.get(endpoint, params=params)

    with _pyspy_record("cold_with_subjects", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = measure_endpoint_latency(
            client,
            endpoint,
            params,
            warmup_runs=0,
            measurement_runs=_PROFILE_REQUESTS,
        )

    s = stats.get_stats()
    print(
        f"\n  cold_with_subjects — mean={s['mean_ms']:.1f}ms  "
        f"p50={s['median_ms']:.1f}ms  p95={s['p95_ms']:.1f}ms"
    )


def test_profile_cold_without_subjects(client: TestClient):
    """Flamegraph for the pure Bayesian fallback path."""
    if not COLD_WITHOUT_SUBJECTS_USER_IDS:
        pytest.skip("No cold users without subjects configured")

    user_id = COLD_WITHOUT_SUBJECTS_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "auto"}

    for _ in range(_PROFILE_WARMUP):
        client.get(endpoint, params=params)

    with _pyspy_record("cold_without_subjects", duration=_PROFILE_REQUESTS * 2 + 10):
        stats, _ = measure_endpoint_latency(
            client,
            endpoint,
            params,
            warmup_runs=0,
            measurement_runs=_PROFILE_REQUESTS,
        )

    s = stats.get_stats()
    print(
        f"\n  cold_without_subjects — mean={s['mean_ms']:.1f}ms  "
        f"p50={s['median_ms']:.1f}ms  p95={s['p95_ms']:.1f}ms"
    )
