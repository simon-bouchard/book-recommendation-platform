# tests/load/models/conftest.py
"""
Pytest configuration for models load tests.

Owns the session-scoped result accumulator and the baseline JSON writer that
runs after all load tests complete.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.load._stats import LatencyStats
from tests.load.models._constants import (
    CONCURRENCY_LEVELS,
    REQUESTS_PER_LEVEL,
    SUSTAINED_DURATION_S,
    SUSTAINED_WORKERS,
    WARMUP_RUNS,
)

os.environ["OTEL_ENVIRONMENT"] = "test"
os.environ["TEST_RUN_ID"] = datetime.now().strftime("%Y%m%d_%H%M%S")

_BASELINES_DIR = Path(__file__).parent / "performance_baselines"


@pytest.fixture(scope="session")
def performance_results() -> dict[str, LatencyStats]:
    """
    Session-scoped accumulator for all load test results.

    Writes a timestamped JSON baseline to performance_baselines/ after the
    full session completes. The schema matches the integration test baselines
    so analyze_baseline.py and compare_results.py can consume both without
    modification; extra keys (workers, throughput_rps, etc.) are ignored by
    the current scripts and can be leveraged once those scripts are extended.
    """
    results: dict[str, LatencyStats] = {}
    yield results

    _BASELINES_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = _BASELINES_DIR / f"load_baseline_{timestamp}.json"

    export: dict[str, Any] = {
        "timestamp": timestamp,
        "test_type": "load",
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "requests_per_level": REQUESTS_PER_LEVEL,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "sustained_workers": SUSTAINED_WORKERS,
            "sustained_duration_s": SUSTAINED_DURATION_S,
        },
        "results": {name: stats.get_stats() for name, stats in results.items()},
    }

    with open(output_file, "w") as f:
        json.dump(export, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"LOAD BASELINE SAVED TO: {output_file}")
    print(f"{'=' * 80}")
    print(f"\nTo analyze: python analyze_baseline.py {output_file.name}")
    print(f"Or latest:  python analyze_baseline.py --latest")


@pytest.fixture
def performance_report(
    performance_results: dict[str, LatencyStats],
) -> dict[str, LatencyStats]:
    """
    Test-scoped result collector.

    Tests write their LatencyStats objects here; teardown merges them into
    the session accumulator and prints a compact per-test summary.
    """
    report: dict[str, LatencyStats] = {}
    yield report

    for name, stats in report.items():
        performance_results[name] = stats

    print("\n" + "-" * 80)
    for name, stats in report.items():
        s = stats.get_stats()
        if s.get("mean_ms"):
            extra_parts = []
            if "throughput_rps" in s:
                extra_parts.append(f"rps={s['throughput_rps']:.1f}")
            if "concurrency_overhead_ratio" in s:
                extra_parts.append(f"overhead={s['concurrency_overhead_ratio']:.2f}x")
            extra = f"  ({', '.join(extra_parts)})" if extra_parts else ""
            print(
                f"{name}: mean={s['mean_ms']:.2f}ms  "
                f"p95={s['p95_ms']:.2f}ms  "
                f"failures={s['failures']}{extra}"
            )
    print("-" * 80)
