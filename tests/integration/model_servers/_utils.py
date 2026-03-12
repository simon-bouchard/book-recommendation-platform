# tests/integration/model_servers/_utils.py
"""
Shared utilities, test data constants, and assertion helpers for model server
integration tests.

Kept separate from conftest.py so that test modules can import directly from
here using a normal relative import. conftest.py is a pytest-reserved file and
cannot be imported as a regular module.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable, Coroutine, Dict, List

# ---------------------------------------------------------------------------
# Measurement configuration
# ---------------------------------------------------------------------------

WARMUP_RUNS = 2
MEASUREMENT_RUNS = 10

# ---------------------------------------------------------------------------
# Test data
#
# User and book ID sets are intentionally shared with the application-level
# latency suite so that results from both layers are directly comparable.
# ---------------------------------------------------------------------------

WARM_USER_IDS = [11676, 98391, 189835, 153662, 23902, 171118, 235105, 76499, 16795, 248718]
COLD_USER_IDS_WITH_SUBJECTS = [
    248965,
    249650,
    249939,
    250634,
    251575,
    251744,
    252628,
    253310,
    258352,
    259734,
]
COLD_USER_IDS_WITHOUT_SUBJECTS = [278860, 278855, 52702]
TEST_BOOK_IDS = [
    1666,
    45959,
    402,
    27,
    41636,
    166,
    44327,
    3240,
    45503,
    49865,
    43852,
    208,
    41810,
    12372,
    3158,
    729,
    2015,
    46695,
    46839,
    45820,
]

# Subject indices must exist in the trained subject vocabulary. Update this
# list if the subject vocabulary changes between training runs.
TEST_SUBJECT_INDICES = [1, 2, 3, 5, 8]


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


class LatencyStats:
    """Collects and analyzes latency measurements for performance testing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.measurements: List[float] = []

    def add(self, duration_ms: float) -> None:
        self.measurements.append(duration_ms)

    def get_stats(self) -> Dict[str, float]:
        if not self.measurements:
            return {}
        return {
            "min_ms": min(self.measurements),
            "max_ms": max(self.measurements),
            "mean_ms": statistics.mean(self.measurements),
            "median_ms": statistics.median(self.measurements),
            "p95_ms": _percentile(self.measurements, 95),
            "p99_ms": _percentile(self.measurements, 99),
            "stdev_ms": (
                statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0
            ),
            "count": len(self.measurements),
        }

    def __str__(self) -> str:
        stats = self.get_stats()
        if not stats:
            return f"{self.name}: No measurements"
        return (
            f"{self.name}:\n"
            f"  Mean:   {stats['mean_ms']:.2f}ms\n"
            f"  Median: {stats['median_ms']:.2f}ms\n"
            f"  P95:    {stats['p95_ms']:.2f}ms\n"
            f"  P99:    {stats['p99_ms']:.2f}ms\n"
            f"  Range:  [{stats['min_ms']:.2f}ms - {stats['max_ms']:.2f}ms]\n"
            f"  Stdev:  {stats['stdev_ms']:.2f}ms\n"
            f"  Count:  {stats['count']}"
        )


def _percentile(data: List[float], percentile: float) -> float:
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


async def measure_latency(
    name: str,
    coro_factory: Callable[[], Coroutine],
    warmup_runs: int = WARMUP_RUNS,
    measurement_runs: int = MEASUREMENT_RUNS,
) -> LatencyStats:
    """
    Measure async operation latency after discarding warmup runs.

    Warmup runs prime the server's internal caches and the connection pool
    so that measurements reflect steady-state latency rather than cold-start
    overhead.

    Args:
        name: Label attached to the returned LatencyStats instance.
        coro_factory: Zero-argument callable returning the coroutine to time.
                      Called once per warmup run and once per measurement run.
        warmup_runs: Number of un-timed runs to execute before measurement.
        measurement_runs: Number of timed runs to collect.

    Returns:
        LatencyStats populated with wall-clock measurements in milliseconds.
    """
    stats = LatencyStats(name)

    for _ in range(warmup_runs):
        await coro_factory()

    for _ in range(measurement_runs):
        start = time.perf_counter()
        await coro_factory()
        stats.add((time.perf_counter() - start) * 1000)

    return stats


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_l2_normalized(vector: List[float], tol: float = 1e-5) -> None:
    """Assert that a vector has unit L2 norm within the given tolerance."""
    norm = math.sqrt(sum(x**2 for x in vector))
    assert abs(norm - 1.0) < tol, f"Vector is not L2-normalized: norm={norm:.6f}"


def assert_scores_descending(scores: List[float]) -> None:
    """Assert that a score list is in non-increasing order."""
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Scores not sorted descending at index {i}: {scores[i]:.4f} < {scores[i + 1]:.4f}"
        )
