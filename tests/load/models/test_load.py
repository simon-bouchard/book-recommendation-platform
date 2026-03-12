# tests/load/models/test_load.py
"""
Latency and concurrency load tests for the recommendation and similarity API.

Fires real HTTP requests at a live Gunicorn server (set PERF_TEST_BASE_URL).
All concurrency is genuine: asyncio coroutines each hold independent TCP
connections from the shared httpx connection pool, so multiple outstanding
requests are handled in parallel by Gunicorn workers. This is fundamentally
different from the integration tests, which run through ASGITransport on a
single event loop and cannot observe real worker-level parallelism.

Two test categories are covered:

  Ramp tests   -- fire REQUESTS_PER_LEVEL requests at each concurrency level
                  (1, 2, 5, 10, 20 workers) and record per-request latency.
                  The concurrency_overhead_ratio in each result is mean latency
                  at N workers divided by mean latency at 1 worker; a value of
                  1.0 means no degradation.

  Sustained tests -- N workers loop continuously for SUSTAINED_DURATION_S
                     seconds, measuring both latency distribution and aggregate
                     throughput (requests per second). This reveals memory
                     pressure and connection-pool exhaustion that only appear
                     under continuous load.

For maximum-load profiling during a sustained test, attach py-spy to a
Gunicorn worker PID from a separate terminal:

    py-spy record --pid <worker_pid> --output profile.json --format speedscope

Or run tests/load/models/test_profile.py, which automates PID discovery via psutil.

Usage:
    export PERF_TEST_BASE_URL=http://localhost:8000
    pytest tests/load/models/test_load.py -v -s

    # Run only ramp tests:
    pytest tests/load/models/test_load.py -v -s -k "ramp"

    # Run only sustained tests:
    pytest tests/load/models/test_load.py -v -s -k "sustained"
"""

import asyncio
import itertools
import time
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest

from tests.load._stats import LatencyStats
from tests.load.models._constants import (
    COLD_WITHOUT_SUBJECTS_USER_IDS,
    COLD_WITH_SUBJECTS_USER_IDS,
    CONCURRENCY_LEVELS,
    REQUESTS_PER_LEVEL,
    SUSTAINED_DURATION_S,
    SUSTAINED_WORKERS,
    TEST_BOOK_IDS,
    WARM_USER_IDS,
    WARMUP_RUNS,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")

# A factory is a zero-argument async callable returning (duration_ms, status_code).
RequestFactory = Callable[[], Coroutine[Any, Any, tuple[float, int]]]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

_RECOMMEND_SCENARIOS = [
    {
        "name": "warm_behavioral",
        "user_ids": WARM_USER_IDS,
        "mode": "behavioral",
        "skip_if_empty": False,
    },
    {
        "name": "warm_subject",
        "user_ids": WARM_USER_IDS,
        "mode": "subject",
        "skip_if_empty": False,
    },
    {
        "name": "cold_with_subjects",
        "user_ids": COLD_WITH_SUBJECTS_USER_IDS,
        "mode": "subject",
        "skip_if_empty": True,
    },
    {
        "name": "cold_without_subjects",
        "user_ids": COLD_WITHOUT_SUBJECTS_USER_IDS,
        "mode": "subject",
        "skip_if_empty": True,
    },
]

_SIMILARITY_SCENARIOS = [
    {"name": "similarity_subject", "mode": "subject"},
    {"name": "similarity_als", "mode": "als"},
    {"name": "similarity_hybrid", "mode": "hybrid"},
]

# Subset of recommendation scenarios worth sustaining for 30 s each.
# All similarity modes are included for similarity sustained tests.
_SUSTAINED_RECOMMEND_SCENARIOS = [
    s for s in _RECOMMEND_SCENARIOS if s["name"] in ("warm_behavioral", "cold_with_subjects")
]


# ---------------------------------------------------------------------------
# Request factories
# ---------------------------------------------------------------------------


def _make_recommend_factory(
    client: httpx.AsyncClient,
    user_ids: list[int],
    mode: str,
) -> RequestFactory:
    """
    Return a factory that cycles through user_ids to avoid saturating Redis
    with repeated identical cache keys, which would measure cache latency
    rather than pipeline latency.
    """
    user_cycle = itertools.cycle(user_ids)

    async def factory() -> tuple[float, int]:
        user_id = next(user_cycle)
        start = time.perf_counter()
        response = await client.get(
            "/profile/recommend",
            params={"user": str(user_id), "_id": True, "top_n": 200, "mode": mode},
        )
        return (time.perf_counter() - start) * 1000, response.status_code

    return factory


def _make_similarity_factory(
    client: httpx.AsyncClient,
    mode: str,
) -> RequestFactory:
    """
    Return a factory that cycles through TEST_BOOK_IDS.

    422 responses (book has no ALS data) are counted as failures in the stats
    but do not raise, allowing the test to continue with the remaining IDs.
    """
    book_cycle = itertools.cycle(TEST_BOOK_IDS)

    async def factory() -> tuple[float, int]:
        book_id = next(book_cycle)
        start = time.perf_counter()
        response = await client.get(
            f"/book/{book_id}/similar",
            params={"mode": mode, "top_k": 200},
        )
        return (time.perf_counter() - start) * 1000, response.status_code

    return factory


# ---------------------------------------------------------------------------
# Core measurement helpers
# ---------------------------------------------------------------------------


async def _run_ramp(
    factory: RequestFactory,
    concurrency_levels: list[int],
    requests_per_level: int,
    warmup_runs: int,
    base_name: str,
) -> dict[int, LatencyStats]:
    """
    Exercise a request factory at increasing concurrency levels.

    At each level, fires `workers` requests simultaneously for
    ceil(requests_per_level / workers) rounds, accumulating per-request
    latency into a LatencyStats object. After all levels are measured the
    concurrency_overhead_ratio is computed relative to the single-worker
    baseline and stored in each stats object's extra metadata.

    Returns a mapping of workers -> LatencyStats for the caller to register
    with performance_report.
    """
    for _ in range(warmup_runs):
        await factory()

    results: dict[int, LatencyStats] = {}

    for workers in concurrency_levels:
        stats = LatencyStats(
            f"{base_name}_ramp_w{workers}",
            workers=workers,
            test_type="ramp",
        )
        rounds = max(1, requests_per_level // workers)

        for _ in range(rounds):
            batch = await asyncio.gather(*[factory() for _ in range(workers)])
            for duration_ms, status_code in batch:
                if status_code == 200:
                    stats.add(duration_ms)
                else:
                    stats.failures += 1

        results[workers] = stats

    _attach_overhead_ratios(results)
    return results


async def _run_sustained(
    factory: RequestFactory,
    workers: int,
    duration_s: int,
    warmup_runs: int,
    base_name: str,
) -> LatencyStats:
    """
    Run N workers continuously firing requests for duration_s seconds.

    Each worker loops independently: as soon as one request completes the
    worker immediately dispatches another, keeping the server saturated at
    exactly `workers` concurrent in-flight requests throughout the window.
    This reflects how production traffic actually arrives: not in discrete
    synchronised batches, but as a continuous overlapping stream.

    throughput_rps in the returned stats is total requests (successes +
    failures) divided by the actual wall-clock duration.
    """
    for _ in range(warmup_runs):
        await factory()

    stats = LatencyStats(
        f"{base_name}_sustained_w{workers}",
        workers=workers,
        test_type="sustained",
        target_duration_s=duration_s,
    )

    loop = asyncio.get_event_loop()
    deadline = loop.time() + duration_s

    async def worker_loop() -> None:
        while loop.time() < deadline:
            duration_ms, status_code = await factory()
            if status_code == 200:
                stats.add(duration_ms)
            else:
                stats.failures += 1

    wall_start = time.perf_counter()
    await asyncio.gather(*[worker_loop() for _ in range(workers)])
    wall_elapsed = time.perf_counter() - wall_start

    total_requests = stats.count + stats.failures
    stats.extra["throughput_rps"] = round(total_requests / wall_elapsed, 2)
    stats.extra["success_rate"] = round(stats.count / total_requests, 4) if total_requests else 0.0
    stats.extra["actual_duration_s"] = round(wall_elapsed, 2)

    return stats


def _attach_overhead_ratios(ramp_results: dict[int, LatencyStats]) -> None:
    """
    Compute and store concurrency_overhead_ratio in each ramp result.

    The ratio is mean latency at N workers divided by mean latency at 1 worker.
    A value of 1.0 means no degradation; 1.5 means 50 % overhead. Stored in
    stats.extra so it is automatically included in get_stats() output.
    """
    baseline = ramp_results.get(1)
    if baseline is None:
        return
    baseline_mean = baseline.get_stats().get("mean_ms", 0.0)
    if baseline_mean == 0.0:
        return
    for workers, stats in ramp_results.items():
        s = stats.get_stats()
        if s.get("mean_ms"):
            stats.extra["concurrency_overhead_ratio"] = round(s["mean_ms"] / baseline_mean, 3)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def _print_ramp_result(name: str, workers: int, stats: LatencyStats) -> None:
    s = stats.get_stats()
    overhead = s.get("concurrency_overhead_ratio", "-")
    print(
        f"\n  {name} w={workers}: "
        f"mean={s.get('mean_ms', 0):.1f}ms  "
        f"p95={s.get('p95_ms', 0):.1f}ms  "
        f"failures={s['failures']}  "
        f"overhead={overhead}"
    )


def _print_sustained_result(name: str, stats: LatencyStats) -> None:
    s = stats.get_stats()
    print(
        f"\n  {name}: "
        f"mean={s.get('mean_ms', 0):.1f}ms  "
        f"p95={s.get('p95_ms', 0):.1f}ms  "
        f"rps={s.get('throughput_rps', 0):.1f}  "
        f"success_rate={s.get('success_rate', 0):.3f}  "
        f"failures={s['failures']}"
    )


# ---------------------------------------------------------------------------
# Ramp tests — recommendations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", _RECOMMEND_SCENARIOS, ids=lambda s: s["name"])
async def test_ramp_recommendations(
    http_client: httpx.AsyncClient,
    performance_report: dict[str, LatencyStats],
    scenario: dict,
) -> None:
    """
    Ramp test across all recommendation scenarios.

    Exercises behavioral (ALS dot-product) and subject (FAISS + Bayesian blend)
    paths for both warm and cold users at CONCURRENCY_LEVELS, giving a complete
    picture of how each pipeline degrades under load.
    """
    if scenario["skip_if_empty"] and not scenario["user_ids"]:
        pytest.skip(f"No user IDs configured for scenario '{scenario['name']}'")

    factory = _make_recommend_factory(http_client, scenario["user_ids"], scenario["mode"])
    results = await _run_ramp(
        factory,
        CONCURRENCY_LEVELS,
        REQUESTS_PER_LEVEL,
        WARMUP_RUNS,
        scenario["name"],
    )

    for workers, stats in results.items():
        performance_report[stats.name] = stats
        _print_ramp_result(scenario["name"], workers, stats)


# ---------------------------------------------------------------------------
# Ramp tests — similarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", _SIMILARITY_SCENARIOS, ids=lambda s: s["name"])
async def test_ramp_similarity(
    http_client: httpx.AsyncClient,
    performance_report: dict[str, LatencyStats],
    scenario: dict,
) -> None:
    """
    Ramp test across subject, ALS, and hybrid similarity modes.

    Comparing the overhead ratios across modes shows whether FAISS and ALS
    retrieval scale independently under concurrent load. 422 responses for
    books without ALS data are tracked as failures but do not abort the test.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    factory = _make_similarity_factory(http_client, scenario["mode"])
    results = await _run_ramp(
        factory,
        CONCURRENCY_LEVELS,
        REQUESTS_PER_LEVEL,
        WARMUP_RUNS,
        scenario["name"],
    )

    for workers, stats in results.items():
        performance_report[stats.name] = stats
        _print_ramp_result(scenario["name"], workers, stats)


# ---------------------------------------------------------------------------
# Sustained tests — recommendations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", _SUSTAINED_RECOMMEND_SCENARIOS, ids=lambda s: s["name"])
async def test_sustained_recommendations(
    http_client: httpx.AsyncClient,
    performance_report: dict[str, LatencyStats],
    scenario: dict,
) -> None:
    """
    Sustained load test for the most important recommendation paths.

    Runs SUSTAINED_WORKERS concurrent workers for SUSTAINED_DURATION_S seconds.
    The warm behavioral path is the highest-volume production path; cold with
    subjects is the most expensive cold path. Watching mean latency and
    throughput_rps over the window reveals connection pool exhaustion, worker
    queue buildup, and GIL contention around numpy operations that only appear
    under continuous load.
    """
    if scenario["skip_if_empty"] and not scenario["user_ids"]:
        pytest.skip(f"No user IDs configured for scenario '{scenario['name']}'")

    factory = _make_recommend_factory(http_client, scenario["user_ids"], scenario["mode"])
    stats = await _run_sustained(
        factory,
        SUSTAINED_WORKERS,
        SUSTAINED_DURATION_S,
        WARMUP_RUNS,
        scenario["name"],
    )

    performance_report[stats.name] = stats
    _print_sustained_result(scenario["name"], stats)


# ---------------------------------------------------------------------------
# Sustained tests — similarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", _SIMILARITY_SCENARIOS, ids=lambda s: s["name"])
async def test_sustained_similarity(
    http_client: httpx.AsyncClient,
    performance_report: dict[str, LatencyStats],
    scenario: dict,
) -> None:
    """
    Sustained load test for all three similarity modes.

    Comparing throughput_rps across subject, ALS, and hybrid under identical
    worker counts shows whether mode-specific bottlenecks (FAISS search vs ALS
    factor retrieval) limit concurrency differently. Hybrid is expected to show
    lower RPS than subject or ALS alone; a disproportionate drop indicates
    overhead in the score fusion step beyond simple additivity.
    """
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    factory = _make_similarity_factory(http_client, scenario["mode"])
    stats = await _run_sustained(
        factory,
        SUSTAINED_WORKERS,
        SUSTAINED_DURATION_S,
        WARMUP_RUNS,
        scenario["name"],
    )

    performance_report[stats.name] = stats
    _print_sustained_result(scenario["name"], stats)
