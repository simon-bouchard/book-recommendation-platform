# tests/integration/models/test_models_performance.py
"""
Performance test suite for recommendation and similarity models.
Measures latency at the API level to establish baseline metrics before refactoring.
Results are automatically saved to performance_baselines/ directory.
"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

WARMUP_RUNS = 10
MEASUREMENT_RUNS = 50

_CONFIG = json.loads((Path(__file__).parent / "test_data_config.json").read_text())

WARM_USER_IDS = _CONFIG["warm_user_ids"]
COLD_WITH_SUBJECTS_USER_IDS = _CONFIG["cold_with_subjects_user_ids"]
COLD_WITHOUT_SUBJECTS_USER_IDS = _CONFIG["cold_without_subjects_user_ids"]
TEST_BOOK_IDS = _CONFIG["test_book_ids"]["all"]


class LatencyStats:
    """Collects and analyzes latency measurements for performance testing."""

    def __init__(self, name: str):
        self.name = name
        self.measurements: List[float] = []

    def add(self, duration_ms: float):
        self.measurements.append(duration_ms)

    def get_stats(self) -> Dict[str, float]:
        if not self.measurements:
            return {}

        return {
            "min_ms": min(self.measurements),
            "max_ms": max(self.measurements),
            "mean_ms": statistics.mean(self.measurements),
            "median_ms": statistics.median(self.measurements),
            "p95_ms": self._percentile(self.measurements, 95),
            "p99_ms": self._percentile(self.measurements, 99),
            "stdev_ms": statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0,
            "count": len(self.measurements),
        }

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def __str__(self) -> str:
        stats = self.get_stats()
        if not stats:
            return f"{self.name}: No measurements"

        return (
            f"{self.name}:\n"
            f"  Mean: {stats['mean_ms']:.2f}ms\n"
            f"  Median: {stats['median_ms']:.2f}ms\n"
            f"  P95: {stats['p95_ms']:.2f}ms\n"
            f"  P99: {stats['p99_ms']:.2f}ms\n"
            f"  Range: [{stats['min_ms']:.2f}ms - {stats['max_ms']:.2f}ms]\n"
            f"  Stdev: {stats['stdev_ms']:.2f}ms\n"
            f"  Count: {stats['count']}"
        )


async def measure_endpoint_latency(
    client: httpx.AsyncClient,
    endpoint: str,
    params: Dict = None,
    warmup_runs: int = WARMUP_RUNS,
    measurement_runs: int = MEASUREMENT_RUNS,
) -> Tuple[LatencyStats, Dict]:
    """
    Measure endpoint latency after discarding warmup runs.

    Returns a tuple of (LatencyStats, last_response_json). Warmup runs prime
    the connection pool so measurements reflect steady-state latency rather
    than cold-start overhead.
    """
    stats = LatencyStats(f"GET {endpoint}")

    for _ in range(warmup_runs):
        await client.get(endpoint, params=params)

    last_response = None
    for _ in range(measurement_runs):
        start = time.perf_counter()
        response = await client.get(endpoint, params=params)
        duration_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200, (
            f"Request failed with {response.status_code}: {response.text}"
        )
        stats.add(duration_ms)
        last_response = response.json()

    return stats, last_response


@pytest.fixture(scope="session")
def performance_results():
    """
    Session-scoped fixture that accumulates all performance results and writes
    them to a timestamped JSON file after the session ends.
    """
    results = {}
    yield results

    output_dir = Path(__file__).parent / "performance_baselines"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"baseline_{timestamp}.json"

    export_data = {
        "timestamp": timestamp,
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "measurement_runs": MEASUREMENT_RUNS,
        },
        "results": {name: stats.get_stats() for name, stats in results.items()},
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"BASELINE RESULTS SAVED TO: {output_file}")
    print(f"{'=' * 80}")
    print(f"\nTo analyze: python analyze_baseline.py {output_file.name}")
    print("Or latest:  python analyze_baseline.py --latest")


@pytest.fixture
def performance_report(performance_results):
    """
    Test-scoped fixture that collects results from a single test and merges
    them into the session-scoped accumulator on teardown.
    """
    report = {}
    yield report

    for name, stats in report.items():
        performance_results[name] = stats

    print("\n" + "-" * 80)
    for test_name, stats in report.items():
        print(f"{test_name}: {stats.get_stats()['mean_ms']:.2f}ms")
    print("-" * 80)


# ============================================================================
# Recommendation tests
# ============================================================================


@pytest.mark.parametrize("user_id", WARM_USER_IDS[:3])
async def test_warm_user_behavioral_latency(
    client: httpx.AsyncClient, performance_report: Dict, user_id: int
):
    """
    Tests recommendation latency for warm users via the ALS (behavioral) strategy.

    This is the primary warm-user path the UI exercises: the client always
    sends mode=behavioral when it knows the user is warm.
    """
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "behavioral"}

    stats, response = await measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list)
    assert len(response) > 0
    assert len(response) <= 200

    for item in response[:5]:
        assert "item_idx" in item
        assert "title" in item
        assert "score" in item

    performance_report[f"recommend_warm_user_{user_id}_behavioral"] = stats


@pytest.mark.parametrize("user_id", WARM_USER_IDS[:3])
async def test_warm_user_subject_mode_latency(
    client: httpx.AsyncClient, performance_report: Dict, user_id: int
):
    """
    Tests a warm user on the subject (cold) path.

    The UI sends mode=subject regardless of warmth when the user navigates
    to a subject-discovery context. Comparing this profile against behavioral
    isolates path-specific cost versus shared overhead (filter, enrich).
    """
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"}

    stats, response = await measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list)
    assert len(response) > 0

    performance_report[f"recommend_warm_user_{user_id}_subject"] = stats


@pytest.mark.parametrize("user_id", COLD_WITH_SUBJECTS_USER_IDS[:3])
async def test_cold_user_with_subjects_latency(
    client: httpx.AsyncClient, performance_report: Dict, user_id: int
):
    """
    Tests cold user with favorite subjects via the subject similarity path.

    Critical path: embedding + similarity + Bayesian blending. The UI
    always sends mode=subject for cold users.
    """
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"}

    stats, response = await measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list)
    assert len(response) > 0

    performance_report[f"recommend_cold_with_subjects_user_{user_id}"] = stats


@pytest.mark.parametrize("user_id", COLD_WITHOUT_SUBJECTS_USER_IDS[:3])
async def test_cold_user_without_subjects_latency(
    client: httpx.AsyncClient, performance_report: Dict, user_id: int
):
    """
    Tests cold user without favorite subjects via the subject path.

    With no subject embeddings to score against, the pipeline falls back to
    pure Bayesian popularity. This is the fastest cold path and sets the lower
    bound for cold-user latency. The UI sends mode=subject for all cold users
    regardless of whether they have subjects set.
    """
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"}

    stats, response = await measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list)
    assert len(response) > 0

    performance_report[f"recommend_cold_without_subjects_user_{user_id}"] = stats


@pytest.mark.parametrize("w", [0.3, 0.6, 0.9])
async def test_cold_recommendations_varying_w(
    client: httpx.AsyncClient, performance_report: Dict, w: float
):
    """Tests cold recommendation performance across subject weight values."""
    if not COLD_WITH_SUBJECTS_USER_IDS:
        pytest.skip("No cold users with subjects configured")

    user_id = COLD_WITH_SUBJECTS_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject", "w": w}

    stats, response = await measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list)
    assert len(response) > 0

    performance_report[f"recommend_cold_w_{w}"] = stats


@pytest.mark.parametrize("top_n", [50, 200, 500])
async def test_recommendations_varying_top_n(
    client: httpx.AsyncClient, performance_report: Dict, top_n: int
):
    """
    Tests how recommendation latency scales with top_n.

    Uses a warm user on the behavioral path, which is the highest-volume
    production scenario.
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs configured")

    user_id = WARM_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": top_n, "mode": "behavioral"}

    stats, response = await measure_endpoint_latency(client, endpoint, params)

    assert len(response) <= top_n

    performance_report[f"recommend_top_n_{top_n}"] = stats


# ============================================================================
# Similarity tests
# ============================================================================


@pytest.mark.parametrize(
    "book_id,mode",
    [
        *[(bid, "subject") for bid in TEST_BOOK_IDS[:3]],
        *[(bid, "als") for bid in TEST_BOOK_IDS[:2]],
        *[(bid, "hybrid") for bid in TEST_BOOK_IDS[:2]],
    ],
)
async def test_similar_books_latency(
    client: httpx.AsyncClient, performance_report: Dict, book_id: int, mode: str
):
    """Tests similarity search latency across modes."""
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": mode, "top_k": 200}

    try:
        stats, response = await measure_endpoint_latency(client, endpoint, params)

        assert isinstance(response, list)
        assert len(response) > 0
        assert len(response) <= 200

        for item in response[:5]:
            assert "item_idx" in item
            assert "title" in item
            assert "score" in item

        performance_report[f"similar_book_{book_id}_mode_{mode}"] = stats

    except AssertionError as e:
        if "422" in str(e):
            pytest.skip(f"Book {book_id} has no {mode} data, skipping")
        raise


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
async def test_hybrid_similarity_varying_alpha(
    client: httpx.AsyncClient, performance_report: Dict, alpha: float
):
    """Tests hybrid similarity performance across alpha values."""
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    book_id = TEST_BOOK_IDS[0]
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": "hybrid", "alpha": alpha, "top_k": 200}

    try:
        stats, response = await measure_endpoint_latency(client, endpoint, params)
        assert isinstance(response, list)

        performance_report[f"similar_hybrid_alpha_{alpha}"] = stats

    except AssertionError as e:
        if "422" in str(e):
            pytest.skip(f"Book {book_id} has no hybrid data, skipping")
        raise


@pytest.mark.parametrize("top_k", [10, 50, 200, 500])
async def test_similarity_varying_top_k(
    client: httpx.AsyncClient, performance_report: Dict, top_k: int
):
    """Tests how similarity search scales with top_k."""
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs configured")

    book_id = TEST_BOOK_IDS[0]
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": "subject", "top_k": top_k}

    stats, response = await measure_endpoint_latency(client, endpoint, params)

    assert len(response) <= top_k

    performance_report[f"similar_top_k_{top_k}"] = stats
