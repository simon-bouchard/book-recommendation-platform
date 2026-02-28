# tests/integration/models/test_models_performance.py
"""
Performance test suite for recommendation and similarity models.
Measures latency at API level to establish baseline metrics before refactoring.
Results are automatically saved to performance_baselines/ directory.
"""

import pytest
import time
import statistics
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

WARMUP_RUNS = 2
MEASUREMENT_RUNS = 10

WARM_USER_IDS = [11676, 98391, 189835, 153662, 23902, 171118, 235105, 76499, 16795, 248718]
COLD_WITH_SUBJECTS_USER_IDS = [
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
COLD_WITHOUT_SUBJECTS_USER_IDS = [278860, 278855, 52702]
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


def measure_endpoint_latency(
    client: TestClient,
    endpoint: str,
    params: Dict = None,
    warmup_runs: int = WARMUP_RUNS,
    measurement_runs: int = MEASUREMENT_RUNS,
) -> Tuple[LatencyStats, Dict]:
    """
    Measures endpoint latency with warmup runs.
    Returns tuple of (LatencyStats, last_response_json).
    """
    stats = LatencyStats(f"GET {endpoint}")

    for _ in range(warmup_runs):
        client.get(endpoint, params=params)

    last_response = None
    for _ in range(measurement_runs):
        start = time.perf_counter()
        response = client.get(endpoint, params=params)
        duration_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200, f"Request failed: {response.status_code}"
        stats.add(duration_ms)
        last_response = response.json()

    return stats, last_response


@pytest.fixture(scope="session")
def performance_results():
    """Session-scoped fixture that collects all performance results."""
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
    print(f"Or latest:  python analyze_baseline.py --latest")


@pytest.fixture
def client(db: Session):
    """Creates test client for FastAPI app."""
    from main import app

    return TestClient(app)


@pytest.fixture
def performance_report(performance_results):
    """Test-scoped fixture that adds results to session-scoped collection."""
    report = {}
    yield report

    for name, stats in report.items():
        performance_results[name] = stats

    print("\n" + "-" * 80)
    for test_name, stats in report.items():
        print(f"{test_name}: {stats.get_stats()['mean_ms']:.2f}ms")
    print("-" * 80)


@pytest.mark.parametrize(
    "user_id,mode",
    [
        *[(uid, "auto") for uid in WARM_USER_IDS[:3]],
        *[(uid, "behavioral") for uid in WARM_USER_IDS[:2]],
    ],
)
def test_warm_user_recommendations_latency(
    client: TestClient, performance_report: Dict, user_id: int, mode: str
):
    """Tests recommendation latency for warm users using ALS strategy."""
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": mode}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list), "Response should be a list"
    assert len(response) > 0, "Should return recommendations"
    assert len(response) <= 200, "Should not exceed top_n"

    for item in response[:5]:
        assert "item_idx" in item
        assert "title" in item
        assert "score" in item

    test_key = f"recommend_warm_user_{user_id}_mode_{mode}"
    performance_report[test_key] = stats


@pytest.mark.parametrize("user_id", WARM_USER_IDS[:3])
def test_warm_user_forced_subject_mode_latency(
    client: TestClient, performance_report: Dict, user_id: int
):
    """
    Tests warm user forced to use subject-based (cold) strategy.
    Critical path: tests if subject similarity scales for high-engagement users.
    """
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list)
    assert len(response) > 0

    test_key = f"recommend_warm_user_{user_id}_forced_subject_mode"
    performance_report[test_key] = stats


@pytest.mark.parametrize("user_id", COLD_WITH_SUBJECTS_USER_IDS[:3])
def test_cold_user_with_subjects_latency(
    client: TestClient, performance_report: Dict, user_id: int
):
    """
    Tests cold user WITH favorite subjects.
    Critical path: similarity computation + Bayesian blending.
    """
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "auto"}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list)
    assert len(response) > 0

    test_key = f"recommend_cold_with_subjects_user_{user_id}"
    performance_report[test_key] = stats


@pytest.mark.parametrize("user_id", COLD_WITHOUT_SUBJECTS_USER_IDS[:3])
def test_cold_user_without_subjects_latency(
    client: TestClient, performance_report: Dict, user_id: int
):
    """
    Tests cold user WITHOUT favorite subjects (pure Bayesian fallback).
    Critical path: fastest cold recommendation (no embedding/similarity computation).
    """
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "auto"}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list)
    assert len(response) > 0

    test_key = f"recommend_cold_without_subjects_user_{user_id}"
    performance_report[test_key] = stats


@pytest.mark.parametrize("w", [0.3, 0.6, 0.9])
def test_cold_recommendations_varying_w(client: TestClient, performance_report: Dict, w: float):
    """
    Tests cold recommendation with varying w parameter.
    Tests similarity vs Bayesian weight balance impact on performance.
    """
    if not COLD_WITH_SUBJECTS_USER_IDS:
        pytest.skip("No cold users with subjects provided")

    user_id = COLD_WITH_SUBJECTS_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "auto", "w": w}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    assert isinstance(response, list)
    assert len(response) > 0

    test_key = f"recommend_cold_w_{w}"
    performance_report[test_key] = stats


@pytest.mark.parametrize("top_n", [50, 200, 500])
def test_recommendations_varying_top_n(client: TestClient, performance_report: Dict, top_n: int):
    """Tests how recommendation latency scales with top_n parameter."""
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs provided")

    user_id = WARM_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": top_n, "mode": "auto"}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    assert len(response) <= top_n

    test_key = f"recommend_top_n_{top_n}"
    performance_report[test_key] = stats


@pytest.mark.parametrize(
    "book_id,mode",
    [
        *[(bid, "subject") for bid in TEST_BOOK_IDS[:3]],
        *[(bid, "als") for bid in TEST_BOOK_IDS[:2]],
        *[(bid, "hybrid") for bid in TEST_BOOK_IDS[:2]],
    ],
)
def test_similar_books_latency(
    client: TestClient, performance_report: Dict, book_id: int, mode: str
):
    """Tests similarity search latency across different modes."""
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": mode, "top_k": 200}

    try:
        stats, response = measure_endpoint_latency(
            client, endpoint, params, measurement_runs=MEASUREMENT_RUNS
        )

        assert isinstance(response, list)
        assert len(response) > 0
        assert len(response) <= 200

        for item in response[:5]:
            assert "item_idx" in item
            assert "title" in item
            assert "score" in item

        test_key = f"similar_book_{book_id}_mode_{mode}"
        performance_report[test_key] = stats

    except AssertionError as e:
        if "422" in str(e):
            pytest.skip(f"Book {book_id} doesn't have {mode} data")
        raise


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_hybrid_similarity_varying_alpha(
    client: TestClient, performance_report: Dict, alpha: float
):
    """Tests hybrid similarity performance with different alpha values."""
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs provided")

    book_id = TEST_BOOK_IDS[0]
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": "hybrid", "alpha": alpha, "top_k": 200}

    try:
        stats, response = measure_endpoint_latency(client, endpoint, params)
        assert isinstance(response, list)

        test_key = f"similar_hybrid_alpha_{alpha}"
        performance_report[test_key] = stats

    except AssertionError as e:
        if "422" in str(e):
            pytest.skip(f"Book {book_id} doesn't have hybrid data")
        raise


@pytest.mark.parametrize("top_k", [10, 50, 200, 500])
def test_similarity_varying_top_k(client: TestClient, performance_report: Dict, top_k: int):
    """Tests how similarity search scales with top_k parameter."""
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs provided")

    book_id = TEST_BOOK_IDS[0]
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": "subject", "top_k": top_k}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    assert len(response) <= top_k

    test_key = f"similar_top_k_{top_k}"
    performance_report[test_key] = stats


def test_concurrent_recommendation_requests(client: TestClient, performance_report: Dict):
    """Tests latency under simulated concurrent load."""
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs provided")

    import concurrent.futures

    endpoint = "/profile/recommend"
    user_ids = WARM_USER_IDS[:5]

    def make_request(user_id: int) -> float:
        start = time.perf_counter()
        response = client.get(
            endpoint, params={"user": str(user_id), "_id": True, "top_n": 200, "mode": "auto"}
        )
        duration_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200
        return duration_ms

    sequential_stats = LatencyStats("concurrent_sequential")
    for user_id in user_ids:
        for _ in range(3):
            duration = make_request(user_id)
            sequential_stats.add(duration)

    concurrent_stats = LatencyStats("concurrent_parallel_5")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for user_id in user_ids:
            for _ in range(3):
                futures.append(executor.submit(make_request, user_id))

        for future in concurrent.futures.as_completed(futures):
            duration = future.result()
            concurrent_stats.add(duration)

    performance_report["concurrent_sequential"] = sequential_stats
    performance_report["concurrent_parallel_5"] = concurrent_stats

    seq_mean = sequential_stats.get_stats()["mean_ms"]
    conc_mean = concurrent_stats.get_stats()["mean_ms"]

    print(f"\nConcurrency overhead: {((conc_mean / seq_mean) - 1) * 100:.1f}%")


if __name__ == "__main__":
    import os

    os.environ.setdefault("TESTING", "1")
    pytest.main([__file__, "-v", "-s"])
