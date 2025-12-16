# test/performance/models/test_models_performance.py
"""
Performance/Latency Test Suite for Models Module
=================================================

Tests the recommendation and similarity endpoints at the API level to establish
baseline performance metrics before refactoring.

Usage:
    pytest tests/integration/models/test_models_performance.py -v
    pytest tests/integration/models/test_models_performance.py -v --benchmark-only
"""

import pytest
import time
import statistics
from typing import Dict, List, Tuple
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import os

# Test configuration
WARMUP_RUNS = 2
MEASUREMENT_RUNS = 10


# ============================================================================
# Test Data IDs (you should replace these with your actual test IDs)
# ============================================================================

# Warm users (>=10 ratings)
WARM_USER_IDS = [11676, 98391, 189835, 153662, 23902, 171118, 235105, 76499, 16795, 248718]

# Cold users (<10 ratings)
COLD_USER_IDS = [248965, 249650, 249939, 250634, 251575, 251744, 252628, 253310, 258352, 259734]

# Users with no favorite subjects
NO_SUBJECT_USER_IDS = [278860]

# Books for similarity testing
# Should include books with:
# - ALS data (for behavioral similarity)
# - Real subjects (for subject similarity)
# - Mix of popular and niche books
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


# ============================================================================
# Performance Measurement Utilities
# ============================================================================


class LatencyStats:
    """Collect and analyze latency statistics."""

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
    Measure endpoint latency with warmup runs.

    Returns:
        Tuple of (LatencyStats, last_response_json)
    """
    stats = LatencyStats(f"GET {endpoint}")

    # Warmup runs (not measured)
    for _ in range(warmup_runs):
        client.get(endpoint, params=params)

    # Measurement runs
    last_response = None
    for _ in range(measurement_runs):
        start = time.perf_counter()
        response = client.get(endpoint, params=params)
        duration_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200, f"Request failed: {response.status_code}"
        stats.add(duration_ms)
        last_response = response.json()

    return stats, last_response


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def client(db: Session):
    """Create a test client for the FastAPI app."""
    from main import app

    return TestClient(app)


@pytest.fixture
def performance_report():
    """Collect all performance stats for final report."""
    report = {}
    yield report

    # Print summary report
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 80)
    for test_name, stats in report.items():
        print(f"\n{test_name}")
        print("-" * 80)
        print(stats)
    print("\n" + "=" * 80)


# ============================================================================
# Recommendation Endpoint Tests
# ============================================================================


@pytest.mark.parametrize(
    "user_id,mode",
    [
        *[(uid, "auto") for uid in WARM_USER_IDS[:3]],  # Test first 3 warm users
        *[(uid, "behavioral") for uid in WARM_USER_IDS[:2]],  # Explicit behavioral mode
    ],
)
def test_warm_user_recommendations_latency(
    client: TestClient, performance_report: Dict, user_id: int, mode: str
):
    """Test recommendation latency for warm users (>=10 ratings)."""
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": mode}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    # Verify response structure
    assert isinstance(response, list), "Response should be a list"
    assert len(response) > 0, "Should return recommendations"
    assert len(response) <= 200, "Should not exceed top_n"

    # Verify each recommendation has required fields
    for item in response[:5]:  # Check first 5
        assert "item_idx" in item
        assert "title" in item
        assert "score" in item

    test_key = f"recommend_warm_user_{user_id}_mode_{mode}"
    performance_report[test_key] = stats


@pytest.mark.parametrize("user_id", COLD_USER_IDS[:3])
def test_cold_user_recommendations_latency(
    client: TestClient, performance_report: Dict, user_id: int
):
    """Test recommendation latency for cold users (<10 ratings)."""
    endpoint = "/profile/recommend"
    params = {
        "user": str(user_id),
        "_id": True,
        "top_n": 200,
        "mode": "auto",  # Should route to cold strategy
    }

    stats, response = measure_endpoint_latency(client, endpoint, params)

    # Verify response
    assert isinstance(response, list)
    assert len(response) > 0

    test_key = f"recommend_cold_user_{user_id}"
    performance_report[test_key] = stats


@pytest.mark.parametrize("user_id", NO_SUBJECT_USER_IDS[:2])
def test_no_subject_user_recommendations_latency(
    client: TestClient, performance_report: Dict, user_id: int
):
    """Test recommendation latency for users with no favorite subjects."""
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200, "mode": "auto"}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    # Verify response (should still return results, likely Bayesian-based)
    assert isinstance(response, list)
    assert len(response) > 0

    test_key = f"recommend_no_subjects_user_{user_id}"
    performance_report[test_key] = stats


@pytest.mark.parametrize("top_n", [50, 200, 500])
def test_recommendations_varying_top_n(client: TestClient, performance_report: Dict, top_n: int):
    """Test how recommendation latency scales with top_n."""
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs provided")

    user_id = WARM_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": top_n, "mode": "auto"}

    stats, response = measure_endpoint_latency(client, endpoint, params)

    assert len(response) <= top_n

    test_key = f"recommend_top_n_{top_n}"
    performance_report[test_key] = stats


# ============================================================================
# Book Similarity Endpoint Tests
# ============================================================================


@pytest.mark.parametrize(
    "book_id,mode",
    [
        *[(bid, "subject") for bid in TEST_BOOK_IDS[:3]],
        *[(bid, "als") for bid in TEST_BOOK_IDS[:2]],  # Only books with ALS data
        *[(bid, "hybrid") for bid in TEST_BOOK_IDS[:2]],
    ],
)
def test_similar_books_latency(
    client: TestClient, performance_report: Dict, book_id: int, mode: str
):
    """Test similarity search latency across different modes."""
    endpoint = f"/book/{book_id}/similar"
    params = {"mode": mode, "top_k": 200}

    # Handle potential 422 for ALS mode if book doesn't have ALS data
    try:
        stats, response = measure_endpoint_latency(
            client, endpoint, params, measurement_runs=MEASUREMENT_RUNS
        )

        # Verify response
        assert isinstance(response, list)
        assert len(response) > 0
        assert len(response) <= 200

        # Verify each result has required fields
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
    """Test how hybrid similarity performs with different alpha values."""
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
    """Test how similarity search scales with top_k."""
    if not TEST_BOOK_IDS:
        pytest.skip("No test book IDs provided")

    book_id = TEST_BOOK_IDS[0]
    endpoint = f"/book/{book_id}/similar"
    params = {
        "mode": "subject",  # Use subject mode as it should work for all books
        "top_k": top_k,
    }

    stats, response = measure_endpoint_latency(client, endpoint, params)

    assert len(response) <= top_k

    test_key = f"similar_top_k_{top_k}"
    performance_report[test_key] = stats


# ============================================================================
# Stress/Load Tests
# ============================================================================


def test_concurrent_recommendation_requests(client: TestClient, performance_report: Dict):
    """Test latency under simulated concurrent load."""
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs provided")

    import concurrent.futures

    endpoint = "/profile/recommend"
    user_ids = WARM_USER_IDS[:5]  # Use first 5 users

    def make_request(user_id: int) -> float:
        """Make a single request and return duration in ms."""
        start = time.perf_counter()
        response = client.get(
            endpoint, params={"user": str(user_id), "_id": True, "top_n": 200, "mode": "auto"}
        )
        duration_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200
        return duration_ms

    # Sequential baseline
    sequential_stats = LatencyStats("sequential_requests")
    for user_id in user_ids:
        for _ in range(3):  # 3 requests per user
            duration = make_request(user_id)
            sequential_stats.add(duration)

    # Concurrent requests (5 threads)
    concurrent_stats = LatencyStats("concurrent_requests_5_threads")
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

    # Verify concurrent isn't dramatically slower
    seq_mean = sequential_stats.get_stats()["mean_ms"]
    conc_mean = concurrent_stats.get_stats()["mean_ms"]

    print(f"\nConcurrency overhead: {((conc_mean / seq_mean) - 1) * 100:.1f}%")


# ============================================================================
# Model Loading/Cache Tests
# ============================================================================


def test_cold_start_latency(client: TestClient, performance_report: Dict):
    """
    Test first request latency (cold start) vs subsequent requests.
    This simulates what happens after model reload or server restart.
    """
    if not WARM_USER_IDS:
        pytest.skip("No warm user IDs provided")

    # Force model reload
    admin_secret = os.getenv("ADMIN_SECRET")
    if admin_secret:
        response = client.post("/admin/reload_models", params={"secret": admin_secret})
        assert response.status_code == 200
        time.sleep(2)  # Give models time to fully reload

    user_id = WARM_USER_IDS[0]
    endpoint = "/profile/recommend"
    params = {"user": str(user_id), "_id": True, "top_n": 200}

    # First request (cold start)
    cold_stats = LatencyStats("cold_start_first_request")
    start = time.perf_counter()
    response = client.get(endpoint, params=params)
    duration_ms = (time.perf_counter() - start) * 1000
    assert response.status_code == 200
    cold_stats.add(duration_ms)

    # Subsequent requests (warm cache)
    warm_stats = LatencyStats("warm_cache_requests")
    for _ in range(10):
        start = time.perf_counter()
        response = client.get(endpoint, params=params)
        duration_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200
        warm_stats.add(duration_ms)

    performance_report["cold_start"] = cold_stats
    performance_report["warm_cache"] = warm_stats

    print(
        f"\nCold start overhead: {cold_stats.get_stats()['mean_ms'] - warm_stats.get_stats()['mean_ms']:.2f}ms"
    )


# ============================================================================
# Export Results for Comparison
# ============================================================================


def test_export_baseline_results(performance_report: Dict):
    """
    Export performance results to JSON for later comparison.
    Run this after refactoring to compare results.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    output_dir = Path(__file__).parent / "performance_baselines"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"baseline_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "measurement_runs": MEASUREMENT_RUNS,
        },
        "results": {name: stats.get_stats() for name, stats in performance_report.items()},
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline results exported to: {output_file}")


if __name__ == "__main__":
    # Allow running directly for quick tests
    import os

    os.environ.setdefault("TESTING", "1")
    pytest.main([__file__, "-v", "-s"])
