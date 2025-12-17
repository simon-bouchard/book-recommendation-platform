# tests/integration/models/analyze_baseline.py
"""
Analyzes a single performance baseline and provides summary statistics.
Shows average times grouped by request type.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RequestTypeStats:
    """Statistics for a group of requests of the same type."""

    request_type: str
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    def __str__(self) -> str:
        return (
            f"{self.request_type}:\n"
            f"  Count: {self.count}\n"
            f"  Mean: {self.mean_ms:.2f}ms\n"
            f"  Median: {self.median_ms:.2f}ms\n"
            f"  P95: {self.p95_ms:.2f}ms\n"
            f"  Range: [{self.min_ms:.2f}ms - {self.max_ms:.2f}ms]\n"
            f"  Stdev: {self.stdev_ms:.2f}ms"
        )


class BaselineAnalyzer:
    """Analyzes performance baseline and groups results by request type."""

    REQUEST_TYPE_PATTERNS = {
        "recommend_warm_als": lambda k: "recommend_warm_user_" in k
        and "mode_auto" in k
        or "mode_behavioral" in k,
        "recommend_warm_subject": lambda k: "recommend_warm_user_" in k and "forced_subject" in k,
        "recommend_cold_with_subjects": lambda k: "recommend_cold_with_subjects" in k,
        "recommend_cold_without_subjects": lambda k: "recommend_cold_without_subjects" in k,
        "recommend_varying_w": lambda k: "recommend_cold_w_" in k,
        "recommend_varying_topn": lambda k: "recommend_top_n_" in k,
        "similar_subject": lambda k: "similar_book_" in k and "mode_subject" in k,
        "similar_als": lambda k: "similar_book_" in k and "mode_als" in k,
        "similar_hybrid": lambda k: "similar_book_" in k
        and "mode_hybrid" in k
        or "similar_hybrid_alpha" in k,
        "similar_varying_topk": lambda k: "similar_top_k_" in k,
        "concurrent_load": lambda k: "concurrent_" in k,
        "cold_start": lambda k: k == "cold_start",
        "warm_cache": lambda k: k == "warm_cache",
    }

    def __init__(self, baseline_file: Path):
        self.baseline_file = baseline_file

        with open(baseline_file) as f:
            self.data = json.load(f)

        self.results = self.data.get("results", {})
        self.grouped = self._group_results()

    def _group_results(self) -> Dict[str, List[Dict]]:
        """Groups test results by request type."""
        grouped = defaultdict(list)

        for test_name, stats in self.results.items():
            matched = False
            for request_type, pattern_func in self.REQUEST_TYPE_PATTERNS.items():
                if pattern_func(test_name):
                    grouped[request_type].append({"test_name": test_name, **stats})
                    matched = True
                    break

            if not matched:
                grouped["other"].append({"test_name": test_name, **stats})

        return dict(grouped)

    def get_aggregated_stats(self) -> List[RequestTypeStats]:
        """Computes aggregated statistics for each request type."""
        aggregated = []

        for request_type, tests in sorted(self.grouped.items()):
            if not tests:
                continue

            means = [t["mean_ms"] for t in tests]
            medians = [t["median_ms"] for t in tests]
            p95s = [t["p95_ms"] for t in tests]
            mins = [t["min_ms"] for t in tests]
            maxs = [t["max_ms"] for t in tests]
            stdevs = [t["stdev_ms"] for t in tests]

            stats = RequestTypeStats(
                request_type=request_type,
                count=len(tests),
                mean_ms=sum(means) / len(means),
                median_ms=sum(medians) / len(medians),
                p95_ms=sum(p95s) / len(p95s),
                min_ms=min(mins),
                max_ms=max(maxs),
                stdev_ms=sum(stdevs) / len(stdevs),
            )
            aggregated.append(stats)

        return aggregated

    def print_summary(self):
        """Prints formatted summary of baseline performance."""
        print("=" * 80)
        print(f"PERFORMANCE BASELINE ANALYSIS")
        print("=" * 80)
        print(f"\nBaseline: {self.baseline_file.name}")
        print(f"Timestamp: {self.data.get('timestamp', 'unknown')}")
        print(f"Total test cases: {len(self.results)}")

        print("\n" + "=" * 80)
        print("SUMMARY BY REQUEST TYPE")
        print("=" * 80)

        aggregated = self.get_aggregated_stats()

        for stats in aggregated:
            print(f"\n{stats}")

        print("\n" + "=" * 80)
        print("PERFORMANCE CHARACTERISTICS")
        print("=" * 80)

        self._print_performance_insights(aggregated)

    def _print_performance_insights(self, stats: List[RequestTypeStats]):
        """Prints insights about relative performance."""
        stats_by_type = {s.request_type: s for s in stats}

        print("\nRecommendation Performance:")

        rec_types = [
            ("recommend_warm_als", "Warm (ALS)"),
            ("recommend_warm_subject", "Warm (forced subject)"),
            ("recommend_cold_with_subjects", "Cold (with subjects)"),
            ("recommend_cold_without_subjects", "Cold (without subjects)"),
        ]

        for key, label in rec_types:
            if key in stats_by_type:
                s = stats_by_type[key]
                print(f"  {label:30s} {s.mean_ms:7.2f}ms (median: {s.median_ms:7.2f}ms)")

        print("\nSimilarity Performance:")

        sim_types = [
            ("similar_subject", "Subject mode"),
            ("similar_als", "ALS mode"),
            ("similar_hybrid", "Hybrid mode"),
        ]

        for key, label in sim_types:
            if key in stats_by_type:
                s = stats_by_type[key]
                print(f"  {label:30s} {s.mean_ms:7.2f}ms (median: {s.median_ms:7.2f}ms)")

        if "cold_start" in stats_by_type and "warm_cache" in stats_by_type:
            cold = stats_by_type["cold_start"]
            warm = stats_by_type["warm_cache"]
            overhead = cold.mean_ms - warm.mean_ms
            overhead_pct = (overhead / warm.mean_ms) * 100
            print(f"\nCold Start Analysis:")
            print(f"  First request: {cold.mean_ms:.2f}ms")
            print(f"  Cached request: {warm.mean_ms:.2f}ms")
            print(f"  Overhead: {overhead:.2f}ms ({overhead_pct:.1f}%)")

        if "concurrent_load" in stats_by_type:
            conc = stats_by_type["concurrent_load"]
            print(f"\nConcurrency:")
            print(f"  Average latency under load: {conc.mean_ms:.2f}ms")

    def export_csv(self, output_file: Path):
        """Exports summary statistics to CSV."""
        import csv

        aggregated = self.get_aggregated_stats()

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Request Type",
                    "Count",
                    "Mean (ms)",
                    "Median (ms)",
                    "P95 (ms)",
                    "Min (ms)",
                    "Max (ms)",
                    "Stdev (ms)",
                ]
            )

            for stats in aggregated:
                writer.writerow(
                    [
                        stats.request_type,
                        stats.count,
                        f"{stats.mean_ms:.2f}",
                        f"{stats.median_ms:.2f}",
                        f"{stats.p95_ms:.2f}",
                        f"{stats.min_ms:.2f}",
                        f"{stats.max_ms:.2f}",
                        f"{stats.stdev_ms:.2f}",
                    ]
                )

        print(f"\nCSV exported to: {output_file}")

    def print_detailed_breakdown(self):
        """Prints detailed breakdown of all individual tests."""
        print("\n" + "=" * 80)
        print("DETAILED TEST BREAKDOWN")
        print("=" * 80)

        for request_type, tests in sorted(self.grouped.items()):
            print(f"\n{request_type.upper().replace('_', ' ')}")
            print("-" * 80)

            for test in sorted(tests, key=lambda t: t["mean_ms"], reverse=True):
                print(f"  {test['test_name']:60s} {test['mean_ms']:7.2f}ms")


def find_latest_baseline(baseline_dir: Path) -> Path:
    """Finds the most recent baseline file."""
    baselines = sorted(baseline_dir.glob("baseline_*.json"), reverse=True)

    if not baselines:
        raise ValueError(f"No baseline files found in {baseline_dir}")

    return baselines[0]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze performance baseline")
    parser.add_argument("baseline", nargs="?", help="Baseline file to analyze")
    parser.add_argument("--latest", action="store_true", help="Analyze latest baseline")
    parser.add_argument("--csv", action="store_true", help="Export to CSV")
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown")

    args = parser.parse_args()

    baseline_dir = Path(__file__).parent / "performance_baselines"

    if args.latest:
        try:
            baseline_file = find_latest_baseline(baseline_dir)
            print(f"Using latest baseline: {baseline_file.name}\n")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        if not args.baseline:
            parser.print_help()
            sys.exit(1)

        baseline_file = (
            baseline_dir / args.baseline
            if not Path(args.baseline).is_absolute()
            else Path(args.baseline)
        )

        if not baseline_file.exists():
            print(f"Error: Baseline file not found: {baseline_file}")
            sys.exit(1)

    analyzer = BaselineAnalyzer(baseline_file)
    analyzer.print_summary()

    if args.detailed:
        analyzer.print_detailed_breakdown()

    if args.csv:
        csv_file = baseline_file.with_suffix(".csv")
        analyzer.export_csv(csv_file)


if __name__ == "__main__":
    main()
