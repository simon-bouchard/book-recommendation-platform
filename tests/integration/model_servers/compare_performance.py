# tests/integration/model_servers/compare_performance.py
"""
Compare two model server performance baseline JSON files.

Results are grouped by operation type (embedder, subject_sim, als_sim,
hybrid_sim, subject_recs) and show both HTTP and compute latency deltas
side by side, making it easy to see whether a BLAS or matmul configuration
change improved the compute path, the end-to-end path, or both.

Usage:
    python tests/integration/model_servers/compare_performance.py before.json after.json
    python tests/integration/model_servers/compare_performance.py --auto
    python tests/integration/model_servers/compare_performance.py --auto --detailed
    python tests/integration/model_servers/compare_performance.py --auto --fail-on-regression
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Operation-type classification
# ---------------------------------------------------------------------------

OPERATION_TYPES: List[Tuple[str, str]] = [
    ("embedder", "Embedder"),
    ("subject_sim", "Similarity / Subject"),
    ("als_sim", "Similarity / ALS"),
    ("hybrid_sim", "Similarity / Hybrid"),
    ("subject_recs", "Similarity / Subject Recs"),
    ("als_recs", "ALS Recs"),
]

_PATTERNS: Dict[str, callable] = {
    "embedder": lambda k: k.startswith("embedder_"),
    "subject_sim": lambda k: "subject_sim" in k,
    "als_sim": lambda k: "als_sim" in k,
    "hybrid_sim": lambda k: "hybrid_sim" in k,
    "subject_recs": lambda k: "subject_recs" in k,
    "als_recs": lambda k: k.startswith("als_"),
}


def classify(test_name: str) -> str:
    for type_key, pattern_fn in _PATTERNS.items():
        if pattern_fn(test_name):
            return type_key
    return "other"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TestComparison:
    """Before/after comparison for a single test (HTTP and compute median_ms)."""

    test_name: str
    type_key: str
    before_http: float
    after_http: float
    before_http_stdev: Optional[float]
    after_http_stdev: Optional[float]
    before_compute: Optional[float]
    after_compute: Optional[float]
    before_compute_stdev: Optional[float]
    after_compute_stdev: Optional[float]

    @property
    def http_diff_ms(self) -> float:
        return self.after_http - self.before_http

    @property
    def http_diff_pct(self) -> float:
        return (self.http_diff_ms / self.before_http * 100) if self.before_http > 0 else 0.0

    @property
    def compute_diff_ms(self) -> Optional[float]:
        if self.before_compute is None or self.after_compute is None:
            return None
        return self.after_compute - self.before_compute

    @property
    def compute_diff_pct(self) -> Optional[float]:
        if self.before_compute is None or self.after_compute is None or self.before_compute == 0:
            return None
        return ((self.after_compute - self.before_compute) / self.before_compute) * 100


@dataclass
class GroupComparison:
    """Aggregated before/after comparison for one operation-type group."""

    type_key: str
    label: str
    n_tests: int
    before_http: float
    after_http: float
    before_compute: Optional[float]
    after_compute: Optional[float]

    @property
    def http_diff_ms(self) -> float:
        return self.after_http - self.before_http

    @property
    def http_diff_pct(self) -> float:
        return (self.http_diff_ms / self.before_http * 100) if self.before_http > 0 else 0.0

    @property
    def compute_diff_ms(self) -> Optional[float]:
        if self.before_compute is None or self.after_compute is None:
            return None
        return self.after_compute - self.before_compute

    @property
    def compute_diff_pct(self) -> Optional[float]:
        if self.before_compute is None or self.after_compute is None or self.before_compute == 0:
            return None
        return ((self.after_compute - self.before_compute) / self.before_compute) * 100

    @property
    def is_regression(self) -> bool:
        return self.http_diff_pct > 5.0

    @property
    def is_improvement(self) -> bool:
        return self.http_diff_pct < -5.0

    @property
    def status(self) -> str:
        if self.is_regression:
            return "REGRESSION"
        if self.is_improvement:
            return "IMPROVEMENT"
        return "neutral"


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class ModelServerComparator:
    """Compare two model server performance baseline JSON files."""

    def __init__(self, before_file: Path, after_file: Path) -> None:
        self.before_file = before_file
        self.after_file = after_file

        with open(before_file) as f:
            self.before_data = json.load(f)
        with open(after_file) as f:
            self.after_data = json.load(f)

        self._test_comparisons = self._build_test_comparisons()
        self._group_comparisons = self._build_group_comparisons()

    def _build_test_comparisons(self) -> List[TestComparison]:
        before = self.before_data.get("results", {})
        after = self.after_data.get("results", {})
        common = set(before) & set(after)

        comparisons = []
        for name in sorted(common):
            b = before[name]
            a = after[name]
            if not b or not a:
                continue
            bh = b.get("median_ms")
            ah = a.get("median_ms")
            if bh is None or ah is None:
                continue
            bc = b.get("compute", {}).get("median_ms")
            ac = a.get("compute", {}).get("median_ms")
            comparisons.append(
                TestComparison(
                    test_name=name,
                    type_key=classify(name),
                    before_http=bh,
                    after_http=ah,
                    before_http_stdev=b.get("stdev_ms"),
                    after_http_stdev=a.get("stdev_ms"),
                    before_compute=bc,
                    after_compute=ac,
                    before_compute_stdev=b.get("compute", {}).get("stdev_ms"),
                    after_compute_stdev=a.get("compute", {}).get("stdev_ms"),
                )
            )
        return comparisons

    def _build_group_comparisons(self) -> List[GroupComparison]:
        buckets: Dict[str, List[TestComparison]] = defaultdict(list)
        for tc in self._test_comparisons:
            buckets[tc.type_key].append(tc)

        groups = []
        for type_key, label in OPERATION_TYPES:
            tests = buckets.get(type_key, [])
            if not tests:
                continue

            before_http = sum(t.before_http for t in tests) / len(tests)
            after_http = sum(t.after_http for t in tests) / len(tests)

            compute_tests = [
                t for t in tests if t.before_compute is not None and t.after_compute is not None
            ]
            before_compute = (
                sum(t.before_compute for t in compute_tests) / len(compute_tests)
                if compute_tests
                else None
            )
            after_compute = (
                sum(t.after_compute for t in compute_tests) / len(compute_tests)
                if compute_tests
                else None
            )

            groups.append(
                GroupComparison(
                    type_key=type_key,
                    label=label,
                    n_tests=len(tests),
                    before_http=before_http,
                    after_http=after_http,
                    before_compute=before_compute,
                    after_compute=after_compute,
                )
            )
        return groups

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_report(self, detailed: bool = False) -> bool:
        """Print the comparison report. Returns True if no regressions found."""
        W = 88
        print("=" * W)
        print("MODEL SERVER PERFORMANCE COMPARISON")
        print("=" * W)
        print(f"\n  Before : {self.before_file.name}  ({self.before_data.get('timestamp', '?')})")
        print(f"  After  : {self.after_file.name}  ({self.after_data.get('timestamp', '?')})")

        self._print_grouped_summary(W)

        if detailed:
            self._print_detailed_breakdown(W)

        return self._print_verdict(W)

    def _print_grouped_summary(self, width: int) -> None:
        has_compute = any(g.before_compute is not None for g in self._group_comparisons)

        print(f"\n{'=' * width}")
        if has_compute:
            print(
                "BY OPERATION TYPE  (median ms — HTTP | Compute, averaged across parametrized cases)"
            )
        else:
            print("BY OPERATION TYPE  (median ms, averaged across parametrized cases)")
        print("=" * width)

        col = 30
        if has_compute:
            print(
                f"\n  {'Operation':<{col}} "
                f"{'HTTP Before':>11}  {'HTTP After':>10}  {'HTTP Diff':>10}  {'HTTP%':>6}  "
                f"{'Cmp Before':>10}  {'Cmp After':>9}  {'Cmp%':>6}  Status"
            )
            print(f"  {'-' * (col + 78)}")
        else:
            print(
                f"\n  {'Operation':<{col}} "
                f"{'Before':>9}  {'After':>9}  {'Diff':>9}  {'Diff %':>7}  Status"
            )
            print(f"  {'-' * (col + 50)}")

        for g in self._group_comparisons:
            sign = "+" if g.http_diff_ms >= 0 else ""
            status = f"  << {g.status}" if g.status != "neutral" else ""

            if has_compute:
                if g.before_compute is not None and g.after_compute is not None:
                    cmp_sign = "+" if (g.compute_diff_pct or 0) >= 0 else ""
                    cmp_col = (
                        f"{g.before_compute:>9.2f}ms  "
                        f"{g.after_compute:>8.2f}ms  "
                        f"{cmp_sign}{g.compute_diff_pct:>5.1f}%"
                    )
                else:
                    cmp_col = f"{'n/a':>9}   {'n/a':>8}   {'n/a':>6}"

                print(
                    f"  {g.label:<{col}} "
                    f"{g.before_http:>10.2f}ms  "
                    f"{g.after_http:>9.2f}ms  "
                    f"{sign}{g.http_diff_ms:>8.2f}ms  "
                    f"{sign}{g.http_diff_pct:>5.1f}%  "
                    f"{cmp_col}"
                    f"{status}"
                )
            else:
                print(
                    f"  {g.label:<{col}} "
                    f"{g.before_http:>8.2f}ms "
                    f"{g.after_http:>8.2f}ms "
                    f"{sign}{g.http_diff_ms:>8.2f}ms "
                    f"{sign}{g.http_diff_pct:>6.1f}%"
                    f"{status}"
                )

    def _print_detailed_breakdown(self, width: int) -> None:
        buckets: Dict[str, List[TestComparison]] = defaultdict(list)
        for tc in self._test_comparisons:
            buckets[tc.type_key].append(tc)

        print(f"\n{'=' * width}")
        print("INDIVIDUAL TESTS  (median ms)")
        print("=" * width)

        for type_key, label in OPERATION_TYPES:
            tests = buckets.get(type_key)
            if not tests:
                continue
            print(f"\n  {label}")
            print(f"  {'-' * 78}")

            for t in sorted(tests, key=lambda x: x.http_diff_pct, reverse=True):
                sign = "+" if t.http_diff_ms >= 0 else ""
                flag = (
                    "  << REGRESSION"
                    if t.http_diff_pct > 5.0
                    else ("  << improvement" if t.http_diff_pct < -5.0 else "")
                )
                bh_stdev = f"±{t.before_http_stdev:.1f}" if t.before_http_stdev is not None else ""
                ah_stdev = f"±{t.after_http_stdev:.1f}" if t.after_http_stdev is not None else ""
                http_part = (
                    f"{t.before_http:>7.2f}ms{bh_stdev:<6} -> "
                    f"{t.after_http:>7.2f}ms{ah_stdev:<6}  "
                    f"{sign}{t.http_diff_ms:>6.2f}ms ({sign}{t.http_diff_pct:.1f}%)"
                )
                if t.compute_diff_pct is not None:
                    cmp_sign = "+" if (t.compute_diff_pct or 0) >= 0 else ""
                    bc_stdev = (
                        f"±{t.before_compute_stdev:.1f}"
                        if t.before_compute_stdev is not None
                        else ""
                    )
                    ac_stdev = (
                        f"±{t.after_compute_stdev:.1f}" if t.after_compute_stdev is not None else ""
                    )
                    cmp_part = (
                        f"  cmp: {t.before_compute:>6.2f}ms{bc_stdev:<6} -> "
                        f"{t.after_compute:>6.2f}ms{ac_stdev:<6} "
                        f"({cmp_sign}{t.compute_diff_pct:.1f}%)"
                    )
                else:
                    cmp_part = ""
                print(f"    {t.test_name:<42} {http_part}{cmp_part}{flag}")

    def _print_verdict(self, width: int) -> bool:
        regressions = [g for g in self._group_comparisons if g.is_regression]
        improvements = [g for g in self._group_comparisons if g.is_improvement]

        print(f"\n{'=' * width}")
        if regressions:
            print(f"VERDICT: {len(regressions)} group(s) regressed (>5% slower HTTP):")
            for g in regressions:
                print(f"  {g.label}  {g.http_diff_pct:+.1f}%")
            print("=" * width)
            return False

        if improvements:
            print(f"VERDICT: {len(improvements)} group(s) improved (>5% faster HTTP):")
            for g in improvements:
                cmp_note = ""
                if g.compute_diff_pct is not None:
                    cmp_sign = "+" if g.compute_diff_pct >= 0 else ""
                    cmp_note = f"  (compute: {cmp_sign}{g.compute_diff_pct:.1f}%)"
                print(f"  {g.label}  {g.http_diff_pct:+.1f}%{cmp_note}")
        else:
            print("VERDICT: Performance is stable — no significant changes detected.")
        print("=" * width)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_latest_baselines(baseline_dir: Path) -> Tuple[Path, Path]:
    """Return (before, after) as the two most recently modified baseline files."""
    baselines = sorted(baseline_dir.glob("model_servers_*.json"), key=lambda p: p.stat().st_mtime)
    if len(baselines) < 2:
        raise ValueError(
            f"Need at least 2 baseline files in {baseline_dir}, found {len(baselines)}"
        )
    return baselines[-2], baselines[-1]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two model server performance baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("before", nargs="?", help="Before baseline filename or path")
    parser.add_argument("after", nargs="?", help="After baseline filename or path")
    parser.add_argument(
        "--auto", action="store_true", help="Auto-select the two most recently modified baselines"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show per-test breakdown under each group"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if any group regresses (useful in CI)",
    )
    args = parser.parse_args()

    baseline_dir = Path(__file__).parent / "performance_baselines"

    if args.auto:
        try:
            before_file, after_file = find_latest_baselines(baseline_dir)
            print("Auto-detected baselines:")
            print(f"  Before : {before_file.name}")
            print(f"  After  : {after_file.name}\n")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        if not args.before or not args.after:
            parser.print_help()
            sys.exit(1)

        def resolve(p: str) -> Path:
            path = Path(p)
            return path if path.is_absolute() else baseline_dir / path

        before_file = resolve(args.before)
        after_file = resolve(args.after)

        for f in (before_file, after_file):
            if not f.exists():
                print(f"Error: file not found: {f}")
                sys.exit(1)

    comparator = ModelServerComparator(before_file, after_file)
    no_regressions = comparator.print_report(detailed=args.detailed)

    if args.fail_on_regression and not no_regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
