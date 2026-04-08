# tests/integration/models/compare_performance.py
"""
Compare two performance baseline JSON files produced by test_models_performance.py.

Results are grouped by request type (als_recs, subject_recs, popular_recs,
subject_sim, als_sim, hybrid_sim) so the summary mirrors the structure of
analyze_baseline.py and is easy to read at a glance.

Usage:
    python tests/integration/models/compare_performance.py before.json after.json
    python tests/integration/models/compare_performance.py --auto
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Request-type classification
# Kept in sync with analyze_baseline.REQUEST_TYPE_PATTERNS.
# ---------------------------------------------------------------------------

REQUEST_TYPES: List[Tuple[str, str]] = [
    ("recommend_als", "Recommend  ALS recs"),
    ("recommend_subject", "Recommend  Subject recs"),
    ("recommend_popular", "Recommend  Popular (cold no subjects)"),
    ("similar_subject", "Similarity Subject"),
    ("similar_als", "Similarity ALS"),
    ("similar_hybrid", "Similarity Hybrid"),
    ("concurrent", "Concurrency"),
]

_PATTERNS: Dict[str, callable] = {
    "recommend_als": lambda k: (
        ("recommend_warm_user_" in k and ("mode_auto" in k or "mode_behavioral" in k))
        or "recommend_top_n_" in k
    ),
    "recommend_subject": lambda k: (
        ("recommend_warm_user_" in k and "forced_subject" in k)
        or "recommend_cold_with_subjects" in k
        or "recommend_cold_w_" in k
    ),
    "recommend_popular": lambda k: "recommend_cold_without_subjects" in k,
    "similar_subject": lambda k: "similar_book_" in k
    and "mode_subject" in k
    or "similar_top_k_" in k,
    "similar_als": lambda k: "similar_book_" in k and "mode_als" in k,
    "similar_hybrid": lambda k: (
        ("similar_book_" in k and "mode_hybrid" in k) or "similar_hybrid_alpha_" in k
    ),
    "concurrent": lambda k: "concurrent_" in k,
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
class GroupComparison:
    """Aggregated before/after comparison for one request-type group."""

    type_key: str
    label: str
    n_tests: int
    before_median: float
    after_median: float

    @property
    def diff_ms(self) -> float:
        return self.after_median - self.before_median

    @property
    def diff_pct(self) -> float:
        return (self.diff_ms / self.before_median * 100) if self.before_median > 0 else 0.0

    @property
    def is_regression(self) -> bool:
        return self.diff_pct > 5.0

    @property
    def is_improvement(self) -> bool:
        return self.diff_pct < -5.0

    @property
    def status(self) -> str:
        if self.is_regression:
            return "REGRESSION"
        if self.is_improvement:
            return "IMPROVEMENT"
        return "neutral"


@dataclass
class TestComparison:
    """Before/after comparison for a single individual test (median_ms)."""

    test_name: str
    type_key: str
    before: float
    after: float

    @property
    def diff_ms(self) -> float:
        return self.after - self.before

    @property
    def diff_pct(self) -> float:
        return (self.diff_ms / self.before * 100) if self.before > 0 else 0.0

    @property
    def is_regression(self) -> bool:
        return self.diff_pct > 5.0

    @property
    def is_improvement(self) -> bool:
        return self.diff_pct < -5.0


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class PerformanceComparator:
    """Compare two performance baseline JSON files."""

    def __init__(self, before_file: Path, after_file: Path):
        self.before_file = before_file
        self.after_file = after_file

        with open(before_file) as f:
            self.before_data = json.load(f)
        with open(after_file) as f:
            self.after_data = json.load(f)

        self._test_comparisons: List[TestComparison] = self._build_test_comparisons()
        self._group_comparisons: List[GroupComparison] = self._build_group_comparisons()

    def _build_test_comparisons(self) -> List[TestComparison]:
        before = self.before_data.get("results", {})
        after = self.after_data.get("results", {})
        common = set(before) & set(after)

        comparisons = []
        for name in sorted(common):
            b = before[name].get("median_ms")
            a = after[name].get("median_ms")
            if b is None or a is None:
                continue
            comparisons.append(
                TestComparison(
                    test_name=name,
                    type_key=classify(name),
                    before=b,
                    after=a,
                )
            )
        return comparisons

    def _build_group_comparisons(self) -> List[GroupComparison]:
        dict(REQUEST_TYPES)
        buckets: Dict[str, List[TestComparison]] = defaultdict(list)
        for tc in self._test_comparisons:
            buckets[tc.type_key].append(tc)

        groups = []
        for type_key, label in REQUEST_TYPES:
            tests = buckets.get(type_key, [])
            if not tests:
                continue
            groups.append(
                GroupComparison(
                    type_key=type_key,
                    label=label,
                    n_tests=len(tests),
                    before_median=sum(t.before for t in tests) / len(tests),
                    after_median=sum(t.after for t in tests) / len(tests),
                )
            )
        return groups

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_report(self, detailed: bool = False) -> bool:
        """
        Print the comparison report. Returns True if no regressions were found.
        """
        W = 80
        print("=" * W)
        print("PERFORMANCE COMPARISON")
        print("=" * W)
        print(f"\n  Before : {self.before_file.name}  ({self.before_data.get('timestamp', '?')})")
        print(f"  After  : {self.after_file.name}  ({self.after_data.get('timestamp', '?')})")

        self._print_grouped_summary(W)

        if detailed:
            self._print_detailed_breakdown(W)

        return self._print_verdict(W)

    def _print_grouped_summary(self, width: int) -> None:
        print(f"\n{'=' * width}")
        print("BY REQUEST TYPE  (median ms, averaged across parametrized cases)")
        print("=" * width)

        col = 36
        print(
            f"\n  {'Request Type':<{col}} {'Before':>9}  {'After':>9}  {'Diff':>9}  {'Diff %':>7}  Status"
        )
        print(f"  {'-' * (col + 52)}")

        for g in self._group_comparisons:
            sign = "+" if g.diff_ms >= 0 else ""
            status = f"  << {g.status}" if g.status != "neutral" else ""
            print(
                f"  {g.label:<{col}} "
                f"{g.before_median:>8.2f}ms "
                f"{g.after_median:>8.2f}ms "
                f"{sign}{g.diff_ms:>8.2f}ms "
                f"{sign}{g.diff_pct:>6.1f}%"
                f"{status}"
            )

    def _print_detailed_breakdown(self, width: int) -> None:
        dict(REQUEST_TYPES)
        buckets: Dict[str, List[TestComparison]] = defaultdict(list)
        for tc in self._test_comparisons:
            buckets[tc.type_key].append(tc)

        print(f"\n{'=' * width}")
        print("INDIVIDUAL TESTS  (median ms)")
        print("=" * width)

        for type_key, label in REQUEST_TYPES:
            tests = buckets.get(type_key)
            if not tests:
                continue
            print(f"\n  {label}")
            print(f"  {'-' * 70}")
            for t in sorted(tests, key=lambda x: x.diff_pct, reverse=True):
                sign = "+" if t.diff_ms >= 0 else ""
                flag = (
                    "  << REGRESSION"
                    if t.is_regression
                    else ("  << improvement" if t.is_improvement else "")
                )
                print(
                    f"    {t.test_name:<52} "
                    f"{t.before:>7.2f}ms -> {t.after:>7.2f}ms  "
                    f"{sign}{t.diff_ms:>6.2f}ms ({sign}{t.diff_pct:.1f}%)"
                    f"{flag}"
                )

    def _print_verdict(self, width: int) -> bool:
        regressions = [g for g in self._group_comparisons if g.is_regression]
        improvements = [g for g in self._group_comparisons if g.is_improvement]

        print(f"\n{'=' * width}")
        if regressions:
            print(f"VERDICT: {len(regressions)} group(s) regressed (>5% slower):")
            for g in regressions:
                print(f"  {g.label}  {g.diff_pct:+.1f}%")
            print("=" * width)
            return False

        if improvements:
            print(f"VERDICT: {len(improvements)} group(s) improved (>5% faster):")
            for g in improvements:
                print(f"  {g.label}  {g.diff_pct:+.1f}%")
        else:
            print("VERDICT: Performance is stable — no significant changes detected.")
        print("=" * width)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_latest_baselines(baseline_dir: Path) -> Tuple[Path, Path]:
    """Return (before, after) as the two most recent baseline files."""
    baselines = sorted(baseline_dir.glob("baseline_*.json"), reverse=True)
    if len(baselines) < 2:
        raise ValueError(
            f"Need at least 2 baseline files in {baseline_dir}, found {len(baselines)}"
        )
    return baselines[1], baselines[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare two performance baselines")
    parser.add_argument("before", nargs="?", help="Before baseline filename or path")
    parser.add_argument("after", nargs="?", help="After baseline filename or path")
    parser.add_argument(
        "--auto", action="store_true", help="Auto-select the two most recent baselines"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show per-test breakdown under each group"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if any group regresses",
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

    comparator = PerformanceComparator(before_file, after_file)
    no_regressions = comparator.print_report(detailed=args.detailed)

    if args.fail_on_regression and not no_regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
