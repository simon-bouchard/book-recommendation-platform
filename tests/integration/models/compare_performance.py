# test/integration/models/compare_performance.py
"""
Performance Comparison Tool
============================

Compare performance baseline (before refactor) with current results (after refactor).

Usage:
    python tests/integration/models/compare_performance.py baseline_20241216_143022.json baseline_20241216_150045.json
    python tests/integration/models/compare_performance.py --auto  # Compare latest two baselines
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MetricComparison:
    """Comparison of a single metric between two runs."""

    test_name: str
    metric: str
    before: float
    after: float
    diff_ms: float
    diff_pct: float

    @property
    def is_regression(self) -> bool:
        """Check if this is a performance regression (>5% slower)."""
        return self.diff_pct > 5.0

    @property
    def is_improvement(self) -> bool:
        """Check if this is a performance improvement (>5% faster)."""
        return self.diff_pct < -5.0

    @property
    def status(self) -> str:
        """Get status emoji and label."""
        if self.is_regression:
            return "🔴 REGRESSION"
        elif self.is_improvement:
            return "🟢 IMPROVEMENT"
        else:
            return "⚪ NEUTRAL"


class PerformanceComparator:
    """Compare two performance baseline reports."""

    def __init__(self, before_file: Path, after_file: Path):
        self.before_file = before_file
        self.after_file = after_file

        with open(before_file) as f:
            self.before_data = json.load(f)

        with open(after_file) as f:
            self.after_data = json.load(f)

        self.comparisons: List[MetricComparison] = []
        self._compute_comparisons()

    def _compute_comparisons(self):
        """Compute all metric comparisons."""
        before_results = self.before_data.get("results", {})
        after_results = self.after_data.get("results", {})

        # Find common tests
        common_tests = set(before_results.keys()) & set(after_results.keys())

        for test_name in sorted(common_tests):
            before_metrics = before_results[test_name]
            after_metrics = after_results[test_name]

            # Compare key metrics
            for metric in ["mean_ms", "median_ms", "p95_ms"]:
                if metric in before_metrics and metric in after_metrics:
                    before_val = before_metrics[metric]
                    after_val = after_metrics[metric]
                    diff_ms = after_val - before_val
                    diff_pct = (diff_ms / before_val * 100) if before_val > 0 else 0

                    comp = MetricComparison(
                        test_name=test_name,
                        metric=metric,
                        before=before_val,
                        after=after_val,
                        diff_ms=diff_ms,
                        diff_pct=diff_pct,
                    )
                    self.comparisons.append(comp)

    def get_regressions(self) -> List[MetricComparison]:
        """Get all performance regressions."""
        return [c for c in self.comparisons if c.is_regression]

    def get_improvements(self) -> List[MetricComparison]:
        """Get all performance improvements."""
        return [c for c in self.comparisons if c.is_improvement]

    def get_summary(self) -> Dict:
        """Get comparison summary statistics."""
        regressions = self.get_regressions()
        improvements = self.get_improvements()
        neutral = [c for c in self.comparisons if not c.is_regression and not c.is_improvement]

        return {
            "total_comparisons": len(self.comparisons),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "neutral": len(neutral),
            "max_regression_pct": max((c.diff_pct for c in regressions), default=0),
            "max_improvement_pct": min((c.diff_pct for c in improvements), default=0),
        }

    def print_report(self):
        """Print formatted comparison report."""
        print("=" * 100)
        print("PERFORMANCE COMPARISON REPORT")
        print("=" * 100)

        # Header
        print(f"\n📊 Comparing:")
        print(f"  Before: {self.before_file.name} ({self.before_data.get('timestamp', 'unknown')})")
        print(f"  After:  {self.after_file.name} ({self.after_data.get('timestamp', 'unknown')})")

        # Summary
        summary = self.get_summary()
        print(f"\n📈 Summary:")
        print(f"  Total comparisons: {summary['total_comparisons']}")
        print(f"  🔴 Regressions: {summary['regressions']}")
        print(f"  🟢 Improvements: {summary['improvements']}")
        print(f"  ⚪ Neutral: {summary['neutral']}")

        if summary["regressions"] > 0:
            print(f"  Worst regression: {summary['max_regression_pct']:.1f}% slower")
        if summary["improvements"] > 0:
            print(f"  Best improvement: {abs(summary['max_improvement_pct']):.1f}% faster")

        # Regressions (if any)
        regressions = self.get_regressions()
        if regressions:
            print(f"\n{'=' * 100}")
            print("🔴 PERFORMANCE REGRESSIONS (>5% slower)")
            print("=" * 100)

            # Group by test name
            by_test = {}
            for comp in regressions:
                by_test.setdefault(comp.test_name, []).append(comp)

            for test_name in sorted(by_test.keys()):
                print(f"\n{test_name}:")
                for comp in sorted(by_test[test_name], key=lambda c: c.diff_pct, reverse=True):
                    print(
                        f"  {comp.metric:12s} | Before: {comp.before:7.2f}ms → After: {comp.after:7.2f}ms | "
                        f"{comp.diff_ms:+7.2f}ms ({comp.diff_pct:+6.1f}%)"
                    )

        # Improvements
        improvements = self.get_improvements()
        if improvements:
            print(f"\n{'=' * 100}")
            print("🟢 PERFORMANCE IMPROVEMENTS (>5% faster)")
            print("=" * 100)

            by_test = {}
            for comp in improvements:
                by_test.setdefault(comp.test_name, []).append(comp)

            for test_name in sorted(by_test.keys()):
                print(f"\n{test_name}:")
                for comp in sorted(by_test[test_name], key=lambda c: c.diff_pct):
                    print(
                        f"  {comp.metric:12s} | Before: {comp.before:7.2f}ms → After: {comp.after:7.2f}ms | "
                        f"{comp.diff_ms:+7.2f}ms ({comp.diff_pct:+6.1f}%)"
                    )

        # Detailed comparison (mean_ms only, sorted by absolute change)
        print(f"\n{'=' * 100}")
        print("📋 DETAILED COMPARISON (mean_ms)")
        print("=" * 100)

        mean_comps = [c for c in self.comparisons if c.metric == "mean_ms"]
        mean_comps.sort(key=lambda c: abs(c.diff_pct), reverse=True)

        print(
            f"\n{'Test Name':<50} {'Before':>10} {'After':>10} {'Diff':>10} {'Diff %':>10} {'Status':>15}"
        )
        print("-" * 100)

        for comp in mean_comps:
            status = comp.status
            print(
                f"{comp.test_name:<50} {comp.before:>9.2f}ms {comp.after:>9.2f}ms "
                f"{comp.diff_ms:>+9.2f}ms {comp.diff_pct:>+9.1f}% {status:>15}"
            )

        # Final verdict
        print(f"\n{'=' * 100}")
        if summary["regressions"] > 0:
            print("⚠️  VERDICT: Performance regressions detected!")
            print(f"   {summary['regressions']} test(s) are >5% slower")
            return False
        elif summary["improvements"] > summary["total_comparisons"] * 0.3:
            print("🎉 VERDICT: Significant performance improvements!")
            print(f"   {summary['improvements']} test(s) are >5% faster")
        else:
            print("✅ VERDICT: Performance is stable")
            print("   No significant regressions detected")

        print("=" * 100)
        return True

    def export_html_report(self, output_file: Path):
        """Export comparison as interactive HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Performance Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
        }}
        .card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-collapse: collapse;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .regression {{ color: #e53e3e; }}
        .improvement {{ color: #38a169; }}
        .neutral {{ color: #718096; }}
        .status {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .status.regression {{ background: #fed7d7; color: #c53030; }}
        .status.improvement {{ background: #c6f6d5; color: #22543d; }}
        .status.neutral {{ background: #e2e8f0; color: #4a5568; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Comparison Report</h1>
        <p>Before: {self.before_file.name} ({self.before_data.get("timestamp", "unknown")})</p>
        <p>After: {self.after_file.name} ({self.after_data.get("timestamp", "unknown")})</p>
    </div>

    <div class="summary">
        <div class="card">
            <h3>Total Tests</h3>
            <div class="value">{self.get_summary()["total_comparisons"]}</div>
        </div>
        <div class="card">
            <h3>Regressions</h3>
            <div class="value regression">{self.get_summary()["regressions"]}</div>
        </div>
        <div class="card">
            <h3>Improvements</h3>
            <div class="value improvement">{self.get_summary()["improvements"]}</div>
        </div>
        <div class="card">
            <h3>Neutral</h3>
            <div class="value neutral">{self.get_summary()["neutral"]}</div>
        </div>
    </div>

    <div class="card">
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Metric</th>
                    <th>Before (ms)</th>
                    <th>After (ms)</th>
                    <th>Difference</th>
                    <th>Change %</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

        for comp in sorted(self.comparisons, key=lambda c: abs(c.diff_pct), reverse=True):
            status_class = (
                "regression"
                if comp.is_regression
                else "improvement"
                if comp.is_improvement
                else "neutral"
            )
            status_text = (
                "Regression"
                if comp.is_regression
                else "Improvement"
                if comp.is_improvement
                else "Neutral"
            )

            html += f"""
                <tr>
                    <td>{comp.test_name}</td>
                    <td>{comp.metric}</td>
                    <td>{comp.before:.2f}</td>
                    <td>{comp.after:.2f}</td>
                    <td class="{status_class}">{comp.diff_ms:+.2f}</td>
                    <td class="{status_class}">{comp.diff_pct:+.1f}%</td>
                    <td><span class="status {status_class}">{status_text}</span></td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        with open(output_file, "w") as f:
            f.write(html)

        print(f"\n📄 HTML report exported to: {output_file}")


def find_latest_baselines(baseline_dir: Path) -> Tuple[Path, Path]:
    """Find the two most recent baseline files."""
    baselines = sorted(baseline_dir.glob("baseline_*.json"), reverse=True)

    if len(baselines) < 2:
        raise ValueError(f"Need at least 2 baseline files, found {len(baselines)}")

    return baselines[1], baselines[0]  # Before, After (oldest, newest)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare performance baselines")
    parser.add_argument("before", nargs="?", help="Before baseline file")
    parser.add_argument("after", nargs="?", help="After baseline file")
    parser.add_argument("--auto", action="store_true", help="Auto-compare latest two baselines")
    parser.add_argument("--html", action="store_true", help="Export HTML report")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with error code if regressions detected",
    )

    args = parser.parse_args()

    baseline_dir = Path(__file__).parent / "performance_baselines"

    if args.auto:
        try:
            before_file, after_file = find_latest_baselines(baseline_dir)
            print(f"Auto-detected:")
            print(f"  Before: {before_file.name}")
            print(f"  After:  {after_file.name}")
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
    else:
        if not args.before or not args.after:
            parser.print_help()
            sys.exit(1)

        before_file = (
            baseline_dir / args.before if not Path(args.before).is_absolute() else Path(args.before)
        )
        after_file = (
            baseline_dir / args.after if not Path(args.after).is_absolute() else Path(args.after)
        )

        if not before_file.exists():
            print(f"❌ Before file not found: {before_file}")
            sys.exit(1)

        if not after_file.exists():
            print(f"❌ After file not found: {after_file}")
            sys.exit(1)

    # Run comparison
    comparator = PerformanceComparator(before_file, after_file)
    no_regressions = comparator.print_report()

    # Export HTML if requested
    if args.html:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = baseline_dir / f"comparison_{timestamp}.html"
        comparator.export_html_report(html_file)

    # Exit with error if regressions detected and flag is set
    if args.fail_on_regression and not no_regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
