# tests/load/models/analyze_load.py
"""
Analyzes a single load test baseline produced by test_load.py.

Displays two views:

  Ramp analysis   -- latency at each concurrency level per scenario, showing
                     how each pipeline degrades as workers increase.
                     The concurrency_overhead_ratio column (mean latency at N
                     workers divided by mean at 1 worker) is the clearest
                     single number for degradation: 1.0 = no degradation,
                     1.5 = 50% overhead.

  Sustained analysis -- steady-state latency, throughput (RPS), and success
                        rate per scenario at the configured worker count.

Usage:
    python tests/load/models/analyze_load.py <baseline_file>
    python tests/load/models/analyze_load.py --latest
    python tests/load/models/analyze_load.py --latest --csv
    python tests/load/models/analyze_load.py --latest --detailed
"""

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Key classification
# Result keys: {scenario}_ramp_w{N}  |  {scenario}_sustained_w{N}
# ---------------------------------------------------------------------------

_SCENARIOS = [
    "warm_behavioral",
    "warm_subject",
    "cold_with_subjects",
    "cold_without_subjects",
    "similarity_subject",
    "similarity_als",
    "similarity_hybrid",
]

_SCENARIO_LABELS: dict[str, str] = {
    "warm_behavioral": "Warm / ALS behavioral",
    "warm_subject": "Warm / Subject path",
    "cold_with_subjects": "Cold / With subjects",
    "cold_without_subjects": "Cold / No subjects (Bayesian)",
    "similarity_subject": "Similarity / Subject (FAISS)",
    "similarity_als": "Similarity / ALS",
    "similarity_hybrid": "Similarity / Hybrid",
}


def _scenario_of(key: str) -> str | None:
    """Return the scenario name embedded in a result key, or None."""
    for s in _SCENARIOS:
        if key.startswith(s + "_ramp_") or key.startswith(s + "_sustained_"):
            return s
    return None


def _test_type_of(key: str) -> str | None:
    if "_ramp_w" in key:
        return "ramp"
    if "_sustained_w" in key:
        return "sustained"
    return None


def _workers_of(key: str) -> int | None:
    for marker in ("_ramp_w", "_sustained_w"):
        idx = key.find(marker)
        if idx != -1:
            try:
                return int(key[idx + len(marker) :])
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RampRow:
    """One row in the ramp table: latency stats at a given concurrency level."""

    workers: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    failures: int
    overhead_ratio: float | None


@dataclass
class SustainedRow:
    """One row in the sustained table: steady-state metrics for a scenario."""

    scenario: str
    workers: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    throughput_rps: float
    success_rate: float
    failures: int


@dataclass
class ScenarioRamp:
    """All ramp rows for one scenario, sorted by workers."""

    scenario: str
    rows: list[RampRow] = field(default_factory=list)

    def sorted_rows(self) -> list[RampRow]:
        return sorted(self.rows, key=lambda r: r.workers)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class LoadBaselineAnalyzer:
    """
    Parses a load baseline JSON file and provides formatted analysis views.

    Ramp results (test_type == "ramp") are grouped by scenario and displayed
    as a concurrency-level table. Sustained results (test_type == "sustained")
    are collected into a single summary table.
    """

    def __init__(self, baseline_file: Path) -> None:
        self.baseline_file = baseline_file
        with open(baseline_file) as f:
            self.data: dict[str, Any] = json.load(f)
        self.results: dict[str, dict] = self.data.get("results", {})
        self.config: dict[str, Any] = self.data.get("config", {})
        self._ramp_by_scenario: dict[str, ScenarioRamp] = {}
        self._sustained_rows: list[SustainedRow] = []
        self._parse()

    def _parse(self) -> None:
        for key, stats in self.results.items():
            scenario = _scenario_of(key)
            test_type = _test_type_of(key)
            workers = _workers_of(key)
            if scenario is None or test_type is None or workers is None:
                continue

            if test_type == "ramp":
                if scenario not in self._ramp_by_scenario:
                    self._ramp_by_scenario[scenario] = ScenarioRamp(scenario=scenario)
                self._ramp_by_scenario[scenario].rows.append(
                    RampRow(
                        workers=workers,
                        mean_ms=stats.get("mean_ms", 0.0),
                        median_ms=stats.get("median_ms", 0.0),
                        p95_ms=stats.get("p95_ms", 0.0),
                        failures=stats.get("failures", 0),
                        overhead_ratio=stats.get("concurrency_overhead_ratio"),
                    )
                )

            elif test_type == "sustained":
                self._sustained_rows.append(
                    SustainedRow(
                        scenario=scenario,
                        workers=workers,
                        mean_ms=stats.get("mean_ms", 0.0),
                        median_ms=stats.get("median_ms", 0.0),
                        p95_ms=stats.get("p95_ms", 0.0),
                        throughput_rps=stats.get("throughput_rps", 0.0),
                        success_rate=stats.get("success_rate", 0.0),
                        failures=stats.get("failures", 0),
                    )
                )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def print_summary(self, detailed: bool = False) -> None:
        W = 80
        print("=" * W)
        print("LOAD TEST BASELINE ANALYSIS")
        print("=" * W)
        print(f"\nBaseline : {self.baseline_file.name}")
        print(f"Timestamp: {self.data.get('timestamp', 'unknown')}")
        print(f"Results  : {len(self.results)} entries")
        self._print_config()
        self._print_ramp_section(W, detailed)
        self._print_sustained_section(W)

    def _print_config(self) -> None:
        if not self.config:
            return
        print("\nConfiguration:")
        levels = self.config.get("concurrency_levels", [])
        print(f"  Concurrency levels   : {levels}")
        print(f"  Requests per level   : {self.config.get('requests_per_level', '?')}")
        print(f"  Sustained workers    : {self.config.get('sustained_workers', '?')}")
        print(f"  Sustained duration   : {self.config.get('sustained_duration_s', '?')}s")

    def _print_ramp_section(self, width: int, detailed: bool) -> None:
        if not self._ramp_by_scenario:
            return

        print(f"\n{'=' * width}")
        print("RAMP TESTS  — latency degradation by concurrency level")
        print("=" * width)
        print(
            "\n  overhead_ratio = mean latency at N workers / mean at 1 worker\n"
            "  1.00 = no degradation   1.20 = 20% overhead   2.00 = doubled\n"
        )

        for scenario in _SCENARIOS:
            ramp = self._ramp_by_scenario.get(scenario)
            if ramp is None:
                continue

            label = _SCENARIO_LABELS.get(scenario, scenario)
            print(f"\n  {label}")
            print(
                f"  {'Workers':>8}  {'Mean':>9}  {'Median':>9}  {'P95':>9}  {'Overhead':>10}  {'Failures':>8}"
            )
            print(f"  {'-' * 64}")

            for row in ramp.sorted_rows():
                overhead = f"{row.overhead_ratio:.2f}x" if row.overhead_ratio is not None else "  -"
                flag = "  !" if row.overhead_ratio is not None and row.overhead_ratio > 1.5 else ""
                print(
                    f"  {row.workers:>8}  "
                    f"{row.mean_ms:>8.1f}ms  "
                    f"{row.median_ms:>8.1f}ms  "
                    f"{row.p95_ms:>8.1f}ms  "
                    f"{overhead:>10}  "
                    f"{row.failures:>8}"
                    f"{flag}"
                )

        if detailed:
            self._print_ramp_insights()

    def _print_ramp_insights(self) -> None:
        print("\n  Degradation summary (worst overhead per scenario):")
        for scenario in _SCENARIOS:
            ramp = self._ramp_by_scenario.get(scenario)
            if ramp is None:
                continue
            rows_with_ratio = [r for r in ramp.rows if r.overhead_ratio is not None]
            if not rows_with_ratio:
                continue
            worst = max(rows_with_ratio, key=lambda r: r.overhead_ratio)
            label = _SCENARIO_LABELS.get(scenario, scenario)
            status = (
                "POOR"
                if worst.overhead_ratio > 1.5
                else ("FAIR" if worst.overhead_ratio > 1.2 else "GOOD")
            )
            print(
                f"    {label:<40}  "
                f"worst: {worst.overhead_ratio:.2f}x at w={worst.workers}  [{status}]"
            )

    def _print_sustained_section(self, width: int) -> None:
        if not self._sustained_rows:
            return

        print(f"\n{'=' * width}")
        print("SUSTAINED TESTS  — steady-state throughput and latency")
        print("=" * width)

        col = 36
        print(
            f"\n  {'Scenario':<{col}} "
            f"{'Workers':>7}  {'Mean':>9}  {'P95':>9}  "
            f"{'RPS':>7}  {'Success':>8}  {'Failures':>8}"
        )
        print(f"  {'-' * (col + 60)}")

        for row in sorted(
            self._sustained_rows,
            key=lambda r: _SCENARIOS.index(r.scenario) if r.scenario in _SCENARIOS else 99,
        ):
            label = _SCENARIO_LABELS.get(row.scenario, row.scenario)
            success_pct = f"{row.success_rate * 100:.1f}%"
            flag = "  !" if row.success_rate < 0.99 else ""
            print(
                f"  {label:<{col}} "
                f"{row.workers:>7}  "
                f"{row.mean_ms:>8.1f}ms  "
                f"{row.p95_ms:>8.1f}ms  "
                f"{row.throughput_rps:>7.1f}  "
                f"{success_pct:>8}  "
                f"{row.failures:>8}"
                f"{flag}"
            )

    def export_csv(self, output_file: Path) -> None:
        """Export both ramp and sustained results to a single CSV."""
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "test_type",
                    "scenario",
                    "workers",
                    "mean_ms",
                    "median_ms",
                    "p95_ms",
                    "failures",
                    "overhead_ratio",
                    "throughput_rps",
                    "success_rate",
                ]
            )

            for scenario in _SCENARIOS:
                ramp = self._ramp_by_scenario.get(scenario)
                if ramp:
                    for row in ramp.sorted_rows():
                        writer.writerow(
                            [
                                "ramp",
                                scenario,
                                row.workers,
                                f"{row.mean_ms:.2f}",
                                f"{row.median_ms:.2f}",
                                f"{row.p95_ms:.2f}",
                                row.failures,
                                f"{row.overhead_ratio:.3f}"
                                if row.overhead_ratio is not None
                                else "",
                                "",
                                "",
                            ]
                        )

            for row in self._sustained_rows:
                writer.writerow(
                    [
                        "sustained",
                        row.scenario,
                        row.workers,
                        f"{row.mean_ms:.2f}",
                        f"{row.median_ms:.2f}",
                        f"{row.p95_ms:.2f}",
                        row.failures,
                        "",
                        f"{row.throughput_rps:.2f}",
                        f"{row.success_rate:.4f}",
                    ]
                )

        print(f"\nCSV exported to: {output_file}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_latest_load_baseline(baseline_dir: Path) -> Path:
    candidates = sorted(baseline_dir.glob("load_baseline_*.json"), reverse=True)
    if not candidates:
        raise ValueError(f"No load baseline files found in {baseline_dir}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a load test baseline")
    parser.add_argument("baseline", nargs="?", help="Baseline filename or path")
    parser.add_argument("--latest", action="store_true", help="Use the most recent baseline")
    parser.add_argument("--csv", action="store_true", help="Export results to CSV")
    parser.add_argument(
        "--detailed", action="store_true", help="Show per-scenario degradation summary"
    )
    args = parser.parse_args()

    baseline_dir = Path(__file__).parent / "performance_baselines"

    if args.latest:
        try:
            baseline_file = find_latest_load_baseline(baseline_dir)
            print(f"Using latest baseline: {baseline_file.name}\n")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.baseline:
        baseline_file = (
            Path(args.baseline)
            if Path(args.baseline).is_absolute()
            else baseline_dir / args.baseline
        )
        if not baseline_file.exists():
            print(f"Error: file not found: {baseline_file}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    analyzer = LoadBaselineAnalyzer(baseline_file)
    analyzer.print_summary(detailed=args.detailed)

    if args.csv:
        analyzer.export_csv(baseline_file.with_suffix(".csv"))


if __name__ == "__main__":
    main()
