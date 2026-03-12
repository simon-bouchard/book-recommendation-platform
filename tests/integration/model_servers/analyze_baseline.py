# tests/integration/model_servers/analyze_baseline.py
"""
Print latency baseline results from a JSON file in human-readable format.

Usage:
    python analyze_baseline.py                        # most recent baseline
    python analyze_baseline.py --latest               # most recent baseline
    python analyze_baseline.py model_servers_20260306_142301.json
    python analyze_baseline.py --compare model_servers_A.json model_servers_B.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BASELINES_DIR = Path(__file__).parent / "performance_baselines"

_COL_WIDTH = 52
_NUM_WIDTH = 8


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_baseline(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_path(filename: str | None) -> Path:
    """
    Resolve a baseline file path.

    If filename is None or '--latest', returns the most recently modified
    file in the baselines directory. If filename is a bare name (no slashes),
    looks it up in the baselines directory. Otherwise treats it as a path.
    """
    if filename is None or filename == "--latest":
        return _latest_baseline()

    candidate = Path(filename)
    if not candidate.is_absolute() and "/" not in filename and "\\" not in filename:
        candidate = _BASELINES_DIR / filename

    if not candidate.exists():
        print(f"File not found: {candidate}", file=sys.stderr)
        sys.exit(1)

    return candidate


def _latest_baseline() -> Path:
    if not _BASELINES_DIR.exists():
        print(f"Baselines directory not found: {_BASELINES_DIR}", file=sys.stderr)
        sys.exit(1)

    files = sorted(_BASELINES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("No baseline files found.", file=sys.stderr)
        sys.exit(1)

    return files[-1]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _row(label: str, *values: str) -> str:
    return f"  {label:<{_COL_WIDTH}}" + "".join(f"{v:>{_NUM_WIDTH}}" for v in values)


def print_baseline(data: dict, path: Path) -> None:
    config = data.get("config", {})
    results = data.get("results", {})

    print(f"\nFile      : {path.name}")
    print(f"Timestamp : {data.get('timestamp', 'unknown')}")
    print(f"Warmup    : {config.get('warmup_runs', '?')} runs")
    print(f"Measured  : {config.get('measurement_runs', '?')} runs")
    print(f"Tests     : {len(results)}")

    if not results:
        print("\n  (no results)")
        return

    header = _row("Test", "mean", "median", "p95", "p99", "min", "max")
    separator = "  " + "-" * (_COL_WIDTH + _NUM_WIDTH * 6)

    print(f"\n{header}")
    print(separator)

    for name, stats in sorted(results.items()):
        if not stats:
            print(_row(name, *["n/a"] * 6))
            continue
        print(
            _row(
                name,
                f"{stats['mean_ms']:.1f}ms",
                f"{stats['median_ms']:.1f}ms",
                f"{stats['p95_ms']:.1f}ms",
                f"{stats['p99_ms']:.1f}ms",
                f"{stats['min_ms']:.1f}ms",
                f"{stats['max_ms']:.1f}ms",
            )
        )

    print()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def print_comparison(data_a: dict, path_a: Path, data_b: dict, path_b: Path) -> None:
    results_a = data_a.get("results", {})
    results_b = data_b.get("results", {})
    all_keys = sorted(results_a.keys() | results_b.keys())

    print(f"\nComparing:")
    print(f"  A: {path_a.name}  ({data_a.get('timestamp', '?')})")
    print(f"  B: {path_b.name}  ({data_b.get('timestamp', '?')})")

    header = _row("Test", "A mean", "B mean", "delta", "change")
    separator = "  " + "-" * (_COL_WIDTH + _NUM_WIDTH * 4)

    print(f"\n{header}")
    print(separator)

    for name in all_keys:
        stats_a = results_a.get(name)
        stats_b = results_b.get(name)

        if not stats_a or not stats_b:
            label = "(A only)" if not stats_b else "(B only)"
            mean = stats_a["mean_ms"] if stats_a else stats_b["mean_ms"]
            print(_row(name, f"{mean:.1f}ms", "-", "-", label))
            continue

        mean_a = stats_a["mean_ms"]
        mean_b = stats_b["mean_ms"]
        delta = mean_b - mean_a
        pct = (delta / mean_a) * 100 if mean_a else 0

        delta_str = f"{delta:+.1f}ms"
        pct_str = f"{pct:+.1f}%"

        print(_row(name, f"{mean_a:.1f}ms", f"{mean_b:.1f}ms", delta_str, pct_str))

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print latency baseline results in human-readable format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    show_p = subparsers.add_parser("show", help="Print a single baseline file.")
    show_p.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Baseline filename or path. Defaults to the most recent file.",
    )

    compare_p = subparsers.add_parser("compare", help="Compare two baseline files.")
    compare_p.add_argument("file_a", help="First (older) baseline file.")
    compare_p.add_argument("file_b", help="Second (newer) baseline file.")

    subparsers.add_parser("list", help="List all available baseline files.")

    return parser


def cmd_show(args) -> None:
    path = resolve_path(args.file)
    data = load_baseline(path)
    print_baseline(data, path)


def cmd_compare(args) -> None:
    path_a = resolve_path(args.file_a)
    path_b = resolve_path(args.file_b)
    data_a = load_baseline(path_a)
    data_b = load_baseline(path_b)
    print_comparison(data_a, path_a, data_b, path_b)


def cmd_list(_args) -> None:
    if not _BASELINES_DIR.exists():
        print(f"Baselines directory not found: {_BASELINES_DIR}", file=sys.stderr)
        sys.exit(1)

    files = sorted(_BASELINES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("No baseline files found.")
        return

    for f in files:
        try:
            data = load_baseline(f)
            n = len(data.get("results", {}))
            ts = data.get("timestamp", "?")
            print(f"  {f.name}  ({ts})  {n} tests")
        except (json.JSONDecodeError, OSError):
            print(f"  {f.name}  (unreadable)")


def main() -> None:
    parser = _build_parser()

    # Support bare invocation with a positional filename and no subcommand,
    # matching the usage hint printed by conftest.py after each run.
    if len(sys.argv) == 1 or (
        len(sys.argv) == 2
        and not sys.argv[1].startswith("-")
        and sys.argv[1] not in ("show", "compare", "list")
    ):
        filename = sys.argv[1] if len(sys.argv) == 2 else None
        path = resolve_path(filename)
        data = load_baseline(path)
        print_baseline(data, path)
        return

    args = parser.parse_args()

    if args.command == "show" or args.command is None:
        cmd_show(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
