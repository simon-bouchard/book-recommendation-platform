# tests/load/_stats.py
"""
Latency statistics accumulator shared across all load test suites.

Extends the integration-test LatencyStats with failure tracking and arbitrary
extra metadata (workers, throughput_rps, etc.) that load tests need to attach
to a result before writing it to a baseline JSON file.
"""

import statistics
from typing import Any


class LatencyStats:
    """
    Collects successful request latencies and tracks failures for a named series.

    Extra keyword arguments passed at construction time (e.g. workers=5,
    test_type="ramp") are stored in self.extra and merged into get_stats()
    output, which keeps the baseline JSON schema open for extension without
    requiring subclassing.
    """

    def __init__(self, name: str, **extra: Any) -> None:
        self.name = name
        self.measurements: list[float] = []
        self.failures: int = 0
        self.extra: dict[str, Any] = dict(extra)

    def add(self, duration_ms: float) -> None:
        """Record a successful request latency in milliseconds."""
        self.measurements.append(duration_ms)

    @property
    def count(self) -> int:
        return len(self.measurements)

    def get_stats(self) -> dict[str, Any]:
        """
        Return a flat dictionary of all statistics suitable for JSON serialisation.

        Returns only count and failures when no successful measurements exist,
        so callers always get a valid (if sparse) dict rather than an empty one.
        """
        if not self.measurements:
            return {"count": 0, "failures": self.failures, **self.extra}

        s = sorted(self.measurements)
        n = len(s)
        stats: dict[str, Any] = {
            "min_ms": s[0],
            "max_ms": s[-1],
            "mean_ms": statistics.mean(s),
            "median_ms": statistics.median(s),
            "p95_ms": s[min(int(n * 0.95), n - 1)],
            "p99_ms": s[min(int(n * 0.99), n - 1)],
            "stdev_ms": statistics.stdev(s) if n > 1 else 0.0,
            "count": n,
            "failures": self.failures,
        }
        stats.update(self.extra)
        return stats

    def __str__(self) -> str:
        s = self.get_stats()
        if not s.get("mean_ms"):
            return f"{self.name}: no measurements (failures={self.failures})"
        lines = [
            self.name + ":",
            f"  Mean:     {s['mean_ms']:.2f}ms",
            f"  Median:   {s['median_ms']:.2f}ms",
            f"  P95:      {s['p95_ms']:.2f}ms",
            f"  P99:      {s['p99_ms']:.2f}ms",
            f"  Range:    [{s['min_ms']:.2f}ms - {s['max_ms']:.2f}ms]",
            f"  Stdev:    {s['stdev_ms']:.2f}ms",
            f"  Count:    {s['count']}",
            f"  Failures: {s['failures']}",
        ]
        for k, v in self.extra.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
