# tests/load/models/setup_test_data.py
"""
Probes a candidate pool of book IDs against the live server to determine which
ones work correctly for every similarity mode, then writes a verified test data
config file.

A book is only included in the verified set if it returns 200 for ALL THREE
similarity modes: subject, als, and hybrid. This guarantees that load and
profile tests never receive unexpected 422 responses regardless of which mode
is under test.

  - subject 422: book has no subject embedding in the FAISS index
  - als/hybrid 422: book has no ALS factors in the similarity server

Requests are fired concurrently to keep total probe time short.

Run this once before executing load tests. The config file is read by
_constants.py and replaces the hardcoded fallback IDs automatically.

Usage:
    export PERF_TEST_BASE_URL=http://localhost:8000
    python tests/load/models/setup_test_data.py

    # Request a specific number of verified IDs (default: 20):
    python tests/load/models/setup_test_data.py --target 30

    # Dry run — print results without writing the config file:
    python tests/load/models/setup_test_data.py --dry-run
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------
# Roughly 2x the default target of 20. Expand this list freely — the script
# probes all of them and selects only those that pass every mode check.

_CANDIDATE_BOOK_IDS: list[int] = [
    # Original test set
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
    # Additional candidates
    1,
    2,
    3,
    5,
    10,
    15,
    20,
    25,
    30,
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    800,
]

_SIMILARITY_MODES = ("subject", "als", "hybrid")

_CONFIG_FILE = Path(__file__).parent / "test_data_config.json"
_PROBE_CONCURRENCY = 10
_PROBE_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    book_id: int
    subject_ok: bool
    als_ok: bool
    hybrid_ok: bool

    @property
    def all_modes_ok(self) -> bool:
        return self.subject_ok and self.als_ok and self.hybrid_ok

    def summary(self) -> str:
        flags = {
            "subject": self.subject_ok,
            "als": self.als_ok,
            "hybrid": self.hybrid_ok,
        }
        passing = [m for m, ok in flags.items() if ok]
        failing = [m for m, ok in flags.items() if not ok]
        if self.all_modes_ok:
            return "all modes OK"
        return f"OK: {passing}  FAIL: {failing}"


# ---------------------------------------------------------------------------
# Probe logic
# ---------------------------------------------------------------------------


async def _probe_mode(
    client: httpx.AsyncClient,
    book_id: int,
    mode: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, bool]:
    """
    Return (mode, passed) for a single book/mode combination.

    200 -> passed.
    422 -> failed (no embedding or ALS factors for this mode).
    Any other status or network error -> failed and logged.
    """
    async with semaphore:
        try:
            response = await client.get(
                f"/book/{book_id}/similar",
                params={"mode": mode, "top_k": 1},
            )
            if response.status_code == 200:
                return mode, True
            elif response.status_code == 422:
                return mode, False
            else:
                print(
                    f"  WARNING: book {book_id} mode={mode} "
                    f"returned unexpected status {response.status_code}"
                )
                return mode, False
        except httpx.RequestError as exc:
            print(f"  WARNING: request failed for book {book_id} mode={mode}: {exc}")
            return mode, False


async def _probe_book(
    client: httpx.AsyncClient,
    book_id: int,
    semaphore: asyncio.Semaphore,
) -> ProbeResult:
    """Probe all three similarity modes for a single book concurrently."""
    mode_results = await asyncio.gather(
        *[_probe_mode(client, book_id, mode, semaphore) for mode in _SIMILARITY_MODES]
    )
    results = dict(mode_results)
    return ProbeResult(
        book_id=book_id,
        subject_ok=results["subject"],
        als_ok=results["als"],
        hybrid_ok=results["hybrid"],
    )


async def probe_candidates(
    base_url: str,
    candidates: list[int],
) -> list[ProbeResult]:
    """
    Probe all candidate IDs concurrently and return a ProbeResult per book.

    Uses a shared semaphore across all mode probes to cap total concurrent
    requests at _PROBE_CONCURRENCY regardless of how many books or modes
    are being probed simultaneously.
    """
    semaphore = asyncio.Semaphore(_PROBE_CONCURRENCY)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(_PROBE_TIMEOUT),
    ) as client:
        tasks = [_probe_book(client, book_id, semaphore) for book_id in candidates]
        probe_results: list[ProbeResult] = []

        for coro in asyncio.as_completed(tasks):
            result = await coro
            print(f"  book {result.book_id:>6}: {result.summary()}")
            probe_results.append(result)

    # Restore original candidate ordering for determinism.
    order = {bid: i for i, bid in enumerate(candidates)}
    probe_results.sort(key=lambda r: order.get(r.book_id, 9999))
    return probe_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe candidate book IDs across all similarity modes and write "
            "a verified test data config."
        )
    )
    parser.add_argument(
        "--target",
        type=int,
        default=20,
        help="Number of fully-verified book IDs to select (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing the config file",
    )
    args = parser.parse_args()

    base_url = os.environ.get("PERF_TEST_BASE_URL", "").rstrip("/")
    if not base_url:
        print("Error: PERF_TEST_BASE_URL is not set.")
        print("  export PERF_TEST_BASE_URL=http://localhost:8000")
        sys.exit(1)

    print(f"\nProbing {len(_CANDIDATE_BOOK_IDS)} candidate book IDs")
    print(f"Target server : {base_url}")
    print(f"Modes checked : {list(_SIMILARITY_MODES)}")
    print(f"Target count  : {args.target} fully-verified IDs")
    print(f"Concurrency   : {_PROBE_CONCURRENCY} parallel requests\n")

    start = time.perf_counter()
    probe_results = asyncio.run(probe_candidates(base_url, _CANDIDATE_BOOK_IDS))
    elapsed = time.perf_counter() - start

    verified = [r for r in probe_results if r.all_modes_ok]
    partial = [r for r in probe_results if not r.all_modes_ok]

    print(f"\nProbe complete in {elapsed:.1f}s")
    print(f"  Fully verified (all modes) : {len(verified)}")
    print(f"  Partial / failed           : {len(partial)}")

    if partial:
        print("\n  Partial results (excluded from verified set):")
        for r in partial:
            print(f"    book {r.book_id:>6}: {r.summary()}")

    if len(verified) < args.target:
        print(
            f"\nWARNING: only {len(verified)} books pass all modes, "
            f"fewer than the requested {args.target}. "
            f"Add more candidates to _CANDIDATE_BOOK_IDS in this script and re-run."
        )

    selected = [r.book_id for r in verified[: args.target]]

    print(f"\nSelected {len(selected)} fully-verified book IDs:")
    print(f"  {selected}")

    if args.dry_run:
        print("\nDry run — config file not written.")
        return

    config = {
        "verified_book_ids_with_als": selected,
        "probe_metadata": {
            "base_url": base_url,
            "candidates_probed": len(_CANDIDATE_BOOK_IDS),
            "modes_checked": list(_SIMILARITY_MODES),
            "fully_verified": len(verified),
            "partial_or_failed": len(partial),
            "elapsed_s": round(elapsed, 2),
        },
    }

    with open(_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig written to: {_CONFIG_FILE}")
    print("You can now run load tests — _constants.py will use the verified IDs.")


if __name__ == "__main__":
    main()
