# tests/integration/model_servers/bench_serialization.py
"""
Benchmark Pydantic serialization/validation cost vs raw orjson for each
model server response type, and optionally measure FastAPI threadpool dispatch
overhead by timing the /health endpoint on each running server.

Measures two sides of the round-trip:
  Server  : Pydantic model construction + model_dump_json()
            vs orjson.dumps(raw dict)
  Client  : Model.model_validate_json(bytes)
            vs orjson.loads(bytes)

Usage:
    python tests/integration/model_servers/bench_serialization.py
    python tests/integration/model_servers/bench_serialization.py --n 2000
    python tests/integration/model_servers/bench_serialization.py --http
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import httpx
import orjson

from model_servers._shared.contracts import (
    AlsRecsResponse,
    EmbedResponse,
    SimResponse,
    ScoredItem,
    SubjectRecsResponse,
)

_DEFAULT_N = 1000
_COL = 46
_NUM = 9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_ms(fn, n: int) -> float:
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1000


def _row(label: str, pydantic_ms: float, orjson_ms: float) -> str:
    saving = pydantic_ms - orjson_ms
    pct = (saving / pydantic_ms * 100) if pydantic_ms else 0
    return (
        f"  {label:<{_COL}}"
        f"{pydantic_ms:>{_NUM}.3f}ms"
        f"{orjson_ms:>{_NUM}.3f}ms"
        f"{saving:>{_NUM}.3f}ms"
        f"  ({pct:.0f}%)"
    )


def _header(title: str) -> str:
    sep = "  " + "-" * (_COL + _NUM * 3 + 10)
    return (
        f"\n  {title}\n"
        f"{sep}\n"
        f"  {'Operation':<{_COL}}"
        f"{'Pydantic':>{_NUM}}"
        f"{'orjson':>{_NUM}}"
        f"{'saving':>{_NUM}}"
    )


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------


def bench_scored_item_response(n: int, k: int, model_cls) -> tuple[float, float]:
    """Benchmark a response wrapping list[ScoredItem] at the given k."""
    item_ids = list(range(k))
    scores = [float(i) * 0.001 for i in range(k)]
    raw_dict = {"results": [{"item_idx": iid, "score": s} for iid, s in zip(item_ids, scores)]}
    raw_bytes = orjson.dumps(raw_dict)

    # Server: build Pydantic model + serialize
    server_pydantic = _mean_ms(
        lambda: model_cls(
            results=[ScoredItem(item_idx=iid, score=s) for iid, s in zip(item_ids, scores)]
        ).model_dump_json(),
        n,
    )
    # Server: raw orjson
    server_orjson = _mean_ms(lambda: orjson.dumps(raw_dict), n)

    # Client: Pydantic validation
    client_pydantic = _mean_ms(lambda: model_cls.model_validate_json(raw_bytes), n)
    # Client: raw orjson
    client_orjson = _mean_ms(lambda: orjson.loads(raw_bytes), n)

    return (server_pydantic, server_orjson, client_pydantic, client_orjson)


_SERVERS = {
    "embedder":   8001,
    "similarity": 8002,
    "als":        8003,
    "metadata":   8004,
}


async def bench_threadpool_dispatch(n: int) -> None:
    """
    Time GET /health on each server — zero compute, minimal payload.

    The median latency here is the FastAPI threadpool dispatch floor:
    the minimum HTTP cost any sync route can achieve regardless of what
    it does. Compare this against compute-heavy endpoint HTTP times to
    see how much headroom exists.
    """
    print("\n  Threadpool dispatch floor  (GET /health, zero compute)")
    print("  " + "-" * 62)
    print(f"  {'Server':<16} {'median':>8} {'p95':>8} {'p99':>8} {'min':>8}")

    warmup = max(10, n // 10)

    for name, port in _SERVERS.items():
        url = f"http://localhost:{port}"
        try:
            async with httpx.AsyncClient(base_url=url, timeout=5.0) as client:
                for _ in range(warmup):
                    await client.get("/health")
                times = []
                for _ in range(n):
                    t = time.perf_counter()
                    await client.get("/health")
                    times.append((time.perf_counter() - t) * 1000)
            times.sort()
            p50 = times[n // 2]
            p95 = times[int(n * 0.95)]
            p99 = times[int(n * 0.99)]
            mn  = times[0]
            print(f"  {name:<16} {p50:>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms {mn:>7.1f}ms")
        except Exception as e:
            print(f"  {name:<16} unreachable ({e})")

    print()


def bench_embed_response(n: int, dim: int) -> tuple[float, float, float, float]:
    """Benchmark EmbedResponse (vector of floats)."""
    vector = [float(i) * 0.001 for i in range(dim)]
    raw_dict = {"vector": vector}
    raw_bytes = orjson.dumps(raw_dict)

    server_pydantic = _mean_ms(
        lambda: EmbedResponse(vector=vector).model_dump_json(), n
    )
    server_orjson = _mean_ms(lambda: orjson.dumps(raw_dict), n)
    client_pydantic = _mean_ms(lambda: EmbedResponse.model_validate_json(raw_bytes), n)
    client_orjson = _mean_ms(lambda: orjson.loads(raw_bytes), n)

    return server_pydantic, server_orjson, client_pydantic, client_orjson


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Pydantic vs orjson serialization.")
    parser.add_argument("--n", type=int, default=_DEFAULT_N, help="Iterations per case")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding vector dimension")
    parser.add_argument("--http", action="store_true", help="Also benchmark threadpool dispatch via /health")
    args = parser.parse_args()

    n = args.n
    print(f"\nIterations per case: {n}")

    cases: dict[str, tuple] = {}

    for k in (200, 500):
        for label, cls in (
            ("AlsRecsResponse", AlsRecsResponse),
            ("SimResponse (subject_sim / als_sim / hybrid_sim)", SimResponse),
            ("SubjectRecsResponse", SubjectRecsResponse),
        ):
            key = f"{label} k={k}"
            cases[key] = bench_scored_item_response(n, k, cls)

    cases[f"EmbedResponse dim={args.embed_dim}"] = bench_embed_response(n, args.embed_dim)

    # --- Server table ---
    print(_header("Server  (model construction + model_dump_json  vs  orjson.dumps)"))
    server_total_pydantic = server_total_orjson = 0.0
    for label, (sp, so, cp, co) in cases.items():
        print(_row(label, sp, so))
        server_total_pydantic += sp
        server_total_orjson += so
    print(_row("TOTAL (all cases summed)", server_total_pydantic, server_total_orjson))

    # --- Client table ---
    print(_header("Client  (model_validate_json  vs  orjson.loads)"))
    client_total_pydantic = client_total_orjson = 0.0
    for label, (sp, so, cp, co) in cases.items():
        print(_row(label, cp, co))
        client_total_pydantic += cp
        client_total_orjson += co
    print(_row("TOTAL (all cases summed)", client_total_pydantic, client_total_orjson))

    # --- Round-trip summary ---
    print(f"\n  {'Round-trip saving per request (server + client)':}")
    for label, (sp, so, cp, co) in cases.items():
        total_saving = (sp - so) + (cp - co)
        print(f"    {label:<{_COL}}  {total_saving:+.3f}ms")

    print()

    if args.http:
        asyncio.run(bench_threadpool_dispatch(n))


if __name__ == "__main__":
    main()
