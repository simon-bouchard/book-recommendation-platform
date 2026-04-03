#!/usr/bin/env python3
"""
Benchmark raw MySQL query latency for the recommendation hot path.

Run directly in the prod/integration environment:
    python tests/integration/models/benchmark_queries.py

Measures the two query types on the hot path:
  1. User fetch  — users + user_fav_subjects (two queries, one connection)
  2. Filter      — interactions IN clause with varying candidate counts

Results show p50/p95/min across 30 warm runs.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

os.environ.setdefault("SECURE_MODE", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

WARM_USER_IDS = [11676, 98391, 189835]
COLD_USER_IDS = [248965, 249650, 249939]
WARMUP = 10
RUNS = 30


def stats(times_ms: list, label: str) -> None:
    times_ms.sort()
    n = len(times_ms)
    print(
        f"  {label:<50}  "
        f"p50={times_ms[n // 2]:6.2f}ms  "
        f"p95={times_ms[int(n * 0.95)]:6.2f}ms  "
        f"min={times_ms[0]:6.2f}ms"
    )


async def bench(coro_factory) -> list:
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        await coro_factory()
        times.append((time.perf_counter() - t0) * 1000)
    return times


async def query_user_fetch(pool, user_id: int) -> None:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT user_id, country, age, filled_age FROM users WHERE user_id = %s",
                (user_id,),
            )
            row = await cur.fetchone()
            if row:
                await cur.execute(
                    "SELECT subject_idx FROM user_fav_subjects WHERE user_id = %s",
                    (row[0],),
                )
                await cur.fetchall()


async def query_filter(pool, user_id: int, n: int) -> None:
    candidate_ids = list(range(1, n + 1))
    placeholders = ",".join(["%s"] * n)
    sql = (
        f"SELECT item_idx FROM interactions "
        f"WHERE user_id = %s AND item_idx IN ({placeholders})"
    )
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, [user_id] + candidate_ids)
            await cur.fetchall()


async def main() -> None:
    from app.database import init_aiomysql_pool, get_aiomysql_pool

    print("Initialising pool...")
    await init_aiomysql_pool()
    pool = get_aiomysql_pool()

    print("\n── User fetch (users + user_fav_subjects, one connection) ──────────────")
    for uid in WARM_USER_IDS + COLD_USER_IDS:
        label = "warm" if uid in WARM_USER_IDS else "cold"
        for _ in range(WARMUP):
            await query_user_fetch(pool, uid)
        t = await bench(lambda u=uid: query_user_fetch(pool, u))
        stats(t, f"{label}  user_id={uid}")

    print("\n── Filter IN clause — warm user, varying candidate count ───────────────")
    uid = WARM_USER_IDS[0]
    for n in [50, 100, 200, 250]:
        for _ in range(WARMUP):
            await query_filter(pool, uid, n)
        t = await bench(lambda u=uid, c=n: query_filter(pool, u, c))
        stats(t, f"user_id={uid}  IN({n})")

    print("\n── Filter IN clause — cold users IN(250) ───────────────────────────────")
    for uid in COLD_USER_IDS:
        for _ in range(WARMUP):
            await query_filter(pool, uid, 250)
        t = await bench(lambda u=uid: query_filter(pool, u, 250))
        stats(t, f"user_id={uid}  IN(250)")


if __name__ == "__main__":
    asyncio.run(main())
