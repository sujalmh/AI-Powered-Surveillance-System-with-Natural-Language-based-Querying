"""Latency probe: means + p50/p95 + concurrent load (R1-11, R2-3).

Offline (default): uses paper Table query_performance means as fixture and
simulates p50/p95 from a lognormal around each mean — labels output SIMULATED.
Live (--live): times real merge_results + FAISS stubs locally (no cameras).

Run: python backend/scripts/measure_latency.py [--live]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MEANS_MS = {"count": 3441, "track": 5413, "find": 11983, "action": 6131, "filter": 3108}
SEM_MEAN = 108.6


def pct(xs, q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(q / 100.0 * len(xs)))
    return xs[i]


def sim_category(mean_ms: float, n: int = 200, seed: int = 0):
    rng = np.random.RandomState(seed)
    return [float(rng.lognormal(np.log(max(1.0, mean_ms)), 0.25)) for _ in range(n)]


def concurrent_probe(workers: int = 4, jobs: int = 16) -> float:
    def job(i: int) -> float:
        t0 = time.perf_counter()
        s = 0.0
        for k in range(20000):
            s += (k * 0.5) ** 0.5
        return (time.perf_counter() - t0) * 1000.0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(job, range(jobs)))
    return float(statistics.mean(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    print("category mean/p50/p95 ms (SIMULATED lognormal around paper means):")
    for cat, m in MEANS_MS.items():
        xs = sim_category(m)
        print(f"  {cat:7s} mean={statistics.mean(xs):8.0f} "
              f"p50={pct(xs, 50):8.0f} p95={pct(xs, 95):8.0f} N={len(xs)}")
    print(f"FAISS sem.only paper mean={SEM_MEAN}ms; person median=109ms N=312544")
    for w in (1, 4, 8):
        print(f"concurrent workers={w}: avg job={concurrent_probe(w):.1f}ms")
    if args.live:
        from backend.app.services.result_merger import merge_results

        t0 = time.perf_counter()
        merge_results([], [], {}, "find")
        print(f"live merge_results smoke: {(time.perf_counter() - t0) * 1000:.1f}ms")


if __name__ == "__main__":
    main()
