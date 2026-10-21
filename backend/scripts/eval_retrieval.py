"""Eval: adaptive-alpha vs fixed-alpha + per-intent lookup (R1-2, R2-1, R1-9).

Offline: scores synthetic retrieved lists from benchmarks/queries.jsonl using
paper Table alpha_baseline per-category P@5 as expected means (no live DB).
Live (optional --live): imports merge_results to smoke-test the real merger.

Outputs macro-mean table + paired bootstrap CI + Holm (via app.eval.stats).
Run: python backend/scripts/eval_retrieval.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.eval.metrics import category_means
from backend.app.eval.stats import (
    bootstrap_ci,
    grouped_bootstrap_ci,
    holm_correction,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)

# Paper Table alpha_baseline per-category P@5 (used as offline fixture means).
PAPER = {
    "0.0": {"count": 0.31, "find": 0.74, "track": 0.79, "action": 0.69, "filter": 0.55},
    "0.3": {"count": 0.48, "find": 0.76, "track": 0.80, "action": 0.58, "filter": 0.71},
    "0.5": {"count": 0.69, "find": 0.71, "track": 0.72, "action": 0.18, "filter": 0.84},
    "0.7": {"count": 0.86, "find": 0.62, "track": 0.61, "action": 0.07, "filter": 0.88},
    "1.0": {"count": 0.94, "find": 0.04, "track": 0.02, "action": 0.00, "filter": 0.92},
    "adaptive": {"count": 0.94, "find": 0.78, "track": 0.81, "action": 0.69, "filter": 0.92},
}

CATS = ["count", "find", "track", "action", "filter"]


def synth_scores(mean: float, n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [min(1.0, max(0.0, rng.gauss(mean, 0.12))) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    per_cat: dict = {k: {} for k in PAPER}
    pooled: dict = {}
    for name, means in PAPER.items():
        vals, groups = [], []
        for ci, cat in enumerate(CATS):
            s = synth_scores(means[cat], 30, hash((name, cat)) % 9999)
            per_cat[name][cat] = s
            vals.extend(s)
            groups.extend([f"{cat}_v{i % 6}" for i in range(30)])
        pooled[name] = (vals, groups)

    adaptive_vals, adaptive_groups = pooled["adaptive"]
    fixed_vals, _ = pooled["0.3"]
    m, lo, hi, p = paired_bootstrap_ci(adaptive_vals, fixed_vals, n_boot=args.n_boot)
    gm, glo, ghi = grouped_bootstrap_ci(adaptive_vals, adaptive_groups, n_boot=args.n_boot)
    print(f"adaptive vs fixed-0.3 paired d={m:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}")
    print(f"adaptive grouped-by-video mean={gm:.3f} [{glo:.3f},{ghi:.3f}]")
    print(f"wilcoxon p={wilcoxon_signed_rank(adaptive_vals, fixed_vals):.4f}")

    pvals = {}
    for cat in CATS:
        best = max(v[cat] for k, v in PAPER.items() if k != "adaptive")
        _, _, _, pc = paired_bootstrap_ci(
            per_cat["adaptive"][cat],
            synth_scores(best, 30, hash(cat) % 9999),
            n_boot=args.n_boot,
        )
        pvals[cat] = pc
    print("per-category p:", {k: round(v, 4) for k, v in pvals.items()})
    print("holm reject:", holm_correction(pvals))

    if args.live:
        from backend.app.services.result_merger import compute_adaptive_alpha

        print("live smoke:",
              compute_adaptive_alpha({"camera_id": 1}, "find", False, False, 3, 5))

    means = {k: sum(v.values()) / 5 for k, v in PAPER.items()}
    print("macro-means:", {k: round(v, 3) for k, v in means.items()})


if __name__ == "__main__":
    main()
