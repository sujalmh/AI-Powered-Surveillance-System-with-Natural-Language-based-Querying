"""Eval: fusion ablation with fairer baselines (R1-7, R2-1).

Compares on synthetic 768/384 Gaussian embeddings (seeded, no model download):
concat (novel, MultimodalFusion) vs visual-only / text-only (cosine) vs
zero-pad add / norm-matched add / random-proj add (5 seeds) / RRF late fusion.

Run: python backend/scripts/eval_fusion.py [--n 200 --seeds 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.eval.baselines import (
    norm_matched_add,
    random_proj_add,
    rrf_fuse,
    zero_pad_add,
)
from backend.app.services.fusion import MultimodalFusion


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def trial(n: int, seed: int):
    rng = np.random.RandomState(seed)
    base_v = rng.randn(768).astype(np.float32)
    base_t = rng.randn(384).astype(np.float32)
    base_v /= np.linalg.norm(base_v) + 1e-9
    base_t /= np.linalg.norm(base_t) + 1e-9
    gallery_v = [base_v + 0.3 * rng.randn(768).astype(np.float32) for _ in range(n)]
    gallery_t = [base_t + 0.3 * rng.randn(384).astype(np.float32) for _ in range(n)]
    gallery_v = [g / (np.linalg.norm(g) + 1e-9) for g in gallery_v]
    gallery_t = [g / (np.linalg.norm(g) + 1e-9) for g in gallery_t]
    qv = (base_v + 0.1 * rng.randn(768).astype(np.float32)).astype(np.float32)
    qt = (base_t + 0.1 * rng.randn(384).astype(np.float32)).astype(np.float32)
    qv /= np.linalg.norm(qv) + 1e-9
    qt /= np.linalg.norm(qt) + 1e-9

    fuse = MultimodalFusion(0.6, 0.4)
    g_concat = [fuse.fuse(v, t) for v, t in zip(gallery_v, gallery_t)]
    q_concat = fuse.fuse(qv, qt)
    s_concat = [cos(q_concat, g) for g in g_concat]
    s_vis = [cos(qv, g) for g in gallery_v]
    s_txt = [cos(qt, g) for g in gallery_t]
    g_zp = [zero_pad_add(v, t) for v, t in zip(gallery_v, gallery_t)]
    q_zp = zero_pad_add(qv, qt)
    s_zp = [float(q_zp[:768] @ g[:768]) for g in g_zp]
    order_vis = sorted(range(n), key=lambda i: -s_vis[i])
    order_txt = sorted(range(n), key=lambda i: -s_txt[i])
    fused_order = rrf_fuse([str(i) for i in order_vis], [str(i) for i in order_txt])
    rank = {d: r for r, d in enumerate(fused_order)}
    s_rrf = [1.0 / (1 + rank[str(i)]) for i in range(n)]
    proj = []
    for sd in range(5):
        g_p = [random_proj_add(v, t, seed=sd) for v, t in zip(gallery_v, gallery_t)]
        q_p = random_proj_add(qv, qt, seed=sd)
        proj.append(float(np.mean([cos(q_p, g) for g in g_p])))
    return {
        "concat": float(np.mean(sorted(s_concat, reverse=True)[:5])),
        "visual": float(np.mean(sorted(s_vis, reverse=True)[:5])),
        "text": float(np.mean(sorted(s_txt, reverse=True)[:5])),
        "zero_pad": float(np.mean(sorted(s_zp, reverse=True)[:5])),
        "rrf_top": s_rrf[0],
        "randproj_mean": float(np.mean(proj)),
        "randproj_seeds": proj,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    acc = None
    for s in range(args.seeds):
        r = trial(args.n, 1000 + s)
        acc = r if acc is None else {k: acc[k] + r[k] if isinstance(r[k], float) else r[k] for k in r}
    assert acc is not None
    for k, v in acc.items():
        if isinstance(v, float):
            print(f"{k}: {v / args.seeds:.4f}")


if __name__ == "__main__":
    main()
