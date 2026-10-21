"""Reviewer baselines for C1/C2 — additive, no change to novel code.

C1 (novel, untouched): backend/app/services/result_merger.py::compute_adaptive_alpha
  + merge_results (alpha*struct + (1-alpha)*sem on per-query [0,1] scores).

This module only ADDS competitors (R1-2, R1-7, R2-1):
- fixed-alpha sweep {0.0,0.3,0.5,0.7,1.0}
- per-intent lookup (count->1.0, action->0.0, filter->1.0, find/track->0.3)
- z-score score-norm variant (tests R1-4 calibration)
- RRF late fusion on two indices (rank-based, k=60)
- same-space additive baselines (zero-pad, norm-matched, random-proj seeded)

All functions are pure (lists/arrays) so tests run without FAISS/Mongo.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
import math

import numpy as np


# ---------------- C1: alpha competitors ----------------

FIXED_ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def fixed_alpha_score(s_struct: float, s_sem: float, alpha: float) -> float:
    return alpha * float(s_struct) + (1.0 - alpha) * float(s_sem)


def per_intent_lookup_alpha(category: str) -> float:
    """Strong non-learned competitor derived from Table alpha_baseline.

    count->1.0, action->0.0, filter->1.0, find/track->0.3.
    """
    c = (category or "").lower()
    if c == "count":
        return 1.0
    if c == "action":
        return 0.0
    if c == "filter":
        return 1.0
    return 0.3


def zscore_norm(scores: Sequence[float]) -> List[float]:
    """Alternative calibration (R1-4): z-score then sigmoid to [0,1]."""
    xs = [float(s) for s in scores]
    if not xs:
        return []
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var) + 1e-9
    return [1.0 / (1.0 + math.exp(-(x - m) / sd)) for x in xs]


def rrf_fuse(rank_a: Sequence[str], rank_b: Sequence[str], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion (parameter-free baseline, R1-7/R2-1).

    score(d) = 1/(k+rank_a) + 1/(k+rank_b). Needs two indices; the paper's
    concat uses one FAISS index instead.
    """
    scores: Dict[str, float] = {}
    for r, d in enumerate(rank_a, start=1):
        scores[d] = scores.get(d, 0.0) + 1.0 / (k + r)
    for r, d in enumerate(rank_b, start=1):
        scores[d] = scores.get(d, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


# ---------------- C2: same-space additive variants ----------------

def zero_pad_add(
    v: np.ndarray, t: np.ndarray, w_v: float = 0.6, w_t: float = 0.4
) -> np.ndarray:
    """Fairer same-space baseline (reviewer R1-7): zero-pad text to 768 then add."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    if t.shape[0] < v.shape[0]:
        tp = np.zeros_like(v)
        tp[: t.shape[0]] = t
        t = tp
    elif t.shape[0] > v.shape[0]:
        t = t[: v.shape[0]]
    out = (w_v * v + w_t * t).astype(np.float32)
    n = float(np.linalg.norm(out)) + 1e-6
    return (out / n).astype(np.float32)


def norm_matched_add(
    v: np.ndarray, t: np.ndarray, w_v: float = 0.6, w_t: float = 0.4
) -> np.ndarray:
    """Norm-matched addition: unit-norm each branch before weighted add."""
    v = np.asarray(v, dtype=np.float32).reshape(-1).astype(np.float64)
    t = np.asarray(t, dtype=np.float32).reshape(-1).astype(np.float64)
    v = v / (np.linalg.norm(v) + 1e-9)
    t = t / (np.linalg.norm(t) + 1e-9)
    if t.shape[0] < v.shape[0]:
        tp = np.zeros_like(v)
        tp[: t.shape[0]] = t
        t = tp
    else:
        t = t[: v.shape[0]]
    out = (w_v * v + w_t * t).astype(np.float32)
    n = float(np.linalg.norm(out)) + 1e-6
    return (out / n).astype(np.float32)


def random_proj_add(
    v: np.ndarray, t: np.ndarray, seed: int = 0, w_v: float = 1.0, w_t: float = 1.0
) -> np.ndarray:
    """Paper's illustrative baseline: single-seed Gaussian P(768x384), v + P t.

    Kept for reproduction; zero_pad/norm_matched are the fairer variants.
    """
    rng = np.random.RandomState(seed)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    P = rng.randn(v.shape[0], t.shape[0]).astype(np.float32) / math.sqrt(t.shape[0])
    out = (w_v * v + w_t * (P @ t)).astype(np.float32)
    n = float(np.linalg.norm(out)) + 1e-6
    return (out / n).astype(np.float32)
