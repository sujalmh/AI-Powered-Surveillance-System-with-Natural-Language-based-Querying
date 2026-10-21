"""Real person-pair / single-camera track retrieval on the released descriptors.

There are no identity labels and no track ids in the release (all
``track_id=-1``).  This script therefore builds deterministic pseudo-tracks
from the 1,778 shipped 1152-d descriptors (visual-only SigLIP cosine + 3 s
temporal gate, threshold selected on a held-out calibration block), splits
whole tracks into calibration/test, and evaluates two things:

1. **Retrieval**: given one test detection, retrieve other detections of the
   same pseudo-track from the test split.
2. **Pair classification**: classify the 2,016 released metadata-derived
   same/different-track pairs (1,008 calibration + 1,008 test).  The decision
   threshold for every method is selected on calibration pairs only; AUC is
   threshold-free.

The benchmark is a *released-subset reproducibility track*: it is not the
paper's 312k-descriptor / 4-camera identity benchmark, and the tracks are
pseudo-labels rather than human identity annotations.  Visual-only is
therefore optimistically biased because tracks were formed visually; this is
stated in the results.

Run:
    python backend/scripts/eval_fusion.py
    python backend/scripts/eval_fusion.py --seeds 5 --n-boot 3000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval import subset
from backend.app.eval.baselines import (
    norm_matched_add,
    random_proj_add_batch,
    rrf_fuse,
    zero_pad_add,
)
from backend.app.eval.metrics import (
    f1_at_10_paper,
    f1_at_10_standard,
    precision_at_k,
    rank_of_first_relevant,
    recall_at_k,
)
from backend.app.eval.stats import (
    bootstrap_ci,
    format_p,
    format_sd,
    grouped_bootstrap_ci,
    paired_bootstrap_ci,
)
from backend.app.services.fusion import MultimodalFusion

BENCHMARK = ROOT / "backend" / "benchmarks"
RESULTS = ROOT / "results"
CAL_BLOCK_END = subset.parse_det_ts("2026-03-01T17:09:00")
CAL_BLOCK2_START = subset.parse_det_ts("2026-03-01T18:55:00")
METHODS = [
    "visual", "text", "concat", "zero_pad", "norm_matched", "rrf",
    "learned_proj", "randproj",
]


# ---------------------------------------------------------------------------
# Track split
# ---------------------------------------------------------------------------

def load_tracks() -> Tuple[
    List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray,
    List[int], List[Dict[str, Any]], Dict[str, Any],
]:
    dets = subset.load_person_meta(sort_by_time=False)
    vectors = subset.load_index_vectors(subset.PERSON_INDEX, 1152)
    visual, text = subset.split_person_vectors(vectors)
    cal_mask = np.array(
        [d["timestamp"] < CAL_BLOCK_END or d["timestamp"] >= CAL_BLOCK2_START for d in dets]
    )
    cal_dets = [d for d, m in zip(dets, cal_mask) if m]
    cal_visual = visual[cal_mask]
    threshold, grid = subset.tune_track_threshold(cal_dets, cal_visual)
    track_of, tracks, edges = subset.build_tracks(
        dets, visual, max_gap=3.0, threshold=threshold
    )
    order = sorted(tracks, key=lambda t: (t["start"], t["track_id"]))
    cal_tracks = {t["track_id"] for i, t in enumerate(order) if i % 4 == 0}
    test_tracks = {t["track_id"] for t in order} - cal_tracks
    split_of = {
        tid: ("calibration" if tid in cal_tracks else "test") for tid in range(len(tracks))
    }
    track_name = {
        tid: f"{split_of[tid]}_t{tid:05d}" for tid in range(len(tracks))
    }
    track_meta = {
        "threshold": threshold,
        "threshold_grid": grid,
        "selected_rule": "strictest threshold within 5% of best calibration objective",
        "n_tracks": len(tracks),
        "n_calibration_tracks": len(cal_tracks),
        "n_test_tracks": len(test_tracks),
        "n_calibration_detections": sum(1 for t in track_of if t in cal_tracks),
        "n_test_detections": sum(1 for t in track_of if t not in cal_tracks),
        "n_tracks_ge2": sum(1 for t in tracks if t["size"] >= 2),
        "n_link_edges": len(edges),
        "max_gap_seconds": 3.0,
        "max_duration_seconds": 60.0,
        "max_track_size": 20,
    }
    return dets, vectors, visual, text, track_of, tracks, {
        "cal_tracks": cal_tracks,
        "test_tracks": test_tracks,
        "split_of": split_of,
        "track_name": track_name,
        "meta": track_meta,
    }


# ---------------------------------------------------------------------------
# Triangular fusion baselines
# ---------------------------------------------------------------------------

def fuse_detached(
    fuse: MultimodalFusion, visual_emb: np.ndarray, text_emb: np.ndarray
) -> np.ndarray:
    """Call ``MultimodalFusion.fuse`` without its in-place input mutation.

    ``MultimodalFusion.fuse`` casts with ``np.asarray(..., dtype=np.float32)``
    and then does ``v *= w`` / ``t *= w``.  For an already-float32 input that
    is a view, so the caller's branch matrix is scaled in place.  Production
    call sites pass freshly encoded vectors, but the evaluator reuses the
    matrices, so it must pass copies.  (The production bug is reported, not
    fixed here, because this change set is scoped to the evaluation code.)
    """
    v = np.array(visual_emb, dtype=np.float32, copy=True)
    t = np.array(text_emb, dtype=np.float32, copy=True)
    return fuse.fuse(v, t)


def build_method_galleries(
    visual: np.ndarray,
    text: np.ndarray,
    concat: np.ndarray,
    learned_W: np.ndarray,
) -> Dict[str, np.ndarray]:
    fuse = MultimodalFusion(0.6, 0.4)
    # Reconstruct the shipped index exactly from the two unit branches.
    reconstructed = np.vstack(
        [fuse_detached(fuse, v, t) for v, t in zip(visual, text)]
    )
    if not np.allclose(reconstructed, concat, atol=1e-4):
        diff = float(np.max(np.abs(reconstructed - concat)))
        raise RuntimeError(f"concat reconstruction mismatch: {diff}")

    zero = np.vstack([zero_pad_add(v, t) for v, t in zip(visual, text)])
    norm = np.vstack([norm_matched_add(v, t) for v, t in zip(visual, text)])
    learned = visual + (text @ learned_W)
    learned = learned / (np.linalg.norm(learned, axis=1, keepdims=True) + 1e-6)
    return {
        "visual": visual.astype(np.float32),
        "text": text.astype(np.float32),
        "concat": concat.astype(np.float32),
        "zero_pad": zero.astype(np.float32),
        "norm_matched": norm.astype(np.float32),
        "learned_proj": learned.astype(np.float32),
    }


def fit_ridge_projection(
    visual: np.ndarray, text: np.ndarray, train_idx: Sequence[int], lam: float = 1.0
) -> np.ndarray:
    """Ridge map text (384) -> visual (768), fit on calibration detections."""
    X = text[list(train_idx)].astype(np.float64)
    Y = visual[list(train_idx)].astype(np.float64)
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float64)
    B = X.T @ Y
    W = np.linalg.solve(A, B)
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def ranking_from_scores(
    scores: np.ndarray, candidates: Sequence[int], k: int = 10
) -> List[int]:
    cand = np.asarray(list(candidates), dtype=np.int64)
    if cand.size == 0:
        return []
    order = np.argsort(-scores[cand], kind="stable")[:k]
    return [int(cand[i]) for i in order]


def retrieval_metrics(
    scores: np.ndarray, candidates: Sequence[int], relevant: set
) -> Dict[str, Any]:
    ranking = ranking_from_scores(scores, candidates, 10)
    p5 = precision_at_k(ranking, relevant, 5)
    r10 = recall_at_k(ranking, relevant, 10)
    top = ranking[0] if ranking else -1
    return {
        "p5": p5,
        "r10": r10,
        "f1": f1_at_10_standard(ranking, relevant),
        "f1_paper": f1_at_10_paper(ranking, relevant),
        "rank_first_relevant": rank_of_first_relevant(ranking, relevant),
        "top1_index": top,
        "top1_score": float(scores[top]) if top >= 0 else 0.0,
        "top5_indices": ranking[:5],
    }


def evaluate_method(
    name: str,
    gallery: np.ndarray,
    query_indices: Sequence[int],
    candidates_by_query: Dict[int, List[int]],
    relevant_by_query: Dict[int, set],
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for qi in query_indices:
        scores = gallery @ gallery[qi]
        m = retrieval_metrics(scores, candidates_by_query[qi], relevant_by_query[qi])
        rows.append({"method": name, "seed": seed, "qid": int(qi), **m})
    return rows


def evaluate_rrf(
    visual: np.ndarray,
    text: np.ndarray,
    query_indices: Sequence[int],
    candidates_by_query: Dict[int, List[int]],
    relevant_by_query: Dict[int, set],
) -> List[Dict[str, Any]]:
    rows = []
    for qi in query_indices:
        cand = candidates_by_query[qi]
        cand_arr = np.asarray(cand, dtype=np.int64)
        sv = visual[cand_arr] @ visual[qi]
        st = text[cand_arr] @ text[qi]
        order_v = np.argsort(-sv, kind="stable")
        order_t = np.argsort(-st, kind="stable")
        rank_v = [str(cand_arr[i]) for i in order_v]
        rank_t = [str(cand_arr[i]) for i in order_t]
        fused = rrf_fuse(rank_v, rank_t, k=60)[:10]
        rank = [int(x) for x in fused]
        rel = relevant_by_query[qi]
        pos_v = {str(c): i + 1 for i, c in enumerate(cand_arr[order_v])}
        pos_t = {str(c): i + 1 for i, c in enumerate(cand_arr[order_t])}
        top = rank[0] if rank else -1
        top_score = (
            1.0 / (60 + pos_v[str(top)]) + 1.0 / (60 + pos_t[str(top)])
            if top >= 0 else 0.0
        )
        rows.append(
            {
                "method": "rrf", "seed": None, "qid": int(qi),
                "p5": precision_at_k(rank, rel, 5),
                "r10": recall_at_k(rank, rel, 10),
                "f1": f1_at_10_standard(rank, rel),
                "f1_paper": f1_at_10_paper(rank, rel),
                "rank_first_relevant": rank_of_first_relevant(rank, rel),
                "top1_index": top,
                "top1_score": float(top_score),
                "top5_indices": rank[:5],
            }
        )
    return rows


def summarize(rows: Sequence[Dict[str, Any]], n_boot: int, seed: int = 4321) -> Dict[str, Any]:
    p5 = [r["p5"] for r in rows]
    r10 = [r["r10"] for r in rows]
    f1 = [r["f1"] for r in rows]
    f1p = [r["f1_paper"] for r in rows]
    m, lo, hi = bootstrap_ci(p5, n_boot=n_boot, seed=seed)
    return {
        "n_queries": len(rows),
        "p5": float(np.mean(p5)),
        "r10": float(np.mean(r10)),
        "f1": float(np.mean(f1)),
        "f1_paper": float(np.mean(f1p)),
        "p5_ci": [float(lo), float(hi)],
        "p5_boot_mean": float(m),
    }


# ---------------------------------------------------------------------------
# Pair classification
# ---------------------------------------------------------------------------

def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average (tie-aware) ranks, 1-based."""
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = average_ranks(np.asarray(scores, dtype=np.float64))
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def select_pair_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> float:
    """Pick the calibration threshold maximising positive-class F1.

    Ties in scores are respected: only prefixes that end at a tie boundary are
    considered.  Predicted-same is ``score >= threshold``.  Ties are broken by
    accuracy, then by the higher threshold (more conservative).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(scores)
    if n == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    l_sorted = labels[order]
    tp_cum = np.cumsum(l_sorted)
    fp_cum = np.cumsum(1 - l_sorted)
    n_pos = int(labels.sum())
    best_key = (-1.0, -1.0, -np.inf)
    best_thr = float(s_sorted[0])
    for k in range(n):
        if k + 1 < n and s_sorted[k + 1] == s_sorted[k]:
            continue
        tp = float(tp_cum[k])
        fp = float(fp_cum[k])
        fn = float(n_pos - tp)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        acc = (tp + (len(labels) - n_pos - fp)) / float(n)
        key = (f1, acc, float(s_sorted[k]))
        if key > best_key:
            best_key = key
            best_thr = float(s_sorted[k])
    return best_thr


def pair_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Dict[str, float]:
    pred = (np.asarray(scores) >= threshold).astype(np.int64)
    tp = float(np.sum((pred == 1) & (labels == 1)))
    fp = float(np.sum((pred == 1) & (labels == 0)))
    fn = float(np.sum((pred == 0) & (labels == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "auc": auc_score(scores, labels),
        "accuracy": float(np.mean(pred == labels)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def pair_scores_for_method(
    name: str,
    gallery: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    rrf_cache: Optional[Dict[Tuple[str, int], Dict[int, float]]] = None,
) -> np.ndarray:
    if name == "rrf":
        assert rrf_cache is not None
        out = []
        for r in rows:
            qi = int(r["query_row_index"])
            gj = int(r["gallery_row_index"])
            out.append(float(rrf_cache[(r["split"], qi)][gj]))
        return np.asarray(out, dtype=np.float64)
    out = []
    for r in rows:
        qi = int(r["query_row_index"])
        gj = int(r["gallery_row_index"])
        out.append(float(gallery[qi] @ gallery[gj]))
    return np.asarray(out, dtype=np.float64)


def build_rrf_pair_cache(
    visual: np.ndarray,
    text: np.ndarray,
    split_members: Dict[str, Sequence[int]],
) -> Dict[Tuple[str, int], Dict[int, float]]:
    """RRF score of every gallery detection for each pair query, per split.

    Ranks are computed over the full split (not just detections that appear in
    the pair file), mirroring the retrieval protocol.
    """
    cache: Dict[Tuple[str, int], Dict[int, float]] = {}
    for split, members in split_members.items():
        member_list = [j for j in members if 0 <= j < len(visual)]
        for qi in member_list:
            cand = [j for j in member_list if j != qi]
            sv = visual[cand] @ visual[qi]
            st = text[cand] @ text[qi]
            order_v = np.argsort(-sv, kind="stable")
            order_t = np.argsort(-st, kind="stable")
            rank_v = {cand[i]: k + 1 for k, i in enumerate(order_v)}
            rank_t = {cand[i]: k + 1 for k, i in enumerate(order_t)}
            cache[(split, qi)] = {
                j: 1.0 / (60 + rank_v[j]) + 1.0 / (60 + rank_t[j])
                for j in cand
            }
    return cache


def paired_auc_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """Stratified paired bootstrap for ΔAUC (a - b)."""
    labels = np.asarray(labels, dtype=np.int64)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    n_pos, n_neg = len(pos), len(neg)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([
            np.asarray([pos[rng.randrange(n_pos)] for _ in range(n_pos)]),
            np.asarray([neg[rng.randrange(n_neg)] for _ in range(n_neg)]),
        ])
        lab = labels[idx]
        deltas.append(
            auc_score(np.asarray(scores_a)[idx], lab)
            - auc_score(np.asarray(scores_b)[idx], lab)
        )
    deltas.sort()
    m = float(np.mean(np.asarray(scores_a)) - np.mean(np.asarray(scores_b)))
    observed = auc_score(np.asarray(scores_a), labels) - auc_score(
        np.asarray(scores_b), labels
    )
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot) - 1]
    p = sum(1 for x in deltas if x <= 0) / n_boot if observed >= 0 else \
        sum(1 for x in deltas if x >= 0) / n_boot
    return observed, float(lo), float(hi), min(1.0, 2.0 * p)


def holm_adjusted(pvals: Dict[str, float]) -> Dict[str, float]:
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out = {}
    prev = 0.0
    for i, k in enumerate(order):
        adj = min(1.0, max(prev, pvals[k] * (m - i)))
        out[k] = adj
        prev = adj
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=3000)
    args = ap.parse_args()

    dets, vectors, visual, text, track_of, tracks, split = load_tracks()
    cal_tracks = split["cal_tracks"]
    test_tracks = split["test_tracks"]
    track_name = split["track_name"]
    track_meta = split["meta"]

    # Training data for the learned projection: calibration tracks only.
    cal_idx = [i for i, t in enumerate(track_of) if t in cal_tracks]
    W = fit_ridge_projection(visual, text, cal_idx, lam=1.0)

    # The shipped concat is the reference descriptor.
    fuse = MultimodalFusion(0.6, 0.4)
    concat = np.vstack([fuse_detached(fuse, v, t) for v, t in zip(visual, text)])
    galleries = build_method_galleries(visual, text, concat, W)

    test_idx = [i for i, t in enumerate(track_of) if t in test_tracks]
    test_set = set(test_idx)
    relevant_by_query: Dict[int, set] = {}
    candidates_by_query: Dict[int, List[int]] = {}
    query_indices: List[int] = []
    for qi in test_idx:
        tid = track_of[qi]
        rel = {
            j
            for j in test_idx
            if j != qi
            and track_of[j] == tid
            and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) > 0.5
        }
        if not rel:
            continue
        # Same-track detections closer than 0.5 s are unjudged near-duplicates:
        # exclude them from the gallery instead of counting them as negatives.
        cand = [
            j for j in test_idx
            if j != qi
            and not (
                track_of[j] == tid
                and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) <= 0.5
            )
        ]
        relevant_by_query[qi] = rel
        candidates_by_query[qi] = cand
        query_indices.append(qi)
    if not query_indices:
        raise RuntimeError("no test queries with a same-track positive")

    n_pos = sum(len(relevant_by_query[q]) for q in query_indices)
    n_neg = sum(
        len(candidates_by_query[q]) - len(relevant_by_query[q]) for q in query_indices
    )

    # Pair file (released metadata-derived same/different-track pairs).
    pair_rows = [
        json.loads(line)
        for line in (BENCHMARK / "person_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pair_counts = Counter((p["split"], p["same"]) for p in pair_rows)
    cal_pairs = [p for p in pair_rows if p["split"] == "calibration"]
    test_pairs = [p for p in pair_rows if p["split"] == "test"]
    cal_labels = np.asarray([1 if p["same"] else 0 for p in cal_pairs], dtype=np.int64)
    test_labels = np.asarray([1 if p["same"] else 0 for p in test_pairs], dtype=np.int64)
    rrf_cache = build_rrf_pair_cache(
        visual, text,
        {"calibration": cal_idx, "test": test_idx},
    )

    # Deterministic retrieval methods.
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name in ["visual", "text", "concat", "zero_pad", "norm_matched", "learned_proj"]:
        all_rows[name] = evaluate_method(
            name, galleries[name], query_indices, candidates_by_query, relevant_by_query
        )
    all_rows["rrf"] = evaluate_rrf(
        visual, text, query_indices, candidates_by_query, relevant_by_query
    )

    # Stochastic random-projection baseline: 5 seeds.
    rp_rows: Dict[int, List[Dict[str, Any]]] = {}
    rp_pair_scores: Dict[int, np.ndarray] = {}
    for seed in range(args.seeds):
        g = random_proj_add_batch(visual, text, seed=seed)
        rp_rows[seed] = evaluate_method(
            f"randproj_seed{seed}", g, query_indices,
            candidates_by_query, relevant_by_query, seed=seed,
        )
        rp_pair_scores[seed] = pair_scores_for_method("randproj", g, test_pairs)

    # Retrieval summaries.
    summaries: Dict[str, Any] = {}
    for name, rows in all_rows.items():
        summaries[name] = summarize(rows, args.n_boot)
    rp_p5 = np.asarray([[r["p5"] for r in rp_rows[s]] for s in range(args.seeds)])
    rp_r10 = np.asarray([[r["r10"] for r in rp_rows[s]] for s in range(args.seeds)])
    rp_f1 = np.asarray([[r["f1"] for r in rp_rows[s]] for s in range(args.seeds)])
    rp_f1p = np.asarray([[r["f1_paper"] for r in rp_rows[s]] for s in range(args.seeds)])
    summaries["randproj"] = {
        "n_queries": len(query_indices),
        "p5": float(rp_p5.mean()),
        "p5_sd": float(rp_p5.mean(axis=1).std(ddof=0)),
        "r10": float(rp_r10.mean()),
        "r10_sd": float(rp_r10.mean(axis=1).std(ddof=0)),
        "f1": float(rp_f1.mean()),
        "f1_sd": float(rp_f1.mean(axis=1).std(ddof=0)),
        "f1_paper": float(rp_f1p.mean()),
        "f1_paper_sd": float(rp_f1p.mean(axis=1).std(ddof=0)),
        "seeds": args.seeds,
    }

    # Paired bootstrap vs concat, Holm-corrected over all competitors.
    concat_rows = {r["qid"]: r for r in all_rows["concat"]}
    significance: Dict[str, Any] = {}
    raw_p: Dict[str, float] = {}
    for name, rows in all_rows.items():
        if name == "concat":
            continue
        qids = [r["qid"] for r in rows]
        a = [r["p5"] for r in rows]
        b = [concat_rows[q]["p5"] for q in qids]
        dm, lo, hi, p = paired_bootstrap_ci(a, b, n_boot=args.n_boot, seed=7)
        af = [r["f1"] for r in rows]
        bf = [concat_rows[q]["f1"] for q in qids]
        fm, flo, fhi, fp = paired_bootstrap_ci(af, bf, n_boot=args.n_boot, seed=7)
        significance[name] = {
            "delta_p5": float(dm), "p5_lo": float(lo), "p5_hi": float(hi), "p5_p": float(p),
            "delta_f1": float(fm), "f1_lo": float(flo), "f1_hi": float(fhi), "f1_p": float(fp),
        }
        raw_p[name] = float(p)
    qids = [r["qid"] for r in rp_rows[0]]
    avg_p5 = np.mean([[r["p5"] for r in rp_rows[s]] for s in range(args.seeds)], axis=0)
    avg_f1 = np.mean([[r["f1"] for r in rp_rows[s]] for s in range(args.seeds)], axis=0)
    b_p5 = [concat_rows[q]["p5"] for q in qids]
    b_f1 = [concat_rows[q]["f1"] for q in qids]
    dm, lo, hi, p = paired_bootstrap_ci(list(avg_p5), b_p5, n_boot=args.n_boot, seed=7)
    fm, flo, fhi, fp = paired_bootstrap_ci(list(avg_f1), b_f1, n_boot=args.n_boot, seed=7)
    significance["randproj"] = {
        "delta_p5": float(dm), "p5_lo": float(lo), "p5_hi": float(hi), "p5_p": float(p),
        "delta_f1": float(fm), "f1_lo": float(flo), "f1_hi": float(fhi), "f1_p": float(fp),
        "note": "per-query metrics averaged over seeds before pairing",
    }
    raw_p["randproj"] = float(p)
    p_adj = holm_adjusted(raw_p)
    for name in significance:
        significance[name]["holm_adjusted_p"] = float(p_adj[name])
        significance[name]["holm_reject"] = bool(p_adj[name] <= 0.05)

    # Pair classification: thresholds come from calibration pairs only.
    pair_results: Dict[str, Any] = {}
    pair_thresholds: Dict[str, float] = {}
    pair_test_scores: Dict[str, np.ndarray] = {}
    for name in ["visual", "text", "concat", "zero_pad", "norm_matched", "learned_proj", "rrf"]:
        cal_scores = pair_scores_for_method(
            name, galleries.get(name, visual), cal_pairs, rrf_cache
        )
        test_scores = pair_scores_for_method(
            name, galleries.get(name, visual), test_pairs, rrf_cache
        )
        thr = select_pair_threshold(cal_scores, cal_labels)
        pair_thresholds[name] = thr
        pair_test_scores[name] = test_scores
        pair_results[name] = {
            "calibration": pair_metrics(cal_scores, cal_labels, thr),
            "test": pair_metrics(test_scores, test_labels, thr),
        }
    # Random projection: per-seed threshold and metrics.
    rp_pair_cal = []
    rp_pair_test = []
    for seed in range(args.seeds):
        g = random_proj_add_batch(visual, text, seed=seed)
        cal_s = pair_scores_for_method("randproj", g, cal_pairs)
        test_s = pair_scores_for_method("randproj", g, test_pairs)
        thr = select_pair_threshold(cal_s, cal_labels)
        rp_pair_cal.append(pair_metrics(cal_s, cal_labels, thr))
        rp_pair_test.append(pair_metrics(test_s, test_labels, thr))
    pair_results["randproj"] = {
        "seeds": args.seeds,
        "calibration": {
            k: float(np.mean([m[k] for m in rp_pair_cal]))
            for k in ("auc", "accuracy", "precision", "recall", "f1")
        },
        "test": {
            k: float(np.mean([m[k] for m in rp_pair_test]))
            for k in ("auc", "accuracy", "precision", "recall", "f1")
        },
        "test_sd": {
            k: float(np.std([m[k] for m in rp_pair_test], ddof=0))
            for k in ("auc", "accuracy", "precision", "recall", "f1")
        },
        "threshold_mean": float(np.mean([m["threshold"] for m in rp_pair_cal])),
    }

    # Paired AUC comparisons vs concat, Holm-corrected.
    pair_significance: Dict[str, Any] = {}
    pair_raw_p: Dict[str, float] = {}
    for name in ["visual", "text", "zero_pad", "norm_matched", "learned_proj", "rrf"]:
        dm, lo, hi, p = paired_auc_bootstrap(
            pair_test_scores[name], pair_test_scores["concat"], test_labels,
            n_boot=args.n_boot, seed=31,
        )
        pair_significance[name] = {
            "delta_auc": float(dm), "lo": float(lo), "hi": float(hi), "p": float(p),
        }
        pair_raw_p[name] = float(p)
    # Random projection: bootstrap on seed-averaged per-pair scores (documented
    # approximation; averaging scores is not identical to averaging AUC).
    rp_avg_scores = np.mean(
        [rp_pair_scores[s] for s in range(args.seeds)], axis=0
    )
    dm, lo, hi, p = paired_auc_bootstrap(
        rp_avg_scores, pair_test_scores["concat"], test_labels,
        n_boot=args.n_boot, seed=31,
    )
    pair_significance["randproj"] = {
        "delta_auc": float(dm), "lo": float(lo), "hi": float(hi), "p": float(p),
        "note": "AUC of per-pair scores averaged over seeds (approximation).",
    }
    pair_raw_p["randproj"] = float(p)
    pair_p_adj = holm_adjusted(pair_raw_p)
    for name in pair_significance:
        pair_significance[name]["holm_adjusted_p"] = float(pair_p_adj[name])
        pair_significance[name]["holm_reject"] = bool(pair_p_adj[name] <= 0.05)

    # Grouped bootstrap by pseudo-track.
    group_of = {q: track_name[track_of[q]] for q in query_indices}
    grouped = {}
    for name in ["concat", "visual", "text", "rrf", "learned_proj"]:
        rows = all_rows[name]
        m, lo, hi = grouped_bootstrap_ci(
            [r["p5"] for r in rows],
            [group_of[r["qid"]] for r in rows],
            n_boot=args.n_boot,
            seed=13,
        )
        grouped[name] = {"p5": float(m), "lo": float(lo), "hi": float(hi)}

    # Per-query CSV.
    with (RESULTS / "fusion_per_query.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "seed", "qid", "detection_id", "track_id", "split", "n_relevant",
            "n_gallery", "p5", "r10", "f1", "f1_paper", "rank_first_relevant",
            "top1_detection_id", "top1_track_id", "top1_score",
        ])
        for name, rows in all_rows.items():
            for r in rows:
                qi = r["qid"]
                top = r["top1_index"]
                w.writerow([
                    name, "" if r["seed"] is None else r["seed"], qi,
                    dets[qi]["detection_id"], track_name[track_of[qi]], "test",
                    len(relevant_by_query[qi]), len(candidates_by_query[qi]),
                    f"{r['p5']:.6f}", f"{r['r10']:.6f}", f"{r['f1']:.6f}",
                    f"{r['f1_paper']:.6f}", r["rank_first_relevant"],
                    dets[top]["detection_id"] if top >= 0 else "",
                    track_name[track_of[top]] if top >= 0 else "",
                    f"{r['top1_score']:.6f}",
                ])
        for seed, rows in rp_rows.items():
            for r in rows:
                qi = r["qid"]
                top = r["top1_index"]
                w.writerow([
                    "randproj", seed, qi, dets[qi]["detection_id"],
                    track_name[track_of[qi]], "test",
                    len(relevant_by_query[qi]), len(candidates_by_query[qi]),
                    f"{r['p5']:.6f}", f"{r['r10']:.6f}", f"{r['f1']:.6f}",
                    f"{r['f1_paper']:.6f}", r["rank_first_relevant"],
                    dets[top]["detection_id"] if top >= 0 else "",
                    track_name[track_of[top]] if top >= 0 else "",
                    f"{r['top1_score']:.6f}",
                ])

    # JSON.
    results = {
        "scope": {
            "name": "released single-camera person-descriptor subset",
            "not": "full 4-camera / 312k-descriptor identity benchmark",
            "camera_ids": sorted({d["camera_id"] for d in dets}),
            "note": (
                "No true identity labels or track ids are released (track_id=-1). "
                "Same-track labels are deterministic pseudo-labels formed from the "
                "shipped visual descriptors; visual-only is therefore optimistic."
            ),
        },
        "dataset": {
            "person_descriptors": len(dets),
            "vector_dim": 1152,
            "visual_dim": 768,
            "text_dim": 384,
            "time_start": min(d["timestamp"] for d in dets).isoformat(),
            "time_end": max(d["timestamp"] for d in dets).isoformat(),
            "tracks": track_meta,
            "pair_file": {
                "total": len(pair_rows),
                "calibration_same": pair_counts[("calibration", True)],
                "calibration_different": pair_counts[("calibration", False)],
                "test_same": pair_counts[("test", True)],
                "test_different": pair_counts[("test", False)],
                "label_rule": (
                    "same = two detections in the same deterministic visual+temporal "
                    "pseudo-track at least 0.5 s apart; different = another track in "
                    "the same split, sampled within 30 s when possible."
                ),
            },
            "retrieval_protocol": {
                "queries": len(query_indices),
                "gallery": len(test_idx),
                "mean_gallery_per_query": float(np.mean([len(candidates_by_query[q]) for q in query_indices])),
                "positives": n_pos,
                "negatives": n_neg,
                "query_filter": "test detections with >=1 same-track detection at |dt| > 0.5 s",
                "gallery_filter": (
                    "all test detections except the query and same-track detections "
                    "within 0.5 s (unjudged near-duplicates)"
                ),
                "relevance": "same pseudo-track and |dt| > 0.5 s",
            },
        },
        "methods": {
            "visual": "cosine on the 768-d SigLIP branch",
            "text": "cosine on the 384-d MiniLM attribute-text branch",
            "concat": "MultimodalFusion(0.6,0.4), 1152-d, reproduces the shipped index",
            "zero_pad": "text zero-padded to 768 then 0.6*v + 0.4*t",
            "norm_matched": "unit-norm branches, zero-pad text, 0.6/0.4 weighted add",
            "randproj": f"Gaussian P(768x384)/sqrt(384), 0.6/0.4 add, {args.seeds} seeds",
            "rrf": "Reciprocal Rank Fusion of visual and text rankings, k=60",
            "learned_proj": "ridge text->visual map (384->768, lambda=1) fit on calibration tracks only, then add",
        },
        "test": {
            "retrieval": summaries,
            "pairs": pair_results,
        },
        "significance_vs_concat": {
            "retrieval": significance,
            "pairs": pair_significance,
        },
        "grouped_bootstrap_by_track": grouped,
        "limitations": [
            "Single camera (camera_0); cross-camera identity and generalisation are untestable.",
            "Pseudo-track labels, not human identity annotations; no IAA.",
            "Tracks were formed with visual-only cosine, so visual-only retrieval is optimistically biased.",
            "Track threshold is selected on a small calibration block (169 detections) with a self-supervised objective, not identity labels.",
            "Pair thresholds are tuned on calibration pairs; test pairs are track-disjoint but come from the same two-hour window.",
            "The paper's 312k-descriptor / 4-camera 0.785->0.807 fusion result is not reproduced and is not replaced by these numbers.",
            "The learned projection is a linear ridge baseline, not a jointly trained Re-ID model.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fusion_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Markdown.
    lines: List[str] = []
    lines.append("# Released-subset person fusion results\n")
    lines.append(
        "**Scope.** `camera_0` only, 1,778 shipped 1152-d descriptors, "
        f"{track_meta['n_test_detections']} in the test track split. Same-track labels are "
        "deterministic pseudo-labels from visual+temporal matching, not human identities.\n"
    )
    lines.append(
        f"- Person descriptors: {len(dets)}\n"
        f"- Pseudo-tracks: {track_meta['n_tracks']} "
        f"({track_meta['n_test_tracks']} test / {track_meta['n_calibration_tracks']} calibration)\n"
        f"- Test retrieval queries: {len(query_indices)}; mean gallery: "
        f"{results['dataset']['retrieval_protocol']['mean_gallery_per_query']:.0f}\n"
        f"- Same-track positives in the retrieval pool: {n_pos}\n"
        f"- Different-track negatives in the retrieval pool: {n_neg}\n"
        f"- Pair file: {len(pair_rows)} pairs "
        f"(test {pair_counts[('test', True)]}+ / {pair_counts[('test', False)]}-; "
        f"calibration {pair_counts[('calibration', True)]}+ / {pair_counts[('calibration', False)]}-)\n"
        f"- Track threshold: {track_meta['threshold']:.2f} "
        f"(grid in `fusion_results.json`); max gap 3 s, max duration 60 s, max track size 20.\n"
    )
    lines.append("\n## Person retrieval (same-track pseudo-labels)\n")
    lines.append("| Method | P@5 | R@10 | F1@10 (P@10,R@10) | F1@10 paper (P@5,R@10) | P@5 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name in METHODS:
        s = summaries[name]
        ci = s.get("p5_ci")
        ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "mean±SD over seeds"
        if name == "randproj":
            lines.append(
                f"| {name} | {s['p5']:.3f} ± {format_sd(s['p5_sd'])} | "
                f"{s['r10']:.3f} ± {format_sd(s['r10_sd'])} | "
                f"{s['f1']:.3f} ± {format_sd(s['f1_sd'])} | "
                f"{s['f1_paper']:.3f} ± {format_sd(s['f1_paper_sd'])} | "
                f"{ci_txt} |"
            )
        else:
            lines.append(
                f"| {name} | {s['p5']:.3f} | {s['r10']:.3f} | {s['f1']:.3f} | "
                f"{s['f1_paper']:.3f} | {ci_txt} |"
            )
    lines.append(
        "\n`±` is the standard deviation over the "
        f"{summaries['randproj']['seeds']} random-projection seeds; a spread below "
        "0.001 is shown as `< 0.001` (it is not zero). The raw standard deviations "
        "are in `fusion_results.json`."
    )
    lines.append("\n## Person-pair classification (threshold tuned on calibration pairs)\n")
    lines.append("| Method | Threshold | AUC | Accuracy | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in METHODS:
        s = pair_results[name]["test"]
        if name == "randproj":
            sd = pair_results[name]["test_sd"]
            lines.append(
                f"| {name} | {pair_results[name]['threshold_mean']:.3f} | "
                f"{s['auc']:.3f} ± {format_sd(sd['auc'])} | "
                f"{s['accuracy']:.3f} ± {format_sd(sd['accuracy'])} | "
                f"{s['precision']:.3f} ± {format_sd(sd['precision'])} | "
                f"{s['recall']:.3f} ± {format_sd(sd['recall'])} | "
                f"{s['f1']:.3f} ± {format_sd(sd['f1'])} |"
            )
        else:
            lines.append(
                f"| {name} | {s['threshold']:.3f} | {s['auc']:.3f} | {s['accuracy']:.3f} | "
                f"{s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |"
            )
    lines.append("\n## Paired bootstrap vs concat (Holm over competitors)\n")
    lines.append(
        f"P-values are two-sided paired bootstrap with {args.n_boot} resamples. "
        f"`< {2.0 / args.n_boot:.4f}` is the resolution floor (2/{args.n_boot}): no "
        "resample fell on the null side, so the p-value is below the floor and not "
        "exactly zero. The raw p-values are in `fusion_results.json`.\n"
    )
    lines.append("| Task | Method | Δ [95% CI] | raw p | Holm p | reject |")
    lines.append("|---|---|---:|---:|---:|---|")
    for name, s in significance.items():
        lines.append(
            f"| retrieval P@5 | {name} | {s['delta_p5']:+.3f} "
            f"[{s['p5_lo']:+.3f}, {s['p5_hi']:+.3f}] | {format_p(s['p5_p'], args.n_boot)} | "
            f"{format_p(s['holm_adjusted_p'], args.n_boot)} | {s['holm_reject']} |"
        )
    for name, s in pair_significance.items():
        lines.append(
            f"| pair AUC | {name} | {s['delta_auc']:+.3f} "
            f"[{s['lo']:+.3f}, {s['hi']:+.3f}] | {format_p(s['p'], args.n_boot)} | "
            f"{format_p(s['holm_adjusted_p'], args.n_boot)} | {s['holm_reject']} |"
        )
    lines.append("\n## Grouped bootstrap by pseudo-track (P@5)\n")
    lines.append("| Method | Mean | 95% CI |")
    lines.append("|---|---:|---|")
    for name, s in grouped.items():
        lines.append(f"| {name} | {s['p5']:.3f} | [{s['lo']:.3f}, {s['hi']:.3f}] |")
    lines.append("\n## What cannot be reproduced\n")
    for item in results["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    (RESULTS / "fusion_tables.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        f"tracks={track_meta['n_tracks']} threshold={track_meta['threshold']:.2f} "
        f"test_queries={len(query_indices)} positives={n_pos} "
        f"pairs={len(pair_rows)} ({len(cal_pairs)} cal/{len(test_pairs)} test)"
    )
    for name in METHODS:
        s = summaries[name]
        ps = pair_results[name]["test"]
        print(
            f"{name:14s} P@5={s['p5']:.3f} R@10={s['r10']:.3f} F1={s['f1']:.3f} "
            f"pairAUC={ps['auc']:.3f} pairAcc={ps['accuracy']:.3f}"
        )
    print(f"wrote {RESULTS / 'fusion_results.json'}")
    print(f"wrote {RESULTS / 'fusion_tables.md'}")
    print(f"wrote {RESULTS / 'fusion_per_query.csv'}")


if __name__ == "__main__":
    main()
