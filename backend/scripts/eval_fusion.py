"""Real person-pair / single-camera track retrieval on the released descriptors.

There are no identity labels and no track ids in the release (all
``track_id=-1``).  The PRIMARY protocol therefore builds deterministic
pseudo-tracks from detector/attribute metadata only (camera, colour,
attribute-phrase overlap, 3 s temporal gate) -- no visual cosine -- splits
whole tracks into calibration/test, and evaluates two things:

1. **Retrieval**: given one test detection, retrieve other detections of the
   same pseudo-track from the test split.
2. **Pair classification**: classify metadata-track-derived same/different
   pairs.  The decision threshold for every method is selected on
   calibration pairs only; **threshold-free pair AUC is the primary
   metric**.

A visual-cosine track formation is retained as a *sensitivity* (point
estimates only) to demonstrate the direction of the formation bias.  The
benchmark is a *released-subset reproducibility track*: it is not the
paper's 312k-descriptor / 4-camera identity benchmark, and the tracks are
pseudo-labels rather than human identity annotations.

Run:
    python backend/scripts/eval_fusion.py
    python backend/scripts/eval_fusion.py --seeds 5 --n-boot 3000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
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
    holm_correction,
    paired_bootstrap_ci,
)
from backend.app.services.fusion import MultimodalFusion, fuse_gated

BENCHMARK = ROOT / "backend" / "benchmarks"
RESULTS = ROOT / "results"
CAL_BLOCK_END = subset.parse_det_ts("2026-03-01T17:09:00")
CAL_BLOCK2_START = subset.parse_det_ts("2026-03-01T18:55:00")
METHODS = [
    "visual", "text", "concat", "gated", "zero_pad", "norm_matched", "rrf",
    "learned_proj", "randproj",
]
GATE_LAMBDA = 4.0


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
# C2': metadata-only track formation (no visual cosine) + quality estimators
# ---------------------------------------------------------------------------

def attr_phrases(det: Dict[str, Any]) -> set:
    """Attribute-phrase set from the released attribute text (no labels)."""
    parts = [
        p.strip().lower()
        for p in str(det.get("attribute_text") or "").split(",")
    ]
    return {p for p in parts if p and p != "person"}


def attr_jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_tracks_metadata(
    dets: Sequence[Dict[str, Any]],
    max_gap: float = 3.0,
    max_duration: float = 60.0,
    max_size: int = 20,
    min_jaccard: float = 0.5,
) -> Tuple[List[int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy metadata-only linking: same camera + same colour + phrase overlap.

    A detection joins the active track with the highest attribute-phrase
    Jaccard similarity, provided the colours match (case-insensitive), the
    Jaccard is at least ``min_jaccard``, the last detection is at most
    ``max_gap`` seconds older, and the duration/size caps hold.  No visual
    embedding is consulted, so visual-branch retrieval evaluated on these
    tracks is not optimistically biased by construction.  The rule is fixed
    and documented (no threshold is tuned on embeddings).
    """
    n = len(dets)
    order = sorted(
        range(n), key=lambda i: (dets[i]["timestamp"], dets[i]["faiss_id"])
    )
    phrases = [attr_phrases(d) for d in dets]
    colors = [str(d.get("color") or "").lower() for d in dets]
    cams = [d.get("camera_id") for d in dets]
    track_of = [-1] * n
    tracks: List[Dict[str, Any]] = []
    active: List[Tuple[int, int]] = []
    edges: List[Dict[str, Any]] = []
    for i in order:
        ts_i = dets[i]["timestamp"]
        best_tid, best_j, best_sim = -1, -1, -1.0
        for tid, j in active:
            dt = (ts_i - dets[j]["timestamp"]).total_seconds()
            if dt <= 0 or dt > max_gap:
                continue
            if (ts_i - tracks[tid]["start"]).total_seconds() > max_duration:
                continue
            if tracks[tid]["size"] >= max_size:
                continue
            if cams[i] != cams[j] or colors[i] != colors[j]:
                continue
            sim = attr_jaccard(phrases[i], phrases[j])
            if sim > best_sim or (sim == best_sim and tid < best_tid):
                best_sim, best_tid, best_j = sim, tid, j
        if best_tid >= 0 and best_sim >= min_jaccard:
            track_of[i] = best_tid
            edges.append(
                {
                    "query_index": i,
                    "gallery_index": best_j,
                    "track_id": best_tid,
                    "same": True,
                    "dt": (ts_i - dets[best_j]["timestamp"]).total_seconds(),
                    "jaccard": best_sim,
                }
            )
            for k, (tid, j) in enumerate(active):
                if tid == best_tid:
                    active[k] = (tid, i)
                    break
        else:
            tid = len(tracks)
            track_of[i] = tid
            tracks.append(
                {"track_id": tid, "indices": [], "start": ts_i, "end": ts_i, "size": 0}
            )
            active.append((tid, i))
        tracks[track_of[i]]["indices"].append(i)
        tracks[track_of[i]]["size"] += 1
        tracks[track_of[i]]["start"] = min(tracks[track_of[i]]["start"], ts_i)
        tracks[track_of[i]]["end"] = max(tracks[track_of[i]]["end"], ts_i)
        active = [
            (tid, j)
            for tid, j in active
            if (ts_i - dets[j]["timestamp"]).total_seconds() <= max_gap
        ]
    for tr in tracks:
        tr["size"] = len(tr["indices"])
        tr["track_id_str"] = f"t{tr['track_id']:05d}"
    return track_of, tracks, edges


def split_tracks_by_time(
    tracks: Sequence[Dict[str, Any]], n_folds: int = 4
) -> Tuple[set, set]:
    """Whole-track calibration/test split: every 4th track (by start time)."""
    order = sorted(range(len(tracks)),
                   key=lambda t: (tracks[t]["start"], tracks[t]["track_id"]))
    cal = {t for i, t in enumerate(order) if i % n_folds == 0}
    return cal, set(order) - cal


def estimate_qualities(
    dets: Sequence[Dict[str, Any]],
    frames: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Per-detection branch qualities from released metadata only.

    * Text quality ``qt``: attribute-text richness =
      ``min(1, n_phrases / 4)`` where phrases are the comma-separated
      attributes.  A bare ``"person"`` carries no attribute signal (0.0).
    * Visual quality ``qv``: occlusion proxy from the nearest scene frame's
      person count, ``1 / sqrt(max(1, count))``.  Crowded frames are more
      likely to contain occlusions and small crops.

    No new labels are used; both inputs ship in the release.
    """
    frame_ts = [fr["frame_ts"] for fr in frames]
    frame_counts = [
        fr["count"] if fr.get("count") is not None else 1 for fr in frames
    ]
    qv = np.zeros(len(dets), dtype=np.float64)
    qt = np.zeros(len(dets), dtype=np.float64)
    fallback = 0
    for i, d in enumerate(dets):
        n_phr = len(attr_phrases(d))
        qt[i] = min(1.0, n_phr / 4.0)
        ts = d["timestamp"]
        best_j, best_dt = -1, None
        for j, fts in enumerate(frame_ts):
            dt = abs((ts - fts).total_seconds())
            if best_dt is None or dt < best_dt:
                best_dt, best_j = dt, j
        if best_j < 0:
            qv[i] = 1.0
            fallback += 1
        else:
            qv[i] = 1.0 / math.sqrt(max(1, int(frame_counts[best_j] or 1)))
    info = {
        "qt_definition": "min(1, n_attribute_phrases/4)",
        "qv_definition": "1/sqrt(max(1, nearest_scene_frame_person_count))",
        "mean_qv": float(np.mean(qv)),
        "mean_qt": float(np.mean(qt)),
        "frames_without_match": fallback,
    }
    return qv, qt, info


def build_pairs_from_tracks(
    track_of: Sequence[int],
    tracks: Sequence[Dict[str, Any]],
    dets: Sequence[Dict[str, Any]],
    split_of: Dict[int, str],
    seed: int = 0,
    max_pairs_per_split: int = 1008,
) -> List[Dict[str, Any]]:
    """Deterministic same/different-track pairs mirroring the released rule.

    Same = two detections of one pseudo-track at least 0.5 s apart;
    different = another track in the same split, sampled within 30 s when
    possible (one negative per positive, seeded).  Totals are capped at
    ``max_pairs_per_split`` per split (balanced) for bootstrap tractability;
    actual counts are reported.
    """
    rng = random.Random(seed)
    by_track: Dict[int, List[int]] = {}
    for i, t in enumerate(track_of):
        by_track.setdefault(t, []).append(i)
    split_members: Dict[str, List[int]] = {"calibration": [], "test": []}
    for i, t in enumerate(track_of):
        split_members[split_of[t]].append(i)
    rows: List[Dict[str, Any]] = []
    pid = 0
    for split in ("calibration", "test"):
        positives: List[Tuple[int, int]] = []
        for t, members in by_track.items():
            if split_of.get(t) != split:
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    dt = abs((dets[i]["timestamp"] - dets[j]["timestamp"]).total_seconds())
                    if dt > 0.5:
                        positives.append((i, j))
        rng.shuffle(positives)
        cap = max_pairs_per_split // 2
        positives = positives[:cap]
        members = split_members[split]
        for i, j in positives:
            cands_near = [
                m for m in members
                if track_of[m] != track_of[i]
                and abs((dets[m]["timestamp"] - dets[i]["timestamp"]).total_seconds()) <= 30.0
            ]
            pool = cands_near or [
                m for m in members if track_of[m] != track_of[i]
            ]
            if not pool:
                continue
            m = pool[rng.randrange(len(pool))]
            for qi, gj, same in ((i, j, True), (i, m, False)):
                pid += 1
                rows.append(
                    {
                        "pid": f"meta_{pid:06d}",
                        "split": split,
                        "query_row_index": int(qi),
                        "gallery_row_index": int(gj),
                        "same": bool(same),
                        "track_a": int(track_of[qi]),
                        "track_b": int(track_of[gj]),
                        "dt_seconds": float(
                            abs((dets[qi]["timestamp"] - dets[gj]["timestamp"]).total_seconds())
                        ),
                        "camera_id": dets[qi].get("camera_id"),
                    }
                )
    return rows


def probe_public_reid() -> Dict[str, Any]:
    """Time-boxed public Re-ID mini-split probe (local-disk check only).

    The environment is CPU-only and offline, so no dataset download is
    attempted.  This function checks the ``REID_DATASET_DIR`` environment
    variable and a short list of conventional local paths for a verifiable
    public Re-ID source (Market-1501 / MSMT17 style: bounding-box images +
    identity labels).  Whatever is found (or not) is returned for the
    provenance log; the caller falls back to a held-out temporal split of
    the released data when nothing verifiable resolves.
    """
    candidates: List[str] = []
    env_dir = os.environ.get("REID_DATASET_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates += [
        "data/reid/Market-1501-v15.09.15",
        "data/reid/MSMT17",
        "data/reid",
        "../Market-1501-v15.09.15",
    ]
    checked = []
    for cand in candidates:
        p = (ROOT / cand) if not os.path.isabs(cand) else Path(cand)
        bounding = p / "bounding_box_train"
        query = p / "query"
        has_layout = bounding.is_dir() and query.is_dir()
        checked.append({"path": str(p), "exists": p.exists(),
                        "has_market_layout": bool(has_layout)})
        if has_layout:
            return {
                "resolved": True, "path": str(p),
                "checked": checked,
                "note": "public Re-ID source found on local disk",
            }
    return {
        "resolved": False, "path": None,
        "checked": checked,
        "note": (
            "No verifiable public Re-ID source on local disk; download not "
            "attempted (offline, CPU-only). Fell back to a held-out temporal "
            "split (early/late halves by test-timestamp median) of the "
            "released data, reported as temporal_halves."
        ),
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
    qv: Optional[np.ndarray] = None,
    qt: Optional[np.ndarray] = None,
    gate_lambda: float = GATE_LAMBDA,
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
    out = {
        "visual": visual.astype(np.float32),
        "text": text.astype(np.float32),
        "concat": concat.astype(np.float32),
        "zero_pad": zero.astype(np.float32),
        "norm_matched": norm.astype(np.float32),
        "learned_proj": learned.astype(np.float32),
    }
    if qv is not None and qt is not None:
        gated = np.vstack(
            [
                fuse_gated(
                    np.array(v, dtype=np.float32, copy=True),
                    np.array(t, dtype=np.float32, copy=True),
                    float(qv[i]), float(qt[i]), lam=gate_lambda,
                )
                for i, (v, t) in enumerate(zip(visual, text))
            ]
        )
        out["gated"] = gated.astype(np.float32)
    return out


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


def kfold_by_track(
    dets: Sequence[Dict[str, Any]],
    visual: np.ndarray,
    text: np.ndarray,
    galleries: Dict[str, np.ndarray],
    track_of: Sequence[int],
    tracks: Sequence[Dict[str, Any]],
    n_folds: int = 5,
) -> Dict[str, Any]:
    """5-fold CV over pseudo-tracks: P@5 per fold for deterministic methods.

    Fold assignment is round-robin over tracks sorted by (start, track_id),
    so every fold holds out whole tracks (~174/5 tracks each). Within a fold
    the gallery is restricted to the held-out tracks' detections, mirroring
    the main retrieval protocol (|dt| > 0.5 s relevance, near-duplicates
    excluded). RRF is recomputed per fold over the fold gallery; randproj
    uses seed 0 only (spread across seeds is reported in the main table).
    """
    from backend.app.eval.baselines import random_proj_add_batch

    order = sorted(range(len(tracks)),
                   key=lambda t: (tracks[t]["start"], tracks[t]["track_id"]))
    fold_of_track = {t: (i % n_folds) for i, t in enumerate(order)}
    fold_ids = [fold_of_track[t] for t in track_of]
    rp_gallery = random_proj_add_batch(visual, text, seed=0)
    fold_galleries = dict(galleries)
    fold_galleries["randproj_s0"] = rp_gallery

    per_fold: List[Dict[str, float]] = []
    for fold in range(n_folds):
        members = [i for i, f in enumerate(fold_ids) if f == fold]
        member_set = set(members)
        qis = []
        rel: Dict[int, set] = {}
        cand: Dict[int, List[int]] = {}
        for qi in members:
            tid = track_of[qi]
            r = {j for j in members if j != qi and track_of[j] == tid
                 and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) > 0.5}
            if not r:
                continue
            c = [j for j in members if j != qi and not (
                track_of[j] == tid
                and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) <= 0.5)]
            if not c:
                continue
            qis.append(qi)
            rel[qi] = r
            cand[qi] = c
        fold_row: Dict[str, float] = {"fold": float(fold), "n_queries": float(len(qis))}
        for name in ["visual", "text", "concat", "gated", "zero_pad", "norm_matched",
                     "learned_proj", "randproj_s0"]:
            g = fold_galleries[name]
            p5s = []
            for qi in qis:
                scores = g @ g[qi]
                m = retrieval_metrics(scores, cand[qi], rel[qi])
                p5s.append(m["p5"])
            fold_row[name] = float(np.mean(p5s)) if p5s else 0.0
        # RRF per fold.
        p5s = []
        for qi in qis:
            c = cand[qi]
            carr = np.asarray(c, dtype=np.int64)
            sv = visual[carr] @ visual[qi]
            st = text[carr] @ text[qi]
            ov = np.argsort(-sv, kind="stable")
            ot = np.argsort(-st, kind="stable")
            rv = [str(carr[i]) for i in ov]
            rt = [str(carr[i]) for i in ot]
            rank = [int(x) for x in rrf_fuse(rv, rt, k=60)[:10]]
            p5s.append(precision_at_k(rank, rel[qi], 5))
        fold_row["rrf"] = float(np.mean(p5s)) if p5s else 0.0
        per_fold.append(fold_row)

    summary: Dict[str, Any] = {"n_folds": n_folds, "n_tracks": len(tracks),
                               "per_fold": per_fold}
    for name in ["visual", "text", "concat", "gated", "zero_pad", "norm_matched",
                 "learned_proj", "randproj_s0", "rrf"]:
        vals = [f[name] for f in per_fold]
        summary[name] = {"mean": float(np.mean(vals)),
                         "sd": float(np.std(vals, ddof=0)),
                         "min": float(min(vals)), "max": float(max(vals))}
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=3000)
    args = ap.parse_args()

    dets, vectors, visual, text, vis_track_of, vis_tracks, split = load_tracks()
    vis_cal_tracks = split["cal_tracks"]
    vis_test_tracks = split["test_tracks"]
    vis_track_meta = split["meta"]

    # C2' primary: metadata-only track formation (no visual cosine).
    meta_track_of, meta_tracks, meta_edges = build_tracks_metadata(dets)
    meta_cal_tracks, meta_test_tracks = split_tracks_by_time(meta_tracks)
    meta_split_of = {
        tid: ("calibration" if tid in meta_cal_tracks else "test")
        for tid in range(len(meta_tracks))
    }
    frames = subset.load_scene_frames()
    qv, qt, quality_info = estimate_qualities(dets, frames)

    # Working variables below refer to the PRIMARY (metadata-formed) tracks.
    # Visual-formed tracks are retained only for the bias sensitivity.
    track_of, tracks = meta_track_of, meta_tracks
    cal_tracks, test_tracks = meta_cal_tracks, meta_test_tracks
    track_name = {
        tid: f"{meta_split_of[tid]}_t{tid:05d}" for tid in range(len(meta_tracks))
    }
    track_meta = {
        "formation": "metadata-only (camera + colour + attribute-phrase Jaccard>=0.5 + 3s gate); no visual cosine",
        "min_jaccard": 0.5,
        "n_tracks": len(meta_tracks),
        "n_calibration_tracks": len(meta_cal_tracks),
        "n_test_tracks": len(meta_test_tracks),
        "n_calibration_detections": sum(1 for t in meta_track_of if t in meta_cal_tracks),
        "n_test_detections": sum(1 for t in meta_track_of if t not in meta_cal_tracks),
        "n_tracks_ge2": sum(1 for t in meta_tracks if t["size"] >= 2),
        "n_link_edges": len(meta_edges),
        "max_gap_seconds": 3.0,
        "max_duration_seconds": 60.0,
        "max_track_size": 20,
    }

    # Training data for the learned projection: calibration tracks only.
    cal_idx = [i for i, t in enumerate(track_of) if t in cal_tracks]
    W = fit_ridge_projection(visual, text, cal_idx, lam=1.0)

    # The shipped concat is the reference descriptor.
    fuse = MultimodalFusion(0.6, 0.4)
    concat = np.vstack([fuse_detached(fuse, v, t) for v, t in zip(visual, text)])
    galleries = build_method_galleries(visual, text, concat, W, qv, qt)

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

    # C2' primary pairs: built from the metadata-formed tracks (no visual
    # cosine anywhere in formation).  Thresholds come from calibration only.
    provenance = probe_public_reid()
    pair_rows = build_pairs_from_tracks(
        track_of, tracks, dets, meta_split_of, seed=0
    )
    pair_counts = Counter((p["split"], p["same"]) for p in pair_rows)
    cal_pairs = [p for p in pair_rows if p["split"] == "calibration"]
    test_pairs = [p for p in pair_rows if p["split"] == "test"]
    if not cal_pairs or not test_pairs:
        raise RuntimeError("metadata track pairs are empty")
    cal_labels = np.asarray([1 if p["same"] else 0 for p in cal_pairs], dtype=np.int64)
    test_labels = np.asarray([1 if p["same"] else 0 for p in test_pairs], dtype=np.int64)
    rrf_cache = build_rrf_pair_cache(
        visual, text,
        {"calibration": cal_idx, "test": test_idx},
    )

    # Deterministic retrieval methods.
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name in ["visual", "text", "concat", "gated", "zero_pad", "norm_matched", "learned_proj"]:
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
    for name in ["visual", "text", "concat", "gated", "zero_pad", "norm_matched", "learned_proj", "rrf"]:
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
    for name in ["visual", "text", "zero_pad", "norm_matched", "learned_proj", "gated", "rrf"]:
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

    # C2' head-to-head: gated vs static concat and gated vs RRF, on both the
    # threshold-free pair AUC (primary) and retrieval P@5.  Holm over the
    # two comparisons per task.
    gated_head_to_head: Dict[str, Any] = {}
    for task, s_a, s_b, lab, seed in [
        ("pair_auc", pair_test_scores["gated"], pair_test_scores["concat"], test_labels, 61),
        ("pair_auc", pair_test_scores["gated"], pair_test_scores["rrf"], test_labels, 63),
    ]:
        dm, lo, hi, p = paired_auc_bootstrap(s_a, s_b, lab, n_boot=args.n_boot, seed=seed)
        gated_head_to_head.setdefault(task, []).append(
            {"delta_auc": float(dm), "lo": float(lo), "hi": float(hi), "p": float(p)}
        )
    g_rows = {r["qid"]: r for r in all_rows["gated"]}
    c_rows = {r["qid"]: r for r in all_rows["concat"]}
    r_rows = {r["qid"]: r for r in all_rows["rrf"]}
    qids_h = [r["qid"] for r in all_rows["gated"]]
    for other, other_rows, seed in [("concat", c_rows, 65), ("rrf", r_rows, 67)]:
        a = [g_rows[q]["p5"] for q in qids_h]
        b = [other_rows[q]["p5"] for q in qids_h]
        dm, lo, hi, p = paired_bootstrap_ci(a, b, n_boot=args.n_boot, seed=seed)
        gated_head_to_head.setdefault("retrieval_p5", []).append(
            {"other": other, "delta_p5": float(dm), "lo": float(lo),
             "hi": float(hi), "p": float(p)}
        )
    for i, entry in enumerate(gated_head_to_head["pair_auc"]):
        entry["other"] = ["concat", "rrf"][i]
    for task in ("pair_auc", "retrieval_p5"):
        fam = {e["other"]: e["p"] for e in gated_head_to_head[task]}
        adj = holm_adjusted(fam)
        rej = holm_correction(fam, alpha=0.05)
        for e in gated_head_to_head[task]:
            e["holm_adjusted_p"] = float(adj[e["other"]])
            e["holm_reject"] = bool(rej[e["other"]])

    # C2' held-out temporal split: test pairs halved by gallery-timestamp
    # median (early/late).  Same calibration thresholds; threshold-free AUC
    # per half and method (point estimates -- the CIs live in the main
    # pair-AUC table).  This is the documented fallback for the unresolved
    # public Re-ID probe (see provenance).
    gal_ts = np.asarray(
        [dets[int(p["gallery_row_index"])]["timestamp"].timestamp() for p in test_pairs],
        dtype=np.float64,
    )
    t_median = float(np.median(gal_ts))
    temporal_halves: Dict[str, Any] = {"median_gallery_ts": t_median, "halves": {}}
    for half, mask in [
        ("early", gal_ts <= t_median),
        ("late", gal_ts > t_median),
    ]:
        idx = np.flatnonzero(mask)
        lab_h = test_labels[idx]
        half_out: Dict[str, float] = {"n_pairs": int(len(idx))}
        for name, scores in pair_test_scores.items():
            half_out[name] = float(auc_score(np.asarray(scores)[idx], lab_h))
        temporal_halves["halves"][half] = half_out

    # Sensitivity: visual-cosine track formation (point estimates only).
    # Demonstrates the formation-bias direction without spending bootstrap
    # budget on a knowingly biased labelling.
    vis_test_idx = [i for i, t in enumerate(vis_track_of) if t in vis_test_tracks]
    vis_test_set = set(vis_test_idx)
    vis_rel: Dict[int, set] = {}
    vis_cand: Dict[int, List[int]] = {}
    vis_qis: List[int] = []
    for qi in vis_test_idx:
        tid = vis_track_of[qi]
        rel = {
            j for j in vis_test_idx if j != qi and vis_track_of[j] == tid
            and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) > 0.5
        }
        if not rel:
            continue
        cand = [
            j for j in vis_test_idx if j != qi and not (
                vis_track_of[j] == tid
                and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) <= 0.5
            )
        ]
        vis_rel[qi] = rel
        vis_cand[qi] = cand
        vis_qis.append(qi)
    vis_sensitivity: Dict[str, Any] = {
        "formation": "visual-cosine (knowingly biased toward the visual branch; point estimates only)",
        "n_queries": len(vis_qis),
        "retrieval_p5": {},
        "pair_auc": {},
    }
    for name, gal in galleries.items():
        p5s = []
        for qi in vis_qis:
            scores = gal @ gal[qi]
            m = retrieval_metrics(scores, vis_cand[qi], vis_rel[qi])
            p5s.append(m["p5"])
        vis_sensitivity["retrieval_p5"][name] = float(np.mean(p5s)) if p5s else 0.0
    try:
        released_pairs = [
            json.loads(line)
            for line in (BENCHMARK / "person_pairs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rel_test = [p for p in released_pairs if p["split"] == "test"]
        rel_labels = np.asarray([1 if p["same"] else 0 for p in rel_test], dtype=np.int64)
        for name, gal in galleries.items():
            sc = pair_scores_for_method(name, gal, rel_test)
            vis_sensitivity["pair_auc"][name] = float(auc_score(sc, rel_labels))
        # RRF on the released pairs needs its own cache (guarded: only pairs
        # whose query is a visual-test member are scored).
        vis_rrf_cache = build_rrf_pair_cache(
            visual, text, {"test": vis_test_idx}
        )
        rel_test_rrf = [p for p in rel_test if ("test", int(p["query_row_index"])) in vis_rrf_cache]
        if rel_test_rrf:
            sc_rrf = pair_scores_for_method("rrf", visual, rel_test_rrf, vis_rrf_cache)
            lab_rrf = np.asarray([1 if p["same"] else 0 for p in rel_test_rrf], dtype=np.int64)
            vis_sensitivity["pair_auc"]["rrf"] = float(auc_score(sc_rrf, lab_rrf))
            vis_sensitivity["n_rrf_pairs"] = len(rel_test_rrf)
        vis_sensitivity["n_released_test_pairs"] = len(rel_test)
    except Exception as exc:
        vis_sensitivity["pair_auc_error"] = f"{type(exc).__name__}: {exc}"

    # Grouped bootstrap by pseudo-track.
    group_of = {q: track_name[track_of[q]] for q in query_indices}
    grouped = {}
    for name in ["concat", "gated", "visual", "text", "rrf", "learned_proj"]:
        rows = all_rows[name]
        m, lo, hi = grouped_bootstrap_ci(
            [r["p5"] for r in rows],
            [group_of[r["qid"]] for r in rows],
            n_boot=args.n_boot,
            seed=13,
        )
        grouped[name] = {"p5": float(m), "lo": float(lo), "hi": float(hi)}

    # K-fold over pseudo-tracks (k=5): each fold holds out ~1/5 of the tracks
    # as queries, gallery = all detections in the held-out tracks except the
    # query and its |dt| <= 0.5 s near-duplicates. Reports P@5 mean+/-SD
    # across folds for the deterministic methods (single seed 0 for randproj).
    kfold = kfold_by_track(
        dets, visual, text, galleries, track_of, tracks, n_folds=5)

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
            "primary_metric": "threshold-free pair AUC on metadata-formed tracks",
            "note": (
                "No true identity labels or track ids are released (track_id=-1). "
                "Same-track labels are deterministic pseudo-labels formed from "
                "detector/attribute metadata only (camera, colour, "
                "attribute-phrase overlap, temporal gate); no visual cosine is "
                "used in formation, so the visual branch is not favoured by "
                "construction. A visual-cosine formation is reported as a "
                "sensitivity only."
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
            "quality_estimators": quality_info,
            "gate_lambda": GATE_LAMBDA,
            "pair_file": {
                "total": len(pair_rows),
                "calibration_same": pair_counts[("calibration", True)],
                "calibration_different": pair_counts[("calibration", False)],
                "test_same": pair_counts[("test", True)],
                "test_different": pair_counts[("test", False)],
                "label_rule": (
                    "same = two detections in the same deterministic "
                    "metadata-only pseudo-track at least 0.5 s apart; "
                    "different = another track in the same split, sampled "
                    "within 30 s when possible (one negative per positive, "
                    "seed 0, capped at 1008 pairs per split)."
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
            "gated": (
                "quality-gated concat fuse_gated(v,t,qv,qt): gw=sigmoid(4*(qv-qt)), "
                "tie reduces to static (0.6,0.4); qv/qv from released metadata only"
            ),
            "zero_pad": "text zero-padded to 768 then 0.6*v + 0.4*t",
            "norm_matched": "unit-norm branches, zero-pad text, 0.6/0.4 weighted add",
            "randproj": f"Gaussian P(768x384)/sqrt(384), 0.6/0.4 add, {args.seeds} seeds",
            "rrf": "Reciprocal Rank Fusion of visual and text rankings, k=60",
            "learned_proj": "ridge text->visual map (384->768, lambda=1) fit on calibration tracks only, then add",
        },
        "test": {
            "retrieval": summaries,
            "pairs": pair_results,
            "temporal_halves_pair_auc": temporal_halves,
        },
        "significance_vs_concat": {
            "retrieval": significance,
            "pairs": pair_significance,
        },
        "gated_head_to_head": gated_head_to_head,
        "grouped_bootstrap_by_track": grouped,
        "kfold_by_track": kfold,
        "visual_formation_sensitivity": vis_sensitivity,
        "visual_track_meta": vis_track_meta,
        "provenance": {
            "public_reid_probe": provenance,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": f"python backend/scripts/eval_fusion.py --seeds {args.seeds} --n-boot {args.n_boot}",
        },
        "limitations": [
            "Single camera (camera_0); cross-camera identity and generalisation are untestable.",
            "Pseudo-track labels, not human identity annotations; no IAA.",
            "Metadata-only formation removes the visual-cosine bias by construction, but colour/attribute matching has its own failure modes (colour changes, sparse attributes); reported as pseudo-labels, not identities.",
            "Pair thresholds are tuned on calibration pairs; test pairs are track-disjoint but come from the same two-hour window.",
            "The held-out temporal split is early/late halves of the same two-hour window, not an independent deployment; no public Re-ID source resolved offline.",
            "The paper's 312k-descriptor / 4-camera 0.785->0.807 fusion result is not reproduced and is not replaced by these numbers.",
            "The learned projection is a linear ridge baseline, not a jointly trained Re-ID model.",
            "Quality estimators (occlusion proxy, attribute richness) are unvalidated proxies from released metadata, not measured image quality.",
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
        "deterministic pseudo-labels from detector/attribute metadata (no visual cosine), "
        "not human identities. Threshold-free pair AUC is the primary metric.\n"
    )
    lines.append(
        f"- Person descriptors: {len(dets)}\n"
        f"- Pseudo-tracks (metadata-only formation): {track_meta['n_tracks']} "
        f"({track_meta['n_test_tracks']} test / {track_meta['n_calibration_tracks']} calibration)\n"
        f"- Test retrieval queries: {len(query_indices)}; mean gallery: "
        f"{results['dataset']['retrieval_protocol']['mean_gallery_per_query']:.0f}\n"
        f"- Same-track positives in the retrieval pool: {n_pos}\n"
        f"- Different-track negatives in the retrieval pool: {n_neg}\n"
        f"- Pairs (metadata-formed): {len(pair_rows)} pairs "
        f"(test {pair_counts[('test', True)]}+ / {pair_counts[('test', False)]}-; "
        f"calibration {pair_counts[('calibration', True)]}+ / {pair_counts[('calibration', False)]}-)\n"
        f"- Track rule: same camera + same colour + attribute-phrase Jaccard>=0.5, "
        f"max gap 3 s, max duration 60 s, max track size 20 (fixed rule, no embedding threshold).\n"
        f"- Quality estimators: {quality_info['qv_definition']}; "
        f"{quality_info['qt_definition']} "
        f"(mean qv {quality_info['mean_qv']:.3f}, mean qt {quality_info['mean_qt']:.3f}).\n"
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
    lines.append("\n## K-fold over pseudo-tracks (P@5, k=5)\n")
    lines.append(
        "Whole-track 5-fold CV (round-robin over tracks sorted by start time; "
        "gallery restricted to the held-out fold). Randproj uses seed 0 only; "
        "the 5-seed spread is in the main table above.\n"
    )
    lines.append("| Method | Mean | SD | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ["visual", "text", "concat", "gated", "zero_pad", "norm_matched",
                 "learned_proj", "randproj_s0", "rrf"]:
        s = kfold[name]
        lines.append(
            f"| {name} | {s['mean']:.3f} | {s['sd']:.3f} | {s['min']:.3f} | "
            f"{s['max']:.3f} |")
    lines.append("\n## Gated vs static vs RRF head-to-head (Holm over 2 per task)\n")
    lines.append("| Task | Comparison | Δ [95% CI] | raw p | Holm p | reject |")
    lines.append("|---|---|---:|---:|---:|---|")
    for e in gated_head_to_head["retrieval_p5"]:
        lines.append(
            f"| retrieval P@5 | gated vs {e['other']} | {e['delta_p5']:+.3f} "
            f"[{e['lo']:+.3f}, {e['hi']:+.3f}] | {format_p(e['p'], args.n_boot)} | "
            f"{format_p(e['holm_adjusted_p'], args.n_boot)} | {e['holm_reject']} |"
        )
    for e in gated_head_to_head["pair_auc"]:
        lines.append(
            f"| pair AUC | gated vs {e['other']} | {e['delta_auc']:+.3f} "
            f"[{e['lo']:+.3f}, {e['hi']:+.3f}] | {format_p(e['p'], args.n_boot)} | "
            f"{format_p(e['holm_adjusted_p'], args.n_boot)} | {e['holm_reject']} |"
        )
    lines.append("\n## Held-out temporal split (pair AUC by early/late test half)\n")
    lines.append(
        "Test pairs halved by gallery-timestamp median; same calibration "
        "thresholds as the main table. Point estimates (CIs are in the main "
        "pair-AUC table).\n"
    )
    lines.append("| Half | " + " | ".join(m for m in METHODS if m != "randproj") + " |")
    lines.append("|---|" + "---:|" * (len(METHODS) - 1))
    for half in ("early", "late"):
        h = temporal_halves["halves"][half]
        lines.append(
            f"| {half} (n={h['n_pairs']}) | "
            + " | ".join(f"{h[m]:.3f}" for m in METHODS if m != "randproj") + " |"
        )
    rp_cal_auc = pair_results["randproj"]["test"]["auc"]
    lines.append(
        f"\nrandproj seed-mean test AUC: {rp_cal_auc:.3f} "
        f"(± {format_sd(pair_results['randproj']['test_sd']['auc'])})."
    )
    lines.append("\n## Visual-formation sensitivity (point estimates, biased labelling)\n")
    lines.append(
        "Same methods evaluated on visual-cosine pseudo-tracks "
        f"({vis_track_meta['n_tracks']} tracks, {vis_sensitivity['n_queries']} queries). "
        "The visual branch is favoured by construction here; these numbers exist "
        "only to show the bias direction versus the metadata-formed primary table.\n"
    )
    lines.append("| Method | Retrieval P@5 | Pair AUC |")
    lines.append("|---|---:|---:|")
    for m in METHODS:
        p5 = vis_sensitivity["retrieval_p5"].get(m)
        auc = vis_sensitivity["pair_auc"].get(m)
        lines.append(
            f"| {m} | {p5:.3f} " if p5 is not None else f"| {m} | n/a "
        )
        lines[-1] += f"| {auc:.3f} |" if auc is not None else "| n/a |"
    lines.append("\n## Provenance (public Re-ID probe)\n")
    lines.append(f"- Probe resolved: `{provenance['resolved']}`.")
    for c in provenance["checked"]:
        lines.append(
            f"  - `{c['path']}`: exists={c['exists']}, "
            f"market_layout={c['has_market_layout']}"
        )
    lines.append(f"- {provenance['note']}")
    lines.append("\n## What cannot be reproduced\n")
    for item in results["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    (RESULTS / "fusion_tables.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        f"tracks={track_meta['n_tracks']} formation=metadata-only "
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
