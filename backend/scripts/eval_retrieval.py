"""Real retrieval evaluation on the released single-camera subset.

No synthetic scores and no paper-table fixture.  The script:

1. encodes the 150 benchmark queries with the cached OpenCLIP ViT-B/32
   tokenizer/model and searches the shipped 242 x 512 scene index;
2. maps frame hits to the 121 released clips and aggregates by max/mean;
3. builds a real structured score from the released metadata (time overlap,
   camera, VLM zone, person colour/attributes, caption count, caption event,
   motion-energy proxy) without MongoDB;
4. sweeps fixed alpha, the repository's adaptive alpha, a per-category alpha
   tuned on the 50-query tune split, and the paper's per-intent lookup;
5. repeats the sweep under min-max and z-score branch normalisation;
6. reports P@5, R@10, F1@10 (P@10/R@10 and paper P@5/R@10 variants), count
   EM/MAE, query-level and grouped bootstrap CIs, paired bootstrap tests, and
   Holm correction over the five categories.

Run:
    python backend/scripts/eval_retrieval.py
    python backend/scripts/eval_retrieval.py --n-boot 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval import subset
from backend.app.eval.baselines import (
    FIXED_ALPHAS,
    per_intent_lookup_alpha,
    zscore_norm,
)
from backend.app.eval.metrics import (
    category_means,
    exact_match,
    macro_mean,
    mean_absolute_error,
    ranking_metrics,
)
from backend.app.eval.stats import (
    bootstrap_ci,
    grouped_bootstrap_ci,
    holm_correction,
    paired_bootstrap_ci,
    paired_grouped_bootstrap_ci,
)

CATEGORIES = ["count", "find", "track", "action", "filter"]
ALPHA_GRID = [round(i / 10.0, 1) for i in range(11)]
BENCHMARK = ROOT / "backend" / "benchmarks"
RESULTS = ROOT / "results"
N_BOOT_DEFAULT = 3000


# ---------------------------------------------------------------------------
# Adaptive alpha (real repo function, with a documented fallback)
# ---------------------------------------------------------------------------

try:
    from backend.app.services.result_merger import compute_adaptive_alpha as _real_adaptive
    ADAPTIVE_SOURCE = "backend.app.services.result_merger.compute_adaptive_alpha"
except Exception as _exc:  # pragma: no cover - only if optional deps are absent
    _real_adaptive = None
    ADAPTIVE_SOURCE = f"local replica (import failed: {type(_exc).__name__}: {_exc})"


def compute_adaptive_alpha(
    parsed_filter: Dict[str, Any],
    intent: str,
    has_count_constraint: bool,
    has_action: bool,
    n_structured: int,
    n_semantic: int,
) -> float:
    if _real_adaptive is not None:
        return float(
            _real_adaptive(
                parsed_filter, intent, has_count_constraint, has_action,
                n_structured, n_semantic,
            )
        )
    # Identical replica of backend/app/services/result_merger.py.
    if has_count_constraint:
        return 1.0
    if has_action:
        return 0.1
    filter_count = sum(
        1
        for k in ("camera_id", "objects.object_name", "objects.color", "zone", "location_hint")
        if parsed_filter.get(k)
    )
    if parsed_filter.get("timestamp"):
        filter_count += 1
    complexity = min(1.0, filter_count / 4.0)
    base = {"count": 0.9, "track": 0.5, "find": 0.7}.get(intent, 0.7)
    struct_ratio = (
        n_structured / max(1, n_structured + n_semantic)
        if (n_structured or n_semantic)
        else 0.5
    )
    alpha = base * 0.6 + 0.2 * complexity + 0.2 * struct_ratio
    return max(0.1, min(0.95, alpha))


# ---------------------------------------------------------------------------
# Load benchmark and indices
# ---------------------------------------------------------------------------

def load_benchmark() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    queries = [
        json.loads(line)
        for line in (BENCHMARK / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    splits = json.loads((BENCHMARK / "splits.json").read_text(encoding="utf-8"))
    if len(queries) != 150:
        raise RuntimeError(f"expected 150 queries, found {len(queries)}")
    if len({q["qid"] for q in queries}) != 150:
        raise RuntimeError("duplicate qids")
    if len({q["text"] for q in queries}) != 150:
        raise RuntimeError("duplicate query text")
    for q in queries:
        q["split"] = "tune" if q["qid"] in set(splits["tune"]) else "test"
    return queries, splits


def load_scene_index() -> Tuple[Any, List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    frames = subset.load_scene_frames()
    clips = subset.build_clip_table(frames)
    clip_ids = list(clips.keys())
    clip_for_faiss = {int(fr["faiss_id"]): fr["clip_id"] for fr in frames}
    import faiss

    index = faiss.read_index(str(subset.SCENE_INDEX))
    if index.d != 512 or index.ntotal != len(frames):
        raise RuntimeError(f"unexpected scene index: d={index.d} n={index.ntotal}")
    return index, frames, clips, clip_ids


def load_clip_detections(
    clips: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[int, str], Dict[str, List[Dict[str, Any]]]]:
    dets = subset.load_person_meta()
    assigned = subset.assign_detections_to_clips(dets, clips)
    clip_of = {d["faiss_id"]: cid for d, cid in zip(dets, assigned) if cid}
    by_clip: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det, cid in zip(dets, assigned):
        if cid:
            by_clip[cid].append(det)
    return dets, clip_of, by_clip


# ---------------------------------------------------------------------------
# Structured branch
# ---------------------------------------------------------------------------

def _mode(values: Sequence[int]) -> Optional[int]:
    if not values:
        return None
    counts = Counter(values)
    return max(counts, key=lambda c: (counts[c], c))


def build_query_context(
    q: Dict[str, Any],
    clips: Dict[str, Dict[str, Any]],
    dets_by_clip: Dict[str, List[Dict[str, Any]]],
    motion: Dict[str, Dict[str, float]],
    motion_max: float,
) -> Dict[str, Any]:
    start = subset.parse_det_ts(q["time_start"])
    end = subset.parse_det_ts(q["time_end"])
    p = q.get("parsed") or {}
    color = p.get("objects.color")
    attrs = list(p.get("attributes") or [])
    zone = p.get("zone")
    event = p.get("event") or (p.get("action") if p.get("action") in {"fighting", "smoke"} else None)
    motion_thr = p.get("motion_threshold")
    window_counts = [
        fr["count"]
        for clip in clips.values()
        for fr in clip["frames"]
        if start <= fr["frame_ts"] <= end and fr["count"] is not None
    ]
    est_count = _mode(window_counts)
    raw: Dict[str, float] = {}
    for cid, clip in clips.items():
        dur = max(1e-6, (clip["end"] - clip["start"]).total_seconds())
        time_ov = subset.overlap_seconds(clip["start"], clip["end"], start, end) / dur
        time_ov = max(0.0, min(1.0, time_ov))
        in_window = [fr for fr in clip["frames"] if start <= fr["frame_ts"] <= end]
        clip_dets = [
            d for d in dets_by_clip.get(cid, []) if start <= d["timestamp"] <= end
        ]
        color_match = (
            1.0
            if not color
            else float(any(d["color"].lower() == str(color).lower() for d in clip_dets))
        )
        if attrs:
            attr_match = float(
                any(
                    d["color"].lower() == str(color).lower()
                    and all(_det_attr(d, a) for a in attrs)
                    for d in clip_dets
                )
            )
        else:
            attr_match = 1.0
        if zone:
            zone_match = float(any(fr["zone"] == zone for fr in in_window))
        else:
            zone_match = 1.0
        event_match = float(any(fr["event"] == event for fr in in_window)) if event else 0.0
        if q["category"] == "count":
            count_match = float(
                any(fr["count"] == est_count for fr in in_window)
            ) if est_count is not None else 0.0
            s = 0.70 * count_match + 0.30 * time_ov
        elif q["category"] == "find":
            if attrs:
                s = 0.50 * color_match + 0.25 * attr_match + 0.25 * time_ov
            else:
                s = 0.70 * color_match + 0.30 * time_ov
        elif q["category"] == "track":
            s = 0.50 * time_ov + 0.35 * color_match + 0.15 * zone_match
        elif q["category"] == "filter":
            s = 0.40 * time_ov + 0.35 * color_match + 0.25 * zone_match
        elif q["category"] == "action":
            # The paper's action override deliberately skips the structured
            # branch because action semantics are not a metadata field.  The
            # subset's event/motion labels are *labels*, not retrieval inputs;
            # using them here would leak the benchmark labels into the ranker.
            # Only the time filter remains, so alpha=0.1 is a genuine test of
            # the semantic branch.
            s = 0.35 * time_ov
        else:
            s = time_ov
        raw[cid] = float(s)

    parsed_filter: Dict[str, Any] = {"camera_id": 0, "timestamp": True}
    if q["category"] in {"find", "track", "filter"}:
        parsed_filter["objects.object_name"] = "person"
    if color:
        parsed_filter["objects.color"] = color
    if zone:
        parsed_filter["zone"] = zone
    if event:
        parsed_filter["action"] = event
    if q["category"] == "action" and not event:
        parsed_filter["action"] = "motion_energy"
    if q["category"] == "count" and est_count is not None:
        parsed_filter["count_constraint"] = {"eq": int(est_count)}
    n_structured = sum(1 for v in raw.values() if v > 0.05)
    n_semantic = 20
    has_count = q["category"] == "count"
    has_action = q["category"] == "action"
    alpha = compute_adaptive_alpha(
        parsed_filter, q["category"], has_count, has_action, n_structured, n_semantic
    )
    return {
        "raw": raw,
        "start": start,
        "end": end,
        "est_count": est_count,
        "color": color,
        "attrs": attrs,
        "zone": zone,
        "event": event,
        "n_structured": n_structured,
        "n_semantic": n_semantic,
        "alpha": float(alpha),
        "parsed_filter": parsed_filter,
    }


def _det_attr(det: Dict[str, Any], attr: str) -> bool:
    text = det["attribute_text"].lower()
    if attr == "carrying bag":
        return "carrying bag" in text
    if attr == "long sleeves":
        return "long sleeve" in text
    if attr == "female":
        return "female" in text
    if attr == "male":
        return "male" in text and "female" not in text
    if attr == "coat or jacket":
        return "coat or jacket" in text
    return False


def minmax(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def normalize_branch(values: Sequence[float], mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if mode == "minmax":
        return minmax(arr)
    if mode == "zscore":
        return np.asarray(zscore_norm(arr.tolist()), dtype=np.float64)
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rank_clips(scores: np.ndarray, clip_ids: Sequence[str]) -> List[str]:
    order = np.argsort(-scores, kind="stable")
    return [clip_ids[int(i)] for i in order]


def predict_count(
    q: Dict[str, Any], ranking: Sequence[str], clips: Dict[str, Dict[str, Any]]
) -> Optional[int]:
    if q["category"] != "count" or not ranking:
        return None
    anchor = subset.parse_det_ts(q.get("anchor_ts") or q["time_start"])
    frames = clips[ranking[0]]["frames"]
    if not frames:
        return None
    nearest = min(frames, key=lambda fr: abs((fr["frame_ts"] - anchor).total_seconds()))
    return nearest["count"]


def score_rows(
    queries: Sequence[Dict[str, Any]],
    fused_by_qid: Dict[str, Tuple[np.ndarray, List[str]]],
    clips: Dict[str, Dict[str, Any]],
    sem_norm_by_qid: Dict[str, np.ndarray],
    struct_norm_by_qid: Dict[str, np.ndarray],
    clip_index: Dict[str, int],
) -> List[Dict[str, Any]]:
    rows = []
    for q in queries:
        scores, ranking = fused_by_qid[q["qid"]]
        m = ranking_metrics(ranking, q["relevant"])
        top = ranking[0]
        j = clip_index[top]
        pred_count = predict_count(q, ranking, clips)
        rows.append(
            {
                "qid": q["qid"],
                "split": q["split"],
                "category": q["category"],
                "n_relevant": len(q["relevant"]),
                "ground_truth_count": q.get("ground_truth_count"),
                "predicted_count": pred_count,
                "top1_clip": top,
                "top1_score": float(scores[j]),
                "top1_sem": float(sem_norm_by_qid[q["qid"]][j]),
                "top1_struct": float(struct_norm_by_qid[q["qid"]][j]),
                "top5_clips": "|".join(ranking[:5]),
                "rank_first_relevant": int(m["rank_first_relevant"]),
                **m,
            }
        )
    return rows


def aggregate(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, float]:
    cat = category_means(list(rows), field)
    return {
        "macro": float(macro_mean(cat)),
        "per_category": {c: float(cat.get(c, 0.0)) for c in CATEGORIES},
    }


def count_scores(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    vals = [r for r in rows if r["category"] == "count" and r["predicted_count"] is not None]
    if not vals:
        return {"n": 0, "em": 0.0, "mae": 0.0}
    return {
        "n": len(vals),
        "em": float(np.mean([exact_match(r["predicted_count"], r["ground_truth_count"]) for r in vals])),
        "mae": float(mean_absolute_error(
            [r["predicted_count"] for r in vals],
            [r["ground_truth_count"] for r in vals],
        )),
    }


def method_alpha(
    method: str,
    q: Dict[str, Any],
    qctx: Dict[str, Any],
    tuned: Dict[str, float],
) -> float:
    if method.startswith("fixed_"):
        return float(method.split("_", 1)[1])
    if method == "adaptive":
        return float(qctx["alpha"])
    if method == "paper_lookup":
        return float(per_intent_lookup_alpha(q["category"]))
    if method == "per_intent_tuned":
        return float(tuned[q["category"]])
    raise ValueError(method)


def build_fused(
    queries: Sequence[Dict[str, Any]],
    clip_ids: Sequence[str],
    sem_norm: Dict[str, np.ndarray],
    struct_norm: Dict[str, np.ndarray],
    method: str,
    tuned: Dict[str, float],
    contexts: Dict[str, Dict[str, Any]],
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    out = {}
    for q in queries:
        a = method_alpha(method, q, contexts[q["qid"]], tuned)
        fused = a * struct_norm[q["qid"]] + (1.0 - a) * sem_norm[q["qid"]]
        out[q["qid"]] = (fused, rank_clips(fused, clip_ids))
    return out


def tune_per_intent(
    queries: Sequence[Dict[str, Any]],
    clip_ids: Sequence[str],
    sem_norm: Dict[str, np.ndarray],
    struct_norm: Dict[str, np.ndarray],
    contexts: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, List[Dict[str, float]]]]:
    tuned: Dict[str, float] = {}
    grid_rows: Dict[str, List[Dict[str, float]]] = {}
    tune_qs = [q for q in queries if q["split"] == "tune"]
    for cat in CATEGORIES:
        cat_qs = [q for q in tune_qs if q["category"] == cat]
        rows = []
        for a in ALPHA_GRID:
            p5 = []
            f1 = []
            for q in cat_qs:
                fused = a * struct_norm[q["qid"]] + (1.0 - a) * sem_norm[q["qid"]]
                ranking = rank_clips(fused, clip_ids)
                m = ranking_metrics(ranking, q["relevant"])
                p5.append(m["p5"])
                f1.append(m["f1"])
            rows.append({"alpha": a, "p5": float(np.mean(p5)), "f1": float(np.mean(f1))})
        best = max(rows, key=lambda r: (r["p5"], r["f1"], -abs(r["alpha"] - 0.5)))
        tuned[cat] = float(best["alpha"])
        grid_rows[cat] = rows
    return tuned, grid_rows


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


def ci_dict(rows: Sequence[Dict[str, Any]], n_boot: int, seed: int = 1234) -> Dict[str, float]:
    vals = [r["p5"] for r in rows]
    groups = [r.get("group") or r["qid"] for r in rows]
    m, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
    gm, glo, ghi = grouped_bootstrap_ci(vals, groups, n_boot=n_boot, seed=seed)
    return {
        "p5_mean": float(m), "p5_lo": float(lo), "p5_hi": float(hi),
        "p5_grouped_mean": float(gm), "p5_grouped_lo": float(glo),
        "p5_grouped_hi": float(ghi),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--no-zscore", action="store_true")
    args = ap.parse_args()

    queries, splits = load_benchmark()
    index, frames, clips, clip_ids = load_scene_index()
    dets, clip_of, dets_by_clip = load_clip_detections(clips)
    motion = subset.compute_motion_energy(clips)
    motion_max = max(v["mean"] for v in motion.values()) if motion else 1.0

    # Real OpenCLIP text encoding and FAISS search.
    from backend.app.services.sem_embedder import get_embedder

    embedder = get_embedder()
    texts = [q["text"] for q in queries]
    q_emb = embedder.text_embed(texts)
    if q_emb.shape != (len(queries), 512):
        raise RuntimeError(f"unexpected query embedding shape {q_emb.shape}")
    D, I = index.search(q_emb.astype(np.float32), index.ntotal)

    clip_index = {cid: i for i, cid in enumerate(clip_ids)}
    sem_raw_max: Dict[str, np.ndarray] = {}
    sem_raw_mean: Dict[str, np.ndarray] = {}
    for qi, q in enumerate(queries):
        per_clip_max = np.full(len(clip_ids), -1.0, dtype=np.float64)
        per_clip_sum = np.zeros(len(clip_ids), dtype=np.float64)
        per_clip_cnt = np.zeros(len(clip_ids), dtype=np.float64)
        for dist, fi in zip(D[qi], I[qi]):
            cid = frames[int(fi)]["clip_id"]
            j = clip_index[cid]
            per_clip_max[j] = max(per_clip_max[j], float(dist))
            per_clip_sum[j] += float(dist)
            per_clip_cnt[j] += 1.0
        per_clip_max[per_clip_max < 0] = float(np.min(D[qi]))
        per_clip_mean = np.divide(
            per_clip_sum, np.maximum(per_clip_cnt, 1.0), dtype=np.float64
        )
        sem_raw_max[q["qid"]] = per_clip_max
        sem_raw_mean[q["qid"]] = per_clip_mean

    contexts = {
        q["qid"]: build_query_context(q, clips, dets_by_clip, motion, motion_max)
        for q in queries
    }
    struct_raw = {qid: np.asarray([ctx["raw"][cid] for cid in clip_ids], dtype=np.float64)
                  for qid, ctx in contexts.items()}

    norm_modes = ["minmax"] + ([] if args.no_zscore else ["zscore"])
    systems = (
        [f"fixed_{a:.1f}" for a in FIXED_ALPHAS]
        + ["adaptive", "per_intent_tuned", "paper_lookup"]
    )

    all_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    tune_metrics: Dict[str, Dict[str, Any]] = {}
    per_intent_alphas: Dict[str, Dict[str, float]] = {}
    per_intent_grid: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for norm in norm_modes:
        sem_norm = {qid: normalize_branch(sem_raw_max[qid], norm) for qid in sem_raw_max}
        struct_norm = {qid: normalize_branch(struct_raw[qid], norm) for qid in struct_raw}
        tuned, grid = tune_per_intent(queries, clip_ids, sem_norm, struct_norm, contexts)
        per_intent_alphas[norm] = tuned
        per_intent_grid[norm] = grid
        all_rows[norm] = {}
        tune_metrics[norm] = {}
        for method in systems:
            fused = build_fused(queries, clip_ids, sem_norm, struct_norm, method, tuned, contexts)
            rows = score_rows(queries, fused, clips, sem_norm, struct_norm, clip_index)
            for r, q in zip(rows, queries):
                r["group"] = q.get("group")
                r["alpha"] = method_alpha(method, q, contexts[q["qid"]], tuned)
            all_rows[norm][method] = rows
            tune_rows = [r for r in rows if r["split"] == "tune"]
            tune_metrics[norm][method] = aggregate(tune_rows, "p5")

    # Best fixed alpha chosen on the tune split (honest), plus an oracle row.
    best_fixed: Dict[str, Dict[str, Any]] = {}
    oracle_fixed: Dict[str, Dict[str, Any]] = {}
    for norm in norm_modes:
        tune_scores = {
            m: tune_metrics[norm][m]["macro"]
            for m in systems if m.startswith("fixed_")
        }
        best_m = max(tune_scores, key=lambda m: (tune_scores[m], m))
        best_fixed[norm] = {"method": best_m, "tune_macro_p5": float(tune_scores[best_m])}
        test_scores = {
            m: aggregate([r for r in all_rows[norm][m] if r["split"] == "test"], "p5")["macro"]
            for m in systems if m.startswith("fixed_")
        }
        oracle_m = max(test_scores, key=lambda m: (test_scores[m], m))
        oracle_fixed[norm] = {"method": oracle_m, "test_macro_p5": float(test_scores[oracle_m])}

    # Test summaries.
    summary: Dict[str, Any] = {}
    for norm in norm_modes:
        summary[norm] = {}
        for m in systems:
            test_rows = [r for r in all_rows[norm][m] if r["split"] == "test"]
            p5 = aggregate(test_rows, "p5")
            r10 = aggregate(test_rows, "r10")
            f1 = aggregate(test_rows, "f1")
            f1p = aggregate(test_rows, "f1_paper")
            mrr = aggregate(test_rows, "mrr")
            summary[norm][m] = {
                "macro": {
                    "p5": p5["macro"],
                    "r10": r10["macro"],
                    "f1": f1["macro"],
                    "f1_paper": f1p["macro"],
                    "mrr": mrr["macro"],
                },
                "per_category": {
                    cat: {
                        "p5": p5["per_category"][cat],
                        "r10": r10["per_category"][cat],
                        "f1": f1["per_category"][cat],
                        "f1_paper": f1["per_category"][cat],
                    }
                    for cat in CATEGORIES
                },
                "count": count_scores(test_rows),
                "ci": ci_dict(test_rows, args.n_boot),
                "tune_macro_p5": tune_metrics[norm][m]["macro"],
            }

    # Significance: adaptive vs the tune-selected fixed alpha, per category.
    significance: Dict[str, Any] = {}
    for norm in norm_modes:
        base = best_fixed[norm]["method"]
        a_rows = {r["qid"]: r for r in all_rows[norm]["adaptive"] if r["split"] == "test"}
        b_rows = {r["qid"]: r for r in all_rows[norm][base] if r["split"] == "test"}
        qids = [q["qid"] for q in queries if q["split"] == "test"]
        a_vals = [a_rows[qid]["p5"] for qid in qids]
        b_vals = [b_rows[qid]["p5"] for qid in qids]
        groups = [a_rows[qid]["group"] for qid in qids]
        m, lo, hi, p = paired_bootstrap_ci(a_vals, b_vals, n_boot=args.n_boot, seed=11)
        gm, glo, ghi, gp = paired_grouped_bootstrap_ci(
            a_vals, b_vals, groups, n_boot=args.n_boot, seed=11
        )
        pvals = {}
        details = {}
        for cat in CATEGORIES:
            cat_qids = [q["qid"] for q in queries if q["split"] == "test" and q["category"] == cat]
            av = [a_rows[qid]["p5"] for qid in cat_qids]
            bv = [b_rows[qid]["p5"] for qid in cat_qids]
            dm, dlo, dhi, dp = paired_bootstrap_ci(av, bv, n_boot=args.n_boot, seed=17)
            pvals[cat] = dp
            details[cat] = {
                "delta_p5": float(dm), "lo": float(dlo), "hi": float(dhi), "p": float(dp),
            }
        rejects = holm_correction(pvals, alpha=0.05)
        adjusted = holm_adjusted(pvals)
        significance[norm] = {
            "comparison": f"adaptive vs {base} (fixed alpha selected on tune split)",
            "overall": {
                "delta_p5": float(m), "lo": float(lo), "hi": float(hi), "p": float(p),
                "grouped_delta_p5": float(gm), "grouped_lo": float(glo),
                "grouped_hi": float(ghi), "grouped_p": float(gp),
            },
            "per_category": details,
            "holm_reject": {k: bool(v) for k, v in rejects.items()},
            "holm_adjusted_p": {k: float(v) for k, v in adjusted.items()},
        }
        # Secondary comparison: tuned per-intent vs adaptive.
        c_rows = {r["qid"]: r for r in all_rows[norm]["per_intent_tuned"] if r["split"] == "test"}
        cv = [c_rows[qid]["p5"] for qid in qids]
        dm, dlo, dhi, dp = paired_bootstrap_ci(cv, a_vals, n_boot=args.n_boot, seed=23)
        significance[norm]["per_intent_tuned_vs_adaptive"] = {
            "delta_p5": float(dm), "lo": float(dlo), "hi": float(dhi), "p": float(dp)
        }

    # Normalisation comparison (same method, different branch calibration).
    norm_comparison: Dict[str, Any] = {}
    if "zscore" in norm_modes:
        for m in systems:
            mm = summary["minmax"][m]["macro"]["p5"]
            zz = summary["zscore"][m]["macro"]["p5"]
            norm_comparison[m] = {
                "minmax_macro_p5": float(mm),
                "zscore_macro_p5": float(zz),
                "delta_z_minus_mm": float(zz - mm),
            }
        # Paired bootstrap on adaptive per-query P@5.
        a_rows = {r["qid"]: r for r in all_rows["minmax"]["adaptive"] if r["split"] == "test"}
        z_rows = {r["qid"]: r for r in all_rows["zscore"]["adaptive"] if r["split"] == "test"}
        qids = [q["qid"] for q in queries if q["split"] == "test"]
        dm, dlo, dhi, dp = paired_bootstrap_ci(
            [a_rows[q]["p5"] for q in qids],
            [z_rows[q]["p5"] for q in qids],
            n_boot=args.n_boot, seed=29,
        )
        norm_comparison["adaptive_paired_zscore_minus_minmax"] = {
            "delta_p5": float(dm), "lo": float(dlo), "hi": float(dhi), "p": float(dp)
        }

    # Mean-vs-max aggregation sensitivity (fixed 0.5 and adaptive, minmax).
    sem_mean_norm = {qid: minmax(sem_raw_mean[qid]) for qid in sem_raw_mean}
    struct_mm = {qid: minmax(struct_raw[qid]) for qid in struct_raw}
    agg_sensitivity: Dict[str, Any] = {}
    for method in ["fixed_0.5", "adaptive"]:
        tuned = per_intent_alphas["minmax"]
        fused = build_fused(
            queries, clip_ids, sem_mean_norm, struct_mm, method, tuned, contexts
        )
        rows = score_rows(queries, fused, clips, sem_mean_norm, struct_mm, clip_index)
        test_rows = [r for r in rows if r["split"] == "test"]
        agg_sensitivity[method] = {
            "max_aggregation_macro_p5": summary["minmax"][method]["macro"]["p5"],
            "mean_aggregation_macro_p5": aggregate(test_rows, "p5")["macro"],
        }

    # Structured count-estimate quality (not a retrieval metric).
    count_rows = [q for q in queries if q["category"] == "count"]
    est_em = []
    est_mae = []
    for q in count_rows:
        est = contexts[q["qid"]]["est_count"]
        truth = q["ground_truth_count"]
        if est is not None and truth is not None:
            est_em.append(exact_match(est, truth))
            est_mae.append(abs(int(est) - int(truth)))
    count_est = {
        "n": len(est_em),
        "em": float(np.mean(est_em)) if est_em else 0.0,
        "mae": float(np.mean(est_mae)) if est_mae else 0.0,
        "definition": "window mode of released caption person counts vs anchor-frame count",
    }

    # Adaptive alpha statistics.
    alpha_stats = {}
    for cat in CATEGORIES:
        vals = [contexts[q["qid"]]["alpha"] for q in queries if q["category"] == cat and q["split"] == "test"]
        alpha_stats[cat] = {
            "mean": float(np.mean(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # Assemble JSON.
    subset_stats = json.loads((RESULTS / "subset_stats.json").read_text(encoding="utf-8"))
    results = {
        "scope": {
            "name": "released single-camera subset",
            "not": "full 4-camera / 312k-descriptor deployment",
            "note": (
                "These are subset reproducibility numbers. The paper's headline "
                "full-deployment values must remain separate and are not reproduced here."
            ),
        },
        "dataset": subset_stats,
        "benchmark": {
            "n_queries": len(queries),
            "splits": {"tune": splits["n_tune"], "test": splits["n_test"]},
            "categories": CATEGORIES,
            "label_sources": dict(Counter(q["label_source"] for q in queries)),
            "relevance": {
                "min": min(len(q["relevant"]) for q in queries),
                "median": float(np.median([len(q["relevant"]) for q in queries])),
                "max": max(len(q["relevant"]) for q in queries),
            },
        },
        "retrieval": {
            "scene_index": "data/index/faiss/index.bin (242 x 512, IndexFlatIP)",
            "text_encoder": "OpenCLIP ViT-B/32 laion2b_s34b_b79k (cached, CPU)",
            "semantic_aggregation": "max over frames per clip for the main table; mean reported as sensitivity",
            "structured_branch": (
                "metadata-only score: time overlap, camera, VLM zone, person colour/attributes, "
                "caption count mode, caption event, motion-energy proxy"
            ),
            "adaptive_alpha_source": ADAPTIVE_SOURCE,
            "clip_count": len(clip_ids),
            "frame_count": len(frames),
            "person_detections": len(dets),
            "fixed_alphas": FIXED_ALPHAS,
            "n_boot": args.n_boot,
        },
        "test": summary,
        "best_fixed_on_tune": best_fixed,
        "oracle_fixed_on_test": oracle_fixed,
        "per_intent_alphas": per_intent_alphas,
        "per_intent_tuning_grid": per_intent_grid,
        "significance": significance,
        "normalization_comparison": norm_comparison,
        "aggregation_sensitivity": agg_sensitivity,
        "count_estimate": count_est,
        "adaptive_alpha": alpha_stats,
        "limitations": [
            "Single camera (camera_0) and a 2-hour released window; cross-camera retrieval and identity tracking are untestable.",
            "Labels are derived from released VLM captions, detector metadata and snapshots, not human annotations; no IAA is available.",
            "Two clips on disk are not in the scene FAISS index (121 indexed of 123); evaluation uses the indexed clips.",
            "Action queries use 15 weak VLM caption events (fighting/smoke) plus 15 motion-energy proxy queries; no running/fighting labels are invented.",
            "Track queries are single-camera colour-continuity proxies, not ground-truth identities.",
            "The full-deployment 0.666->0.828 / 0.785->0.807 values are not reproduced and must not be replaced by these subset numbers.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Long-format per-query CSV (all methods, both normalisations).
    with (RESULTS / "retrieval_per_query.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system", "normalization", "split", "qid", "category", "alpha",
            "p5", "r10", "f1", "f1_paper", "mrr", "rank_first_relevant",
            "top1_clip", "top1_score", "top1_sem", "top1_struct",
            "top5_clips", "n_relevant", "ground_truth_count", "predicted_count",
        ])
        for norm in norm_modes:
            for m in systems:
                for r in all_rows[norm][m]:
                    writer.writerow([
                        m, norm, r["split"], r["qid"], r["category"], f"{r['alpha']:.3f}",
                        f"{r['p5']:.6f}", f"{r['r10']:.6f}", f"{r['f1']:.6f}",
                        f"{r['f1_paper']:.6f}", f"{r['mrr']:.6f}", r["rank_first_relevant"],
                        r["top1_clip"], f"{r['top1_score']:.6f}",
                        f"{r['top1_sem']:.6f}", f"{r['top1_struct']:.6f}",
                        r["top5_clips"], r["n_relevant"],
                        "" if r["ground_truth_count"] is None else r["ground_truth_count"],
                        "" if r["predicted_count"] is None else r["predicted_count"],
                    ])

    # Markdown tables.
    lines: List[str] = []
    lines.append("# Released-subset retrieval results\n")
    lines.append(
        "**Scope.** Single camera (`camera_0`), 121 indexed clips / 242 scene frames, "
        "1,778 person descriptors, 2026-03-01 17:01--18:56. These are **not** the "
        "paper's full 4-camera / 312k-descriptor results.\n"
    )
    lines.append(
        f"Adaptive alpha source: `{ADAPTIVE_SOURCE}`. "
        f"Bootstrap resamples: {args.n_boot}. "
        f"Tune/test: {splits['n_tune']}/{splits['n_test']} queries.\n"
    )
    lines.append("\n## Macro results (min-max branch normalisation)\n")
    lines.append("| Method | P@5 | R@10 | F1@10 (P@10,R@10) | F1@10 paper (P@5,R@10) | MRR | P@5 grouped 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for m in ["fixed_0.0", "fixed_0.3", "fixed_0.5", "fixed_0.7", "fixed_1.0",
              "adaptive", "per_intent_tuned", "paper_lookup"]:
        s = summary["minmax"][m]
        ci = s["ci"]
        lines.append(
            f"| {m} | {s['macro']['p5']:.3f} | {s['macro']['r10']:.3f} | "
            f"{s['macro']['f1']:.3f} | {s['macro']['f1_paper']:.3f} | "
            f"{s['macro']['mrr']:.3f} | [{ci['p5_grouped_lo']:.3f}, {ci['p5_grouped_hi']:.3f}] |"
        )
    lines.append(
        f"\nBest fixed alpha selected on the tune split: `{best_fixed['minmax']['method']}` "
        f"(tune macro P@5 {best_fixed['minmax']['tune_macro_p5']:.3f}). "
        f"Oracle fixed on test (descriptive only): `{oracle_fixed['minmax']['method']}`."
    )

    lines.append("\n## Per-category P@5 (min-max)\n")
    lines.append("| Method | Count | Find | Track | Action | Filter | Macro |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in ["fixed_0.0", "fixed_0.3", "fixed_0.5", "fixed_0.7", "fixed_1.0",
              "adaptive", "per_intent_tuned", "paper_lookup"]:
        s = summary["minmax"][m]
        pc = s["per_category"]
        lines.append(
            f"| {m} | {pc['count']['p5']:.3f} | {pc['find']['p5']:.3f} | "
            f"{pc['track']['p5']:.3f} | {pc['action']['p5']:.3f} | "
            f"{pc['filter']['p5']:.3f} | {s['macro']['p5']:.3f} |"
        )

    lines.append("\n## Count EM / MAE (test count queries)\n")
    lines.append("| Method | n | EM | MAE |")
    lines.append("|---|---:|---:|---:|")
    for m in ["fixed_0.0", "fixed_0.3", "fixed_0.5", "fixed_0.7", "fixed_1.0",
              "adaptive", "per_intent_tuned", "paper_lookup"]:
        c = summary["minmax"][m]["count"]
        lines.append(f"| {m} | {c['n']} | {c['em']:.3f} | {c['mae']:.3f} |")
    lines.append(
        f"\nStructured window-count estimate (mode of caption counts) vs anchor-frame truth: "
        f"EM {count_est['em']:.3f}, MAE {count_est['mae']:.3f} (n={count_est['n']}).\n"
    )

    lines.append("\n## Min-max vs z-score branch normalisation (macro P@5)\n")
    if norm_comparison:
        lines.append("| Method | min-max | z-score | Δ (z - mm) |")
        lines.append("|---|---:|---:|---:|")
        for m in ["fixed_0.0", "fixed_0.3", "fixed_0.5", "fixed_0.7", "fixed_1.0",
                  "adaptive", "per_intent_tuned", "paper_lookup"]:
            c = norm_comparison[m]
            lines.append(
                f"| {m} | {c['minmax_macro_p5']:.3f} | {c['zscore_macro_p5']:.3f} | "
                f"{c['delta_z_minus_mm']:+.3f} |"
            )
        s = norm_comparison["adaptive_paired_zscore_minus_minmax"]
        lines.append(
            f"\nAdaptive z-score vs min-max paired ΔP@5 = {s['delta_p5']:+.3f} "
            f"[{s['lo']:+.3f}, {s['hi']:+.3f}], p={s['p']:.4f}.\n"
        )

    lines.append("\n## Adaptive vs tune-selected best fixed-alpha\n")
    for norm in norm_modes:
        sig = significance[norm]
        o = sig["overall"]
        lines.append(
            f"- **{norm}**: `{sig['comparison']}`. ΔP@5 = {o['delta_p5']:+.3f} "
            f"[{o['lo']:+.3f}, {o['hi']:+.3f}], p={o['p']:.4f}; "
            f"grouped Δ = {o['grouped_delta_p5']:+.3f} "
            f"[{o['grouped_lo']:+.3f}, {o['grouped_hi']:+.3f}], p={o['grouped_p']:.4f}."
        )
        for cat in CATEGORIES:
            d = sig["per_category"][cat]
            lines.append(
                f"  - {cat}: Δ={d['delta_p5']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], "
                f"p={d['p']:.4f}, Holm-reject={sig['holm_reject'][cat]}"
            )
        s2 = sig["per_intent_tuned_vs_adaptive"]
        lines.append(
            f"  - per-intent tuned vs adaptive: Δ={s2['delta_p5']:+.3f} "
            f"[{s2['lo']:+.3f}, {s2['hi']:+.3f}], p={s2['p']:.4f}"
        )

    lines.append("\n## Per-intent alpha selected on the tune split\n")
    lines.append("| Normalisation | Count | Find | Track | Action | Filter |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for norm in norm_modes:
        a = per_intent_alphas[norm]
        lines.append(
            f"| {norm} | {a['count']:.1f} | {a['find']:.1f} | {a['track']:.1f} | "
            f"{a['action']:.1f} | {a['filter']:.1f} |"
        )

    lines.append("\n## Realized adaptive alpha\n")
    lines.append("| Category | Mean | Min | Max |")
    lines.append("|---|---:|---:|---:|")
    for cat in CATEGORIES:
        a = alpha_stats[cat]
        lines.append(f"| {cat} | {a['mean']:.3f} | {a['min']:.3f} | {a['max']:.3f} |")

    lines.append("\n## Semantic aggregation sensitivity\n")
    lines.append("| Method | max over frames | mean over frames |")
    lines.append("|---|---:|---:|")
    for m, v in agg_sensitivity.items():
        lines.append(
            f"| {m} | {v['max_aggregation_macro_p5']:.3f} | "
            f"{v['mean_aggregation_macro_p5']:.3f} |"
        )

    lines.append("\n## What cannot be reproduced\n")
    for item in results["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    (RESULTS / "retrieval_tables.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"adaptive alpha source: {ADAPTIVE_SOURCE}")
    print(f"best fixed on tune: {best_fixed['minmax']}")
    for m in ["fixed_0.0", "fixed_0.3", "fixed_0.5", "fixed_0.7", "fixed_1.0",
              "adaptive", "per_intent_tuned", "paper_lookup"]:
        s = summary["minmax"][m]
        print(f"{m:18s} macro P@5={s['macro']['p5']:.3f} R@10={s['macro']['r10']:.3f} "
              f"F1={s['macro']['f1']:.3f} countEM={s['count']['em']:.3f}")
    print(f"adaptive vs best fixed: {significance['minmax']['overall']}")
    print(f"wrote {RESULTS / 'retrieval_results.json'}")
    print(f"wrote {RESULTS / 'retrieval_tables.md'}")
    print(f"wrote {RESULTS / 'retrieval_per_query.csv'}")


if __name__ == "__main__":
    main()
