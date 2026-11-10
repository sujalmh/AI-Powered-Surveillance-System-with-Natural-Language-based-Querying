"""Segment-disjoint (leak-free) retrieval evaluation on the released subset.

Reuses helpers from ``backend/scripts/eval_retrieval_control.py`` (no
copy-paste-modify) and reuses the OpenCLIP/FAISS retrieval pipeline from it.
Adds a temporal-segment split (three bursts, gap > 120 s), a leakage audit of
the shipped tune/test split, and a single held-out-burst evaluation (forward
chaining: train on the two earlier bursts, score on the held-out burst) with
paired bootstrap CIs and paired tests for:

  * adaptive (metadata prior, additive blend) vs the tune-selected best fixed
    alpha, and
  * combMAX (max operator, prior alpha) vs the prior (additive) blend.

Split construction is clip-disjoint by construction: a query is assigned to the
held-out burst only when *all* of its relevant clips lie in that burst; training
queries are those whose relevant clips avoid the held-out burst entirely.  Cross-
burst queries are reported and excluded from both sides where they would leak.

Outputs (never overwriting existing results):
    results/segment_disjoint_results.json
    results/segment_disjoint_tables.md

Run:
    python backend/scripts/eval_retrieval_segment.py
    python backend/scripts/eval_retrieval_segment.py --n-boot 3000 --held-out 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval import subset
from backend.app.eval import metrics as _metrics
from backend.scripts import eval_retrieval_control as ctrl

RESULTS = ROOT / "results"
BENCHMARK = ROOT / "backend" / "benchmarks"
CATEGORIES = ctrl.CATEGORIES
N_BOOT_DEFAULT = 3000


def fuse_max(a: float, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.maximum(a * s, (1.0 - a) * e)


# ---------------------------------------------------------------------------
# Segments + leakage audit
# ---------------------------------------------------------------------------


def clip_segments(clips: Dict[str, Any]) -> Tuple[List[List[str]], Dict[str, int]]:
    segs: List[List[str]] = []
    cur: List[str] = []
    prev = None
    for cid, c in clips.items():
        if prev is None or (c["start"] - prev).total_seconds() > 120:
            if cur:
                segs.append(cur)
            cur = [cid]
        else:
            cur.append(cid)
        prev = c["end"]
    if cur:
        segs.append(cur)
    seg_of: Dict[str, int] = {}
    for i, s in enumerate(segs):
        for cid in s:
            seg_of[cid] = i
    return segs, seg_of


def relevant_segments(q: Dict[str, Any], seg_of: Dict[str, int]) -> set:
    return {seg_of[c] for c in q["relevant"] if c in seg_of}


def leakage_audit(queries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tune_clips: set = set()
    test_clips: set = set()
    for q in queries:
        if q["split"] == "tune":
            for c in q["relevant"]:
                tune_clips.add(c)
        else:
            for c in q["relevant"]:
                test_clips.add(c)
    shared = tune_clips & test_clips
    test_touch = sum(
        1 for q in queries
        if q["split"] == "test" and any(c in tune_clips for c in q["relevant"])
    )
    tune_touch = sum(
        1 for q in queries
        if q["split"] == "tune" and any(c in test_clips for c in q["relevant"])
    )
    return {
        "n_tune_queries": sum(1 for q in queries if q["split"] == "tune"),
        "n_test_queries": sum(1 for q in queries if q["split"] == "test"),
        "n_tune_relevant_clips": len(tune_clips),
        "n_test_relevant_clips": len(test_clips),
        "n_shared_clips": len(shared),
        "shared_pct_of_test_relevant": (
            float(len(shared) / len(test_clips)) if test_clips else 0.0
        ),
        "n_test_queries_touching_a_tune_clip": int(test_touch),
        "n_test_queries_total": sum(1 for q in queries if q["split"] == "test"),
        "n_tune_queries_touching_a_test_clip": int(tune_touch),
        "n_tune_queries_total": sum(1 for q in queries if q["split"] == "tune"),
    }


# ---------------------------------------------------------------------------
# Tune helpers (segment-scoped)
# ---------------------------------------------------------------------------


def tune_fixed_single(
    tune_qs: Sequence[Dict[str, Any]],
    clip_ids: Sequence[str],
    sem_norm: Dict[str, np.ndarray],
    struct_norm: Dict[str, np.ndarray],
    contexts: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, float]]:
    tune_macro: Dict[str, float] = {}
    for a in ctrl.ALPHA_GRID:
        p5s = []
        for q in tune_qs:
            fused = a * struct_norm[q["qid"]] + (1.0 - a) * sem_norm[q["qid"]]
            ranking = ctrl.rank_clips(fused, clip_ids)
            m = _metrics.ranking_metrics(ranking, q["relevant"])
            p5s.append(m["p5"])
        tune_macro[f"{a:.1f}"] = float(np.mean(p5s)) if p5s else 0.0
    best_key = max(
        tune_macro,
        key=lambda k: (tune_macro[k], -abs(float(k) - 0.5), -float(k)),
    )
    return f"fixed_{best_key}", tune_macro


def tune_per_intent_seg(
    tune_qs: Sequence[Dict[str, Any]],
    clip_ids: Sequence[str],
    sem_norm: Dict[str, np.ndarray],
    struct_norm: Dict[str, np.ndarray],
    contexts: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    tuned: Dict[str, float] = {}
    present_cats = sorted({q["category"] for q in tune_qs})
    for cat in present_cats:
        cat_qs = [q for q in tune_qs if q["category"] == cat]
        rows = []
        for a in ctrl.ALPHA_GRID:
            p5 = []
            f1 = []
            for q in cat_qs:
                fused = a * struct_norm[q["qid"]] + (1.0 - a) * sem_norm[q["qid"]]
                ranking = ctrl.rank_clips(fused, clip_ids)
                m = _metrics.ranking_metrics(ranking, q["relevant"])
                p5.append(m["p5"])
                f1.append(m["f1"])
            rows.append({"alpha": a, "p5": float(np.mean(p5)) if p5 else 0.0,
                         "f1": float(np.mean(f1)) if f1 else 0.0})
        best = max(rows, key=lambda r: (r["p5"], r["f1"], -abs(r["alpha"] - 0.5)))
        tuned[cat] = float(best["alpha"])
    for c in CATEGORIES:
        tuned.setdefault(c, 0.5)
    return tuned


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_method(
    method: str,
    test_qs: Sequence[Dict[str, Any]],
    clip_ids: Sequence[str],
    sem_norm: Dict[str, np.ndarray],
    struct_norm: Dict[str, np.ndarray],
    contexts: Dict[str, Dict[str, Any]],
    tuned: Dict[str, float],
    alpha_prior: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for q in test_qs:
        qid = q["qid"]
        s = struct_norm[qid]
        e = sem_norm[qid]
        a_prior = float(alpha_prior[qid])
        if method.startswith("fixed_"):
            a = float(method.split("_", 1)[1])
        elif method == "adaptive":
            a = float(contexts[qid].get("alpha", a_prior))
        elif method == "combMAX":
            a = a_prior
        elif method == "per_intent_tuned":
            a = float(tuned[q["category"]])
        else:
            a = a_prior
        if method == "combMAX":
            fused = fuse_max(a, s, e)
        else:
            fused = a * s + (1.0 - a) * e
        ranking = ctrl.rank_clips(fused, clip_ids)
        m = _metrics.ranking_metrics(ranking, q["relevant"])
        rows[qid] = {
            "qid": qid,
            "category": q["category"],
            "group": q.get("group") or qid,
            "p5": float(m["p5"]),
            "r10": float(m["r10"]),
            "f1": float(m["f1"]),
            "f1_paper": float(m["f1_paper"]),
            "mrr": float(m["mrr"]),
            "alpha": float(a),
        }
    return rows


def compute_macro_p5(rows: Dict[str, Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    vals = [{"category": r["category"], "v": r["p5"]} for r in rows.values()]
    cm = _metrics.category_means(vals, "v")
    return float(_metrics.macro_mean(cm)), {c: float(cm.get(c, 0.0)) for c in CATEGORIES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--held-out", type=int, default=2,
                    help="segment index to hold out (default 2 = S3, forward chaining)")
    ap.add_argument("--no-zscore", action="store_true")
    args = ap.parse_args()

    queries, splits = ctrl.load_benchmark()
    index, frames, clips, clip_ids = ctrl.load_scene_index()
    dets, clip_of, dets_by_clip = ctrl.load_clip_detections(clips)
    motion = subset.compute_motion_energy(clips)
    motion_max = max(v["mean"] for v in motion.values()) if motion else 1.0

    # Leakage audit of the shipped split.
    leak = leakage_audit(queries)

    # Segments.
    segs, seg_of = clip_segments(clips)
    seg_names = ["S1 (17:01-17:08)", "S2 (18:04-18:28)", "S3 (18:55-18:56)"]
    while len(seg_names) < len(segs):
        seg_names.append(f"S{len(seg_names)+1}")

    # Segment of each query by its relevant clips.
    segset_of_query: Dict[str, set] = {}
    cross_segment_qids: List[str] = []
    for q in queries:
        ss = relevant_segments(q, seg_of)
        segset_of_query[q["qid"]] = ss
        if len(ss) > 1:
            cross_segment_qids.append(q["qid"])

    seg_query_counts = Counter()
    for q in queries:
        # representative segment: if single, that; else -1 for cross
        ss = segset_of_query[q["qid"]]
        if len(ss) == 1:
            seg_query_counts[next(iter(ss))] += 1

    seg_clip_counts = {i: len(segs[i]) for i in range(len(segs))}

    # Disjointness for chosen held-out.
    h = args.held_out
    train_segs = [i for i in range(len(segs)) if i != h]
    # clip-disjoint: test = queries whose relevant clips are ALL in held-out burst
    test_qs = [q for q in queries if segset_of_query[q["qid"]] == {h}]
    # train = queries whose relevant clips avoid held-out burst entirely
    tune_qs = [q for q in queries if h not in segset_of_query[q["qid"]]]
    # dropped: queries touching held-out burst and at least one other burst
    dropped_qs = [q for q in queries if (h in segset_of_query[q["qid"]]
                 and len(segset_of_query[q["qid"]]) > 1)]

    train_clips = set()
    for i in train_segs:
        train_clips.update(segs[i])
    test_clips = set(segs[h])

    tune_rel = set()
    test_rel = set()
    for q in tune_qs:
        for c in q["relevant"]:
            tune_rel.add(c)
    for q in test_qs:
        for c in q["relevant"]:
            test_rel.add(c)
    relevant_overlap = tune_rel & test_rel

    disjointness = {
        "held_out_segment": h,
        "held_out_segment_name": seg_names[h],
        "train_segments": train_segs,
        "train_segment_names": [seg_names[i] for i in train_segs],
        "n_train_clips": len(train_clips),
        "n_test_clips": len(test_clips),
        "n_shared_clips_train_test": len(train_clips & test_clips),
        "n_tune_queries": len(tune_qs),
        "n_test_queries": len(test_qs),
        "n_cross_segment_queries_total": len(cross_segment_qids),
        "n_queries_dropped_for_disjointness": len(dropped_qs),
        "n_tune_relevant_clips": len(tune_rel),
        "n_test_relevant_clips": len(test_rel),
        "n_shared_relevant_clips": len(relevant_overlap),
        "shared_relevant_pct_of_test": (
            float(len(relevant_overlap) / len(test_rel)) if test_rel else 0.0
        ),
        "clip_disjoint": bool(len(relevant_overlap) == 0 and len(train_clips & test_clips) == 0),
    }

    # OpenCLIP retrieval (reuse control pipeline).
    from backend.app.services.sem_embedder import get_embedder

    embedder = get_embedder()
    q_emb = embedder.text_embed([q["text"] for q in queries])
    if q_emb.shape != (len(queries), 512):
        raise RuntimeError(f"unexpected query embedding shape {q_emb.shape}")
    D, I = index.search(q_emb.astype(np.float32), index.ntotal)

    clip_index = {cid: i for i, cid in enumerate(clip_ids)}
    sem_raw_max: Dict[str, np.ndarray] = {}
    for qi, q in enumerate(queries):
        per_clip_max = np.full(len(clip_ids), -1.0, dtype=np.float64)
        for dist, fi in zip(D[qi], I[qi]):
            cid = frames[int(fi)]["clip_id"]
            j = clip_index[cid]
            per_clip_max[j] = max(per_clip_max[j], float(dist))
        per_clip_max[per_clip_max < 0] = float(np.min(D[qi]))
        sem_raw_max[q["qid"]] = per_clip_max

    contexts = {
        q["qid"]: ctrl.build_query_context(q, clips, dets_by_clip, motion, motion_max)
        for q in queries
    }
    struct_raw = {
        qid: np.asarray([ctx["raw"][cid] for cid in clip_ids], dtype=np.float64)
        for qid, ctx in contexts.items()
    }
    alpha_prior = {qid: float(ctx["alpha"]) for qid, ctx in contexts.items()}

    norm_modes = ["minmax"] + ([] if args.no_zscore else ["zscore"])

    summary: Dict[str, Any] = {}
    all_test_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    tuned_per_norm: Dict[str, Dict[str, float]] = {}
    best_fixed_per_norm: Dict[str, str] = {}
    best_fixed_tune_macro: Dict[str, float] = {}

    for norm in norm_modes:
        sem_norm = {qid: ctrl.normalize_branch(sem_raw_max[qid], norm) for qid in sem_raw_max}
        struct_norm = {qid: ctrl.normalize_branch(struct_raw[qid], norm) for qid in struct_raw}

        best_fixed_method, tune_macro_map = tune_fixed_single(
            tune_qs, clip_ids, sem_norm, struct_norm, contexts
        )
        best_fixed_per_norm[norm] = best_fixed_method
        best_fixed_tune_macro[norm] = float(
            tune_macro_map[best_fixed_method.split("_", 1)[1]]
        )

        tuned = tune_per_intent_seg(tune_qs, clip_ids, sem_norm, struct_norm, contexts)
        tuned_per_norm[norm] = tuned

        methods = ["adaptive", "combMAX", best_fixed_method, "per_intent_tuned"]
        for fa in ("0.0", "0.3", "0.5", "0.7", "1.0"):
            methods.append(f"fixed_{fa}")
        methods = list(dict.fromkeys(methods))

        method_rows: Dict[str, Dict[str, Any]] = {}
        for m in methods:
            rows = eval_method(m, test_qs, clip_ids, sem_norm, struct_norm, contexts, tuned, alpha_prior)
            method_rows[m] = rows
        all_test_rows[norm] = method_rows

        summary[norm] = {}
        for m in methods:
            rows = method_rows[m]
            p5s = [r["p5"] for r in rows.values()]
            r10s = [r["r10"] for r in rows.values()]
            f1s = [r["f1"] for r in rows.values()]
            f1ps = [r["f1_paper"] for r in rows.values()]
            mrrs = [r["mrr"] for r in rows.values()]
            macro_p5, per_cat_p5 = compute_macro_p5(rows)
            cm_r10 = _metrics.category_means(
                [{"category": r["category"], "v": r["r10"]} for r in rows.values()], "v"
            )
            macro_r10 = float(_metrics.macro_mean(cm_r10))
            cm_f1 = _metrics.category_means(
                [{"category": r["category"], "v": r["f1"]} for r in rows.values()], "v"
            )
            macro_f1 = float(_metrics.macro_mean(cm_f1))
            cm_f1p = _metrics.category_means(
                [{"category": r["category"], "v": r["f1_paper"]} for r in rows.values()], "v"
            )
            macro_f1p = float(_metrics.macro_mean(cm_f1p))
            cm_mrr = _metrics.category_means(
                [{"category": r["category"], "v": r["mrr"]} for r in rows.values()], "v"
            )
            macro_mrr = float(_metrics.macro_mean(cm_mrr))
            summary[norm][m] = {
                "macro_p5": macro_p5,
                "macro_r10": macro_r10,
                "macro_f1": macro_f1,
                "macro_f1_paper": macro_f1p,
                "macro_mrr": macro_mrr,
                "per_category_p5": per_cat_p5,
                "n_test": len(rows),
            }

    # Significance.
    significance: Dict[str, Any] = {}
    for norm in norm_modes:
        bf = best_fixed_per_norm[norm]
        rows_a = all_test_rows[norm]["adaptive"]
        rows_b = all_test_rows[norm][bf]
        rows_c = all_test_rows[norm]["combMAX"]
        qids = [q["qid"] for q in test_qs]
        a_vals = [rows_a[qid]["p5"] for qid in qids]
        b_vals = [rows_b[qid]["p5"] for qid in qids]
        c_vals = [rows_c[qid]["p5"] for qid in qids]
        groups = [rows_a[qid]["group"] for qid in qids]
        cats = [rows_a[qid]["category"] for qid in qids]

        dm, dlo, dhi, dp = ctrl.paired_bootstrap_ci(a_vals, b_vals, n_boot=args.n_boot, seed=11)
        gm, glo, ghi, gp = ctrl.paired_grouped_bootstrap_ci(
            a_vals, b_vals, groups, n_boot=args.n_boot, seed=11
        )

        def macro_boot(av, bv, cats, n_boot, seed):
            by_cat = defaultdict(list)
            for c, x, y in zip(cats, av, bv):
                by_cat[c].append((float(x), float(y)))
            cat_names = sorted(by_cat)
            obs = float(np.mean([
                np.mean([x for x, _ in by_cat[c]]) - np.mean([y for _, y in by_cat[c]])
                for c in cat_names
            ]))
            rng = random.Random(seed)
            deltas = []
            for _ in range(n_boot):
                per = []
                for c in cat_names:
                    pool = by_cat[c]
                    n = len(pool)
                    if n == 0:
                        per.append(0.0)
                        continue
                    samp = [pool[rng.randrange(n)] for _ in range(n)]
                    per.append(float(np.mean([x for x, _ in samp]) - np.mean([y for _, y in samp])))
                deltas.append(float(np.mean(per)))
            deltas.sort()
            alpha = 0.05
            lo = deltas[int((alpha / 2) * n_boot)]
            hi = deltas[int((1 - alpha / 2) * n_boot) - 1]
            if obs >= 0:
                pp = sum(1 for d in deltas if d <= 0) / n_boot
            else:
                pp = sum(1 for d in deltas if d >= 0) / n_boot
            return {"delta": obs, "lo": float(lo), "hi": float(hi), "p": float(min(1.0, 2.0 * pp))}

        macro_ab = macro_boot(a_vals, b_vals, cats, args.n_boot, 101)

        dm2, dlo2, dhi2, dp2 = ctrl.paired_bootstrap_ci(c_vals, a_vals, n_boot=args.n_boot, seed=21)
        gm2, glo2, ghi2, gp2 = ctrl.paired_grouped_bootstrap_ci(
            c_vals, a_vals, groups, n_boot=args.n_boot, seed=21
        )
        macro_cm = macro_boot(c_vals, a_vals, cats, args.n_boot, 103)

        significance[norm] = {
            "best_fixed_method": bf,
            "adaptive_vs_best_fixed": {
                "delta_p5_pooled": float(dm),
                "lo_pooled": float(dlo),
                "hi_pooled": float(dhi),
                "p_pooled": float(dp),
                "delta_p5_grouped": float(gm),
                "lo_grouped": float(glo),
                "hi_grouped": float(ghi),
                "p_grouped": float(gp),
                "delta_p5_macro": float(macro_ab["delta"]),
                "lo_macro": float(macro_ab["lo"]),
                "hi_macro": float(macro_ab["hi"]),
                "p_macro": float(macro_ab["p"]),
            },
            "combMAX_vs_prior": {
                "delta_p5_pooled": float(dm2),
                "lo_pooled": float(dlo2),
                "hi_pooled": float(dhi2),
                "p_pooled": float(dp2),
                "delta_p5_grouped": float(gm2),
                "lo_grouped": float(glo2),
                "hi_grouped": float(ghi2),
                "p_grouped": float(gp2),
                "delta_p5_macro": float(macro_cm["delta"]),
                "lo_macro": float(macro_cm["lo"]),
                "hi_macro": float(macro_cm["hi"]),
                "p_macro": float(macro_cm["p"]),
            },
        }

    # Verdict.
    try:
        p_ab = significance["minmax"]["adaptive_vs_best_fixed"]["p_pooled"]
        d_ab = significance["minmax"]["adaptive_vs_best_fixed"]["delta_p5_pooled"]
    except KeyError:
        p_ab = 1.0
        d_ab = 0.0
    try:
        d_cm = significance["minmax"]["combMAX_vs_prior"]["delta_p5_pooled"]
        p_cm = significance["minmax"]["combMAX_vs_prior"]["p_pooled"]
    except KeyError:
        d_cm = 0.0
        p_cm = 1.0

    if p_ab < 0.05 and d_ab > 0:
        verdict_ab = "TIE-BREAKS"
    elif p_ab >= 0.05:
        verdict_ab = "TIE-HOLDS"
    else:
        verdict_ab = "TIE-BREAKS"

    try:
        d_cm_z = significance["zscore"]["combMAX_vs_prior"]["delta_p5_pooled"]
        p_cm_z = significance["zscore"]["combMAX_vs_prior"]["p_pooled"]
    except KeyError:
        d_cm_z = 0.0
        p_cm_z = 1.0
    combmax_survives = bool(d_cm_z > 0 and p_cm_z < 0.05)

    verdict = f"adaptive_vs_best_fixed={verdict_ab}; combMAX_zscore_survives={combmax_survives}"

    segment_summary = {
        "segments": [
            {
                "index": i,
                "name": seg_names[i],
                "n_clips": len(segs[i]),
                "n_queries_single_segment": int(seg_query_counts.get(i, 0)),
            }
            for i in range(len(segs))
        ],
        "cross_segment_queries": len(cross_segment_qids),
    }

    results = {
        "scope": {
            "name": "released single-camera subset, segment-disjoint (leak-free) protocol",
            "not": "full 4-camera / 312k-descriptor deployment",
            "note": (
                "Segment-disjoint (temporal) proxy for camera-disjointness. "
                "Alpha selected on training segments only, scored once on the "
                "held-out segment. Released-subset numbers only."
            ),
        },
        "leakage_audit_shipped_split": leak,
        "segments": segment_summary,
        "disjointness": disjointness,
        "held_out": args.held_out,
        "norm_modes": norm_modes,
        "n_boot": args.n_boot,
        "test_queries_by_category": dict(Counter(q["category"] for q in test_qs)),
        "tune_queries_by_category": dict(Counter(q["category"] for q in tune_qs)),
        "best_fixed_per_norm": best_fixed_per_norm,
        "best_fixed_tune_macro_p5": best_fixed_tune_macro,
        "tuned_per_intent_per_norm": tuned_per_norm,
        "summary": summary,
        "significance": significance,
        "verdict": verdict,
        "verdict_adaptive_vs_best_fixed": verdict_ab,
        "combMAX_zscore_survives": combmax_survives,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "segment_disjoint_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Segment-disjoint (leak-free) retrieval results\n")
    lines.append(
        "**Scope.** Released single-camera subset; temporal-segment proxy for "
        "camera-disjointness. Not the 4-camera / 312k deployment.\n"
    )
    lines.append("## Leakage audit of the shipped tune/test split\n")
    lines.append(f"- tune-relevant clips: {leak['n_tune_relevant_clips']}; "
                 f"test-relevant clips: {leak['n_test_relevant_clips']}; "
                 f"shared: {leak['n_shared_clips']} "
                 f"({leak['shared_pct_of_test_relevant']*100:.1f}% of test-relevant).\n")
    lines.append(f"- test queries touching a tune clip: "
                 f"{leak['n_test_queries_touching_a_tune_clip']}/{leak['n_test_queries_total']}.\n")
    lines.append(f"- tune queries touching a test clip: "
                 f"{leak['n_tune_queries_touching_a_test_clip']}/{leak['n_tune_queries_total']}.\n")
    lines.append("\n## Segments\n")
    lines.append("| Segment | Clips | Single-segment queries |")
    lines.append("|---|---:|---:|")
    for seg in segment_summary["segments"]:
        lines.append(f"| {seg['name']} | {seg['n_clips']} | {seg['n_queries_single_segment']} |")
    lines.append(f"\nCross-segment (multi-burst) queries: {segment_summary['cross_segment_queries']}.\n")
    lines.append("\n## Disjointness of this split\n")
    for k, v in disjointness.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"\nHeld-out segment: {seg_names[h]} (index {h}). "
                 f"Train queries: {len(tune_qs)}, test queries: {len(test_qs)}.\n")
    lines.append("\n## Macro P@5 on held-out segment\n")
    for norm in norm_modes:
        lines.append(f"\n### {norm}\n")
        lines.append("| Method | Macro P@5 | R@10 | F1@10 | MRR |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in summary[norm]:
            s = summary[norm][m]
            lines.append(
                f"| {m} | {s['macro_p5']:.4f} | {s['macro_r10']:.4f} | "
                f"{s['macro_f1_paper']:.4f} | {s['macro_mrr']:.4f} |"
            )
        lines.append(f"\nBest fixed alpha selected on tune: `{best_fixed_per_norm[norm]}` "
                     f"(tune macro P@5 {best_fixed_tune_macro[norm]:.4f}).\n")
    lines.append("\n## Adaptive vs best-fixed\n")
    for norm in norm_modes:
        sig = significance[norm]["adaptive_vs_best_fixed"]
        lines.append(
            f"- **{norm}**: ΔP@5 (pooled) = {sig['delta_p5_pooled']:+.4f} "
            f"[{sig['lo_pooled']:+.4f}, {sig['hi_pooled']:+.4f}], "
            f"p={sig['p_pooled']:.4g}; grouped Δ = {sig['delta_p5_grouped']:+.4f} "
            f"[{sig['lo_grouped']:+.4f}, {sig['hi_grouped']:+.4f}], "
            f"p={sig['p_grouped']:.4g}; macro-boot Δ = {sig['delta_p5_macro']:+.4f} "
            f"[{sig['lo_macro']:+.4f}, {sig['hi_macro']:+.4f}], p={sig['p_macro']:.4g}."
        )
    lines.append("\n## combMAX vs prior (additive)\n")
    for norm in norm_modes:
        sig = significance[norm]["combMAX_vs_prior"]
        lines.append(
            f"- **{norm}**: ΔP@5 (pooled) = {sig['delta_p5_pooled']:+.4f} "
            f"[{sig['lo_pooled']:+.4f}, {sig['hi_pooled']:+.4f}], "
            f"p={sig['p_pooled']:.4g}; grouped Δ = {sig['delta_p5_grouped']:+.4f} "
            f"[{sig['lo_grouped']:+.4f}, {sig['hi_grouped']:+.4f}], "
            f"p={sig['p_grouped']:.4g}; macro-boot Δ = {sig['delta_p5_macro']:+.4f} "
            f"[{sig['lo_macro']:+.4f}, {sig['hi_macro']:+.4f}], p={sig['p_macro']:.4g}."
        )
    lines.append(f"\n## Verdict\n{verdict}\n")
    lines.append(f"- adaptive vs best-fixed: **{verdict_ab}**.\n")
    lines.append(f"- combMAX z-score win survives: **{combmax_survives}**.\n")

    out_md = RESULTS / "segment_disjoint_tables.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"leakage: {leak}")
    print(f"segments: {segment_summary}")
    print(f"disjointness: {disjointness}")
    print(f"best_fixed: {best_fixed_per_norm}")
    for norm in norm_modes:
        print(f"[{norm}] adaptive vs best-fixed: {significance[norm]['adaptive_vs_best_fixed']}")
        print(f"[{norm}] combMAX vs prior: {significance[norm]['combMAX_vs_prior']}")
    print(f"verdict: {verdict}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
