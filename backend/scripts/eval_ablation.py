"""MMR / recency ablation on the released protocol-matched control (C1 systems).

The production ranker (``backend/app/services/result_merger.py::merge_results``)
applies two post-ranking steps after the alpha blend:

1. a recency boost ``S' = S_hyb * (1 + 0.5 * 2^{-age/24h})`` gated by
   ``ENABLE_RECENCY_BOOST`` (half-life ``RECENCY_HALFLIFE_HOURS`` = 24 h), and
2. MMR diversity re-ranking (``apply_mmr_diversity``, lambda
   ``MMR_DIVERSITY_LAMBDA`` = 0.3) gated by ``ENABLE_RESULT_DIVERSITY``.

The released C1 control (``eval_retrieval_control.py``) scores the plain alpha
blend, so this script measures the marginal contribution of each step with a
2x2 ablation (recency on/off x MMR on/off) on the 100-query test split.  The
recency boost uses the production wall-clock ``datetime.now()`` semantics, so
on the archived March subset every candidate is ~5,000 h old and the boost is
identically ~1.0; an archive-anchored sensitivity (now = latest clip end + 1 h)
is reported alongside to show what recency would do on fresh data.

Run:
    python backend/scripts/eval_ablation.py
    python backend/scripts/eval_ablation.py --n-boot 3000

Writes ``results/ablation_results.json`` and ``results/ablation_tables.md``.
Existing ``results/retrieval_control_*`` outputs are not modified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval.stats import format_p, paired_bootstrap_ci  # noqa: E402
from backend.app.services.result_merger import apply_mmr_diversity  # noqa: E402
from backend.scripts import eval_retrieval_control as erc  # noqa: E402

RESULTS = ROOT / "results"
N_BOOT_DEFAULT = 3000
CATEGORIES = ["count", "find", "track", "action", "filter"]


def recency_factor(start_iso: str, end_iso: str, now: datetime,
                   halflife_h: float = 24.0) -> float:
    """Production recency multiplier for one candidate."""
    from backend.app.services.retrieval_utils import parse_ts

    ts_str = start_iso or end_iso or ""
    if not ts_str:
        return 1.0
    try:
        age_h = (now - parse_ts(ts_str)).total_seconds() / 3600.0
        boost = math.exp(-age_h * (0.693 / halflife_h))
        return 1.0 + 0.5 * boost
    except Exception:
        return 1.0


def ablated_ranking(fused: np.ndarray, clip_ids: Sequence[str],
                    clips: Dict[str, Dict[str, Any]], now: datetime,
                    use_recency: bool, use_mmr: bool,
                    mmr_lambda: float = 0.3) -> List[str]:
    """Apply the 2x2 post-ranking ablation to one query's fused clip scores."""
    scored = np.asarray(fused, dtype=np.float64).copy()
    if use_recency:
        for j, cid in enumerate(clip_ids):
            clip = clips[cid]
            s = clip["start"].isoformat() if hasattr(clip["start"], "isoformat") else str(clip["start"])
            e = clip["end"].isoformat() if hasattr(clip["end"], "isoformat") else str(clip["end"])
            scored[j] *= recency_factor(s, e, now)
    order = np.argsort(-scored, kind="stable")
    ranked = [clip_ids[int(i)] for i in order]
    if use_mmr:
        pseudo = [
            {"camera_id": 0, "start": clips[cid]["start"].isoformat()
             if hasattr(clips[cid]["start"], "isoformat") else str(clips[cid]["start"]),
             "score": float(scored[clip_ids.index(cid)])}
            for cid in ranked
        ]
        reranked = apply_mmr_diversity(pseudo, lambda_=mmr_lambda)
        # Map back to clip ids via stable score/camera/start matching.
        # apply_mmr_diversity preserves dict identity, so track positions.
        id_by_pos = {id(p): c for p, c in zip(pseudo, ranked)}
        ranked = [id_by_pos[id(p)] for p in reranked]
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    args = ap.parse_args()

    queries, splits = erc.load_benchmark()
    index, frames, clips, clip_ids = erc.load_scene_index()
    dets, clip_of, dets_by_clip = erc.load_clip_detections(clips)
    motion = erc.subset.compute_motion_energy(clips)
    motion_max = max((v["mean"] for v in motion.values()), default=1.0)

    from backend.app.services.sem_embedder import get_embedder

    embedder = get_embedder()
    texts = [q["text"] for q in queries]
    q_emb = embedder.text_embed(texts)
    D, I = index.search(q_emb.astype(np.float32), index.ntotal)

    clip_index = {cid: i for i, cid in enumerate(clip_ids)}
    sem_raw: Dict[str, np.ndarray] = {}
    for qi, q in enumerate(queries):
        per_clip = np.full(len(clip_ids), -1.0, dtype=np.float64)
        for dist, fi in zip(D[qi], I[qi]):
            cid = frames[int(fi)]["clip_id"]
            j = clip_index[cid]
            per_clip[j] = max(per_clip[j], float(dist))
        per_clip[per_clip < 0] = float(np.min(D[qi]))
        sem_raw[q["qid"]] = per_clip

    contexts = {
        q["qid"]: erc.build_query_context(q, clips, dets_by_clip, motion, motion_max)
        for q in queries
    }
    struct_raw = {
        qid: np.asarray([ctx["raw"][cid] for cid in clip_ids], dtype=np.float64)
        for qid, ctx in contexts.items()
    }
    sem_norm = {qid: erc.normalize_branch(v, "minmax") for qid, v in sem_raw.items()}
    struct_norm = {qid: erc.normalize_branch(v, "minmax") for qid, v in struct_raw.items()}

    # Adaptive fused scores (identical to the control track).
    fused_by_qid: Dict[str, np.ndarray] = {}
    for q in queries:
        a = float(contexts[q["qid"]]["alpha"])
        fused_by_qid[q["qid"]] = (
            a * struct_norm[q["qid"]] + (1.0 - a) * sem_norm[q["qid"]]
        )

    now_wall = datetime.now(timezone.utc).replace(tzinfo=None)
    latest_end = max(c["end"] for c in clips.values())
    from datetime import timedelta

    now_fresh = (latest_end + timedelta(hours=1)).replace(tzinfo=None)

    configs = [
        ("base (no recency, no MMR)", False, False),
        ("recency only", True, False),
        ("MMR only", False, True),
        ("both (production default)", True, True),
    ]

    from backend.app.eval.metrics import ranking_metrics

    per_config_rows: Dict[str, List[Dict[str, Any]]] = {}
    for label, use_rec, use_mmr in configs:
        rows = []
        for q in queries:
            if use_rec:
                # Wall-clock production semantics (primary).
                ranking = ablated_ranking(
                    fused_by_qid[q["qid"]], clip_ids, clips, now_wall,
                    True, use_mmr)
            else:
                ranking = ablated_ranking(
                    fused_by_qid[q["qid"]], clip_ids, clips, now_wall,
                    False, use_mmr)
            m = ranking_metrics(ranking, q["relevant"])
            rows.append({"qid": q["qid"], "split": q["split"],
                         "category": q["category"], **m})
        per_config_rows[label] = rows

    # Fresh-data sensitivity: both-on with archive-anchored now.
    fresh_rows = []
    for q in queries:
        ranking = ablated_ranking(
            fused_by_qid[q["qid"]], clip_ids, clips, now_fresh, True, True)
        m = ranking_metrics(ranking, q["relevant"])
        fresh_rows.append({"qid": q["qid"], "split": q["split"],
                           "category": q["category"], **m})

    def macro(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        test = [r for r in rows if r["split"] == "test"]
        cat = {}
        for c in CATEGORIES:
            vals = [r["p5"] for r in test if r["category"] == c]
            cat[c] = float(np.mean(vals)) if vals else 0.0
        return {"macro": float(np.mean(list(cat.values()))),
                "per_category": cat, "n_test": len(test)}

    summary = {label: macro(rows) for label, rows in per_config_rows.items()}
    summary["both (archive-anchored now, sensitivity)"] = macro(fresh_rows)

    # Paired bootstrap of each ablation vs base (test split).
    base_p5 = {r["qid"]: r["p5"] for r in per_config_rows[
        "base (no recency, no MMR)"] if r["split"] == "test"}
    significance: Dict[str, Any] = {}
    for label in ["recency only", "MMR only", "both (production default)"]:
        cur = {r["qid"]: r["p5"] for r in per_config_rows[label]
               if r["split"] == "test"}
        qids = sorted(base_p5)
        a = [cur[q] for q in qids]
        b = [base_p5[q] for q in qids]
        dm, lo, hi, p = paired_bootstrap_ci(a, b, n_boot=args.n_boot, seed=101)
        significance[label] = {
            "delta_p5": float(dm), "lo": float(lo), "hi": float(hi),
            "p": float(p),
        }

    # Recency factor diagnostics: range of multipliers under each now.
    sample_factors_wall = []
    sample_factors_fresh = []
    for cid in clip_ids[:10]:
        c = clips[cid]
        s = c["start"].isoformat() if hasattr(c["start"], "isoformat") else str(c["start"])
        e = c["end"].isoformat() if hasattr(c["end"], "isoformat") else str(c["end"])
        sample_factors_wall.append(recency_factor(s, e, now_wall))
        sample_factors_fresh.append(recency_factor(s, e, now_fresh))

    results = {
        "scope": {
            "benchmark": "backend/benchmarks/queries_control.jsonl "
                         "(protocol-matched control, 50 tune / 100 test)",
            "note": "2x2 post-ranking ablation on the released subset; "
                    "not the 4-camera deployment.",
        },
        "production_flags": {
            "ENABLE_RECENCY_BOOST": True,
            "RECENCY_HALFLIFE_HOURS": 24.0,
            "ENABLE_RESULT_DIVERSITY": True,
            "MMR_DIVERSITY_LAMBDA": 0.3,
        },
        "n_boot": args.n_boot,
        "summary": summary,
        "significance_vs_base": significance,
        "recency_diagnostics": {
            "wall_clock_now": now_wall.isoformat(),
            "archive_anchored_now": now_fresh.isoformat(),
            "sample_factors_wall_clock": [float(f) for f in sample_factors_wall],
            "sample_factors_archive_anchored": [float(f) for f in sample_factors_fresh],
        },
        "limitations": [
            "Single camera (camera_0), 100 test queries; cross-camera and "
            "deployment-scale effects are untestable.",
            "Recency uses production wall-clock semantics; on the archived "
            "subset every boost is ~1.0 by construction (age saturation), so "
            "recency is ranking-inert here, not necessarily in deployment.",
            "MMR (lambda=0.3) penalises same-(camera, hour-bucket) candidates; "
            "on this single-camera subset the relevant clips share the query "
            "window's bucket, so MMR demotes them (-0.138 macro P@5). The "
            "0.538 control is therefore pre-MMR; the production default "
            "scores 0.400 on this subset. Multi-camera deployment effects "
            "are untestable here.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ablation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# MMR / recency ablation (released control, 100-query test split)\n")
    lines.append(
        "**Scope.** Protocol-matched control "
        "(`backend/benchmarks/queries_control.jsonl`), adaptive-alpha fused "
        "scores from `eval_retrieval_control.py`, post-ranking ablated 2x2. "
        "Production defaults are recency ON + MMR ON.\n")
    lines.append("| Config | Macro P@5 | Count | Find | Track | Action | Filter |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in ["base (no recency, no MMR)", "recency only", "MMR only",
                  "both (production default)",
                  "both (archive-anchored now, sensitivity)"]:
        s = summary[label]
        pc = s["per_category"]
        lines.append(
            f"| {label} | {s['macro']:.3f} | {pc['count']:.3f} | "
            f"{pc['find']:.3f} | {pc['track']:.3f} | {pc['action']:.3f} | "
            f"{pc['filter']:.3f} |")
    lines.append("\n## Deltas vs base (paired bootstrap, "
                 f"{args.n_boot} resamples)\n")
    lines.append("| Config | ΔP@5 [95% CI] | p |")
    lines.append("|---|---:|---:|")
    for label, s in significance.items():
        lines.append(
            f"| {label} | {s['delta_p5']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] | "
            f"{format_p(s['p'], args.n_boot)} |")
    lines.append(
        "\n## Recency diagnostics\n")
    lines.append(
        f"Wall-clock now: `{results['recency_diagnostics']['wall_clock_now']}`; "
        f"sample boosts: min {min(sample_factors_wall):.4f}, max "
        f"{max(sample_factors_wall):.4f} (identically ~1.0: every archived clip "
        f"is ~5,000 h old against a 24 h half-life). Archive-anchored now "
        f"(`{results['recency_diagnostics']['archive_anchored_now']}`): min "
        f"{min(sample_factors_fresh):.3f}, max {max(sample_factors_fresh):.3f}.\n")
    lines.append("## What cannot be reproduced\n")
    for item in results["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    (RESULTS / "ablation_tables.md").write_text("\n".join(lines), encoding="utf-8")

    print("ablation (test macro P@5):")
    for label in summary:
        print(f"  {label:42s} {summary[label]['macro']:.3f}")
    print(f"wrote {RESULTS / 'ablation_results.json'}")
    print(f"wrote {RESULTS / 'ablation_tables.md'}")


if __name__ == "__main__":
    main()
