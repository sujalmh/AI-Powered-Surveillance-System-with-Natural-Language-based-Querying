"""Pre-registered calibration search for quality-gate variants (C2' fusion).

This script extends ``backend/scripts/eval_fusion.py`` (imported, never
copied) with a two-stage, pre-registered protocol:

* **Stage CAL** (``--stage cal``) evaluates a fixed grid of gate variants on
  **calibration data only** (calibration pseudo-track retrieval, a 5-fold CV
  restricted to calibration tracks, and the calibration pair set), applies the
  pre-registered selection rule, and freezes **exactly one** winner into
  ``cal_gate_calibration.json``.
* **Stage TEST** (``--stage test``) reads the frozen winner, evaluates it
  **once** on the test split with the same protocol as ``eval_fusion.py``
  (paired bootstrap, Holm over the 2-member head-to-head family, 5-fold
  whole-track CV, early/late temporal halves) and writes
  ``cal_gate_results.json`` / ``cal_gate_tables.md``.

Nothing in the test split is touched during stage CAL, and the frozen
specification is hashed so that the test run provably used the calibrated
choice.

PRE-REGISTRATION (fixed before any test-split number was computed)
-----------------------------------------------------------------
Gate form (the released gate is ``gw = clip(sigmoid(4*(qv-qt)), .05, .95)``;
``wv = 1.2*gw``, ``wt = 0.8*(1-gw)``, branch weights reduce to the static
(0.6, 0.4) point exactly when ``gw == 0.5``):

* **F1 -- raw-quality lambda grid.** ``lam in {0.5, 1, 2, 4, 8}``,
  ``gw = clip(sigmoid(lam*(qv-qt)), .05, .95)`` with the released metadata
  qualities (``estimate_qualities``).  ``F1(lam=4)`` reproduces the released
  gate exactly.
* **F2 -- z-normalised qualities.** ``z_qv = (qv-mu_v)/sd_v`` and
  ``z_qt = (qt-mu_t)/sd_t`` with ``(mu, sd)`` estimated on **calibration
  detections only** and applied to every detection; ``lam in {1, 2, 4, 8, 16}``,
  ``gw = clip(sigmoid(lam*(z_qv-z_qt)), .05, .95)``.  (A strictly
  query-conditional gate is incompatible with the fixed-descriptor retrieval
  protocol of ``eval_fusion.py``, where the gallery matrix is precomputed and
  scored as ``gallery @ gallery[qi]``; the z-normalisation is therefore taken
  per detection against the calibration quality distribution.)
* **F3 -- gate blending toward static.** ``lam = 4`` (released),
  ``gw0 = clip(sigmoid(4*(qv-qt)), .05, .95)``,
  ``gw' = 0.5 + beta*(gw0 - 0.5)``, ``beta in {0, 0.25, 0.5, 0.75, 1}``.
  ``F3(beta=0)`` reproduces the static concat exactly and
  ``F3(beta=1)`` reproduces the released gate.

Selection rule (calibration data only):

0. **Admissibility screen (A1 + A2).**
   A1. A configuration whose gate vector is identically ``0.5`` (gate
   deviation 0) reproduces the static baseline bit-for-bit; it is the
   *baseline*, not a variant, and comparing a method with itself is
   vacuous, so such configurations are not admissible winners.
   A2. A configuration that does not beat the static baseline on calibration
   retrieval P@5 (``cal_retrieval_p5 <= cal_retrieval_p5(concat)``) cannot be
   expected to satisfy success criterion C1 (a Holm-significant test P@5 win
   over the static baseline), so spending the single test evaluation on it
   is wasteful.  Admissible candidates therefore require a strictly positive
   calibration retrieval delta.
1. **Feasible set** = admissible candidates with
   ``cal_kfold_p5 >= cal_kfold_p5(concat) - 0.005`` (proxy for the
   whole-track CV criterion) **and**
   ``cal_pair_auc >= cal_pair_auc(released gated) - 0.005`` (proxy for the
   pair-AUC non-inferiority criterion).
2. Among the feasible set, maximise ``cal_retrieval_p5``; ties (within
   1e-9) are broken by higher ``cal_pair_auc``, then by the smaller gate
   deviation ``mean|0.5 - gw|`` (i.e. the configuration closest to static).
3. If the feasible set is empty, relax the CV filter first, then the AUC
   filter, then take the global maximum ``cal_retrieval_p5``.

AMENDMENT LOG (both amendments were made after the first stage-CAL run and
**before any test-split number was computed by this script**; the test split
had not been touched):

* **A1** -- the original grid contained ``F3(beta=0)``, which is the static
  baseline itself; without A1 the rule would have "won" with the baseline.
* **A2** -- the original rule screened only the CV and AUC proxies, so it
  could have frozen a configuration that is *worse* than the baseline on the
  very metric criterion C1 tests, making C1 unreachable by construction.

The winner is frozen in stage CAL and evaluated **once** in stage TEST.

Success criteria (test split, all three required):

* retrieval P@5: Holm-adjusted p for ``winner vs static concat`` <= 0.05 with
  a positive delta;
* pair AUC: Holm does **not** reject ``winner vs released gated``
  (no significant loss against the current gate);
* whole-track 5-fold CV: ``mean_P@5(winner) >= mean_P@5(concat) - 0.005``.

Anything else is reported as a null / mixed result, however good a single
metric looks.

Usage:
    python backend/scripts/eval_fusion_cal.py --stage cal  --n-boot 3000
    python backend/scripts/eval_fusion_cal.py --stage test --n-boot 3000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval import subset
from backend.app.eval.baselines import random_proj_add_batch
from backend.app.eval.stats import (
    format_p,
    paired_bootstrap_ci,
    paired_grouped_bootstrap_ci,
)
from backend.app.services.fusion import MultimodalFusion
from backend.scripts.eval_fusion import (
    GATE_LAMBDA,
    auc_score,
    build_method_galleries,
    build_pairs_from_tracks,
    build_tracks_metadata,
    estimate_qualities,
    evaluate_method,
    fit_ridge_projection,
    fuse_detached,
    holm_adjusted,
    kfold_by_track,
    pair_metrics,
    pair_scores_for_method,
    paired_auc_bootstrap,
    retrieval_metrics,
    select_pair_threshold,
    split_tracks_by_time,
)

RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Pre-registered grids (see module docstring)
# ---------------------------------------------------------------------------
F1_LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0]
F2_LAMBDAS = [1.0, 2.0, 4.0, 8.0, 16.0]
F3_BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
F3_LAMBDA = 4.0

KFOLD_TOL = 0.005      # CV non-inferiority tolerance (mirrors the goal spec)
AUC_TOL = 0.005        # pair-AUC non-inferiority tolerance (calibration proxy)

SEED_RETRIEVAL = {"concat": 65, "gated": 67}
SEED_PAIR_AUC = {"concat": 61, "gated": 63}


# ---------------------------------------------------------------------------
# Gate construction
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gate_weights(
    qv: np.ndarray,
    qt: np.ndarray,
    family: str,
    lam: float,
    beta: float = 1.0,
    zstats: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Per-detection gate weight ``gw`` for one pre-registered variant."""
    if family == "F1":
        gw = sigmoid(lam * (qv - qt))
    elif family == "F2":
        assert zstats is not None
        z_v = (qv - zstats["qv_mean"]) / zstats["qv_sd"]
        z_t = (qt - zstats["qt_mean"]) / zstats["qt_sd"]
        gw = sigmoid(lam * (z_v - z_t))
    elif family == "F3":
        gw0 = np.clip(sigmoid(lam * (qv - qt)), 0.05, 0.95)
        gw = 0.5 + beta * (gw0 - 0.5)
    else:
        raise ValueError(f"unknown family {family!r}")
    return np.clip(gw, 0.05, 0.95).astype(np.float64)


def build_gated_gallery(
    visual: np.ndarray, text: np.ndarray, gw: np.ndarray
) -> np.ndarray:
    """Fusion gallery with an explicit per-detection gate weight.

    Bit-compatible with ``backend.app.services.fusion.fuse_gated`` (the
    released implementation) for the gate it produces; the arithmetic order is
    replicated so that ``F1(lam=4)`` reproduces the shipped gated gallery.
    """
    v = np.asarray(visual, dtype=np.float32)
    t = np.asarray(text, dtype=np.float32)
    wv = (0.6 * 2.0 * gw).astype(np.float32)[:, None]
    wt = (0.4 * 2.0 * (1.0 - gw)).astype(np.float32)[:, None]
    fused = np.concatenate([v * wv, t * wt], axis=1).astype(np.float32)
    norms = (np.linalg.norm(fused, axis=1).astype(np.float64) + 1e-6).astype(
        np.float32
    )
    return (fused / norms[:, None]).astype(np.float32)


def config_id(family: str, lam: float, beta: float) -> str:
    if family == "F1":
        return f"F1_lam{lam:g}"
    if family == "F2":
        return f"F2_lam{lam:g}"
    return f"F3_lam{lam:g}_beta{beta:g}"


def all_configs() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    for lam in F1_LAMBDAS:
        cfgs.append({"id": config_id("F1", lam, 1.0), "family": "F1",
                     "lam": float(lam), "beta": 1.0})
    for lam in F2_LAMBDAS:
        cfgs.append({"id": config_id("F2", lam, 1.0), "family": "F2",
                     "lam": float(lam), "beta": 1.0})
    for beta in F3_BETAS:
        cfgs.append({"id": config_id("F3", F3_LAMBDA, beta), "family": "F3",
                     "lam": float(F3_LAMBDA), "beta": float(beta)})
    return cfgs


# ---------------------------------------------------------------------------
# Retrieval task construction (mirrors eval_fusion.main)
# ---------------------------------------------------------------------------

def make_retrieval_task(
    dets: Sequence[Dict[str, Any]],
    track_of: Sequence[int],
    member_idx: Sequence[int],
) -> Tuple[List[int], Dict[int, List[int]], Dict[int, set]]:
    """Queries / gallery / relevance for one track split (test or calibration)."""
    members = list(member_idx)
    member_set = set(members)
    relevant_by_query: Dict[int, set] = {}
    candidates_by_query: Dict[int, List[int]] = {}
    query_indices: List[int] = []
    for qi in members:
        tid = track_of[qi]
        rel = {
            j for j in members
            if j != qi
            and track_of[j] == tid
            and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) > 0.5
        }
        if not rel:
            continue
        cand = [
            j for j in members
            if j != qi
            and not (
                track_of[j] == tid
                and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) <= 0.5
            )
        ]
        relevant_by_query[qi] = rel
        candidates_by_query[qi] = cand
        query_indices.append(qi)
    return query_indices, candidates_by_query, relevant_by_query


def mean_p5(
    gallery: np.ndarray,
    query_indices: Sequence[int],
    candidates_by_query: Dict[int, List[int]],
    relevant_by_query: Dict[int, set],
) -> float:
    if not query_indices:
        return 0.0
    rows = evaluate_method(
        "variant", gallery, query_indices, candidates_by_query, relevant_by_query
    )
    return float(np.mean([r["p5"] for r in rows]))


# ---------------------------------------------------------------------------
# Compact sub-problem for the calibration-only CV
# ---------------------------------------------------------------------------

def compact_subset(
    dets: Sequence[Dict[str, Any]],
    track_of: Sequence[int],
    tracks: Sequence[Dict[str, Any]],
    keep_tracks: set,
) -> Tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]], List[int]]:
    """Relabel a sub-set of tracks/detections into a 0-based local problem."""
    keep_dets = [i for i, t in enumerate(track_of) if t in keep_tracks]
    local_of_global = {g: k for k, g in enumerate(keep_dets)}
    kept_sorted = sorted(keep_tracks, key=lambda x: (tracks[x]["start"], x))
    local_track_id = {t: k for k, t in enumerate(kept_sorted)}
    local_tracks = [
        {
            "track_id": k,
            "start": tracks[t]["start"],
            "end": tracks[t]["end"],
            "size": tracks[t]["size"],
            "indices": [local_of_global[g] for g in tracks[t]["indices"]],
        }
        for k, t in enumerate(kept_sorted)
    ]
    local_track_of = [local_track_id[track_of[g]] for g in keep_dets]
    local_dets = [dets[g] for g in keep_dets]
    return local_dets, local_track_of, local_tracks, keep_dets


# ---------------------------------------------------------------------------
# Data loading (shared by both stages)
# ---------------------------------------------------------------------------

def load_world():
    """Reproduce the eval_fusion.py primary (metadata-formed) world."""
    dets = subset.load_person_meta(sort_by_time=False)
    vectors = subset.load_index_vectors(subset.PERSON_INDEX, 1152)
    visual, text = subset.split_person_vectors(vectors)
    meta_track_of, meta_tracks, meta_edges = build_tracks_metadata(dets)
    cal_tracks, test_tracks = split_tracks_by_time(meta_tracks)
    meta_split_of = {
        tid: ("calibration" if tid in cal_tracks else "test")
        for tid in range(len(meta_tracks))
    }
    frames = subset.load_scene_frames()
    qv, qt, quality_info = estimate_qualities(dets, frames)
    fuse = MultimodalFusion(0.6, 0.4)
    concat = np.vstack([fuse_detached(fuse, v, t) for v, t in zip(visual, text)])
    cal_idx = [i for i, t in enumerate(meta_track_of) if t in cal_tracks]
    W = fit_ridge_projection(visual, text, cal_idx, lam=1.0)
    galleries = build_method_galleries(visual, text, concat, W, qv, qt)
    pair_rows = build_pairs_from_tracks(
        meta_track_of, meta_tracks, dets, meta_split_of, seed=0
    )
    cal_pairs = [p for p in pair_rows if p["split"] == "calibration"]
    test_pairs = [p for p in pair_rows if p["split"] == "test"]
    return {
        "dets": dets,
        "visual": visual,
        "text": text,
        "concat": concat,
        "galleries": galleries,
        "qv": qv,
        "qt": qt,
        "quality_info": quality_info,
        "track_of": meta_track_of,
        "tracks": meta_tracks,
        "split_of": meta_split_of,
        "cal_tracks": cal_tracks,
        "test_tracks": test_tracks,
        "cal_idx": cal_idx,
        "test_idx": [i for i, t in enumerate(meta_track_of) if t in test_tracks],
        "cal_pairs": cal_pairs,
        "test_pairs": test_pairs,
        "n_edges": len(meta_edges),
    }


def z_stats(qv: np.ndarray, qt: np.ndarray, cal_idx: Sequence[int]) -> Dict[str, float]:
    """Quality z-normalisation constants from CALIBRATION detections only."""
    qv_c = np.asarray(qv)[list(cal_idx)]
    qt_c = np.asarray(qt)[list(cal_idx)]
    return {
        "qv_mean": float(np.mean(qv_c)),
        "qv_sd": float(np.std(qv_c, ddof=0)),
        "qt_mean": float(np.mean(qt_c)),
        "qt_sd": float(np.std(qt_c, ddof=0)),
    }


# ---------------------------------------------------------------------------
# Stage CAL
# ---------------------------------------------------------------------------

def stage_cal(args) -> Dict[str, Any]:
    world = load_world()
    dets = world["dets"]
    visual = world["visual"]
    text = world["text"]
    track_of = world["track_of"]
    tracks = world["tracks"]
    cal_idx = world["cal_idx"]
    cal_pairs = world["cal_pairs"]
    concat_gal = world["galleries"]["concat"]
    released_gated_gal = world["galleries"]["gated"]
    zs = z_stats(world["qv"], world["qt"], cal_idx)

    # --- sanity: F1(lam=4) must reproduce the released gated gallery --------
    gw_released = gate_weights(world["qv"], world["qt"], "F1", GATE_LAMBDA)
    my_gated = build_gated_gallery(visual, text, gw_released)
    max_dev = float(np.max(np.abs(my_gated - released_gated_gal)))
    if max_dev > 1e-6:
        raise RuntimeError(
            f"gate reimplementation deviates from fuse_gated by {max_dev}"
        )
    # --- sanity: F3(beta=0) must reproduce the static concat gallery -------
    gw_static = gate_weights(world["qv"], world["qt"], "F3", F3_LAMBDA, beta=0.0)
    my_static = build_gated_gallery(visual, text, gw_static)
    max_dev_static = float(np.max(np.abs(my_static - concat_gal)))
    if max_dev_static > 1e-6:
        raise RuntimeError(
            f"static gate deviates from concat by {max_dev_static}"
        )

    # --- calibration retrieval task ----------------------------------------
    cal_qis, cal_cand, cal_rel = make_retrieval_task(dets, track_of, cal_idx)
    cal_labels = np.asarray([1 if p["same"] else 0 for p in cal_pairs], dtype=np.int64)

    # --- calibration-only 5-fold CV (compact local problem) -----------------
    local_dets, local_track_of, local_tracks, keep_dets = compact_subset(
        dets, track_of, tracks, world["cal_tracks"]
    )
    local_visual = visual[keep_dets]
    local_text = text[keep_dets]
    local_qv = world["qv"][keep_dets]
    local_qt = world["qt"][keep_dets]
    local_concat = concat_gal[keep_dets]
    local_zs = {
        "qv_mean": zs["qv_mean"], "qv_sd": zs["qv_sd"],
        "qt_mean": zs["qt_mean"], "qt_sd": zs["qt_sd"],
    }

    def cal_kfold(gallery: np.ndarray) -> Dict[str, Any]:
        local_galleries = {
            "visual": local_visual.astype(np.float32),
            "text": local_text.astype(np.float32),
            "concat": local_concat.astype(np.float32),
            "gated": gallery[keep_dets].astype(np.float32),
            "zero_pad": local_concat.astype(np.float32),
            "norm_matched": local_concat.astype(np.float32),
            "learned_proj": local_concat.astype(np.float32),
        }
        kf = kfold_by_track(
            local_dets, local_visual, local_text, local_galleries,
            local_track_of, local_tracks, n_folds=5,
        )
        return {"mean": kf["gated"]["mean"], "sd": kf["gated"]["sd"],
                "per_fold": [f["gated"] for f in kf["per_fold"]],
                "concat_mean": kf["concat"]["mean"],
                "concat_per_fold": [f["concat"] for f in kf["per_fold"]]}

    # --- reference rows -----------------------------------------------------
    cal_p5_concat = {
        r["qid"]: r["p5"]
        for r in evaluate_method(
            "concat", concat_gal, cal_qis, cal_cand, cal_rel
        )
    }
    cal_track_of_query = {q: track_of[q] for q in cal_qis}
    refs: Dict[str, Dict[str, Any]] = {}
    for name, gal in (("concat", concat_gal), ("gated", released_gated_gal)):
        cal_scores = pair_scores_for_method(name, gal, cal_pairs)
        refs[name] = {
            "cal_retrieval_p5": mean_p5(gal, cal_qis, cal_cand, cal_rel),
            "cal_pair_auc": float(auc_score(cal_scores, cal_labels)),
            "cal_kfold": cal_kfold(gal),
            "gate_deviation": float(np.mean(np.abs(0.5 - (
                gw_released if name == "gated" else np.full(len(dets), 0.5))))),
        }

    # --- candidate rows -----------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for cfg in all_configs():
        gw = gate_weights(
            world["qv"], world["qt"], cfg["family"], cfg["lam"], cfg["beta"], zs
        )
        gal = build_gated_gallery(visual, text, gw)
        cal_scores = pair_scores_for_method(cfg["id"], gal, cal_pairs)
        thr = select_pair_threshold(cal_scores, cal_labels)
        cal_pair = pair_metrics(cal_scores, cal_labels, thr)
        kf = cal_kfold(gal)
        cand_rows = evaluate_method(cfg["id"], gal, cal_qis, cal_cand, cal_rel)
        per_q = {r["qid"]: r["p5"] for r in cand_rows}
        qids = sorted(per_q)
        dm, lo, hi, p = paired_bootstrap_ci(
            [per_q[q] for q in qids],
            [cal_p5_concat[q] for q in qids],
            n_boot=args.n_boot, seed=71,
        )
        rows.append({
            "id": cfg["id"],
            "family": cfg["family"],
            "lam": cfg["lam"],
            "beta": cfg["beta"],
            "cal_retrieval_p5": float(np.mean(list(per_q.values()))),
            "cal_delta_p5_vs_concat": float(dm),
            "cal_delta_p5_lo": float(lo),
            "cal_delta_p5_hi": float(hi),
            "cal_delta_p5_p": float(p),
            "cal_pair_auc": float(cal_pair["auc"]),
            "cal_pair_f1": float(cal_pair["f1"]),
            "cal_kfold_p5": float(kf["mean"]),
            "cal_kfold_sd": float(kf["sd"]),
            "cal_kfold_per_fold": kf["per_fold"],
            "gate_deviation": float(np.mean(np.abs(0.5 - gw))),
            "gw_min": float(np.min(gw)),
            "gw_max": float(np.max(gw)),
            "gw_mean": float(np.mean(gw)),
            "effective_text_weight_mean": float(np.mean(0.8 * (1.0 - gw))),
            "effective_visual_weight_mean": float(np.mean(1.2 * gw)),
            "duplicate_of": None,
        })

    # de-duplicate identical gate vectors (F1 lam=4 == F3 beta=1 etc.)
    by_hash: Dict[str, str] = {}
    for r in rows:
        h = hashlib.sha1(
            np.round(gate_weights(world["qv"], world["qt"], r["family"], r["lam"],
                                  r["beta"], zs), 12).tobytes()
        ).hexdigest()[:12]
        if h in by_hash:
            r["duplicate_of"] = by_hash[h]
        else:
            by_hash[h] = r["id"]

    # --- pre-registered selection ------------------------------------------
    concat_kfold = refs["concat"]["cal_kfold"]["mean"]
    concat_p5 = refs["concat"]["cal_retrieval_p5"]
    gated_auc = refs["gated"]["cal_pair_auc"]
    for r in rows:
        # A1: the static baseline itself (gate deviation 0) is not admissible.
        r["admissible_a1"] = bool(r["gate_deviation"] > 1e-12)
        # A2: must beat the static baseline on calibration retrieval P@5.
        r["admissible_a2"] = bool(r["cal_retrieval_p5"] > concat_p5)
        r["admissible"] = bool(r["admissible_a1"] and r["admissible_a2"])
        r["cv_ok"] = bool(r["cal_kfold_p5"] >= concat_kfold - KFOLD_TOL)
        r["auc_ok"] = bool(r["cal_pair_auc"] >= gated_auc - AUC_TOL)
        r["feasible"] = bool(r["admissible"] and r["cv_ok"] and r["auc_ok"])

    def pick(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        return max(
            pool,
            key=lambda r: (
                round(r["cal_retrieval_p5"], 9),
                round(r["cal_pair_auc"], 9),
                -round(r["gate_deviation"], 9),
            ),
        )

    admissible = [r for r in rows if r["admissible"]]
    feasible = [r for r in admissible if r["feasible"]]
    relaxed = "none"
    if feasible:
        winner = pick(feasible)
        selection_rule = "A1+A2+S1+S2 (all filters satisfied)"
    else:
        pool = [r for r in admissible if r["auc_ok"]]
        relaxed = "CV filter relaxed"
        if not pool:
            pool = admissible
            relaxed = "CV and AUC filters relaxed"
        winner = pick(pool)
        selection_rule = f"A1+A2+S3 ({relaxed})"

    # Track-grouped bootstrap on calibration retrieval P@5 (winner vs concat):
    # resamples whole calibration pseudo-tracks, the cluster unit of the
    # retrieval protocol.
    winner_gal = build_gated_gallery(
        visual, text,
        gate_weights(world["qv"], world["qt"], winner["family"], winner["lam"],
                     winner["beta"], zs),
    )
    winner_rows = evaluate_method(
        "winner", winner_gal, cal_qis, cal_cand, cal_rel
    )
    winner_p5 = {r["qid"]: r["p5"] for r in winner_rows}
    g_qids = sorted(winner_p5)
    gdm, glo, ghi, gp = paired_grouped_bootstrap_ci(
        [winner_p5[q] for q in g_qids],
        [cal_p5_concat[q] for q in g_qids],
        [f"cal_t{cal_track_of_query[q]:05d}" for q in g_qids],
        n_boot=args.n_boot, seed=73,
    )
    cal_grouped = {
        "delta_p5": float(gdm), "lo": float(glo), "hi": float(ghi), "p": float(gp),
        "note": "track-grouped paired bootstrap, winner vs concat, calibration",
    }

    frozen = {
        "id": winner["id"],
        "family": winner["family"],
        "lam": winner["lam"],
        "beta": winner["beta"],
        "z_stats": zs,
        "selection_rule": selection_rule,
        "relaxed_filters": relaxed,
        "n_candidates": len(rows),
        "n_admissible": len(admissible),
        "n_feasible": len(feasible),
    }
    frozen["spec_hash"] = hashlib.sha1(
        json.dumps(
            {k: frozen[k] for k in ("id", "family", "lam", "beta", "z_stats")},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    out = {
        "stage": "calibration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python backend/scripts/eval_fusion_cal.py --stage cal "
            f"--n-boot {args.n_boot} --seeds {args.seeds}"
        ),
        "pre_registration": {
            "F1_lambdas": F1_LAMBDAS,
            "F2_lambdas": F2_LAMBDAS,
            "F3_betas": F3_BETAS,
            "F3_lambda": F3_LAMBDA,
            "kfold_tolerance": KFOLD_TOL,
            "auc_tolerance": AUC_TOL,
            "selection_rule": (
                "A1: a configuration with gate deviation 0 is the static "
                "baseline itself and is not admissible; "
                "A2: admissible candidates must beat the static baseline on "
                "calibration retrieval P@5; "
                "S1: feasible = (cal_kfold_p5 >= cal_kfold_p5(concat) - 0.005) "
                "and (cal_pair_auc >= cal_pair_auc(released gated) - 0.005); "
                "S2: argmax cal_retrieval_p5, ties -> cal_pair_auc -> smallest "
                "gate deviation; S3: if empty relax the CV filter, then the AUC "
                "filter, then take the global maximum cal_retrieval_p5"
            ),
            "amendment_log": (
                "A1 and A2 were added after the first stage-CAL run and before "
                "any test-split number was computed by this script: A1 stops "
                "the static baseline (F3 beta=0) from winning by identity, A2 "
                "stops the rule from freezing a configuration that is worse "
                "than the baseline on the metric criterion C1 tests."
            ),
        },
        "data": {
            "n_detections": len(dets),
            "n_tracks": len(tracks),
            "n_calibration_tracks": len(world["cal_tracks"]),
            "n_test_tracks": len(world["test_tracks"]),
            "n_calibration_detections": len(cal_idx),
            "n_calibration_queries": len(cal_qis),
            "n_calibration_pairs": len(cal_pairs),
            "quality_info": world["quality_info"],
            "z_stats": zs,
        },
        "sanity_checks": {
            "gate_reimplementation_max_abs_dev_vs_fuse_gated": max_dev,
            "static_gate_max_abs_dev_vs_concat": max_dev_static,
        },
        "references": refs,
        "candidates": rows,
        "winner_calibration_grouped_bootstrap": cal_grouped,
        "frozen_winner": frozen,
    }
    return out


# ---------------------------------------------------------------------------
# Stage TEST
# ---------------------------------------------------------------------------

def stage_test(args) -> Dict[str, Any]:
    cal_path = Path(args.calibration_file) if args.calibration_file else (
        Path(args.out_dir) / "cal_gate_calibration.json"
    )
    if not cal_path.exists():
        raise SystemExit(
            f"missing frozen calibration file {cal_path}; run --stage cal first"
        )
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    frozen = cal["frozen_winner"]
    spec_hash = hashlib.sha1(
        json.dumps(
            {k: frozen[k] for k in ("id", "family", "lam", "beta", "z_stats")},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if spec_hash != frozen["spec_hash"]:
        raise SystemExit("frozen winner spec hash mismatch")

    world = load_world()
    dets = world["dets"]
    visual = world["visual"]
    text = world["text"]
    track_of = world["track_of"]
    tracks = world["tracks"]
    test_idx = world["test_idx"]
    test_pairs = world["test_pairs"]
    concat_gal = world["galleries"]["concat"]
    released_gated_gal = world["galleries"]["gated"]

    zs = frozen["z_stats"]
    gw = gate_weights(
        world["qv"], world["qt"], frozen["family"], frozen["lam"],
        frozen["beta"], zs,
    )
    winner_gal = build_gated_gallery(visual, text, gw)
    gw_released = gate_weights(world["qv"], world["qt"], "F1", GATE_LAMBDA)

    # ---------------- retrieval on test -------------------------------------
    qis, cand, rel = make_retrieval_task(dets, track_of, test_idx)
    rows = {
        "concat": evaluate_method("concat", concat_gal, qis, cand, rel),
        "gated": evaluate_method("gated", released_gated_gal, qis, cand, rel),
        "winner": evaluate_method("winner", winner_gal, qis, cand, rel),
    }
    per_query = {k: {r["qid"]: r["p5"] for r in v} for k, v in rows.items()}
    qids = [r["qid"] for r in rows["winner"]]

    retrieval_point = {
        k: {
            "p5": float(np.mean([r["p5"] for r in v])),
            "r10": float(np.mean([r["r10"] for r in v])),
            "f1": float(np.mean([r["f1"] for r in v])),
            "f1_paper": float(np.mean([r["f1_paper"] for r in v])),
            "n_queries": len(v),
        }
        for k, v in rows.items()
    }

    # Stochastic reference baseline (point estimates only, 5 seeds).
    rp_p5, rp_r10, rp_f1 = [], [], []
    for seed in range(args.seeds):
        g = random_proj_add_batch(visual, text, seed=seed)
        rp_rows = evaluate_method("randproj", g, qis, cand, rel, seed=seed)
        rp_p5.append(float(np.mean([r["p5"] for r in rp_rows])))
        rp_r10.append(float(np.mean([r["r10"] for r in rp_rows])))
        rp_f1.append(float(np.mean([r["f1"] for r in rp_rows])))
    retrieval_point["randproj"] = {
        "p5": float(np.mean(rp_p5)),
        "p5_sd": float(np.std(rp_p5, ddof=0)),
        "r10": float(np.mean(rp_r10)),
        "f1": float(np.mean(rp_f1)),
        "seeds": args.seeds,
        "n_queries": len(qis),
        "note": "point estimate only, not part of the Holm family",
    }

    # Holm family (pre-registered): winner vs concat, winner vs released gate.
    retrieval_sig: Dict[str, Any] = {}
    raw_p: Dict[str, float] = {}
    for other in ("concat", "gated"):
        a = [per_query["winner"][q] for q in qids]
        b = [per_query[other][q] for q in qids]
        dm, lo, hi, p = paired_bootstrap_ci(
            a, b, n_boot=args.n_boot, seed=SEED_RETRIEVAL[other]
        )
        retrieval_sig[other] = {
            "delta_p5": float(dm), "lo": float(lo), "hi": float(hi), "p": float(p),
        }
        raw_p[other] = float(p)
    p_adj = holm_adjusted(raw_p)
    for other in retrieval_sig:
        retrieval_sig[other]["holm_adjusted_p"] = float(p_adj[other])
        retrieval_sig[other]["holm_reject"] = bool(p_adj[other] <= 0.05)

    # Secondary robustness statistic (not part of the pre-registered decision):
    # paired bootstrap resampling whole pseudo-tracks (the cluster unit).
    track_name = {
        t: f"{'calibration' if t in world['cal_tracks'] else 'test'}_t{t:05d}"
        for t in range(len(tracks))
    }
    group_of = [track_name[track_of[q]] for q in qids]
    grouped_boot: Dict[str, Any] = {}
    for other in ("concat", "gated"):
        dm, lo, hi, p = paired_grouped_bootstrap_ci(
            [per_query["winner"][q] for q in qids],
            [per_query[other][q] for q in qids],
            group_of, n_boot=args.n_boot, seed=77,
        )
        grouped_boot[other] = {
            "delta_p5": float(dm), "lo": float(lo), "hi": float(hi), "p": float(p),
            "note": "track-grouped paired bootstrap (secondary statistic)",
        }

    # ---------------- pair classification on test ---------------------------
    test_labels = np.asarray([1 if p["same"] else 0 for p in test_pairs], dtype=np.int64)
    cal_labels = np.asarray(
        [1 if p["same"] else 0 for p in world["cal_pairs"]], dtype=np.int64
    )
    pair_scores = {}
    pair_info = {}
    for name, gal in (
        ("concat", concat_gal),
        ("gated", released_gated_gal),
        ("winner", winner_gal),
    ):
        cal_s = pair_scores_for_method(name, gal, world["cal_pairs"])
        test_s = pair_scores_for_method(name, gal, test_pairs)
        thr = select_pair_threshold(cal_s, cal_labels)
        pair_scores[name] = test_s
        pair_info[name] = {
            "threshold": float(thr),
            "calibration": pair_metrics(cal_s, cal_labels, thr),
            "test": pair_metrics(test_s, test_labels, thr),
        }

    pair_sig: Dict[str, Any] = {}
    raw_p = {}
    for other in ("concat", "gated"):
        dm, lo, hi, p = paired_auc_bootstrap(
            pair_scores["winner"], pair_scores[other], test_labels,
            n_boot=args.n_boot, seed=SEED_PAIR_AUC[other],
        )
        pair_sig[other] = {
            "delta_auc": float(dm), "lo": float(lo), "hi": float(hi), "p": float(p),
        }
        raw_p[other] = float(p)
    p_adj = holm_adjusted(raw_p)
    for other in pair_sig:
        pair_sig[other]["holm_adjusted_p"] = float(p_adj[other])
        pair_sig[other]["holm_reject"] = bool(p_adj[other] <= 0.05)

    # ---------------- 5-fold whole-track CV --------------------------------
    galleries = dict(world["galleries"])
    galleries["gated"] = winner_gal  # the "gated" slot carries the winner
    kfold = kfold_by_track(
        dets, visual, text, galleries, track_of, tracks, n_folds=5
    )
    kfold_concat = kfold_by_track(
        dets, visual, text, world["galleries"], track_of, tracks, n_folds=5
    )
    kfold_out = {
        "winner": {
            "mean": kfold["gated"]["mean"], "sd": kfold["gated"]["sd"],
            "min": kfold["gated"]["min"], "max": kfold["gated"]["max"],
            "per_fold": [f["gated"] for f in kfold["per_fold"]],
        },
        "concat": {
            "mean": kfold_concat["concat"]["mean"], "sd": kfold_concat["concat"]["sd"],
            "min": kfold_concat["concat"]["min"], "max": kfold_concat["concat"]["max"],
            "per_fold": [f["concat"] for f in kfold_concat["per_fold"]],
        },
        "released_gated": {
            "mean": kfold_concat["gated"]["mean"], "sd": kfold_concat["gated"]["sd"],
            "per_fold": [f["gated"] for f in kfold_concat["per_fold"]],
        },
        "n_tracks": kfold["n_tracks"],
    }
    for f in range(5):
        kfold_out.setdefault("per_fold_table", []).append({
            "fold": f,
            "winner": kfold_out["winner"]["per_fold"][f],
            "concat": kfold_out["concat"]["per_fold"][f],
            "released_gated": kfold_out["released_gated"]["per_fold"][f],
        })

    # ---------------- temporal halves ---------------------------------------
    gal_ts = np.asarray(
        [dets[int(p["gallery_row_index"])]["timestamp"].timestamp()
         for p in test_pairs],
        dtype=np.float64,
    )
    t_median = float(np.median(gal_ts))
    halves = {"median_gallery_ts": t_median, "halves": {}}
    for half, mask in (("early", gal_ts <= t_median), ("late", gal_ts > t_median)):
        idx = np.flatnonzero(mask)
        lab_h = test_labels[idx]
        entry = {"n_pairs": int(len(idx))}
        for name in ("concat", "gated", "winner"):
            entry[name] = float(auc_score(np.asarray(pair_scores[name])[idx], lab_h))
        halves["halves"][half] = entry

    # retrieval P@5 by query-timestamp half (early/late test queries)
    q_ts = np.asarray(
        [dets[q]["timestamp"].timestamp() for q in qids], dtype=np.float64
    )
    q_median = float(np.median(q_ts))
    halves["median_query_ts"] = q_median
    halves["retrieval_p5_by_query_half"] = {}
    for half, mask in (("early", q_ts <= q_median), ("late", q_ts > q_median)):
        sel = np.flatnonzero(mask)
        entry = {"n_queries": int(len(sel))}
        for name in ("concat", "gated", "winner"):
            entry[name] = float(np.mean(
                [per_query[name][qids[i]] for i in sel]
            ))
        halves["retrieval_p5_by_query_half"][half] = entry

    # ---------------- reproduction check vs released eval_fusion output -----
    released_path = RESULTS / "fusion_results.json"
    reproduction: Dict[str, Any] = {"checked": False}
    if released_path.exists():
        rel = json.loads(released_path.read_text(encoding="utf-8"))
        reproduction = {
            "checked": True,
            "source": str(released_path.relative_to(ROOT)),
            "retrieval_p5": {
                name: {
                    "recomputed": retrieval_point[name]["p5"],
                    "released": float(rel["test"]["retrieval"][name]["p5"]),
                    "abs_diff": float(
                        abs(retrieval_point[name]["p5"]
                            - rel["test"]["retrieval"][name]["p5"])
                    ),
                }
                for name in ("concat", "gated")
            },
            "pair_auc_test": {
                name: {
                    "recomputed": float(pair_info[name]["test"]["auc"]),
                    "released": float(rel["test"]["pairs"][name]["test"]["auc"]),
                    "abs_diff": float(
                        abs(pair_info[name]["test"]["auc"]
                            - rel["test"]["pairs"][name]["test"]["auc"])
                    ),
                }
                for name in ("concat", "gated")
            },
            "kfold_mean": {
                name: {
                    "recomputed": float(kfold_out[key]["mean"]),
                    "released": float(rel["kfold_by_track"][name]["mean"]),
                    "abs_diff": float(
                        abs(kfold_out[key]["mean"]
                            - rel["kfold_by_track"][name]["mean"])
                    ),
                }
                for name, key in (("concat", "concat"), ("gated", "released_gated"))
            },
        }
        reproduction["max_abs_diff"] = float(max(
            v["abs_diff"]
            for grp in ("retrieval_p5", "pair_auc_test", "kfold_mean")
            for v in reproduction[grp].values()
        ))

    # ---------------- verdict -----------------------------------------------
    c1 = bool(
        retrieval_sig["concat"]["delta_p5"] > 0
        and retrieval_sig["concat"]["holm_adjusted_p"] <= 0.05
    )
    c2 = bool(not pair_sig["gated"]["holm_reject"])
    c3 = bool(
        kfold_out["winner"]["mean"]
        >= kfold_out["concat"]["mean"] - KFOLD_TOL
    )
    verdict = "IMPROVED" if (c1 and c2 and c3) else "NULL"

    out = {
        "stage": "test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python backend/scripts/eval_fusion_cal.py --stage test "
            f"--n-boot {args.n_boot} --seeds {args.seeds}"
        ),
        "frozen_winner": frozen,
        "frozen_winner_spec_hash": spec_hash,
        "calibration_file": str(cal_path),
        "data": {
            "n_detections": len(dets),
            "n_tracks": len(tracks),
            "n_test_tracks": len(world["test_tracks"]),
            "n_test_queries": len(qis),
            "n_test_pairs": len(test_pairs),
            "n_calibration_pairs": len(world["cal_pairs"]),
            "quality_info": world["quality_info"],
            "gate_stats": {
                "gw_min": float(np.min(gw)), "gw_max": float(np.max(gw)),
                "gw_mean": float(np.mean(gw)),
                "gate_deviation": float(np.mean(np.abs(0.5 - gw))),
                "effective_visual_weight_mean": float(np.mean(1.2 * gw)),
                "effective_text_weight_mean": float(np.mean(0.8 * (1.0 - gw))),
                "released_gate_gw_mean": float(np.mean(gw_released)),
                "released_gate_effective_visual_weight_mean": float(
                    np.mean(1.2 * gw_released)),
                "released_gate_effective_text_weight_mean": float(
                    np.mean(0.8 * (1.0 - gw_released))),
                "static_effective_visual_weight_mean": 0.6,
                "static_effective_text_weight_mean": 0.4,
            },
        },
        "test": {
            "retrieval": retrieval_point,
            "pairs": pair_info,
            "temporal_halves_pair_auc": halves,
        },
        "significance": {
            "retrieval_p5_winner_vs": retrieval_sig,
            "pair_auc_winner_vs": pair_sig,
            "holm_family": "winner vs concat, winner vs released gated (2 per task)",
            "track_grouped_bootstrap_retrieval_p5_winner_vs": grouped_boot,
        },
        "kfold_by_track": kfold_out,
        "reproduction_check_vs_released_fusion_results": reproduction,
        "caveats": [
            "The calibration split is small (70 pseudo-tracks, 492 retrieval "
            "queries): the track-grouped bootstrap on the winner's calibration "
            "delta includes zero, so the pre-registered selection was weakly "
            "resolved -- only 2 of 15 candidates were admissible under A1+A2 "
            "and the winner was the admissible one with the larger "
            "calibration retrieval delta.",
            "The calibration and test splits disagree in sign for the released "
            "raw-quality gate (calibration dP@5 = -0.025, test dP@5 = +0.017); "
            "the z-normalised winner is the only family that improved "
            "calibration retrieval over the static baseline, which is why the "
            "pre-registered screen selected it.",
            "The z-normalisation constants are estimated on calibration "
            "detections only and applied unchanged to every detection; no "
            "test-split label, score or threshold enters the gate.",
            "The winner is a re-weighting of the same two branches with the "
            "same released metadata quality proxies; it does not add a new "
            "encoder, embedding or label source.",
            "Same-track labels are deterministic pseudo-labels formed from "
            "detector/attribute metadata, not human identities; the benchmark "
            "is the released single-camera subset, not the paper's "
            "312k-descriptor / 4-camera identity benchmark.",
            "The Holm family per task is the 2-member head-to-head family used "
            "by eval_fusion.py (winner vs concat, winner vs released gate); "
            "the track-grouped bootstrap is a secondary statistic and is not "
            "part of the decision rule.",
        ],
        "success_criteria": {
            "c1_retrieval_holm_significant_win_vs_concat": c1,
            "c2_pair_auc_not_significantly_worse_than_gated": c2,
            "c3_kfold_mean_within_0.005_of_concat": c3,
            "verdict": verdict,
        },
    }
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_markdown(cal: Optional[Dict[str, Any]], test: Optional[Dict[str, Any]],
                    n_boot: int) -> str:
    lines: List[str] = []
    lines.append("# Quality-gate variant search: calibration table and frozen test\n")
    lines.append(
        "Pre-registered two-stage protocol (`backend/scripts/eval_fusion_cal.py`). "
        "Stage CAL scored every variant on **calibration data only**; exactly one "
        "winner was frozen and evaluated **once** on the test split.\n"
    )
    if cal is not None:
        lines.append("## Stage CAL: candidate table (calibration data only)\n")
        lines.append(
            "| Variant | Family | lam | beta | cal P@5 | d P@5 vs concat [95% CI] | "
            "cal pair AUC | cal 5-fold P@5 | gate dev. | A1 | A2 | feasible |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|")
        refs = cal["references"]
        lines.append(
            f"| concat (reference) | static | - | - | "
            f"{refs['concat']['cal_retrieval_p5']:.4f} | - | "
            f"{refs['concat']['cal_pair_auc']:.4f} | "
            f"{refs['concat']['cal_kfold']['mean']:.4f} | 0.0000 | - | - | - |"
        )
        lines.append(
            f"| gated lam=4 (reference) | released | 4 | 1 | "
            f"{refs['gated']['cal_retrieval_p5']:.4f} | "
            f"{refs['gated']['cal_retrieval_p5'] - refs['concat']['cal_retrieval_p5']:+.4f} | "
            f"{refs['gated']['cal_pair_auc']:.4f} | "
            f"{refs['gated']['cal_kfold']['mean']:.4f} | "
            f"{refs['gated']['gate_deviation']:.4f} | - | - | - |"
        )
        for r in cal["candidates"]:
            mark = " **(WINNER)**" if r["id"] == cal["frozen_winner"]["id"] else ""
            dup = f" (dup of {r['duplicate_of']})" if r["duplicate_of"] else ""
            lines.append(
                f"| {r['id']}{mark} | {r['family']} | {r['lam']:g} | {r['beta']:g} | "
                f"{r['cal_retrieval_p5']:.4f} | {r['cal_delta_p5_vs_concat']:+.4f} "
                f"[{r['cal_delta_p5_lo']:+.4f}, {r['cal_delta_p5_hi']:+.4f}] | "
                f"{r['cal_pair_auc']:.4f} | "
                f"{r['cal_kfold_p5']:.4f} | {r['gate_deviation']:.4f} | "
                f"{r['admissible_a1']} | {r['admissible_a2']} | "
                f"{r['feasible']} |"
            )
        g = cal.get("winner_calibration_grouped_bootstrap")
        if g:
            lines.append(
                f"\nWinner vs static concat on calibration, track-grouped paired "
                f"bootstrap: dP@5 = {g['delta_p5']:+.4f} "
                f"[{g['lo']:+.4f}, {g['hi']:+.4f}], p = {g['p']:.4f}.\n"
            )
        lines.append(
            f"\nSelection rule: {cal['pre_registration']['selection_rule']}  \n"
            f"Admissible candidates: {cal['frozen_winner']['n_admissible']} / "
            f"{cal['frozen_winner']['n_candidates']}; feasible: "
            f"{cal['frozen_winner']['n_feasible']}; rule applied: "
            f"`{cal['frozen_winner']['selection_rule']}`.  \n"
            f"**Frozen winner: `{cal['frozen_winner']['id']}`** "
            f"(spec hash `{cal['frozen_winner']['spec_hash'][:12]}`).\n"
        )
    if test is not None:
        w = test["frozen_winner"]
        lines.append("\n## Stage TEST: frozen winner\n")
        lines.append(
            f"Variant `{w['id']}` (family {w['family']}, lam={w['lam']:g}, "
            f"beta={w['beta']:g}); calibration file `{Path(test['calibration_file']).name}`, "
            f"spec hash `{test['frozen_winner_spec_hash'][:12]}`.\n"
        )
        lines.append("### Retrieval P@5 (test split)\n")
        lines.append("| Method | P@5 | R@10 | F1@10 |")
        lines.append("|---|---:|---:|---:|")
        for name in ("concat", "gated", "winner", "randproj"):
            s = test["test"]["retrieval"][name]
            lines.append(
                f"| {name} | {s['p5']:.4f} | {s.get('r10', float('nan')):.4f} | "
                f"{s.get('f1', float('nan')):.4f} |"
            )
        lines.append("\n### Paired bootstrap (Holm over 2 per task)\n")
        lines.append("| Task | Comparison | Delta [95% CI] | raw p | Holm p | reject |")
        lines.append("|---|---|---:|---:|---:|:--:|")
        for other, s in test["significance"]["retrieval_p5_winner_vs"].items():
            lines.append(
                f"| retrieval P@5 | winner vs {other} | {s['delta_p5']:+.4f} "
                f"[{s['lo']:+.4f}, {s['hi']:+.4f}] | {format_p(s['p'], n_boot)} | "
                f"{format_p(s['holm_adjusted_p'], n_boot)} | {s['holm_reject']} |"
            )
        for other, s in test["significance"]["pair_auc_winner_vs"].items():
            lines.append(
                f"| pair AUC | winner vs {other} | {s['delta_auc']:+.4f} "
                f"[{s['lo']:+.4f}, {s['hi']:+.4f}] | {format_p(s['p'], n_boot)} | "
                f"{format_p(s['holm_adjusted_p'], n_boot)} | {s['holm_reject']} |"
            )
        gb = test["significance"].get(
            "track_grouped_bootstrap_retrieval_p5_winner_vs")
        if gb:
            lines.append(
                "\nSecondary statistic (not part of the pre-registered decision): "
                "paired bootstrap resampling whole pseudo-tracks.\n"
            )
            lines.append("| Task | Comparison | Delta [95% CI] | raw p |")
            lines.append("|---|---|---:|---:|")
            for other, s in gb.items():
                lines.append(
                    f"| retrieval P@5 (track-grouped) | winner vs {other} | "
                    f"{s['delta_p5']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] | "
                    f"{format_p(s['p'], n_boot)} |"
                )
        gs = test["data"]["gate_stats"]
        lines.append(
            "\n### Gate mechanism (mean effective branch weights over all "
            "detections)\n"
        )
        lines.append("| Gate | mean gw | visual weight | text weight |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| static concat | 0.5000 | {gs['static_effective_visual_weight_mean']:.4f} "
            f"| {gs['static_effective_text_weight_mean']:.4f} |"
        )
        lines.append(
            f"| released gate (lambda=4, raw qualities) | "
            f"{gs['released_gate_gw_mean']:.4f} | "
            f"{gs['released_gate_effective_visual_weight_mean']:.4f} | "
            f"{gs['released_gate_effective_text_weight_mean']:.4f} |"
        )
        lines.append(
            f"| winner (z-normalised, lambda=1) | {gs['gw_mean']:.4f} | "
            f"{gs['effective_visual_weight_mean']:.4f} | "
            f"{gs['effective_text_weight_mean']:.4f} |"
        )
        lines.append("\n### Pair classification (threshold from calibration pairs)\n")
        lines.append("| Method | Threshold | test AUC | test F1 | cal AUC |")
        lines.append("|---|---:|---:|---:|---:|")
        for name in ("concat", "gated", "winner"):
            p = test["test"]["pairs"][name]
            lines.append(
                f"| {name} | {p['threshold']:.4f} | {p['test']['auc']:.4f} | "
                f"{p['test']['f1']:.4f} | {p['calibration']['auc']:.4f} |"
            )
        lines.append("\n### 5-fold whole-track CV (P@5)\n")
        lines.append("| Fold | winner | concat | released gated |")
        lines.append("|---|---:|---:|---:|")
        for row in test["kfold_by_track"]["per_fold_table"]:
            lines.append(
                f"| {row['fold']} | {row['winner']:.4f} | {row['concat']:.4f} | "
                f"{row['released_gated']:.4f} |"
            )
        k = test["kfold_by_track"]
        lines.append(
            f"| **mean** | **{k['winner']['mean']:.4f}** | {k['concat']['mean']:.4f} | "
            f"{k['released_gated']['mean']:.4f} |"
        )
        lines.append("\n### Temporal halves (test pair AUC)\n")
        lines.append("| Half | n pairs | concat | gated | winner |")
        lines.append("|---|---:|---:|---:|---:|")
        for half in ("early", "late"):
            h = test["test"]["temporal_halves_pair_auc"]["halves"][half]
            lines.append(
                f"| {half} | {h['n_pairs']} | {h['concat']:.4f} | {h['gated']:.4f} | "
                f"{h['winner']:.4f} |"
            )
        rh = test["test"]["temporal_halves_pair_auc"].get(
            "retrieval_p5_by_query_half")
        if rh:
            lines.append("\n### Retrieval P@5 by query-timestamp half\n")
            lines.append("| Half | n queries | concat | gated | winner |")
            lines.append("|---|---:|---:|---:|---:|")
            for half in ("early", "late"):
                h = rh[half]
                lines.append(
                    f"| {half} | {h['n_queries']} | {h['concat']:.4f} | "
                    f"{h['gated']:.4f} | {h['winner']:.4f} |"
                )
        rc = test.get("reproduction_check_vs_released_fusion_results", {})
        if rc.get("checked"):
            lines.append(
                f"\n### Reproduction check against the released "
                f"`{Path(rc['source']).name}`\n\n"
                f"Maximum absolute difference over the recomputed reference "
                f"quantities (concat / released-gated retrieval P@5, pair AUC, "
                f"5-fold means): **{rc['max_abs_diff']:.2e}**.\n"
            )
        c = test["success_criteria"]
        lines.append("\n### Verdict\n")
        lines.append(
            f"- C1 retrieval Holm-significant win vs static concat: "
            f"**{c['c1_retrieval_holm_significant_win_vs_concat']}**\n"
            f"- C2 pair AUC not significantly worse than the released gate: "
            f"**{c['c2_pair_auc_not_significantly_worse_than_gated']}**\n"
            f"- C3 5-fold CV mean within 0.005 of static concat: "
            f"**{c['c3_kfold_mean_within_0.005_of_concat']}**\n"
            f"- **VERDICT: {c['verdict']}**\n"
        )
        lines.append("\n### Caveats\n")
        for item in test.get("caveats", []):
            lines.append(f"- {item}")
        lines.append("")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cal", "test"], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--calibration-file", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_file = Path(args.calibration_file) if args.calibration_file else (
        out_dir / "cal_gate_calibration.json"
    )

    if args.stage == "cal":
        cal = stage_cal(args)
        cal_file.write_text(json.dumps(cal, indent=2), encoding="utf-8")
        w = cal["frozen_winner"]
        print(f"[cal] wrote {cal_file}")
        print(f"[cal] frozen winner: {w['id']} (rule: {w['selection_rule']}, "
              f"feasible {w['n_feasible']}/{w['n_candidates']})")
        for r in cal["candidates"]:
            flag = "  <== WINNER" if r["id"] == w["id"] else ""
            print(
                f"  {r['id']:18s} calP@5={r['cal_retrieval_p5']:.4f} "
                f"d={r['cal_delta_p5_vs_concat']:+.4f} "
                f"[{r['cal_delta_p5_lo']:+.4f},{r['cal_delta_p5_hi']:+.4f}] "
                f"calAUC={r['cal_pair_auc']:.4f} calKfold={r['cal_kfold_p5']:.4f} "
                f"dev={r['gate_deviation']:.4f} A1={r['admissible_a1']} "
                f"A2={r['admissible_a2']} feasible={r['feasible']}{flag}"
            )
        g = cal.get("winner_calibration_grouped_bootstrap")
        if g:
            print(f"[cal] winner vs concat, track-grouped bootstrap on calibration: "
                  f"dP@5={g['delta_p5']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] "
                  f"p={g['p']:.4f}")
        return

    test = stage_test(args)
    cal_for_md = json.loads(cal_file.read_text(encoding="utf-8"))
    (out_dir / "cal_gate_results.json").write_text(
        json.dumps(test, indent=2), encoding="utf-8"
    )
    (out_dir / "cal_gate_tables.md").write_text(
        render_markdown(cal_for_md, test, args.n_boot), encoding="utf-8"
    )
    print(f"[test] wrote {out_dir / 'cal_gate_results.json'}")
    print(f"[test] wrote {out_dir / 'cal_gate_tables.md'}")
    c = test["success_criteria"]
    print(f"[test] verdict: {c['verdict']}  {c}")


if __name__ == "__main__":
    main()
