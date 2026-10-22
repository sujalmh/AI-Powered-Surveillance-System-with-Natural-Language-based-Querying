"""Round-2 (cal2) literature-motivated quality-gate search -- C2' fusion.

This script is the *second*, independent round of the quality-gate search
started by ``backend/scripts/eval_fusion_cal.py`` (round 1).  It does **not**
modify that script or its outputs; it imports its helpers and reuses them.

CONTEXT
--------
Round 1 (untracked ``backend/scripts/eval_fusion_cal.py``,
``results/cal_gate_calibration.json`` / ``results/cal_gate_results.json``)
frozen ``F2_lam1`` -- a quality gate computed from **z-normalised** metadata
qualities with lambda = 1, the normalisation constants estimated on
calibration detections only.  Round 1 reported, on the test split,
retrieval P@5 0.4696 (static concat 0.4274, delta +0.0422, Holm
significant), pair AUC 0.7263 (static concat 0.7263, parity) and 5-fold
whole-track CV 0.7018 (static concat 0.6994).

Round 1's own caveat is that its lambda grid started at lambda = 1 (the
boundary of the grid) and that the z-standardisation is a *linear* rescaling
of two quality proxies whose marginals are extremely discrete and skewed
(``qv`` takes 10 distinct values, 45 % of detections sit at 0.447; ``qt``
takes 4 distinct values).  Round 2 therefore asks whether a *published,
non-parametric* way of commensurating two heterogeneous quality signals
does better than the z-standardisation.

LITERATURE MOTIVATION (two families, at most two, as required)
--------------------------------------------------------------
**Family G1 -- "percentile-rank (PIT) normalised quality gate"** (``Grank``).
Score commensuration before fusion is the standard remedy for combining
signals with non-comparable score distributions; the probability integral
transform / empirical CDF is the canonical non-parametric choice:

* Bacellar & co, *Calibrated Fusion for Heterogeneous Graph-Vector Retrieval
  in Multi-Hop QA*, arXiv:2603.28886 (Sec. 3.2): "maps each retriever's
  scores to a common unit-free scale using percentile-rank normalization
  (the probability integral transform, PIT) before fusion"; percentile
  calibration is "directionally more robust than min-max normalization".
* Wu, Crestani & Bi, *Evaluating Score Normalization Methods in Data
  Fusion*, AIRS 2006, LNCS 4182, doi:10.1007/11880592_57: score
  normalisation is "an indispensable step to make scores ... comparable"
  before CombSUM/CombMNZ fusion.
* Montague & Aslam (2001): "normalization matters more than the fusion
  algorithm".
* Scheirer, Stehman, Vinyals, Krishnamurthy, *Robust Fusion: Extreme Value
  Theory for Recognition Score Normalization*, ECCV 2010: a score
  normalisation should be robust to "modeling errors, and parameter
  estimation errors".
* *Know When to Fuse* (arXiv:2409.01357): min-max / z-score / percentile
  scaling of hybrid-retrieval scores.

Applied here: ``u_v = F_cal(qv)`` and ``u_t = F_cal(qt)`` are the empirical
CDFs (mid-rank, tie aware) of the two quality proxies over **calibration
detections only**; the gate is ``gw = clip(sigmoid(lam * (u_v - u_t)),
0.05, 0.95)``.  No linear scale or shape assumption is made about either
marginal.

**Family G2 -- "uncertainty-aware relative-reliability gate with shrinkage
toward the static prior"** (``Ginv``).
Quality-aware *dynamic* fusion (weights that depend on the sample's own
quality) is the documented improvement over static late fusion, and the
weights are expected to be proportional to the *reliability* of each
modality rather than to an arbitrary constant:

* Zhang et al., *Provable Dynamic Fusion for Low-Quality Multimodal Data*,
  ICML 2023 (PMLR 202): dynamic fusion weights ``w_m(x) = alpha_m u_m(x) +
  beta_m`` vary per sample and "provably outperform" static fusion; the
  weight is a *linear, decreasing* function of the modality's uncertainty.
* Che, Nguyen & Luu, *Enhancing person re-identification via Uncertainty
  Feature Fusion Method and Auto-weighted Measure Combination*, Knowledge-
  Based Systems 307 (2025) 112737, arXiv:2405.01101: the combination
  weights are estimated on the *training* split only and are "independent
  of the test set".
* *Adaptive Multimodal Person Recognition* (arXiv:2512.14961): confidence-
  weighted fusion adapts the modality contribution to the reliability of
  each pathway.

Applied here: the visual branch's reliability proxy is ``qv`` and the text
branch's is ``qt``; the *relative* reliability of the two branches is the
log-ratio ``log qv - log qt`` (the inverse-variance / relative-reliability
form, invariant to a common multiplicative quality factor), standardised on
the calibration detections so that the gate is centred on the static
operating point; the gate is ``gw = clip(sigmoid(lam * z(log qv - log qt)),
0.05, 0.95)``.  Because ``qv`` and ``qt`` are bounded proxies whose
outliers (an empty scene frame gives ``qv = 1``) are exactly what pushed
round 1's z-gate into the ``[0.05, 0.95]`` saturation, the log form is
bounded by construction -- the shrinkage behaviour that the QMF formulation
prescribes.

Both families use only the released metadata quality proxies
(``estimate_qualities``) and are calibrated on calibration data only; no
new label, embedding or dataset enters.

PRE-REGISTRATION (fixed before any test-split number was computed here)
----------------------------------------------------------------------
Grids (calibration-only selection):

* ``Grank``: ``lam in {1, 2, 4, 8}``.
* ``Ginv``: ``lam in {0.5, 1, 2, 4}``.

PROVENANCE OF THE GRIDS (full disclosure, no test-split contamination)
----------------------------------------------------------------------
Before this pre-registration was frozen, a **calibration-only exploratory
sweep** of 27 gate configurations was run (stage ``sweep`` of this very
script; its output is ``results/cal2_exploratory_sweep.json`` and it is
reproduced in the report).  The sweep used **only** calibration detections,
calibration pairs, calibration-track retrieval and a calibration-track
5-fold CV -- the test split was not read by it.  The sweep showed that

* the percentile-rank (PIT) gate is the strongest family on calibration
  retrieval P@5 (its best member beats every z-standardised variant and
  the round-1 winner on all three calibration metrics), and
* the log-ratio ("relative reliability") gate is inadmissible on
  calibration retrieval P@5 for every lambda in its grid.

The two families and their lambda grids were therefore frozen as listed
above; the *selection rule* below -- not the sweep -- then chose the
winner, and the winner was evaluated **once** on the test split.  The full
sweep is reported so that no variant is hidden.

Round-1 ``F2_lam1`` is carried as a **reference row** in both tables (it is
already frozen and already evaluated on test by round 1); it is not a
selection candidate and it is not part of the Holm family of this round.

Selection rule (calibration data only), identical in spirit to round 1:

0. **A1** -- a configuration whose gate vector is identically 0.5 is the
   static baseline itself, not a variant; it is inadmissible.
   **A2** -- a configuration that does not beat the static baseline on
   calibration retrieval P@5 cannot satisfy the success criterion C1 on
   test, so admissible candidates require ``cal_retrieval_p5 >
   cal_retrieval_p5(concat)``.
1. **Feasible set** = admissible candidates with ``cal_kfold_p5 >=
   cal_kfold_p5(concat) - 0.005`` **and** ``cal_pair_auc >=
   cal_pair_auc(released gated) - 0.005``.
2. Among the feasible set, maximise ``cal_retrieval_p5``; ties (within 1e-9)
   are broken by higher ``cal_pair_auc``, then by the smaller gate deviation
   ``mean|0.5 - gw|``.
3. If the feasible set is empty, relax the CV filter first, then the AUC
   filter, then take the global maximum ``cal_retrieval_p5``.

The winner is frozen (with a SHA-1 spec hash) in stage CAL and evaluated
**once** in stage TEST.

Success criteria on the test split (all three required for IMPROVED):

* **C1** retrieval P@5: Holm-adjusted p for ``winner vs static concat`` <=
  0.05 with a positive delta;
* **C2** pair AUC: Holm does **not** reject ``winner vs released gated``
  (no significant loss against the released quality gate);
* **C3** whole-track 5-fold CV: ``mean_P@5(winner) >= mean_P@5(concat) -
  0.005``.

Holm families are the 2-member head-to-head families used by
``eval_fusion.py`` / ``eval_fusion_cal.py`` (winner vs concat, winner vs
released gate), one family per task, ``n_boot = 3000``.

Usage:
    python backend/scripts/eval_fusion_cal2.py --stage repro
    python backend/scripts/eval_fusion_cal2.py --stage sweep --n-boot 3000
    python backend/scripts/eval_fusion_cal2.py --stage cal  --n-boot 3000
    python backend/scripts/eval_fusion_cal2.py --stage test --n-boot 3000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval.stats import (
    format_p,
    paired_bootstrap_ci,
    paired_grouped_bootstrap_ci,
)
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
    select_pair_threshold,
    split_tracks_by_time,
)
from backend.scripts.eval_fusion_cal import (
    build_gated_gallery,
    compact_subset,
    make_retrieval_task,
    mean_p5,
    z_stats,
)

RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Pre-registered grids (see module docstring)
# ---------------------------------------------------------------------------
GRANK_LAMBDAS = [1.0, 2.0, 4.0, 8.0]
GINV_LAMBDAS = [0.5, 1.0, 2.0, 4.0]

KFOLD_TOL = 0.005      # CV non-inferiority tolerance
AUC_TOL = 0.005        # pair-AUC non-inferiority tolerance

SEED_RETRIEVAL = {"concat": 65, "gated": 67}
SEED_PAIR_AUC = {"concat": 61, "gated": 63}
SEED_CAL_BOOT = 71
SEED_CAL_GROUPED = 73

# The round-1 winner, carried as a reference row only.
F2_REFERENCE = {"id": "F2_lam1", "family": "F2", "lam": 1.0, "beta": 1.0}


# ---------------------------------------------------------------------------
# Gate construction
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ecdf(reference_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Tie-aware empirical CDF (mid-rank) of ``x`` w.r.t. a sorted reference.

    ``reference_sorted`` is the sorted vector of reference values (the
    calibration detections); the returned value is the mid-rank percentile
    ``(#ref < x + #ref <= x) / 2 / n``, i.e. the probability integral
    transform of the reference distribution evaluated at ``x``.
    """
    ref = np.asarray(reference_sorted, dtype=np.float64)
    xs = np.asarray(x, dtype=np.float64)
    left = np.searchsorted(ref, xs, side="left")
    right = np.searchsorted(ref, xs, side="right")
    return ((left + right) / 2.0) / float(len(ref))


def normaliser_constants(
    family: str,
    qv: np.ndarray,
    qt: np.ndarray,
    cal_idx: Sequence[int],
) -> Dict[str, Any]:
    """Constants of the gate's normaliser, fitted on CALIBRATION only."""
    qv_c = np.asarray(qv, dtype=np.float64)[list(cal_idx)]
    qt_c = np.asarray(qt, dtype=np.float64)[list(cal_idx)]
    if family == "Grank":
        return {
            "kind": "ecdf_midrank",
            "qv_ref_sorted": np.sort(qv_c).tolist(),
            "qt_ref_sorted": np.sort(qt_c).tolist(),
        }
    if family == "Ginv":
        lr = np.log(np.maximum(qv_c, 1e-9)) - np.log(np.maximum(qt_c, 1e-9))
        return {
            "kind": "z_of_log_ratio",
            "lr_mean": float(np.mean(lr)),
            "lr_sd": float(np.std(lr, ddof=0)),
        }
    raise ValueError(f"unknown family {family!r}")


def gate_from_constants(
    family: str,
    lam: float,
    consts: Dict[str, Any],
    qv: np.ndarray,
    qt: np.ndarray,
) -> np.ndarray:
    """Per-detection gate weight ``gw`` for one pre-registered variant."""
    if family == "Grank":
        u_v = _ecdf(np.asarray(consts["qv_ref_sorted"], dtype=np.float64), qv)
        u_t = _ecdf(np.asarray(consts["qt_ref_sorted"], dtype=np.float64), qt)
        drive = u_v - u_t
    elif family == "Ginv":
        lr = np.log(np.maximum(np.asarray(qv, dtype=np.float64), 1e-9)) - np.log(
            np.maximum(np.asarray(qt, dtype=np.float64), 1e-9)
        )
        drive = (lr - consts["lr_mean"]) / consts["lr_sd"]
    else:
        raise ValueError(f"unknown family {family!r}")
    gw = sigmoid(lam * drive)
    return np.clip(gw, 0.05, 0.95).astype(np.float64)


def all_configs() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    for lam in GRANK_LAMBDAS:
        cfgs.append({"id": f"Grank_lam{lam:g}", "family": "Grank", "lam": float(lam)})
    for lam in GINV_LAMBDAS:
        cfgs.append({"id": f"Ginv_lam{lam:g}", "family": "Ginv", "lam": float(lam)})
    return cfgs


# ---------------------------------------------------------------------------
# World (re-uses round-1 helpers; identical primary protocol)
# ---------------------------------------------------------------------------

def load_world():
    """Reproduce the eval_fusion.py primary (metadata-formed) world."""
    from backend.app.eval import subset
    from backend.app.services.fusion import MultimodalFusion

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


def f2_reference_gate(world) -> np.ndarray:
    """Round-1 winner gate (F2_lam1): z-normalised qualities, lambda = 1."""
    zs = z_stats(world["qv"], world["qt"], world["cal_idx"])
    z_v = (world["qv"] - zs["qv_mean"]) / zs["qv_sd"]
    z_t = (world["qt"] - zs["qt_mean"]) / zs["qt_sd"]
    return np.clip(sigmoid(1.0 * (z_v - z_t)), 0.05, 0.95).astype(np.float64)


# ---------------------------------------------------------------------------
# Stage REPRO: re-run of round 1's test stage and a leaf-by-leaf diff
# ---------------------------------------------------------------------------

def _flatten(obj, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}/{i}")
    else:
        yield prefix, obj


def stage_repro(args) -> Dict[str, Any]:
    """Compare the re-run of ``eval_fusion_cal.py --stage test`` with the
    round-1 file, leaf by leaf."""
    rerun_path = Path(args.rerun_file)
    released_path = Path(args.released_file)
    if not rerun_path.exists():
        raise SystemExit(f"missing rerun file {rerun_path}")
    if not released_path.exists():
        raise SystemExit(f"missing round-1 file {released_path}")
    a = json.loads(rerun_path.read_text(encoding="utf-8"))
    b = json.loads(released_path.read_text(encoding="utf-8"))
    fa, fb = dict(_flatten(a)), dict(_flatten(b))
    keys = sorted(set(fa) | set(fb))
    numeric_diffs: List[Dict[str, Any]] = []
    other_diffs: List[Dict[str, Any]] = []
    for k in keys:
        va, vb = fa.get(k, "<missing>"), fb.get(k, "<missing>")
        if isinstance(va, bool) or isinstance(vb, bool):
            same = va == vb
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            same = abs(float(va) - float(vb)) <= 0.0
            if not same:
                numeric_diffs.append({
                    "field": k, "rerun": va, "round1": vb,
                    "abs_diff": abs(float(va) - float(vb)),
                })
            continue
        else:
            same = va == vb
        if not same:
            other_diffs.append({"field": k, "rerun": va, "round1": vb})
    ignored = {"field": "non-numeric fields that may legitimately differ",
               "entries": other_diffs}
    out = {
        "stage": "reproduction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rerun_command": (
            "python backend/scripts/eval_fusion_cal.py --stage test "
            f"--n-boot {args.n_boot} --seeds {args.seeds} "
            f"--out-dir {args.rerun_out_dir} "
            f"--calibration-file {args.released_calibration}"
        ),
        "rerun_file": str(rerun_path),
        "round1_file": str(released_path),
        "n_leaf_fields_compared": len(keys),
        "n_numeric_fields_differing": len(numeric_diffs),
        "numeric_differences": numeric_diffs,
        "non_numeric_differences": other_diffs,
        "verdict": (
            "EXACT_MATCH" if not numeric_diffs else "MISMATCH"
        ),
        "key_numbers": {
            "winner_id": a.get("frozen_winner", {}).get("id"),
            "winner_spec_hash": a.get("frozen_winner_spec_hash"),
            "test_retrieval_p5_winner": a.get("test", {}).get("retrieval", {}).get("winner", {}).get("p5"),
            "test_retrieval_p5_concat": a.get("test", {}).get("retrieval", {}).get("concat", {}).get("p5"),
            "test_retrieval_p5_gated": a.get("test", {}).get("retrieval", {}).get("gated", {}).get("p5"),
            "test_pair_auc_winner": a.get("test", {}).get("pairs", {}).get("winner", {}).get("test", {}).get("auc"),
            "test_pair_auc_concat": a.get("test", {}).get("pairs", {}).get("concat", {}).get("test", {}).get("auc"),
            "test_pair_auc_gated": a.get("test", {}).get("pairs", {}).get("gated", {}).get("test", {}).get("auc"),
            "retrieval_delta_p5_winner_vs_concat": a.get("significance", {}).get("retrieval_p5_winner_vs", {}).get("concat", {}).get("delta_p5"),
            "retrieval_holm_p_winner_vs_concat": a.get("significance", {}).get("retrieval_p5_winner_vs", {}).get("concat", {}).get("holm_adjusted_p"),
            "pair_auc_delta_winner_vs_gated": a.get("significance", {}).get("pair_auc_winner_vs", {}).get("gated", {}).get("delta_auc"),
            "pair_auc_holm_p_winner_vs_gated": a.get("significance", {}).get("pair_auc_winner_vs", {}).get("gated", {}).get("holm_adjusted_p"),
            "kfold_mean_winner": a.get("kfold_by_track", {}).get("winner", {}).get("mean"),
            "kfold_mean_concat": a.get("kfold_by_track", {}).get("concat", {}).get("mean"),
            "kfold_mean_released_gated": a.get("kfold_by_track", {}).get("released_gated", {}).get("mean"),
            "verdict": a.get("success_criteria", {}).get("verdict"),
        },
        "note": ignored,
    }
    return out


# ---------------------------------------------------------------------------
# Calibration-only scaffold shared by the sweep and the selection stage
# ---------------------------------------------------------------------------

def cal_scaffold(world) -> Dict[str, Any]:
    """Everything a gate needs in order to be scored on CALIBRATION data."""
    dets, visual, text = world["dets"], world["visual"], world["text"]
    track_of, tracks = world["track_of"], world["tracks"]
    cal_idx, cal_pairs = world["cal_idx"], world["cal_pairs"]
    concat_gal = world["galleries"]["concat"]

    cal_qis, cal_cand, cal_rel = make_retrieval_task(dets, track_of, cal_idx)
    cal_labels = np.asarray([1 if p["same"] else 0 for p in cal_pairs],
                            dtype=np.int64)

    local_dets, local_track_of, local_tracks, keep_dets = compact_subset(
        dets, track_of, tracks, world["cal_tracks"]
    )
    local_visual = visual[keep_dets].astype(np.float32)
    local_text = text[keep_dets].astype(np.float32)
    local_concat = concat_gal[keep_dets].astype(np.float32)

    def cal_kfold(gallery: np.ndarray) -> Dict[str, Any]:
        lg = {
            "visual": local_visual,
            "text": local_text,
            "concat": local_concat,
            "gated": gallery[keep_dets].astype(np.float32),
            "zero_pad": local_concat,
            "norm_matched": local_concat,
            "learned_proj": local_concat,
        }
        kf = kfold_by_track(
            local_dets, local_visual, local_text, lg,
            local_track_of, local_tracks, n_folds=5,
        )
        return {
            "mean": kf["gated"]["mean"], "sd": kf["gated"]["sd"],
            "per_fold": [f["gated"] for f in kf["per_fold"]],
            "concat_mean": kf["concat"]["mean"],
            "concat_per_fold": [f["concat"] for f in kf["per_fold"]],
        }

    cal_p5_concat = {
        r["qid"]: r["p5"]
        for r in evaluate_method("concat", concat_gal, cal_qis, cal_cand, cal_rel)
    }
    return {
        "cal_qis": cal_qis,
        "cal_cand": cal_cand,
        "cal_rel": cal_rel,
        "cal_labels": cal_labels,
        "cal_p5_concat": cal_p5_concat,
        "cal_kfold": cal_kfold,
    }


def score_gate_on_calibration(
    world: Dict[str, Any],
    scaffold: Dict[str, Any],
    vid: str,
    gw: np.ndarray,
    n_boot: int,
) -> Dict[str, Any]:
    """Calibration retrieval P@5, pair AUC and 5-fold CV for one gate."""
    visual, text = world["visual"], world["text"]
    gal = build_gated_gallery(visual, text, gw)
    cal_scores = pair_scores_for_method(vid, gal, world["cal_pairs"])
    thr = select_pair_threshold(cal_scores, scaffold["cal_labels"])
    cal_pair = pair_metrics(cal_scores, scaffold["cal_labels"], thr)
    kf = scaffold["cal_kfold"](gal)
    cand_rows = evaluate_method(
        vid, gal, scaffold["cal_qis"], scaffold["cal_cand"], scaffold["cal_rel"])
    per_q = {r["qid"]: r["p5"] for r in cand_rows}
    qids = sorted(per_q)
    dm, lo, hi, p = paired_bootstrap_ci(
        [per_q[q] for q in qids],
        [scaffold["cal_p5_concat"][q] for q in qids],
        n_boot=n_boot, seed=SEED_CAL_BOOT,
    )
    return {
        "id": vid,
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
    }


# ---------------------------------------------------------------------------
# Stage SWEEP: the calibration-only exploratory sweep that informed the grids
# ---------------------------------------------------------------------------

def exploratory_gates(world: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    """All 27 configurations of the calibration-only exploratory sweep.

    Reported in full (calibration numbers only) so that no variant is
    hidden.  The sweep never reads the test split.
    """
    qv, qt = world["qv"], world["qt"]
    cal_idx = world["cal_idx"]
    zs = z_stats(qv, qt, cal_idx)
    z_v = (qv - zs["qv_mean"]) / zs["qv_sd"]
    z_t = (qt - zs["qt_mean"]) / zs["qt_sd"]
    z_vw = np.clip(z_v, -2.0, 2.0)
    z_tw = np.clip(z_t, -2.0, 2.0)

    consts_rank = normaliser_constants("Grank", qv, qt, cal_idx)
    consts_inv = normaliser_constants("Ginv", qv, qt, cal_idx)

    out: List[Tuple[str, np.ndarray]] = [
        ("concat_ref", np.full(len(qv), 0.5)),
        ("released_F1_lam4", np.clip(sigmoid(GATE_LAMBDA * (qv - qt)), 0.05, 0.95)),
        ("F2_lam1_round1", np.clip(sigmoid(1.0 * (z_v - z_t)), 0.05, 0.95)),
    ]
    for lam in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        out.append((f"F2z_lam{lam:g}",
                    np.clip(sigmoid(lam * (z_v - z_t)), 0.05, 0.95)))
    for lo in (0.15, 0.25, 0.35):
        out.append((f"F2z_lam1_clip{lo:g}",
                    np.clip(sigmoid(1.0 * (z_v - z_t)), lo, 1.0 - lo)))
    for lam in (0.5, 1.0, 2.0):
        out.append((f"F2w_lam{lam:g}",
                    np.clip(sigmoid(lam * (z_vw - z_tw)), 0.05, 0.95)))
    for lam in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        out.append((f"Grank_lam{lam:g}",
                    gate_from_constants("Grank", lam, consts_rank, qv, qt)))
    for lam in (2.0, 3.0):
        out.append((f"Grank_lam{lam:g}_clip0.15",
                    np.clip(gate_from_constants("Grank", lam, consts_rank, qv, qt),
                            0.15, 0.85)))
    for lam in (0.5, 1.0, 2.0, 3.0):
        out.append((f"Ginv_lam{lam:g}",
                    gate_from_constants("Ginv", lam, consts_inv, qv, qt)))
    return out


def stage_sweep(args) -> Dict[str, Any]:
    """Calibration-only exploratory sweep (no test split is read)."""
    world = load_world()
    scaffold = cal_scaffold(world)
    rows: List[Dict[str, Any]] = []
    for vid, gw in exploratory_gates(world):
        row = score_gate_on_calibration(
            world, scaffold, vid, gw, args.n_boot)
        row["family"] = (
            "static" if vid == "concat_ref"
            else "F1-released" if vid.startswith("released")
            else "F2-z" if vid.startswith("F2z")
            else "F2-winsorised-z" if vid.startswith("F2w")
            else "Grank-PIT" if vid.startswith("Grank")
            else "Ginv-log-ratio" if vid.startswith("Ginv")
            else "round1"
        )
        rows.append(row)
    base = [r for r in rows if r["id"] == "concat_ref"][0]
    for r in rows:
        r["cal_delta_p5_vs_concat"] = r["cal_retrieval_p5"] - base["cal_retrieval_p5"]
        r["cal_delta_kfold_vs_concat"] = (
            r["cal_kfold_p5"] - base["cal_kfold_p5"])
    out = {
        "stage": "exploratory_sweep",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python backend/scripts/eval_fusion_cal2.py --stage sweep "
            f"--n-boot {args.n_boot}"
        ),
        "note": (
            "Calibration-only exploratory sweep that preceded the round-2 "
            "pre-registration. No test-split quantity is computed or read "
            "here: only calibration detections, calibration pairs, "
            "calibration-track retrieval and a calibration-track 5-fold CV."
        ),
        "data": {
            "n_calibration_detections": len(world["cal_idx"]),
            "n_calibration_queries": len(scaffold["cal_qis"]),
            "n_calibration_pairs": len(world["cal_pairs"]),
            "n_calibration_tracks": len(world["cal_tracks"]),
        },
        "rows": rows,
    }
    return out


# ---------------------------------------------------------------------------
# Stage CAL
# ---------------------------------------------------------------------------

def stage_cal(args) -> Dict[str, Any]:
    world = load_world()
    dets, visual, text = world["dets"], world["visual"], world["text"]
    track_of, tracks = world["track_of"], world["tracks"]
    cal_idx, cal_pairs = world["cal_idx"], world["cal_pairs"]
    concat_gal = world["galleries"]["concat"]
    released_gated_gal = world["galleries"]["gated"]
    qv, qt = world["qv"], world["qt"]
    scaffold = cal_scaffold(world)
    cal_qis = scaffold["cal_qis"]
    cal_cand = scaffold["cal_cand"]
    cal_rel = scaffold["cal_rel"]
    cal_labels = scaffold["cal_labels"]
    cal_p5_concat = scaffold["cal_p5_concat"]
    cal_kfold = scaffold["cal_kfold"]

    # sanity: the static gate must reproduce the concat gallery bit-for-bit.
    gw_static = np.full(len(dets), 0.5)
    max_dev_static = float(np.max(np.abs(
        build_gated_gallery(visual, text, gw_static) - concat_gal)))
    if max_dev_static > 1e-6:
        raise RuntimeError(f"static gate deviates from concat by {max_dev_static}")

    # ---- reference rows ----------------------------------------------------
    refs: Dict[str, Dict[str, Any]] = {}
    for name, gal in (("concat", concat_gal), ("gated", released_gated_gal)):
        cal_scores = pair_scores_for_method(name, gal, cal_pairs)
        refs[name] = {
            "cal_retrieval_p5": mean_p5(gal, cal_qis, cal_cand, cal_rel),
            "cal_pair_auc": float(auc_score(cal_scores, cal_labels)),
            "cal_kfold": cal_kfold(gal),
            "gate_deviation": 0.0 if name == "concat" else float(
                np.mean(np.abs(0.5 - np.clip(
                    sigmoid(GATE_LAMBDA * (qv - qt)), 0.05, 0.95)))),
        }
    # round-1 winner carried as a reference row (not a selection candidate)
    gw_f2 = f2_reference_gate(world)
    gal_f2 = build_gated_gallery(visual, text, gw_f2)
    s_f2 = pair_scores_for_method("F2_lam1", gal_f2, cal_pairs)
    thr_f2 = select_pair_threshold(s_f2, cal_labels)
    refs["F2_lam1"] = {
        "cal_retrieval_p5": mean_p5(gal_f2, cal_qis, cal_cand, cal_rel),
        "cal_pair_auc": float(pair_metrics(s_f2, cal_labels, thr_f2)["auc"]),
        "cal_kfold": cal_kfold(gal_f2),
        "gate_deviation": float(np.mean(np.abs(0.5 - gw_f2))),
        "note": "round-1 frozen winner, reference only (not selectable)",
    }

    # implementation cross-check: the recomputed round-1 winner must match
    # the calibration numbers that round 1 recorded for it.
    round1_cross: Dict[str, Any] = {"checked": False}
    r1_cal_path = RESULTS / "cal_gate_calibration.json"
    if r1_cal_path.exists():
        r1 = json.loads(r1_cal_path.read_text(encoding="utf-8"))
        r1_row = next(
            (c for c in r1.get("candidates", []) if c["id"] == "F2_lam1"), None)
        if r1_row is not None:
            diffs = {
                "cal_retrieval_p5": abs(
                    refs["F2_lam1"]["cal_retrieval_p5"]
                    - float(r1_row["cal_retrieval_p5"])),
                "cal_pair_auc": abs(
                    refs["F2_lam1"]["cal_pair_auc"]
                    - float(r1_row["cal_pair_auc"])),
                "cal_kfold_p5": abs(
                    refs["F2_lam1"]["cal_kfold"]["mean"]
                    - float(r1_row["cal_kfold_p5"])),
            }
            round1_cross = {
                "checked": True,
                "source": str(r1_cal_path.relative_to(ROOT)),
                "reference_variant": "F2_lam1",
                "recomputed": {
                    "cal_retrieval_p5": refs["F2_lam1"]["cal_retrieval_p5"],
                    "cal_pair_auc": refs["F2_lam1"]["cal_pair_auc"],
                    "cal_kfold_p5": refs["F2_lam1"]["cal_kfold"]["mean"],
                },
                "round1": {
                    "cal_retrieval_p5": float(r1_row["cal_retrieval_p5"]),
                    "cal_pair_auc": float(r1_row["cal_pair_auc"]),
                    "cal_kfold_p5": float(r1_row["cal_kfold_p5"]),
                },
                "abs_diffs": diffs,
                "max_abs_diff": float(max(diffs.values())),
            }

    # ---- candidate rows ----------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for cfg in all_configs():
        consts = normaliser_constants(cfg["family"], qv, qt, cal_idx)
        gw = gate_from_constants(cfg["family"], cfg["lam"], consts, qv, qt)
        row = score_gate_on_calibration(
            world, scaffold, cfg["id"], gw, args.n_boot)
        row["family"] = cfg["family"]
        row["lam"] = cfg["lam"]
        row["normaliser"] = consts
        rows.append(row)

    # ---- pre-registered selection -----------------------------------------
    concat_kfold = refs["concat"]["cal_kfold"]["mean"]
    concat_p5 = refs["concat"]["cal_retrieval_p5"]
    gated_auc = refs["gated"]["cal_pair_auc"]
    for r in rows:
        r["admissible_a1"] = bool(r["gate_deviation"] > 1e-12)
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

    # track-grouped bootstrap on calibration retrieval P@5 (winner vs concat)
    consts_w = winner["normaliser"]
    gw_w = gate_from_constants(
        winner["family"], winner["lam"], consts_w, qv, qt)
    winner_gal = build_gated_gallery(visual, text, gw_w)
    winner_rows = evaluate_method("winner", winner_gal, cal_qis, cal_cand, cal_rel)
    winner_p5 = {r["qid"]: r["p5"] for r in winner_rows}
    g_qids = sorted(winner_p5)
    track_of_query = {q: track_of[q] for q in cal_qis}
    gdm, glo, ghi, gp = paired_grouped_bootstrap_ci(
        [winner_p5[q] for q in g_qids],
        [cal_p5_concat[q] for q in g_qids],
        [f"cal_t{track_of_query[q]:05d}" for q in g_qids],
        n_boot=args.n_boot, seed=SEED_CAL_GROUPED,
    )

    frozen = {
        "id": winner["id"],
        "family": winner["family"],
        "lam": winner["lam"],
        "normaliser": winner["normaliser"],
        "selection_rule": selection_rule,
        "relaxed_filters": relaxed,
        "n_candidates": len(rows),
        "n_admissible": len(admissible),
        "n_feasible": len(feasible),
    }
    frozen["spec_hash"] = hashlib.sha1(
        json.dumps(
            {k: frozen[k] for k in ("id", "family", "lam", "normaliser")},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    out = {
        "stage": "calibration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python backend/scripts/eval_fusion_cal2.py --stage cal "
            f"--n-boot {args.n_boot}"
        ),
        "pre_registration": {
            "Grank_lambdas": GRANK_LAMBDAS,
            "Ginv_lambdas": GINV_LAMBDAS,
            "kfold_tolerance": KFOLD_TOL,
            "auc_tolerance": AUC_TOL,
            "selection_rule": (
                "A1: a configuration with gate deviation 0 is the static "
                "baseline itself and is not admissible; A2: admissible "
                "candidates must beat the static baseline on calibration "
                "retrieval P@5; S1: feasible = (cal_kfold_p5 >= "
                "cal_kfold_p5(concat) - 0.005) and (cal_pair_auc >= "
                "cal_pair_auc(released gated) - 0.005); S2: argmax "
                "cal_retrieval_p5, ties -> cal_pair_auc -> smallest gate "
                "deviation; S3: if empty relax the CV filter, then the AUC "
                "filter, then take the global maximum cal_retrieval_p5"
            ),
            "families": {
                "Grank": (
                    "percentile-rank (PIT) normalised quality gate: "
                    "gw = clip(sigmoid(lam*(F_cal(qv) - F_cal(qt))), .05, .95); "
                    "F_cal = tie-aware mid-rank empirical CDF over calibration "
                    "detections only. Motivated by arXiv:2603.28886 (PIT "
                    "score commensuration), Wu et al. AIRS 2006 "
                    "(doi:10.1007/11880592_57), Montague & Aslam 2001, "
                    "Scheirer et al. ECCV 2010."
                ),
                "Ginv": (
                    "uncertainty-aware relative-reliability gate: "
                    "gw = clip(sigmoid(lam*z(log qv - log qt)), .05, .95), "
                    "z fitted on calibration detections only. Motivated by "
                    "Zhang et al. ICML 2023 (QMF, dynamic uncertainty-aware "
                    "fusion weights), Che et al. KBS 2025 (auto-weighted "
                    "measure combination fitted on train only), "
                    "arXiv:2512.14961 (confidence-weighted fusion)."
                ),
            },
            "excluded": (
                "F2_lam1 (round-1 winner) is carried as a reference row only: "
                "it was already frozen and already evaluated on test by round "
                "1, so it is not a selection candidate of this round."
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
        },
        "sanity_checks": {
            "static_gate_max_abs_dev_vs_concat": max_dev_static,
            "round1_cross_check": round1_cross,
        },
        "references": refs,
        "candidates": rows,
        "winner_calibration_grouped_bootstrap": {
            "delta_p5": float(gdm), "lo": float(glo), "hi": float(ghi), "p": float(gp),
            "note": "track-grouped paired bootstrap, winner vs concat, calibration",
        },
        "frozen_winner": frozen,
    }
    return out


# ---------------------------------------------------------------------------
# Stage TEST
# ---------------------------------------------------------------------------

def stage_test(args) -> Dict[str, Any]:
    cal_path = (Path(args.calibration_file) if args.calibration_file
                else (Path(args.out_dir) / "cal2_calibration.json"))
    if not cal_path.exists():
        raise SystemExit(
            f"missing frozen calibration file {cal_path}; run --stage cal first"
        )
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    frozen = cal["frozen_winner"]
    spec_hash = hashlib.sha1(
        json.dumps(
            {k: frozen[k] for k in ("id", "family", "lam", "normaliser")},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if spec_hash != frozen["spec_hash"]:
        raise SystemExit("frozen winner spec hash mismatch")

    world = load_world()
    dets, visual, text = world["dets"], world["visual"], world["text"]
    track_of, tracks = world["track_of"], world["tracks"]
    test_idx, test_pairs = world["test_idx"], world["test_pairs"]
    concat_gal = world["galleries"]["concat"]
    released_gated_gal = world["galleries"]["gated"]
    qv, qt = world["qv"], world["qt"]

    # recompute the normaliser constants from the calibration detections and
    # check that they reproduce the frozen ones (guards against drift).
    consts = normaliser_constants(frozen["family"], qv, qt, world["cal_idx"])
    if json.dumps(consts, sort_keys=True) != json.dumps(
        frozen["normaliser"], sort_keys=True
    ):
        raise SystemExit("recomputed normaliser differs from the frozen one")
    gw = gate_from_constants(frozen["family"], frozen["lam"], consts, qv, qt)
    winner_gal = build_gated_gallery(visual, text, gw)
    gw_released = np.clip(sigmoid(GATE_LAMBDA * (qv - qt)), 0.05, 0.95)
    gw_f2 = f2_reference_gate(world)
    f2_gal = build_gated_gallery(visual, text, gw_f2)

    # ---------------- retrieval on test -------------------------------------
    qis, cand, rel = make_retrieval_task(dets, track_of, test_idx)
    rows = {
        "concat": evaluate_method("concat", concat_gal, qis, cand, rel),
        "gated": evaluate_method("gated", released_gated_gal, qis, cand, rel),
        "winner": evaluate_method("winner", winner_gal, qis, cand, rel),
        "F2_lam1": evaluate_method("F2_lam1", f2_gal, qis, cand, rel),
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

    # Secondary: winner vs the round-1 winner F2_lam1 (not in the Holm family).
    dm2, lo2, hi2, p2 = paired_bootstrap_ci(
        [per_query["winner"][q] for q in qids],
        [per_query["F2_lam1"][q] for q in qids],
        n_boot=args.n_boot, seed=79,
    )
    vs_f2 = {
        "delta_p5": float(dm2), "lo": float(lo2), "hi": float(hi2), "p": float(p2),
        "note": "secondary, not part of the pre-registered Holm family",
    }

    # Secondary statistic: paired bootstrap resampling whole pseudo-tracks.
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
    pair_scores: Dict[str, np.ndarray] = {}
    pair_info: Dict[str, Any] = {}
    for name, gal in (
        ("concat", concat_gal),
        ("gated", released_gated_gal),
        ("winner", winner_gal),
        ("F2_lam1", f2_gal),
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

    dm3, lo3, hi3, p3 = paired_auc_bootstrap(
        pair_scores["winner"], pair_scores["F2_lam1"], test_labels,
        n_boot=args.n_boot, seed=69,
    )
    pair_vs_f2 = {
        "delta_auc": float(dm3), "lo": float(lo3), "hi": float(hi3), "p": float(p3),
        "note": "secondary, not part of the pre-registered Holm family",
    }

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
    halves: Dict[str, Any] = {"median_gallery_ts": t_median, "halves": {}}
    for half, mask in (("early", gal_ts <= t_median), ("late", gal_ts > t_median)):
        idx = np.flatnonzero(mask)
        lab_h = test_labels[idx]
        entry = {"n_pairs": int(len(idx))}
        for name in ("concat", "gated", "winner", "F2_lam1"):
            entry[name] = float(auc_score(np.asarray(pair_scores[name])[idx], lab_h))
        halves["halves"][half] = entry

    q_ts = np.asarray(
        [dets[q]["timestamp"].timestamp() for q in qids], dtype=np.float64
    )
    q_median = float(np.median(q_ts))
    halves["median_query_ts"] = q_median
    halves["retrieval_p5_by_query_half"] = {}
    for half, mask in (("early", q_ts <= q_median), ("late", q_ts > q_median)):
        sel = np.flatnonzero(mask)
        entry = {"n_queries": int(len(sel))}
        for name in ("concat", "gated", "winner", "F2_lam1"):
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
                    "abs_diff": float(abs(
                        retrieval_point[name]["p5"]
                        - rel["test"]["retrieval"][name]["p5"])),
                }
                for name in ("concat", "gated")
            },
            "pair_auc_test": {
                name: {
                    "recomputed": float(pair_info[name]["test"]["auc"]),
                    "released": float(rel["test"]["pairs"][name]["test"]["auc"]),
                    "abs_diff": float(abs(
                        pair_info[name]["test"]["auc"]
                        - rel["test"]["pairs"][name]["test"]["auc"])),
                }
                for name in ("concat", "gated")
            },
            "kfold_mean": {
                name: {
                    "recomputed": float(kfold_out[key]["mean"]),
                    "released": float(rel["kfold_by_track"][name]["mean"]),
                    "abs_diff": float(abs(
                        kfold_out[key]["mean"]
                        - rel["kfold_by_track"][name]["mean"])),
                }
                for name, key in (("concat", "concat"), ("gated", "released_gated"))
            },
        }
        reproduction["max_abs_diff"] = float(max(
            v["abs_diff"]
            for grp in ("retrieval_p5", "pair_auc_test", "kfold_mean")
            for v in reproduction[grp].values()
        ))

    # implementation cross-check: the recomputed round-1 winner F2_lam1 must
    # reproduce the test numbers that round 1 recorded for it.  This proves
    # that the round-2 pipeline is equivalent to the round-1 one.
    round1_cross: Dict[str, Any] = {"checked": False}
    r1_path = RESULTS / "cal_gate_results.json"
    if r1_path.exists():
        r1 = json.loads(r1_path.read_text(encoding="utf-8"))
        diffs: Dict[str, float] = {}
        pairs = (
            ("retrieval_p5_winner",
             retrieval_point["F2_lam1"]["p5"],
             float(r1["test"]["retrieval"]["winner"]["p5"])),
            ("pair_auc_test_winner",
             float(pair_info["F2_lam1"]["test"]["auc"]),
             float(r1["test"]["pairs"]["winner"]["test"]["auc"])),
            ("pair_auc_test_concat",
             float(pair_info["concat"]["test"]["auc"]),
             float(r1["test"]["pairs"]["concat"]["test"]["auc"])),
            ("retrieval_p5_concat",
             retrieval_point["concat"]["p5"],
             float(r1["test"]["retrieval"]["concat"]["p5"])),
            ("kfold_mean_concat",
             float(kfold_out["concat"]["mean"]),
             float(r1["kfold_by_track"]["concat"]["mean"])),
        )
        for name, mine, theirs in pairs:
            diffs[name] = abs(float(mine) - float(theirs))
        round1_cross = {
            "checked": True,
            "source": str(r1_path.relative_to(ROOT)),
            "note": (
                "F2_lam1 is recomputed here with the round-2 code path; the "
                "differences against the round-1 file must be exactly zero."
            ),
            "abs_diffs": diffs,
            "max_abs_diff": float(max(diffs.values())),
        }

    # ---------------- verdict -----------------------------------------------
    c1 = bool(
        retrieval_sig["concat"]["delta_p5"] > 0
        and retrieval_sig["concat"]["holm_adjusted_p"] <= 0.05
    )
    c2 = bool(not pair_sig["gated"]["holm_reject"])
    c2b = bool(not pair_sig["concat"]["holm_reject"])
    c3 = bool(
        kfold_out["winner"]["mean"]
        >= kfold_out["concat"]["mean"] - KFOLD_TOL
    )
    verdict = "IMPROVED" if (c1 and c2 and c3) else "NULL"

    out = {
        "stage": "test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python backend/scripts/eval_fusion_cal2.py --stage test "
            f"--n-boot {args.n_boot}"
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
                "f2_lam1_gw_mean": float(np.mean(gw_f2)),
                "f2_lam1_gate_deviation": float(np.mean(np.abs(0.5 - gw_f2))),
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
            "secondary_vs_round1_winner_F2_lam1": {
                "retrieval_p5": vs_f2, "pair_auc": pair_vs_f2,
            },
        },
        "kfold_by_track": kfold_out,
        "reproduction_check_vs_released_fusion_results": reproduction,
        "round1_cross_check": round1_cross,
        "caveats": [
            "The calibration split is small (70 pseudo-tracks, 492 retrieval "
            "queries); the track-grouped bootstrap on the winner's calibration "
            "delta is the honest uncertainty statement for the selection.",
            "F2_lam1 (the round-1 winner) is reported here as a reference "
            "row only; it was frozen and evaluated on test by round 1 "
            "(results/cal_gate_results.json) and was NOT re-selected here.",
            "The frozen normaliser constants are estimated on calibration "
            "detections only and applied unchanged to every detection; no "
            "test-split label, score or threshold enters the gate.",
            "Same-track labels are deterministic pseudo-labels formed from "
            "detector/attribute metadata, not human identities; the benchmark "
            "is the released single-camera subset, not the paper's "
            "312k-descriptor / 4-camera identity benchmark.",
        ],
        "success_criteria": {
            "c1_retrieval_holm_significant_win_vs_concat": c1,
            "c2_pair_auc_not_significantly_worse_than_gated": c2,
            "c2b_pair_auc_not_significantly_worse_than_concat": c2b,
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
    lines.append("# Round-2 literature-motivated quality-gate search (cal2)\n")
    lines.append(
        "Pre-registered two-stage protocol "
        "(`backend/scripts/eval_fusion_cal2.py`). Stage CAL scored every "
        "variant on **calibration data only**; exactly one winner was frozen "
        "and evaluated **once** on the test split. The round-1 winner "
        "`F2_lam1` appears as a reference row only.\n"
    )
    if cal is not None:
        lines.append("## Stage CAL: candidate table (calibration data only)\n")
        lines.append(
            "| Variant | Family | lam | cal P@5 | d P@5 vs concat [95% CI] | "
            "cal pair AUC | cal 5-fold P@5 | d 5-fold | gate dev. | A1 | A2 | feasible |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|")
        refs = cal["references"]
        for name in ("concat", "gated", "F2_lam1"):
            r = refs[name]
            lines.append(
                f"| {name} (reference) | - | - | {r['cal_retrieval_p5']:.4f} | "
                f"{r['cal_retrieval_p5'] - refs['concat']['cal_retrieval_p5']:+.4f} | "
                f"{r['cal_pair_auc']:.4f} | {r['cal_kfold']['mean']:.4f} | "
                f"{r['cal_kfold']['mean'] - refs['concat']['cal_kfold']['mean']:+.4f} | "
                f"{r.get('gate_deviation', 0.0):.4f} | - | - | - |"
            )
        for r in cal["candidates"]:
            mark = " **(WINNER)**" if r["id"] == cal["frozen_winner"]["id"] else ""
            lines.append(
                f"| {r['id']}{mark} | {r['family']} | {r['lam']:g} | "
                f"{r['cal_retrieval_p5']:.4f} | {r['cal_delta_p5_vs_concat']:+.4f} "
                f"[{r['cal_delta_p5_lo']:+.4f}, {r['cal_delta_p5_hi']:+.4f}] | "
                f"{r['cal_pair_auc']:.4f} | {r['cal_kfold_p5']:.4f} | "
                f"{r['cal_kfold_p5'] - refs['concat']['cal_kfold']['mean']:+.4f} | "
                f"{r['gate_deviation']:.4f} | {r['admissible_a1']} | "
                f"{r['admissible_a2']} | {r['feasible']} |"
            )
        g = cal.get("winner_calibration_grouped_bootstrap")
        if g:
            lines.append(
                f"\nWinner vs static concat on calibration, track-grouped "
                f"paired bootstrap: dP@5 = {g['delta_p5']:+.4f} "
                f"[{g['lo']:+.4f}, {g['hi']:+.4f}], p = {g['p']:.4f}.\n"
            )
        fw = cal["frozen_winner"]
        lines.append(
            f"\nAdmissible candidates: {fw['n_admissible']} / {fw['n_candidates']}; "
            f"feasible: {fw['n_feasible']}; rule applied: "
            f"`{fw['selection_rule']}`.  \n"
            f"**Frozen winner: `{fw['id']}`** (spec hash `{fw['spec_hash'][:12]}`).\n"
        )
    if test is not None:
        w = test["frozen_winner"]
        lines.append("\n## Stage TEST: frozen winner\n")
        lines.append(
            f"Variant `{w['id']}` (family {w['family']}, lam={w['lam']:g}); "
            f"calibration file `{Path(test['calibration_file']).name}`, spec hash "
            f"`{test['frozen_winner_spec_hash'][:12]}`.\n"
        )
        lines.append("### Retrieval P@5 (test split)\n")
        lines.append("| Method | P@5 | R@10 | F1@10 |")
        lines.append("|---|---:|---:|---:|")
        for name in ("concat", "gated", "winner", "F2_lam1"):
            s = test["test"]["retrieval"][name]
            lines.append(
                f"| {name} | {s['p5']:.4f} | {s['r10']:.4f} | {s['f1']:.4f} |"
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
        sec = test["significance"].get("secondary_vs_round1_winner_F2_lam1")
        if sec:
            lines.append(
                "\nSecondary (not in the Holm family): winner vs the round-1 "
                "winner `F2_lam1`.\n"
            )
            lines.append("| Task | Delta [95% CI] | raw p |")
            lines.append("|---|---:|---:|")
            lines.append(
                f"| retrieval P@5 | {sec['retrieval_p5']['delta_p5']:+.4f} "
                f"[{sec['retrieval_p5']['lo']:+.4f}, "
                f"{sec['retrieval_p5']['hi']:+.4f}] | "
                f"{format_p(sec['retrieval_p5']['p'], n_boot)} |"
            )
            lines.append(
                f"| pair AUC | {sec['pair_auc']['delta_auc']:+.4f} "
                f"[{sec['pair_auc']['lo']:+.4f}, {sec['pair_auc']['hi']:+.4f}] | "
                f"{format_p(sec['pair_auc']['p'], n_boot)} |"
            )
        gb = test["significance"].get(
            "track_grouped_bootstrap_retrieval_p5_winner_vs")
        if gb:
            lines.append(
                "\nSecondary statistic (not part of the pre-registered "
                "decision): paired bootstrap resampling whole pseudo-tracks.\n"
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
            f"| round-1 winner F2_lam1 (z-standardised, lambda=1) | "
            f"{gs['f2_lam1_gw_mean']:.4f} | "
            f"{1.2 * gs['f2_lam1_gw_mean']:.4f} | "
            f"{0.8 * (1.0 - gs['f2_lam1_gw_mean']):.4f} |"
        )
        lines.append(
            f"| round-2 winner `{w['id']}` (percentile-rank, lambda={w['lam']:g}) | "
            f"{gs['gw_mean']:.4f} | "
            f"{gs['effective_visual_weight_mean']:.4f} | "
            f"{gs['effective_text_weight_mean']:.4f} |"
        )
        lines.append("\n### Pair classification (threshold from calibration pairs)\n")
        lines.append("| Method | Threshold | test AUC | test F1 | cal AUC |")
        lines.append("|---|---:|---:|---:|---:|")
        for name in ("concat", "gated", "winner", "F2_lam1"):
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
        lines.append("| Half | n pairs | concat | gated | winner | F2_lam1 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for half in ("early", "late"):
            h = test["test"]["temporal_halves_pair_auc"]["halves"][half]
            lines.append(
                f"| {half} | {h['n_pairs']} | {h['concat']:.4f} | {h['gated']:.4f} | "
                f"{h['winner']:.4f} | {h['F2_lam1']:.4f} |"
            )
        rh = test["test"]["temporal_halves_pair_auc"].get("retrieval_p5_by_query_half")
        if rh:
            lines.append("\n### Retrieval P@5 by query-timestamp half\n")
            lines.append("| Half | n queries | concat | gated | winner | F2_lam1 |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for half in ("early", "late"):
                h = rh[half]
                lines.append(
                    f"| {half} | {h['n_queries']} | {h['concat']:.4f} | "
                    f"{h['gated']:.4f} | {h['winner']:.4f} | {h['F2_lam1']:.4f} |"
                )
        rc = test.get("reproduction_check_vs_released_fusion_results", {})
        if rc.get("checked"):
            lines.append(
                f"\n### Reproduction check against the released "
                f"`{Path(rc['source']).name}`\n\n"
                f"Maximum absolute difference over the recomputed reference "
                f"quantities: **{rc['max_abs_diff']:.2e}**.\n"
            )
        c = test["success_criteria"]
        lines.append("\n### Verdict\n")
        lines.append(
            f"- C1 retrieval Holm-significant win vs static concat: "
            f"**{c['c1_retrieval_holm_significant_win_vs_concat']}**\n"
            f"- C2 pair AUC not significantly worse than the released gate: "
            f"**{c['c2_pair_auc_not_significantly_worse_than_gated']}**\n"
            f"- C2b pair AUC not significantly worse than static concat: "
            f"**{c['c2b_pair_auc_not_significantly_worse_than_concat']}**\n"
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


def render_sweep_markdown(sweep: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Round-2 exploratory calibration sweep (calibration only)\n")
    lines.append(
        "Every gate configuration explored before the round-2 pre-registration "
        "was frozen.  **No test-split quantity is computed or read here**: the "
        "sweep uses calibration detections, calibration pairs, calibration-track "
        "retrieval and a calibration-track 5-fold CV only.  It is reported in "
        "full so that no variant is hidden.\n"
    )
    lines.append(
        "| Variant | Family | cal P@5 | d P@5 vs concat | cal pair AUC | "
        "cal 5-fold P@5 | d 5-fold vs concat | gate dev. |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sweep["rows"]:
        lines.append(
            f"| {r['id']} | {r['family']} | {r['cal_retrieval_p5']:.4f} | "
            f"{r['cal_delta_p5_vs_concat']:+.4f} | {r['cal_pair_auc']:.4f} | "
            f"{r['cal_kfold_p5']:.4f} | {r['cal_delta_kfold_vs_concat']:+.4f} | "
            f"{r['gate_deviation']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["repro", "sweep", "cal", "test", "render"],
                    required=True)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--calibration-file", default=None)
    ap.add_argument("--rerun-file", default=str(
        RESULTS / "cal2_f2_lam1_rerun.json"))
    ap.add_argument("--released-file", default=str(
        RESULTS / "cal_gate_results.json"))
    ap.add_argument("--released-calibration", default=str(
        RESULTS / "cal_gate_calibration.json"))
    ap.add_argument("--rerun-out-dir", default=str(
        RESULTS.parent / "backend" / "scripts" / ".cal2_rerun"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "repro":
        out = stage_repro(args)
        path = out_dir / "cal2_reproduction.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[repro] wrote {path}")
        print(f"[repro] verdict: {out['verdict']} "
              f"({out['n_leaf_fields_compared']} leaf fields compared, "
              f"{out['n_numeric_fields_differing']} numeric differences)")
        for d in out["non_numeric_differences"]:
            print(f"  non-numeric difference: {d['field']}: "
                  f"{d['rerun']!r} vs {d['round1']!r}")
        return

    cal_file = Path(args.calibration_file) if args.calibration_file else (
        out_dir / "cal2_calibration.json"
    )

    if args.stage == "sweep":
        sweep = stage_sweep(args)
        path = out_dir / "cal2_exploratory_sweep.json"
        path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
        md_path = out_dir / "cal2_exploratory_sweep.md"
        md_path.write_text(render_sweep_markdown(sweep), encoding="utf-8")
        print(f"[sweep] wrote {path}")
        print(f"[sweep] wrote {md_path}")
        print(f"{'id':24s} {'calP@5':>8s} {'dP@5':>8s} {'calAUC':>8s} "
              f"{'calKfold':>9s} {'dKfold':>8s} {'dev':>7s}")
        for r in sweep["rows"]:
            print(f"{r['id']:24s} {r['cal_retrieval_p5']:8.4f} "
                  f"{r['cal_delta_p5_vs_concat']:+8.4f} {r['cal_pair_auc']:8.4f} "
                  f"{r['cal_kfold_p5']:9.4f} {r['cal_delta_kfold_vs_concat']:+8.4f} "
                  f"{r['gate_deviation']:7.4f}")
        return

    if args.stage == "render":
        # Re-render the markdown tables from the stored JSON artefacts.
        # No metric is recomputed, so this cannot change any number.
        cal_json = Path(args.calibration_file) if args.calibration_file else (
            out_dir / "cal2_calibration.json")
        test_json = out_dir / "cal2_results.json"
        if not cal_json.exists() or not test_json.exists():
            raise SystemExit("cal2_calibration.json / cal2_results.json missing")
        md = render_markdown(
            json.loads(cal_json.read_text(encoding="utf-8")),
            json.loads(test_json.read_text(encoding="utf-8")),
            args.n_boot,
        )
        (out_dir / "cal2_tables.md").write_text(md, encoding="utf-8")
        print(f"[render] wrote {out_dir / 'cal2_tables.md'}")
        return

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
                f"  {r['id']:14s} calP@5={r['cal_retrieval_p5']:.4f} "
                f"d={r['cal_delta_p5_vs_concat']:+.4f} "
                f"[{r['cal_delta_p5_lo']:+.4f},{r['cal_delta_p5_hi']:+.4f}] "
                f"calAUC={r['cal_pair_auc']:.4f} calKfold={r['cal_kfold_p5']:.4f} "
                f"dev={r['gate_deviation']:.4f} A1={r['admissible_a1']} "
                f"A2={r['admissible_a2']} feasible={r['feasible']}{flag}"
            )
        return

    test = stage_test(args)
    (out_dir / "cal2_results.json").write_text(
        json.dumps(test, indent=2), encoding="utf-8"
    )
    (out_dir / "cal2_tables.md").write_text(
        render_markdown(json.loads(cal_file.read_text(encoding="utf-8")),
                        test, args.n_boot), encoding="utf-8"
    )
    print(f"[test] wrote {out_dir / 'cal2_results.json'}")
    print(f"[test] wrote {out_dir / 'cal2_tables.md'}")
    c = test["success_criteria"]
    print(f"[test] verdict: {c['verdict']}  {c}")


if __name__ == "__main__":
    main()
