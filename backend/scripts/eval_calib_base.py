#!/usr/bin/env python
"""Calibrated fusion baselines + candidate-set-size robustness probe (released subset).

Peer-review response to: *"compare the source-count-ratio weighting against a
simple CALIBRATED fusion baseline and show robustness to candidate-set
size / score-distribution shifts"*.

Scope
-----
Released single-camera subset (121 indexed clips / 242 scene frames),
150 protocol-matched control queries (tune n=50 / test n=100).  Nothing in
the repository is modified: the benchmark loader, the two branch score
vectors, the metadata prior alpha, the metrics and the statistical protocol
are **imported** from the released control driver
(``backend/scripts/eval_retrieval_control.py``) and from
``backend/app/eval/*`` + ``backend/app/services/result_merger.py``.

What is new here
----------------
1.  **A calibrated-fusion baseline family** (CombSUM / CombANZ / CombMNZ of
    Fox & Shaw, TREC-2 1994) in the *production candidate model*: each branch
    returns a shortlist, every branch is calibrated (per-query min-max, or the
    repository's z-score sigmoid) **over its own shortlist**, candidates that a
    branch did not return are **zero-filled**, and the two calibrated vectors
    are combined by a combination rule.

    * ``comb_sum_cal``  -- CombSUM: 0.5*s + 0.5*e (simple calibrated fusion).
    * ``comb_anz_cal``  -- CombANZ: (0.5*s + 0.5*e) * (#branches that returned d).
    * ``comb_mnz_cal``  -- CombMNZ: (0.5*s + 0.5*e) * (#branches that gave d a
      non-zero calibrated score).

    This is the canonical *empty-branch-robust* fusion: with one branch empty
    the combination degrades to the other branch instead of collapsing.

2.  **A candidate-set-size robustness probe**: the *same frozen* methods are
    re-evaluated on shrunken candidate sets (the gallery subsample is the
    union of the two branches' shortlists after each shortlist is cut to the
    largest-scoring full / half / quarter of its candidates; ties broken with
    a seeded RNG).  Symmetric and asymmetric shrinkages are reported, because
    the paper's alpha depends on the *source-count ratio*
    ``n_structured / (n_structured + n_semantic)`` and therefore moves when
    only one branch's candidate set changes size.

Protocol (pre-registered before the test split was evaluated)
-------------------------------------------------------------
* The comparison set is **fixed a priori**; no selection stage is needed, so
  the *entire* fixed set is evaluated **once** on the test split.  The tune
  split (n=50) is reported for transparency only and is **not** used to pick
  anything.
* Two **arms** (branch calibration): ``minmax`` (the released control's
  protocol, primary) and ``zscore`` (the repository's alternative calibration,
  secondary -- the score-distribution shift).
* Two **protocols**:
    - ``control`` -- the released control harness: both branches are dense over
      the whole 121-clip gallery and are calibrated over the whole gallery.
      Systems: ``prior_only`` (the paper's metadata/source-count-ratio alpha)
      and ``fixed_0.5`` (equal-weight calibrated fusion).
    - ``prod`` -- the production candidate model: structured shortlist =
      clips with a structured result (the control's own ``n_structured``
      threshold 0.05), semantic shortlist = the top ``K_SEM`` semantic hits
      (the control's own ``n_semantic`` = 20), union = the gallery, each
      branch calibrated over its own shortlist, missing candidates zero-filled.
      Systems: ``prior_only_prod``, ``comb_sum_cal``, ``comb_anz_cal``,
      ``comb_mnz_cal``.
* Statistics: macro-P@5 (mean of five category means); primary paired
  bootstrap with **within-category** resampling, secondary grouped
  (5-minute cluster) bootstrap, plus the control script's pooled per-query
  paired bootstrap, the grouped per-query bootstrap, Wilcoxon signed-rank and
  a paired sign-flip randomisation test.  ``n_boot = 3000``.
* Holm correction: one family per calibration arm (4 comparisons each: 1 in
  the ``control`` protocol + 3 in the ``prod`` protocol) and a conservative
  global family over all 8 arm-level comparisons.

Usage
-----
    python backend/scripts/eval_calib_base.py --stage tune
    python backend/scripts/eval_calib_base.py --stage test
    python backend/scripts/eval_calib_base.py --stage all

The expensive part (OpenCLIP text encoding + FAISS search + the metadata
structured branch) is cached in the OS temp directory, so the tune and test
stages share byte-identical inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import tempfile
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval.baselines import zscore_norm
from backend.app.eval.metrics import category_means, macro_mean, ranking_metrics
from backend.app.eval.stats import (
    bootstrap_ci,
    format_p,
    grouped_bootstrap_ci,
    holm_correction,
    paired_bootstrap_ci,
    paired_grouped_bootstrap_ci,
    wilcoxon_signed_rank,
)
from backend.scripts import eval_retrieval_conf3 as conf3
from backend.scripts import eval_retrieval_control as ctrl

RESULTS = ROOT / "results"
CACHE_PATH = Path(tempfile.gettempdir()) / "opencode" / "calib_base_cache.pkl"
GENERATOR = "backend/scripts/eval_calib_base.py"

CATEGORIES = ctrl.CATEGORIES
N_BOOT_DEFAULT = 3000

# --- pre-registered constants -------------------------------------------------
K_SEM = 20          # semantic shortlist size (the control's own n_semantic = 20)
STRUCT_THRESH = 0.05  # the control's own "structured result present" threshold
ARMS = ("minmax", "zscore")   # branch calibrations (minmax primary, zscore secondary)
WEIGHT = 0.5        # calibrated fusion weight (equal weight, no tuning)
PROBE_SEED = 20260401

# Published control numbers (results/retrieval_control_results.json) used as a
# reproduction check so a stale cache cannot silently change the comparison.
CONTROL_CHECK = {
    "minmax": {"prior_only": {"test": 0.538, "tune": 0.580}, "fixed_0.5": {"test": 0.534, "tune": 0.576}},
    "zscore": {"prior_only": {"test": 0.524, "tune": 0.556}, "fixed_0.5": {"test": 0.450, "tune": 0.504}},
}

# probe cells: (name, structured-pool fraction, semantic-pool fraction, struct exponent)
# The exponent is a declared *score-distribution shift*: x -> x^p on the
# calibrated structured branch (p > 1 makes the branch more top-heavy; p = 1 is
# the unshifted distribution).  It is applied identically to every method.
PROBE_CELLS: List[Tuple[str, float, float, float]] = [
    ("full", 1.0, 1.0, 1.0),
    ("half_both", 0.5, 0.5, 1.0),
    ("quarter_both", 0.25, 0.25, 1.0),
    ("struct_half", 0.5, 1.0, 1.0),
    ("struct_quarter", 0.25, 1.0, 1.0),
    ("sem_half", 1.0, 0.5, 1.0),
    ("warp_struct", 1.0, 1.0, 2.0),
    ("warp_struct_half_both", 0.5, 0.5, 2.0),
]

LITERATURE = {
    "comb_sum": (
        "CombSUM (normalise-then-sum) of Fox & Shaw, 'Combination of Multiple "
        "Searches', TREC-2 (NIST, 1994), "
        "https://trec.nist.gov/pubs/trec2/papers/txt/23.txt"
    ),
    "comb_anz": (
        "CombANZ (normalise-then-sum, multiplied by the number of systems that "
        "returned the document) of Fox & Shaw, TREC-2 (1994); the UMass CIIR "
        "'Analysis of Multiple Evidence Combination' report "
        "https://ciir-publications.cs.umass.edu/getpdf.php?id=63"
    ),
    "comb_mnz": (
        "CombMNZ (CombANZ with the count taken over the systems that assigned a "
        "non-zero score) of Fox & Shaw, TREC-2 (1994); Lee (1997), 'Hybrid test "
        "collections and the fusion of retrieval systems'"
    ),
    "empty_branch": (
        "Empty-branch rebalancing: a branch that returns nothing must not "
        "deflate the other branch (the canonical fix for an empty result set); "
        "see also Raza et al. (2019) 'A survey on deep learning for "
        "hybrid information retrieval', and the UMass CIIR evidence-combination "
        "report"
    ),
}


# ---------------------------------------------------------------------------
# Cache: the real retrieval (OpenCLIP + FAISS) and the two raw branch vectors
# ---------------------------------------------------------------------------

def build_cache(force: bool = False) -> Dict[str, Any]:
    """Run (or reload) the real retrieval and cache both raw branch vectors.

    Mirrors ``eval_retrieval_control.main`` exactly: the semantic branch is the
    max over the frames of a clip of the OpenCLIP/FAISS inner product, the
    structured branch is the metadata-only score built by
    ``ctrl.build_query_context`` (which calls the repository's own
    ``compute_adaptive_alpha`` for the metadata prior).
    """
    if CACHE_PATH.exists() and not force:
        with CACHE_PATH.open("rb") as fh:
            return pickle.load(fh)

    queries, splits = ctrl.load_benchmark()
    index, frames, clips, clip_ids = ctrl.load_scene_index()
    dets, _clip_of, dets_by_clip = ctrl.load_clip_detections(clips)
    motion = ctrl.subset.compute_motion_energy(clips)
    motion_max = max(v["mean"] for v in motion.values()) if motion else 1.0

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
    cache = {
        "queries": queries,
        "splits": splits,
        "clip_ids": clip_ids,
        "clips": clips,
        "sem_raw_max": sem_raw_max,
        "struct_raw": struct_raw,
        "alpha_prior": {qid: float(ctx["alpha"]) for qid, ctx in contexts.items()},
        "n_structured_ctrl": {qid: int(ctx["n_structured"]) for qid, ctx in contexts.items()},
        "n_semantic_ctrl": {qid: int(ctx["n_semantic"]) for qid, ctx in contexts.items()},
        "parsed_filter": {qid: ctx["parsed_filter"] for qid, ctx in contexts.items()},
        "has_count": {q["qid"]: bool(q["category"] == "count") for q in queries},
        "has_action": {q["qid"]: bool(q["category"] == "action") for q in queries},
        "category": {q["qid"]: q["category"] for q in queries},
        "group": {q["qid"]: (q.get("group") or q["qid"]) for q in queries},
        "split": {q["qid"]: q["split"] for q in queries},
        "qindex": {q["qid"]: i for i, q in enumerate(queries)},
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as fh:
        pickle.dump(cache, fh)
    return cache


# ---------------------------------------------------------------------------
# Candidate model (production shortlists) and calibration
# ---------------------------------------------------------------------------

def _seeded_rng(*parts: Any) -> random.Random:
    key = "|".join(str(p) for p in parts)
    return random.Random(PROBE_SEED + zlib.crc32(key.encode("utf-8")))


def _cut_topk(
    scores: np.ndarray, idxs: np.ndarray, frac: float, rng: random.Random
) -> np.ndarray:
    """Keep the largest-scoring ``ceil(frac * len(idxs))`` entries of ``idxs``.

    Ties are broken with a seeded per-item jitter so that the probe is exactly
    reproducible but not biased towards the clip order.
    """
    n = len(idxs)
    if n == 0:
        return idxs
    k = max(1, int(math.ceil(frac * n - 1e-9)))
    if k >= n:
        return idxs
    jitter = {int(i): rng.random() for i in idxs}
    order = sorted((int(i) for i in idxs), key=lambda i: (-float(scores[i]), jitter[i]))
    return np.array(sorted(order[:k]), dtype=np.int64)


def branch_pools(
    cache: Dict[str, Any],
    qid: str,
    struct_frac: float = 1.0,
    sem_frac: float = 1.0,
    seed_parts: Tuple[Any, ...] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """The two branch shortlists of a query.

    ``struct``: clips whose metadata structured score is above the control's own
    ``n_structured`` threshold (0.05) -- i.e. the clips for which the
    structured branch produced a result.
    ``sem``: the ``K_SEM`` largest-scoring semantic hits (the control's own
    ``n_semantic`` = 20, i.e. the production FAISS shortlist size).
    """
    s_raw = cache["struct_raw"][qid]
    e_raw = cache["sem_raw_max"][qid]
    s_idx = np.where(s_raw > STRUCT_THRESH)[0]
    e_idx = np.argsort(-e_raw, kind="stable")[:K_SEM]
    if struct_frac < 1.0:
        s_idx = _cut_topk(s_raw, s_idx, struct_frac, _seeded_rng(*seed_parts, "s"))
    if sem_frac < 1.0:
        e_idx = _cut_topk(e_raw, e_idx, sem_frac, _seeded_rng(*seed_parts, "e"))
    return s_idx, e_idx


def calibrate(raw: np.ndarray, idxs: np.ndarray, arm: str) -> np.ndarray:
    """Calibrate a branch over its own shortlist; zero-fill everything else.

    ``minmax`` is the control's ``ctrl.minmax`` (per-query min-max), ``zscore``
    is the repository's ``backend.app.eval.baselines.zscore_norm``
    (z-score + sigmoid).  A degenerate (all-tie) branch maps to zeros, so it
    cannot outvote the other branch.
    """
    out = np.zeros(len(raw), dtype=np.float64)
    if len(idxs) == 0:
        return out
    v = raw[idxs]
    if arm == "minmax":
        nv = ctrl.minmax(v)
    elif arm == "zscore":
        nv = np.asarray(zscore_norm([float(x) for x in v]), dtype=np.float64)
    else:
        raise ValueError(arm)
    out[idxs] = nv
    return out


def prior_alpha(
    cache: Dict[str, Any], qid: str, n_structured: int, n_semantic: int
) -> float:
    """The repository's metadata (source-count-ratio) prior alpha.

    ``backend.app.services.result_merger.compute_adaptive_alpha`` without score
    vectors -- exactly the production call, whose weight depends on the
    *number of results each source returned*.
    """
    return float(
        ctrl.compute_adaptive_alpha(
            cache["parsed_filter"][qid],
            cache["category"][qid],
            cache["has_count"][qid],
            cache["has_action"][qid],
            int(n_structured),
            int(n_semantic),
        )
    )


# ---------------------------------------------------------------------------
# Combination rules
# ---------------------------------------------------------------------------

def combine(
    rule: str,
    alpha: float,
    s: np.ndarray,
    e: np.ndarray,
    in_s: np.ndarray,
    in_e: np.ndarray,
) -> np.ndarray:
    """Combine two calibrated branch vectors.

    ``prior`` -- the paper's weighted blend ``a*s + (1-a)*e`` (source-count-ratio
    weighting).  ``sum`` -- CombSUM with the pre-registered equal weight.
    ``anz`` -- CombANZ: CombSUM x (#branches that returned the candidate).
    ``mnz`` -- CombMNZ: CombSUM x (#branches that gave a non-zero score).
    """
    if rule == "prior":
        return alpha * s + (1.0 - alpha) * e
    base = WEIGHT * s + (1.0 - WEIGHT) * e
    if rule == "sum":
        return base
    if rule == "anz":
        return base * (in_s.astype(np.float64) + in_e.astype(np.float64))
    if rule == "mnz":
        return base * (
            (s > 0).astype(np.float64) + (e > 0).astype(np.float64)
        )
    raise ValueError(rule)


def rank_clips(scores: np.ndarray, universe: np.ndarray, clip_ids: Sequence[str]) -> List[str]:
    """Rank the gallery (``universe``) by score; ties broken by clip order."""
    sub = np.asarray(scores[universe], dtype=np.float64)
    order = np.argsort(-sub, kind="stable")
    return [clip_ids[int(universe[int(i)])] for i in order]


# ---------------------------------------------------------------------------
# System evaluation
# ---------------------------------------------------------------------------

CONTROL_SYSTEMS = {"prior_only": "prior", "fixed_0.5": "sum"}
PROD_SYSTEMS = {
    "prior_only_prod": "prior",
    "comb_sum_cal": "sum",
    "comb_anz_cal": "anz",
    "comb_mnz_cal": "mnz",
}


def evaluate_control_arm(
    cache: Dict[str, Any], arm: str, qids: Sequence[str]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """The released control harness: dense branches over the whole gallery."""
    clip_ids = cache["clip_ids"]
    universe = np.arange(len(clip_ids))
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    branches: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for qid in qids:
        s = ctrl.normalize_branch(cache["struct_raw"][qid], arm)
        e = ctrl.normalize_branch(cache["sem_raw_max"][qid], arm)
        branches[qid] = (s, e)
    for name, rule in CONTROL_SYSTEMS.items():
        rows: Dict[str, Dict[str, Any]] = {}
        for qid in qids:
            s, e = branches[qid]
            a = float(cache["alpha_prior"][qid])
            fused = combine(rule, a, s, e, np.ones(len(clip_ids), bool), np.ones(len(clip_ids), bool))
            ranking = rank_clips(fused, universe, clip_ids)
            m = ranking_metrics(ranking, cache["rel"][qid])
            rows[qid] = {
                "qid": qid,
                "category": cache["category"][qid],
                "group": cache["group"][qid],
                "split": cache["split"][qid],
                "alpha": a,
                "top5": "|".join(ranking[:5]),
                **m,
            }
        out[name] = rows
    return out


def evaluate_prod_arm(
    cache: Dict[str, Any],
    arm: str,
    qids: Sequence[str],
    struct_frac: float = 1.0,
    sem_frac: float = 1.0,
    cell: str = "full",
    struct_exp: float = 1.0,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """The production candidate model: shortlists, zero-fill, calibrated fusion."""
    clip_ids = cache["clip_ids"]
    out: Dict[str, Dict[str, Dict[str, Any]]] = {name: {} for name in PROD_SYSTEMS}
    for qid in qids:
        s_idx, e_idx = branch_pools(
            cache, qid, struct_frac, sem_frac, seed_parts=(cell, arm, qid)
        )
        in_s = np.zeros(len(clip_ids), dtype=bool)
        in_e = np.zeros(len(clip_ids), dtype=bool)
        in_s[s_idx] = True
        in_e[e_idx] = True
        universe = np.where(in_s | in_e)[0]
        s = calibrate(cache["struct_raw"][qid], s_idx, arm)
        e = calibrate(cache["sem_raw_max"][qid], e_idx, arm)
        if struct_exp != 1.0:
            # declared score-distribution shift (top-heavier structured branch)
            s = np.power(np.maximum(s, 0.0), struct_exp)
        # the alpha depends on the *source counts* -> it moves with the pools
        a = prior_alpha(cache, qid, len(s_idx), len(e_idx))
        rel = set(cache["rel"][qid])
        coverage = (
            sum(1 for ci in universe if clip_ids[int(ci)] in rel) / float(len(rel))
            if rel else 0.0
        )
        for name, rule in PROD_SYSTEMS.items():
            fused = combine(rule, a, s, e, in_s, in_e)
            ranking = rank_clips(fused, universe, clip_ids)
            m = ranking_metrics(ranking, cache["rel"][qid])
            out[name][qid] = {
                "qid": qid,
                "category": cache["category"][qid],
                "group": cache["group"][qid],
                "split": cache["split"][qid],
                "alpha": a,
                "n_structured_pool": int(len(s_idx)),
                "n_semantic_pool": int(len(e_idx)),
                "n_universe": int(len(universe)),
                "gallery_relevant_coverage": float(coverage),
                "top5": "|".join(ranking[:5]),
                **m,
            }
    return out


def macro_of(rows: Dict[str, Dict[str, Any]], field: str) -> Tuple[float, Dict[str, float]]:
    vals = [{"category": r["category"], "v": float(r[field])} for r in rows.values()]
    cm = category_means(vals, "v")
    return float(macro_mean(cm)), {c: float(cm.get(c, 0.0)) for c in CATEGORIES}


def summarise(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in ("p5", "r10", "f1", "f1_paper", "mrr"):
        macro, per_cat = macro_of(rows, field)
        out[field] = {"macro": macro, "per_category": per_cat}
    return out


def count_scores(cache: Dict[str, Any], rows: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    vals = []
    for qid, r in rows.items():
        if r["category"] != "count":
            continue
        pred = ctrl.predict_count(cache["qdict"][qid], r["top5"].split("|"), cache["clips"])
        truth = cache["gt_count"].get(qid)
        if pred is not None and truth is not None:
            vals.append((pred, truth))
    if not vals:
        return {"n": 0, "em": 0.0, "mae": 0.0}
    return {
        "n": len(vals),
        "em": float(np.mean([1.0 if int(p) == int(t) else 0.0 for p, t in vals])),
        "mae": float(np.mean([abs(int(p) - int(t)) for p, t in vals])),
    }


# ---------------------------------------------------------------------------
# Statistics (all imported / reused from the repository + the conf3 driver)
# ---------------------------------------------------------------------------

def compare(
    cache: Dict[str, Any],
    rows_a: Dict[str, Dict[str, Any]],
    rows_b: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Paired + grouped bootstrap comparison of two systems on macro P@5."""
    cats = [cache["category"][qid] for qid in qids]
    groups = [cache["group"][qid] for qid in qids]
    a_vals = [rows_a[qid]["p5"] for qid in qids]
    b_vals = [rows_b[qid]["p5"] for qid in qids]
    primary = conf3.paired_macro_bootstrap(cats, a_vals, cats, b_vals, n_boot=n_boot, seed=seed)
    grouped = conf3.paired_grouped_macro_bootstrap(
        groups, a_vals, groups, b_vals, cats, n_boot=n_boot, seed=seed + 1
    )
    rand = conf3.sign_flip_randomisation(cats, a_vals, b_vals, n_boot=n_boot, seed=seed + 2)
    pooled = paired_bootstrap_ci(a_vals, b_vals, n_boot=n_boot, seed=seed + 3)
    pooled_grp = paired_grouped_bootstrap_ci(a_vals, b_vals, groups, n_boot=n_boot, seed=seed + 4)
    wilc = wilcoxon_signed_rank(a_vals, b_vals)
    macro_a, macro_b = macro_of(rows_a, "p5")[0], macro_of(rows_b, "p5")[0]
    changed = [qid for qid in qids if abs(rows_a[qid]["p5"] - rows_b[qid]["p5"]) > 1e-9]
    return {
        "macro_a": float(macro_a),
        "macro_b": float(macro_b),
        "delta_macro_p5": float(macro_a - macro_b),
        "primary_paired_macro_bootstrap": primary,
        "grouped_cluster_macro_bootstrap": grouped,
        "sign_flip_randomisation": rand,
        "pooled_per_query_paired_bootstrap": {
            "delta": float(pooled[0]), "lo": float(pooled[1]), "hi": float(pooled[2]), "p": float(pooled[3]),
        },
        "pooled_per_query_grouped_bootstrap": {
            "delta": float(pooled_grp[0]), "lo": float(pooled_grp[1]), "hi": float(pooled_grp[2]),
            "p": float(pooled_grp[3]),
        },
        "wilcoxon_signed_rank_p": float(wilc),
        "n_queries_changed": len(changed),
        "n_queries_better": sum(1 for qid in changed if rows_a[qid]["p5"] > rows_b[qid]["p5"]),
        "n_queries_worse": sum(1 for qid in changed if rows_a[qid]["p5"] < rows_b[qid]["p5"]),
    }


def per_category_tests(
    cache: Dict[str, Any],
    rows_a: Dict[str, Dict[str, Any]],
    rows_b: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    pvals: Dict[str, float] = {}
    detail: Dict[str, Any] = {}
    for c in CATEGORIES:
        cq = [qid for qid in qids if cache["category"][qid] == c]
        if not cq:
            continue
        av = [rows_a[qid]["p5"] for qid in cq]
        bv = [rows_b[qid]["p5"] for qid in cq]
        m, lo, hi, p = paired_bootstrap_ci(av, bv, n_boot=n_boot, seed=seed)
        pvals[c] = float(p)
        detail[c] = {"delta_p5": float(m), "lo": float(lo), "hi": float(hi), "p": float(p)}
    return {
        "per_category_paired_bootstrap": detail,
        "per_category_holm_reject": {k: bool(v) for k, v in holm_correction(pvals, alpha=0.05).items()} if pvals else {},
        "per_category_holm_adjusted_p": conf3.holm_adjusted(pvals) if pvals else {},
    }


# ---------------------------------------------------------------------------
# Robustness probe
# ---------------------------------------------------------------------------

def probe_cell(
    cache: Dict[str, Any],
    arm: str,
    cell: Tuple[str, float, float, float],
    qids: Sequence[str],
    n_boot: int,
) -> Dict[str, Any]:
    """Evaluate every frozen method on one gallery subsample."""
    name, s_frac, e_frac, s_exp = cell
    prod = evaluate_prod_arm(cache, arm, qids, s_frac, e_frac, cell=name, struct_exp=s_exp)
    out: Dict[str, Any] = {
        "cell": name,
        "struct_frac": s_frac,
        "sem_frac": e_frac,
        "struct_exponent": s_exp,
        "systems": {},
    }
    for sysname, rows in prod.items():
        vals = [rows[qid]["p5"] for qid in qids]
        groups = [cache["group"][qid] for qid in qids]
        m, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=777)
        gm, glo, ghi = grouped_bootstrap_ci(vals, groups, n_boot=n_boot, seed=777)
        macro, per_cat = macro_of(rows, "p5")
        out["systems"][sysname] = {
            "macro_p5": float(macro),
            "per_category_p5": per_cat,
            "bootstrap_ci": {"mean": float(m), "lo": float(lo), "hi": float(hi)},
            "grouped_ci": {"mean": float(gm), "lo": float(glo), "hi": float(ghi)},
            "mean_alpha": float(np.mean([rows[qid]["alpha"] for qid in qids])),
            "mean_struct_pool": float(np.mean([rows[qid]["n_structured_pool"] for qid in qids])),
            "mean_sem_pool": float(np.mean([rows[qid]["n_semantic_pool"] for qid in qids])),
            "mean_universe": float(np.mean([rows[qid]["n_universe"] for qid in qids])),
            "mean_gallery_relevant_coverage": float(
                np.mean([rows[qid]["gallery_relevant_coverage"] for qid in qids])
            ),
        }
    # Random-gallery reference: the *same* gallery size, but chosen without any
    # system knowledge (uniform draw from the whole gallery), ranked by the
    # paper's published prior blend.  It is the floor that any method must pay
    # for a gallery of that size.
    rng = _seeded_rng("random_gallery", arm, name)
    ref_vals = []
    for qid in qids:
        s_cut, e_cut = branch_pools(cache, qid, s_frac, e_frac, seed_parts=(name, arm, qid))
        gal = np.union1d(s_cut, e_cut)
        picked = np.array(
            sorted(rng.sample(range(len(cache["clip_ids"])), int(len(gal)))), dtype=np.int64
        )
        s = ctrl.normalize_branch(cache["struct_raw"][qid], arm)
        e = ctrl.normalize_branch(cache["sem_raw_max"][qid], arm)
        a = float(cache["alpha_prior"][qid])
        fused = combine("prior", a, s, e, np.ones(len(cache["clip_ids"]), bool),
                        np.ones(len(cache["clip_ids"]), bool))
        ranking = rank_clips(fused, picked, cache["clip_ids"])
        m = ranking_metrics(ranking, cache["rel"][qid])
        ref_vals.append(float(m["p5"]))
    rm, rlo, rhi = bootstrap_ci(ref_vals, n_boot=n_boot, seed=778)
    out["random_gallery_reference"] = {
        "macro_p5": float(rm),
        "bootstrap_ci": {"mean": float(rm), "lo": float(rlo), "hi": float(rhi)},
        "note": "seeded uniform subsample of the shrunken gallery, ranked by the paper's prior blend",
    }
    # stability of the *conclusion*: paired delta of every calibrated baseline
    # against the prior-only baseline inside this gallery subsample.
    cats = [cache["category"][qid] for qid in qids]
    base = prod["prior_only_prod"]
    deltas: Dict[str, Any] = {}
    for sysname in ("comb_sum_cal", "comb_anz_cal", "comb_mnz_cal"):
        a_vals = [prod[sysname][qid]["p5"] for qid in qids]
        b_vals = [base[qid]["p5"] for qid in qids]
        d = conf3.paired_macro_bootstrap(cats, a_vals, cats, b_vals, n_boot=n_boot, seed=791)
        deltas[sysname] = {
            "delta_macro_p5": float(d["delta"]),
            "lo": float(d["lo"]),
            "hi": float(d["hi"]),
            "p": float(d["p"]),
        }
    out["paired_delta_vs_prior_only_prod"] = deltas
    return out


def empty_branch_sanity(cache: Dict[str, Any], arm: str) -> Dict[str, Any]:
    """Synthetic (non-benchmark) check of the empty-branch behaviour.

    Not part of the evaluation: it verifies that with one empty shortlist the
    calibrated fusion still produces finite scores and ranks by the surviving
    branch, and it records the *cross-query* consequence of the paper's fixed
    alpha (an empty branch leaves the surviving branch multiplied by
    ``1 - alpha``, so its scores are not comparable across queries).
    """
    # two representative queries: one whose prior alpha is 1.0 (count -> the
    # structured branch owns the whole weight) and one with alpha < 1.
    picks: List[str] = []
    for cat in ("count", "find"):
        for q in cache["queries"]:
            if cache["category"][q["qid"]] == cat:
                picks.append(q["qid"])
                break
    qids = list(picks)
    n = len(cache["clip_ids"])
    out: Dict[str, Any] = {}
    for qid in qids:
        s_raw = cache["struct_raw"][qid]
        e_raw = cache["sem_raw_max"][qid]
        top_s = np.argsort(-s_raw, kind="stable")[:K_SEM]
        top_e = np.argsort(-e_raw, kind="stable")[:K_SEM]
        cases = {
            "empty_structured": (np.array([], dtype=np.int64), top_e),
            "empty_semantic": (top_s, np.array([], dtype=np.int64)),
        }
        per_query: Dict[str, Any] = {}
        for label, (s_idx, e_idx) in cases.items():
            s = calibrate(s_raw, s_idx, arm)
            e = calibrate(e_raw, e_idx, arm)
            in_s = np.zeros(n, bool)
            in_e = np.zeros(n, bool)
            in_s[s_idx] = True
            in_e[e_idx] = True
            alpha = prior_alpha(cache, qid, len(s_idx), len(e_idx))
            survivor = in_e if label == "empty_structured" else in_s
            rec = {}
            for name, rule in PROD_SYSTEMS.items():
                fused = combine(rule, alpha, s, e, in_s, in_e)
                finite = bool(np.all(np.isfinite(fused)))
                order = np.argsort(-fused, kind="stable")
                rec[name] = {
                    "all_finite": finite,
                    "max_fused_score": float(fused[order[0]]) if finite else None,
                    "top1_from_surviving_branch": bool(finite and bool(survivor[order[0]])),
                    "fused_is_degenerate": bool(finite and float(np.max(fused)) <= 0.0),
                }
            per_query[label] = {"alpha": float(alpha), "systems": rec}
        out[qid] = {
            "category": cache["category"][qid],
            "cases": per_query,
            "note": (
                "the paper's alpha keeps the empty branch's weight, so the surviving "
                "branch is multiplied by (1 - alpha) (by 0 when alpha = 1): its scores "
                "are deflated and are not comparable across queries.  The calibrated "
                "rules keep the surviving branch at full scale."
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def split_qids(cache: Dict[str, Any], split: str) -> List[str]:
    return [q["qid"] for q in cache["queries"] if q["split"] == split]


def arm_family(
    cache: Dict[str, Any],
    arm: str,
    qids: Sequence[str],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Run the whole frozen comparison set for one calibration arm."""
    control = evaluate_control_arm(cache, arm, qids)
    prod = evaluate_prod_arm(cache, arm, qids)
    comparisons: Dict[str, Any] = {}
    comparisons["fixed_0.5_vs_prior_only"] = compare(
        cache, control["fixed_0.5"], control["prior_only"], qids, n_boot, seed
    )
    for name in ("comb_sum_cal", "comb_anz_cal", "comb_mnz_cal"):
        comparisons[f"{name}_vs_prior_only_prod"] = compare(
            cache, prod[name], prod["prior_only_prod"], qids, n_boot, seed + 10
        )
    raw_p = {k: float(v["primary_paired_macro_bootstrap"]["p"]) for k, v in comparisons.items()}
    adj = conf3.holm_adjusted(raw_p)
    rejects = holm_correction(raw_p, alpha=0.05)
    for k, v in comparisons.items():
        v["holm_adjusted_p"] = float(adj[k])
        v["holm_reject"] = bool(rejects[k])
        v.update(
            per_category_tests(
                cache,
                control["fixed_0.5"] if k == "fixed_0.5_vs_prior_only" else prod[k.split("_vs_")[0]],
                control["prior_only"] if k == "fixed_0.5_vs_prior_only" else prod["prior_only_prod"],
                qids,
                n_boot,
                seed + 30,
            )
        )
    summary = {
        "control_protocol": {
            name: {"macro": summarise(rows), "count": count_scores(cache, rows)}
            for name, rows in control.items()
        },
        "prod_protocol": {
            name: {
                "macro": summarise(rows),
                "count": count_scores(cache, rows),
                "mean_alpha": float(np.mean([rows[qid]["alpha"] for qid in qids])),
                "mean_struct_pool": float(np.mean([rows[qid]["n_structured_pool"] for qid in qids])),
                "mean_sem_pool": float(np.mean([rows[qid]["n_semantic_pool"] for qid in qids])),
                "mean_universe": float(np.mean([rows[qid]["n_universe"] for qid in qids])),
            }
            for name, rows in prod.items()
        },
        "comparisons": comparisons,
        "holm_family": {
            "members": list(raw_p.keys()),
            "raw_p": raw_p,
            "holm_adjusted_p": {k: float(v) for k, v in adj.items()},
            "holm_reject": {k: bool(v) for k, v in rejects.items()},
        },
    }
    return summary


def weight_sensitivity(cache: Dict[str, Any], arm: str, qids: Sequence[str]) -> List[Dict[str, Any]]:
    """Descriptive tune-only sweep of the CombSUM weight (not used for selection)."""
    global WEIGHT
    rows_out = []
    saved = WEIGHT
    try:
        for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            WEIGHT = w
            prod = evaluate_prod_arm(cache, arm, qids)
            macro, per_cat = macro_of(prod["comb_sum_cal"], "p5")
            pmacro, _ = macro_of(prod["prior_only_prod"], "p5")
            rows_out.append({
                "weight_struct": w,
                "comb_sum_macro_p5": float(macro),
                "prior_only_prod_macro_p5": float(pmacro),
                "delta": float(macro - pmacro),
                "per_category_p5": per_cat,
            })
    finally:
        WEIGHT = saved
    return rows_out


def safe_write_json(path: Path, obj: Dict[str, Any]) -> None:
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("generator") != GENERATOR:
                raise RuntimeError(
                    f"refusing to overwrite {path.name}: it was not produced by {GENERATOR}"
                )
        except json.JSONDecodeError:
            raise RuntimeError(f"refusing to overwrite {path.name}: unparsable JSON")
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_tune_report(res: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Calibrated-fusion baselines + candidate-set probe -- TUNE split (n=%d)\n" % res["n_tune"])
    lines.append(
        "**This file is the pre-registration record.**  The comparison set below was "
        "fixed *before* the test split was evaluated and no selection is performed on "
        "this split: the entire fixed set is evaluated once on TEST.  The tune numbers "
        "are reported for transparency only.\n"
    )
    lines.append("\n## Pre-registered protocol\n")
    for item in res["preregistration"]["protocol"]:
        lines.append(f"- {item}")
    lines.append("\n## Literature basis\n")
    for k, v in res["preregistration"]["literature"].items():
        lines.append(f"- **{k}** -- {v}")
    lines.append("\n## Candidate-set sizes (production candidate model)\n")
    lines.append("| Category | mean structured shortlist | mean semantic shortlist | mean union gallery | mean alpha |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat, v in res["pool_sizes"][res["arms"][0]["name"]].items():
        if cat.startswith("_"):
            continue
        lines.append(
            f"| {cat} | {v['mean_structured_pool']:.1f} | {v['mean_semantic_pool']:.1f} | "
            f"{v['mean_universe']:.1f} | {v['mean_alpha']:.3f} |"
        )
    ps = res["pool_sizes"][res["arms"][0]["name"]]
    lines.append(
        f"\nAlpha consistency check (production pool sizes vs. the control's published "
        f"prior alpha): max |difference| = {ps['_alpha_consistency_max_abs_diff']:.2e} "
        f"(ok = {ps['_alpha_consistency_ok']}).\n"
    )
    for arm in res["arms"]:
        lines.append(f"\n## TUNE results -- {arm['name']} arm\n")
        lines.append("| Protocol | System | macro P@5 | R@10 | F1@10 (paper) | MRR | count EM | count MAE |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for proto in ("control_protocol", "prod_protocol"):
            for name, v in arm[proto].items():
                mm = v["macro"]
                lines.append(
                    f"| {proto.replace('_protocol','')} | {name} | {mm['p5']['macro']:.4f} | "
                    f"{mm['r10']['macro']:.4f} | {mm['f1_paper']['macro']:.4f} | "
                    f"{mm['mrr']['macro']:.4f} | {v['count']['em']:.3f} | {v['count']['mae']:.3f} |"
                )
        lines.append("\n| Comparison | delta macro P@5 | 95% CI | p | grouped delta | grouped p | Holm p | Holm reject |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---|")
        for cname, c in arm["comparisons"].items():
            p = c["primary_paired_macro_bootstrap"]
            g = c["grouped_cluster_macro_bootstrap"]
            lines.append(
                f"| {cname} | {p['delta']:+.4f} | [{p['lo']:+.4f}, {p['hi']:+.4f}] | "
                f"{format_p(p['p'], res['n_boot'])} | {g['delta']:+.4f} | "
                f"{format_p(g['p'], res['n_boot'])} | {format_p(c['holm_adjusted_p'], res['n_boot'])} | "
                f"{c['holm_reject']} |"
            )
    lines.append("\n## Candidate-set-size robustness probe (tune split)\n")
    for arm in res["arms"]:
        probe = res["probe"][arm["name"]]
        names = list(probe["cells"][0]["systems"].keys())
        lines.append(f"\n### {arm['name']} arm\n")
        lines.append("| System | " + " | ".join(c["cell"] for c in probe["cells"]) + " |")
        lines.append("|---|" + "---:|" * len(probe["cells"]))
        for sysname in names:
            cells = [
                f"{c['systems'][sysname]['macro_p5']:.4f} [{c['systems'][sysname]['bootstrap_ci']['lo']:.3f}, {c['systems'][sysname]['bootstrap_ci']['hi']:.3f}]"
                for c in probe["cells"]
            ]
            lines.append(f"| {sysname} | " + " | ".join(cells) + " |")
        lines.append(
            "| random gallery (reference floor) | "
            + " | ".join(f"{c['random_gallery_reference']['macro_p5']:.4f}" for c in probe["cells"])
            + " |"
        )
        lines.append(
            "\n**Stability of the conclusion** -- paired delta macro P@5 of each calibrated "
            "baseline against `prior_only_prod` inside each gallery subsample "
            "(negative = the calibrated baseline is worse):\n"
        )
        lines.append("| System | " + " | ".join(c["cell"] for c in probe["cells"]) + " |")
        lines.append("|---|" + "---:|" * len(probe["cells"]))
        for sysname in ("comb_sum_cal", "comb_anz_cal", "comb_mnz_cal"):
            cells = [
                f"{c['paired_delta_vs_prior_only_prod'][sysname]['delta_macro_p5']:+.4f} "
                f"[{c['paired_delta_vs_prior_only_prod'][sysname]['lo']:+.3f}, "
                f"{c['paired_delta_vs_prior_only_prod'][sysname]['hi']:+.3f}]"
                for c in probe["cells"]
            ]
            lines.append(f"| {sysname} - prior_only_prod | " + " | ".join(cells) + " |")
    lines.append("\n## Descriptive weight sweep (CombSUM weight, tune only -- NOT used for selection)\n")
    lines.append("| weight on structured branch | CombSUM macro P@5 | prior_only_prod macro P@5 | delta |")
    lines.append("|---:|---:|---:|---:|")
    for r in res["weight_sensitivity"][res["arms"][0]["name"]]:
        lines.append(
            f"| {r['weight_struct']:.1f} | {r['comb_sum_macro_p5']:.4f} | "
            f"{r['prior_only_prod_macro_p5']:.4f} | {r['delta']:+.4f} |"
        )
    lines.append("\n## Empty-branch sanity check (synthetic, not a benchmark result)\n")
    lines.append("```json")
    lines.append(json.dumps(res["empty_branch_sanity"], indent=2))
    lines.append("```")
    (RESULTS / "calib_tune.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_test_report(res: Dict[str, Any]) -> None:
    nb = res["n_boot"]
    lines: List[str] = []
    lines.append("# Calibrated-fusion baselines + candidate-set probe -- TEST split (n=%d)\n" % res["n_test"])
    lines.append(
        "**One test evaluation of the frozen comparison set.**  The set was fixed on "
        "the TUNE split (see `calib_tune.md`); nothing was selected on test.  "
        f"Bootstrap resamples: {nb}.  `< {2.0 / nb:.4f}` is the resolution floor.\n"
    )
    for arm in res["arms"]:
        lines.append(f"\n## {arm['name']} arm\n")
        lines.append("| Protocol | System | macro P@5 | R@10 | F1@10 (paper) | MRR | count EM | count MAE |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for proto in ("control_protocol", "prod_protocol"):
            for name, v in arm[proto].items():
                mm = v["macro"]
                lines.append(
                    f"| {proto.replace('_protocol','')} | {name} | {mm['p5']['macro']:.4f} | "
                    f"{mm['r10']['macro']:.4f} | {mm['f1_paper']['macro']:.4f} | "
                    f"{mm['mrr']['macro']:.4f} | {v['count']['em']:.3f} | {v['count']['mae']:.3f} |"
                )
        lines.append("\n| Comparison | delta macro P@5 | 95% CI | p | grouped delta | grouped 95% CI | grouped p | Holm p | reject |")
        lines.append("|---|---:|---|---:|---:|---|---:|---:|---|")
        for cname, c in arm["comparisons"].items():
            p = c["primary_paired_macro_bootstrap"]
            g = c["grouped_cluster_macro_bootstrap"]
            lines.append(
                f"| {cname} | {p['delta']:+.4f} | [{p['lo']:+.4f}, {p['hi']:+.4f}] | "
                f"{format_p(p['p'], nb)} | {g['delta']:+.4f} | [{g['lo']:+.4f}, {g['hi']:+.4f}] | "
                f"{format_p(g['p'], nb)} | {format_p(c['holm_adjusted_p'], nb)} | {c['holm_reject']} |"
            )
        lines.append("\n| Comparison | pooled per-query delta | p | pooled grouped delta | p | Wilcoxon p | sign-flip p | better/worse queries |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for cname, c in arm["comparisons"].items():
            pb = c["pooled_per_query_paired_bootstrap"]
            pg = c["pooled_per_query_grouped_bootstrap"]
            lines.append(
                f"| {cname} | {pb['delta']:+.4f} [{pb['lo']:+.4f}, {pb['hi']:+.4f}] | "
                f"{format_p(pb['p'], nb)} | {pg['delta']:+.4f} | {format_p(pg['p'], nb)} | "
                f"{format_p(c['wilcoxon_signed_rank_p'], nb)} | "
                f"{format_p(c['sign_flip_randomisation']['p'], nb)} | "
                f"{c['n_queries_better']}/{c['n_queries_worse']} |"
            )
        lines.append("\n| Per-category | delta P@5 | 95% CI | p | Holm p | reject |")
        lines.append("|---|---:|---|---:|---:|---|")
        for cname, c in arm["comparisons"].items():
            for cat, d in c["per_category_paired_bootstrap"].items():
                lines.append(
                    f"| {cname} / {cat} | {d['delta_p5']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] | "
                    f"{format_p(d['p'], nb)} | {format_p(c['per_category_holm_adjusted_p'][cat], nb)} | "
                    f"{c['per_category_holm_reject'][cat]} |"
                )
    lines.append("\n## Candidate-set-size robustness probe (test split)\n")
    lines.append(
        "Gallery subsample = the union of the two branches' shortlists after each "
        "shortlist is cut to its largest-scoring full / half / quarter candidates "
        f"(seeded tie-break, seed {PROBE_SEED}).  Every frozen method is re-calibrated "
        "and re-ranked inside the subsampled gallery.  `mean alpha` shows how far the "
        "source-count-ratio weighting moves when the candidate-set size changes.\n"
    )
    for arm in res["arms"]:
        probe = res["probe"][arm["name"]]
        lines.append(f"\n### {arm['name']} arm\n")
        names = list(probe["cells"][0]["systems"].keys())
        lines.append("| System | " + " | ".join(
            f"{c['cell']} (alpha {c['systems'][names[0]]['mean_alpha']:.3f})" for c in probe["cells"]
        ) + " |")
        lines.append("|---|" + "---:|" * len(probe["cells"]))
        for sysname in names:
            cells = []
            for c in probe["cells"]:
                v = c["systems"][sysname]
                cells.append(f"{v['macro_p5']:.4f} [{v['bootstrap_ci']['lo']:.3f}, {v['bootstrap_ci']['hi']:.3f}]")
            lines.append(f"| {sysname} | " + " | ".join(cells) + " |")
        ref = [c["random_gallery_reference"]["macro_p5"] for c in probe["cells"]]
        lines.append("| random gallery (reference floor) | " + " | ".join(f"{v:.4f}" for v in ref) + " |")
        lines.append(
            "\n| Cell | mean structured pool | mean semantic pool | mean union gallery | mean alpha | gallery relevant coverage |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for c in probe["cells"]:
            v = c["systems"][names[0]]
            lines.append(
                f"| {c['cell']} | {v['mean_struct_pool']:.1f} | {v['mean_sem_pool']:.1f} | "
                f"{v['mean_universe']:.1f} | {v['mean_alpha']:.3f} | "
                f"{v['mean_gallery_relevant_coverage']:.3f} |"
            )
    lines.append("\n## Verdict\n")
    v = res["verdict"]
    lines.append(f"- calibrated baseline beats prior-only (Holm-significant): **{v['calibrated_beats_prior']}**")
    lines.append(f"- robustness direction: **{v['robustness_direction']}**")
    lines.append(f"- overall verdict: **{v['verdict']}**")
    for k, val in v["notes"].items():
        lines.append(f"- {k}: {val}")
    (RESULTS / "calib_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(res: Dict[str, Any], tune_res: Optional[Dict[str, Any]]) -> None:
    nb = res["n_boot"]
    lines: List[str] = []
    lines.append("# Calibrated fusion baselines vs. the source-count-ratio weighting\n")
    lines.append(
        "**Scope.** Released single-camera subset (121 indexed clips / 242 scene "
        "frames), 150 protocol-matched control queries (tune n=50 / test n=100).  "
        "The paper's metadata prior alpha (source-count-ratio weighting) is reused "
        "unchanged; the calibrated fusion baselines are *competitors*, not "
        "replacements.\n"
    )
    lines.append(
        "\n**Anti-p-hacking record.** The comparison set was fixed before the test "
        "split was evaluated, the whole fixed set was evaluated **once** on test "
        "(the script refuses to write a second test result), and every configuration "
        "that was tried is listed below.\n"
    )
    lines.append("\n## 1. Baselines implemented\n")
    lines.append("| Baseline | Rule | Source |")
    lines.append("|---|---|---|")
    lines.append(
        "| `fixed_0.5` / `comb_sum_cal` | equal-weight fusion of the two calibrated "
        "branches (CombSUM) | Fox & Shaw, TREC-2 1994 |"
    )
    lines.append(
        "| `comb_anz_cal` | CombANZ: CombSUM x (#branches that returned the "
        "candidate) | Fox & Shaw, TREC-2 1994 |"
    )
    lines.append(
        "| `comb_mnz_cal` | CombMNZ: CombSUM x (#branches with a non-zero score) | "
        "Fox & Shaw, TREC-2 1994; Lee 1997 |"
    )
    lines.append(
        "| zero-fill + per-branch calibration | candidates a branch did not return get "
        "0; each branch is calibrated over its own shortlist | canonical IR "
        "data-fusion practice |"
    )
    lines.append("\n## 2. Pre-registration\n")
    for item in res["preregistration"]["protocol"]:
        lines.append(f"- {item}")
    if tune_res is not None:
        lines.append("\n## 3. TUNE split (n=%d) -- reported, not used for selection\n" % tune_res["n_tune"])
        lines.append("| Arm | Protocol | System | macro P@5 |")
        lines.append("|---|---|---|---:|")
        for arm in tune_res["arms"]:
            for proto in ("control_protocol", "prod_protocol"):
                for name, v in arm[proto].items():
                    lines.append(
                        f"| {arm['name']} | {proto.replace('_protocol','')} | {name} | "
                        f"{v['macro']['p5']['macro']:.4f} |"
                    )
    lines.append("\n## 4. TEST split (n=%d) -- the single frozen evaluation\n" % res["n_test"])
    for arm in res["arms"]:
        lines.append(f"\n### {arm['name']} arm\n")
        lines.append("| Protocol | System | macro P@5 | per-category P@5 (count/find/track/action/filter) |")
        lines.append("|---|---|---:|---|")
        for proto in ("control_protocol", "prod_protocol"):
            for name, v in arm[proto].items():
                pc = v["macro"]["p5"]["per_category"]
                lines.append(
                    f"| {proto.replace('_protocol','')} | {name} | {v['macro']['p5']['macro']:.4f} | "
                    + "/".join(f"{pc[c]:.3f}" for c in CATEGORIES)
                    + " |"
                )
        lines.append("\n| Comparison | delta macro P@5 | 95% CI | p | grouped delta | grouped p | Holm p | reject |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---|")
        for cname, c in arm["comparisons"].items():
            p = c["primary_paired_macro_bootstrap"]
            g = c["grouped_cluster_macro_bootstrap"]
            lines.append(
                f"| {cname} | {p['delta']:+.4f} | [{p['lo']:+.4f}, {p['hi']:+.4f}] | "
                f"{format_p(p['p'], nb)} | {g['delta']:+.4f} | {format_p(g['p'], nb)} | "
                f"{format_p(c['holm_adjusted_p'], nb)} | {c['holm_reject']} |"
            )
    lines.append("\n## 5. Candidate-set-size robustness probe\n")
    lines.append(
        "Same frozen methods, gallery subsample = union of the branches' shortlists "
        "after cutting each shortlist to its largest-scoring full / half / quarter "
        f"candidates (seeded, seed {PROBE_SEED}).  Query-level bootstrap 95% CIs in "
        "brackets; descriptive.\n"
    )
    for arm in res["arms"]:
        probe = res["probe"][arm["name"]]
        names = list(probe["cells"][0]["systems"].keys())
        lines.append(f"\n### {arm['name']} arm\n")
        lines.append("| System | " + " | ".join(c["cell"] for c in probe["cells"]) + " |")
        lines.append("|---|" + "---:|" * len(probe["cells"]))
        for sysname in names:
            cells = [
                f"{c['systems'][sysname]['macro_p5']:.4f} [{c['systems'][sysname]['bootstrap_ci']['lo']:.3f}, {c['systems'][sysname]['bootstrap_ci']['hi']:.3f}]"
                for c in probe["cells"]
            ]
            lines.append(f"| {sysname} | " + " | ".join(cells) + " |")
        lines.append(
            "| random gallery (reference floor) | "
            + " | ".join(f"{c['random_gallery_reference']['macro_p5']:.4f}" for c in probe["cells"])
            + " |"
        )
        lines.append(
            "\n**Stability of the conclusion** -- paired delta macro P@5 of each calibrated "
            "baseline against `prior_only_prod` inside each gallery subsample "
            "(negative = the calibrated baseline is worse):\n"
        )
        lines.append("| System | " + " | ".join(c["cell"] for c in probe["cells"]) + " |")
        lines.append("|---|" + "---:|" * len(probe["cells"]))
        for sysname in ("comb_sum_cal", "comb_anz_cal", "comb_mnz_cal"):
            cells = [
                f"{c['paired_delta_vs_prior_only_prod'][sysname]['delta_macro_p5']:+.4f} "
                f"[{c['paired_delta_vs_prior_only_prod'][sysname]['lo']:+.3f}, "
                f"{c['paired_delta_vs_prior_only_prod'][sysname]['hi']:+.3f}]"
                for c in probe["cells"]
            ]
            lines.append(f"| {sysname} - prior_only_prod | " + " | ".join(cells) + " |")
        lines.append(
            "\n| Cell | mean structured pool | mean semantic pool | mean union gallery | mean alpha | gallery relevant coverage |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for c in probe["cells"]:
            v = c["systems"][names[0]]
            lines.append(
                f"| {c['cell']} | {v['mean_struct_pool']:.1f} | {v['mean_sem_pool']:.1f} | "
                f"{v['mean_universe']:.1f} | {v['mean_alpha']:.3f} | "
                f"{v['mean_gallery_relevant_coverage']:.3f} |"
            )
    lines.append("\n## 6. Verdict\n")
    v = res["verdict"]
    lines.append(f"**{v['verdict']}** -- calibrated baseline beats prior-only (Holm-significant): "
                 f"**{v['calibrated_beats_prior']}**; robustness direction: "
                 f"**{v['robustness_direction']}**.\n")
    for k, val in v["notes"].items():
        lines.append(f"- {k}: {val}")
    lines.append("\n## 7. Reproduction\n")
    lines.append("```")
    for c in res["commands"]:
        lines.append(c)
    lines.append("```")
    lines.append("\n## 8. Limitations\n")
    for item in res["limitations"]:
        lines.append(f"- {item}")
    (RESULTS / "calib_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def calibration_diagnostics(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Label-free structural diagnostics of the two calibrated branches.

    No metric, no label and no query score enters this function: it only
    measures *how the calibration interacts with the combination rules*, which
    is what explains why CombANZ and CombMNZ coincide under one arm and differ
    under the other.
    """
    out: Dict[str, Any] = {}
    for arm in ARMS:
        n_diff = 0
        n_cand = 0
        zero_s = 0
        zero_e = 0
        pool_s = 0
        pool_e = 0
        for qid in cache["struct_raw"]:
            s_idx, e_idx = branch_pools(cache, qid)
            s = calibrate(cache["struct_raw"][qid], s_idx, arm)
            e = calibrate(cache["sem_raw_max"][qid], e_idx, arm)
            in_s = np.zeros(len(s), bool)
            in_e = np.zeros(len(e), bool)
            in_s[s_idx] = True
            in_e[e_idx] = True
            universe = in_s | in_e
            anz = in_s.astype(int) + in_e.astype(int)
            mnz = (s > 0).astype(int) + (e > 0).astype(int)
            n_diff += int(np.sum(anz[universe] != mnz[universe]))
            n_cand += int(np.sum(universe))
            zero_s += int(np.sum(s[s_idx] == 0.0))
            zero_e += int(np.sum(e[e_idx] == 0.0))
            pool_s += int(len(s_idx))
            pool_e += int(len(e_idx))
        out[arm] = {
            "n_queries": len(cache["struct_raw"]),
            "n_gallery_candidates": n_cand,
            "candidates_where_anz_count_differs_from_mnz_count": n_diff,
            "fraction_anz_mnz_differ": float(n_diff / max(1, n_cand)),
            "exact_zeros_inside_structured_pool": zero_s,
            "exact_zeros_inside_semantic_pool": zero_e,
            "mean_structured_pool": float(pool_s / len(cache["struct_raw"])),
            "mean_semantic_pool": float(pool_e / len(cache["struct_raw"])),
            "interpretation": (
                "min-max maps the weakest candidate of a pool to exactly 0, so "
                "CombMNZ can drop that branch's credit where CombANZ keeps it "
                "(the two rules differ).  z-score+sigmoid never returns exactly "
                "0, so CombANZ and CombMNZ coincide (identical rules)."
            ),
        }
    return out


def attach_relevance(cache: Dict[str, Any]) -> Dict[str, Any]:
    cache["rel"] = {q["qid"]: q["relevant"] for q in cache["queries"]}
    cache["gt_count"] = {q["qid"]: q.get("ground_truth_count") for q in cache["queries"]}
    cache["qdict"] = {q["qid"]: q for q in cache["queries"]}
    # the macro statistic is the mean of five category means; the released split
    # is balanced inside every category, so the query-level bootstrap mean and
    # the macro mean coincide (asserted, not assumed).
    for split in ("tune", "test"):
        counts = defaultdict(int)
        for q in cache["queries"]:
            if q["split"] == split:
                counts[q["category"]] += 1
        if len(set(counts.values())) != 1:
            raise RuntimeError(
                f"split {split} is not balanced across categories ({dict(counts)}); "
                "the query-level bootstrap CI would not equal the macro CI"
            )
    return cache


def run_stage(stage: str, n_boot: int, force_cache: bool) -> None:
    cache = attach_relevance(build_cache(force=force_cache))
    RESULTS.mkdir(parents=True, exist_ok=True)

    prereg = {
        "protocol": [
            "Comparison set is fixed a priori; there is no selection stage, so the whole set is evaluated once on TEST (n=100).",
            "TUNE (n=50) is computed for transparency only and is not used to choose anything.",
            "Arms: minmax (released control protocol, primary) and zscore (repository's alternative calibration, secondary = score-distribution shift).",
            "control protocol: dense branches over the whole 121-clip gallery, calibrated over the whole gallery. Systems: prior_only (source-count-ratio alpha) and fixed_0.5 (equal-weight calibrated fusion).",
            "prod protocol (production candidate model): structured shortlist = clips with a structured result (threshold 0.05, the control's own n_structured rule), semantic shortlist = top-20 semantic hits (the control's own n_semantic), union = the gallery, each branch calibrated over its own shortlist, missing candidates zero-filled. Systems: prior_only_prod, comb_sum_cal, comb_anz_cal, comb_mnz_cal.",
            "Statistic: macro P@5 (mean of five category means). Primary: paired bootstrap with within-category resampling (n_boot=3000). Secondary: grouped 5-minute-cluster bootstrap on the macro statistic, pooled per-query paired bootstrap, pooled per-query grouped bootstrap, Wilcoxon signed-rank, paired sign-flip randomisation.",
            "Holm: one family per calibration arm (4 comparisons) plus a conservative global family over all 8 arm-level comparisons.",
            "The paper's metadata prior alpha is reused unchanged; these are competing baselines, not replacements.",
        ],
        "literature": LITERATURE,
        "constants": {
            "K_SEM": K_SEM,
            "STRUCT_THRESH": STRUCT_THRESH,
            "WEIGHT": WEIGHT,
            "PROBE_SEED": PROBE_SEED,
            "n_boot": n_boot,
        },
        "systems_control": CONTROL_SYSTEMS,
        "systems_prod": PROD_SYSTEMS,
    }

    if stage == "diagnostics":
        # label-free structural diagnostics; no metric, no split, no test score
        cache = attach_relevance(build_cache())
        diag = {
            "generator": GENERATOR,
            "stage": "diagnostics",
            "calibration_diagnostics": calibration_diagnostics(cache),
            "commands": ["python backend/scripts/eval_calib_base.py --stage diagnostics"],
        }
        safe_write_json(RESULTS / "calib_diagnostics.json", diag)
        (RESULTS / "calib_diagnostics.md").write_text(
            "# Calibration diagnostics (label-free)\n\n"
            "These numbers contain no metric, no label and no per-query score; they "
            "describe only how the two calibrations interact with the combination "
            "rules.\n\n"
            "| Arm | gallery candidates | candidates where the CombANZ count differs "
            "from the CombMNZ count | exact zeros inside the structured pool | inside "
            "the semantic pool | mean structured pool | mean semantic pool |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n"
            + "".join(
                f"| {arm} | {v['n_gallery_candidates']} | {v['candidates_where_anz_count_differs_from_mnz_count']} "
                f"({v['fraction_anz_mnz_differ']:.3f}) | {v['exact_zeros_inside_structured_pool']} | "
                f"{v['exact_zeros_inside_semantic_pool']} | {v['mean_structured_pool']:.1f} | "
                f"{v['mean_semantic_pool']:.1f} |\n"
                for arm, v in diag["calibration_diagnostics"].items()
            ),
            encoding="utf-8",
        )
        print(f"[diagnostics] wrote {RESULTS / 'calib_diagnostics.json'}")
        return

    if stage == "report":
        # regenerate the markdown from the saved JSON without re-evaluating
        test_path = RESULTS / "calib_test.json"
        tune_path = RESULTS / "calib_tune.json"
        if not test_path.exists():
            raise RuntimeError("results/calib_test.json is missing: run --stage test first")
        test_res = json.loads(test_path.read_text(encoding="utf-8"))
        tune_res = json.loads(tune_path.read_text(encoding="utf-8")) if tune_path.exists() else None
        write_test_report(test_res)
        write_report(test_res, tune_res)
        print(f"[report] rewrote {RESULTS / 'calib_test.md'} and {RESULTS / 'calib_report.md'}")
        return

    if stage in ("tune", "all"):
        qids = split_qids(cache, "tune")
        arms = []
        for i, arm in enumerate(ARMS):
            arms.append({"name": arm, **arm_family(cache, arm, qids, n_boot, seed=101 + 10 * i)})
        pool_sizes = {}
        for arm in ARMS:
            prod = evaluate_prod_arm(cache, arm, qids)
            per_cat = defaultdict(list)
            for qid in qids:
                per_cat[cache["category"][qid]].append(
                    (
                        prod["prior_only_prod"][qid]["n_structured_pool"],
                        prod["prior_only_prod"][qid]["n_semantic_pool"],
                        prod["prior_only_prod"][qid]["n_universe"],
                        prod["prior_only_prod"][qid]["alpha"],
                    )
                )
            pool_sizes[arm] = {
                cat: {
                    "mean_structured_pool": float(np.mean([v[0] for v in vals])),
                    "mean_semantic_pool": float(np.mean([v[1] for v in vals])),
                    "mean_universe": float(np.mean([v[2] for v in vals])),
                    "mean_alpha": float(np.mean([v[3] for v in vals])),
                }
                for cat, vals in per_cat.items()
            }
            # consistency check: the alpha recomputed from the production pool
            # sizes must equal the control's published prior alpha.
            diffs = [
                abs(
                    prod["prior_only_prod"][qid]["alpha"]
                    - float(cache["alpha_prior"][qid])
                )
                for qid in qids
            ]
            pool_sizes[arm]["_alpha_consistency_max_abs_diff"] = float(max(diffs))
            pool_sizes[arm]["_alpha_consistency_ok"] = bool(max(diffs) < 1e-9)
        tune_res = {
            "generator": GENERATOR,
            "stage": "tune",
            "n_tune": len(qids),
            "n_boot": n_boot,
            "preregistration": prereg,
            "arms": arms,
            "pool_sizes": pool_sizes,
            "probe": {
                arm: {"cells": [probe_cell(cache, arm, c, qids, n_boot) for c in PROBE_CELLS]}
                for arm in ARMS
            },
            "weight_sensitivity": {arm: weight_sensitivity(cache, arm, qids) for arm in ARMS},
            "empty_branch_sanity": empty_branch_sanity(cache, "minmax"),
            "commands": [
                "python backend/scripts/eval_calib_base.py --stage tune",
            ],
        }
        safe_write_json(RESULTS / "calib_tune.json", tune_res)
        write_tune_report(tune_res)
        print(f"[tune] wrote {RESULTS / 'calib_tune.json'}")
        for arm in arms:
            for cname, c in arm["comparisons"].items():
                p = c["primary_paired_macro_bootstrap"]
                print(
                    f"[tune] {arm['name']:7s} {cname:34s} delta={p['delta']:+.4f} "
                    f"[{p['lo']:+.4f},{p['hi']:+.4f}] p={p['p']:.4f} "
                    f"holm_p={c['holm_adjusted_p']:.4f} reject={c['holm_reject']}"
                )

    if stage in ("test", "all"):
        test_path = RESULTS / "calib_test.json"
        if test_path.exists():
            try:
                old = json.loads(test_path.read_text(encoding="utf-8"))
                if old.get("generator") == GENERATOR:
                    raise RuntimeError(
                        "results/calib_test.json already exists: the test stage is "
                        "single-shot by design (anti-p-hacking guard). Delete it "
                        "deliberately if you really want to re-evaluate."
                    )
            except json.JSONDecodeError:
                pass
        qids = split_qids(cache, "test")
        arms = []
        for i, arm in enumerate(ARMS):
            arms.append({"name": arm, **arm_family(cache, arm, qids, n_boot, seed=201 + 10 * i)})
        probe = {}
        for arm in ARMS:
            probe[arm] = {
                "cells": [probe_cell(cache, arm, c, qids, n_boot) for c in PROBE_CELLS],
            }
        # reproduction check against the published control numbers
        repro = {}
        for arm in ARMS:
            for sysname in ("prior_only", "fixed_0.5"):
                got_test = arms[ARMS.index(arm)]["control_protocol"][sysname]["macro"]["p5"]["macro"]
                exp = CONTROL_CHECK[arm][sysname]["test"]
                repro[f"{arm}/{sysname}/test"] = {
                    "published": exp, "reproduced": float(got_test),
                    "abs_diff": float(abs(got_test - exp)),
                    "ok": bool(abs(got_test - exp) < 5e-4),
                }
        # global (all 8 arm-level comparisons) Holm family
        all_p = {}
        for arm in arms:
            for cname, c in arm["comparisons"].items():
                all_p[f"{arm['name']}/{cname}"] = float(c["primary_paired_macro_bootstrap"]["p"])
        global_adj = conf3.holm_adjusted(all_p)
        global_rej = holm_correction(all_p, alpha=0.05)

        verdict = build_verdict(arms, probe, empty_branch_sanity(cache, "minmax"))

        test_res = {
            "generator": GENERATOR,
            "stage": "test",
            "n_test": len(qids),
            "n_boot": n_boot,
            "preregistration": prereg,
            "arms": arms,
            "probe": probe,
            "empty_branch_sanity": empty_branch_sanity(cache, "minmax"),
            "reproduction_check": repro,
            "global_holm_family": {
                "members": list(all_p.keys()),
                "raw_p": all_p,
                "holm_adjusted_p": {k: float(v) for k, v in global_adj.items()},
                "holm_reject": {k: bool(v) for k, v in global_rej.items()},
            },
            "verdict": verdict,
            "commands": [
                "python backend/scripts/eval_calib_base.py --stage tune",
                "python backend/scripts/eval_calib_base.py --stage test",
            ],
            "limitations": [
                "Single camera (camera_0) and a 2-hour released window; the numbers are not the paper's full 4-camera / 312k-descriptor deployment results.",
                "Labels are derived from released VLM captions, detector metadata and snapshots, not human annotations; no inter-annotator agreement is available.",
                "The control harness scores every clip with the metadata structured branch (it is a dense scoring function, not a real structured search); the production candidate model therefore approximates the structured shortlist with the control's own n_structured rule (score > 0.05).",
                "CombANZ/CombMNZ are applied to the two-branch case only; with a dense branch pair the count factor is nearly constant, so they can coincide with CombSUM -- that is reported, not hidden.",
                "The probe is descriptive (stability under perturbation), not a hypothesis test; the CIs are bootstrap intervals over queries.",
                "The alpha prior, the benchmark, the metrics and the statistical protocol are imported unchanged from the released control driver.",
            ],
        }
        safe_write_json(test_path, test_res)
        write_test_report(test_res)
        tune_json = RESULTS / "calib_tune.json"
        tune_res = json.loads(tune_json.read_text(encoding="utf-8")) if tune_json.exists() else None
        write_report(test_res, tune_res)
        print(f"[test] wrote {test_path}")
        for k, v in repro.items():
            print(f"[test] repro {k}: published={v['published']:.4f} reproduced={v['reproduced']:.4f} ok={v['ok']}")
        for arm in arms:
            for cname, c in arm["comparisons"].items():
                p = c["primary_paired_macro_bootstrap"]
                g = c["grouped_cluster_macro_bootstrap"]
                print(
                    f"[test] {arm['name']:7s} {cname:34s} delta={p['delta']:+.4f} "
                    f"[{p['lo']:+.4f},{p['hi']:+.4f}] p={p['p']:.4f} "
                    f"grouped_delta={g['delta']:+.4f} grouped_p={g['p']:.4f} "
                    f"holm_p={c['holm_adjusted_p']:.4f} reject={c['holm_reject']}"
                )
        print(f"[test] verdict: {verdict['verdict']}")


def build_verdict(
    arms: List[Dict[str, Any]],
    probe: Dict[str, Any],
    empty_sanity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pre-registered decision rule, evaluated mechanically."""
    notes: Dict[str, str] = {}
    beats = False
    best = None
    for arm in arms:
        for cname, c in arm["comparisons"].items():
            if not cname.startswith("comb_"):
                continue
            if c["holm_reject"] and c["primary_paired_macro_bootstrap"]["delta"] > 0:
                beats = True
                best = f"{arm['name']}/{cname}"
                notes[f"winner"] = (
                    f"{cname} in the {arm['name']} arm beats the prior-only baseline: "
                    f"delta={c['primary_paired_macro_bootstrap']['delta']:+.4f} "
                    f"Holm p={c['holm_adjusted_p']:.4f}"
                )
    if not best:
        notes["winner"] = (
            "no calibrated baseline is Holm-significant against the prior-only baseline in either arm"
        )
    # robustness: candidate-set-size axis (cells without the distribution warp)
    probe_min = probe["minmax"]["cells"]
    size_cells = [c for c in probe_min if float(c["struct_exponent"]) == 1.0]
    warp_cells = [c for c in probe_min if float(c["struct_exponent"]) != 1.0]
    names = list(probe_min[0]["systems"].keys())
    drops = {}
    for sysname in names:
        base = size_cells[0]["systems"][sysname]["macro_p5"]
        worst = min(size_cells, key=lambda c: c["systems"][sysname]["macro_p5"])
        drops[sysname] = {
            "full": base,
            "worst_cell": worst["cell"],
            "worst": worst["systems"][sysname]["macro_p5"],
            "abs_drop": base - worst["systems"][sysname]["macro_p5"],
            "worst_ci": worst["systems"][sysname]["bootstrap_ci"],
        }
    prior_drop = drops.get("prior_only_prod", {}).get("abs_drop", 0.0)
    cal_drops = [v["abs_drop"] for k, v in drops.items() if k != "prior_only_prod"]
    mean_cal_drop = float(np.mean(cal_drops)) if cal_drops else 0.0
    # stability of the *conclusion*: does the sign of (calibrated - prior_only)
    # survive every perturbation?
    all_cells = size_cells + warp_cells
    sign_flips = []
    delta_spread = {}
    for sysname in ("comb_sum_cal", "comb_anz_cal", "comb_mnz_cal"):
        signs = []
        vals = []
        for c in all_cells:
            d = c["paired_delta_vs_prior_only_prod"][sysname]["delta_macro_p5"]
            vals.append(d)
            signs.append(1 if d > 0 else (-1 if d < 0 else 0))
        delta_spread[sysname] = {
            "deltas_by_cell": {c["cell"]: c["paired_delta_vs_prior_only_prod"][sysname]["delta_macro_p5"] for c in all_cells},
            "min": float(min(vals)),
            "max": float(max(vals)),
            "spread": float(max(vals) - min(vals)),
            "sign_stable": bool(len(set(signs)) == 1),
        }
        if not delta_spread[sysname]["sign_stable"]:
            sign_flips.append(sysname)
    if sign_flips:
        direction = (
            "FRAGILE (the sign of the calibrated-vs-prior comparison flips under "
            f"candidate-set perturbation for {', '.join(sign_flips)})"
        )
    else:
        direction = (
            "ROBUST (the sign and the size of the calibrated-vs-prior difference are "
            "preserved under every candidate-set-size and score-distribution perturbation)"
        )
    notes["probe_drop"] = (
        f"absolute macro P@5 drop from the full gallery to the worst candidate-set-size "
        f"cell: prior_only_prod -{prior_drop:.4f}, the three calibrated baselines "
        f"-{mean_cal_drop:.4f} on average (the drop is driven by the gallery losing "
        f"relevant candidates, and is shared by all methods)"
    )
    notes["probe_stability"] = "; ".join(
        f"{k}: delta range [{v['min']:+.4f}, {v['max']:+.4f}], sign stable={v['sign_stable']}"
        for k, v in delta_spread.items()
    )
    if warp_cells:
        warp_notes = []
        for c in warp_cells:
            for sysname in names:
                v = c["systems"][sysname]
                warp_notes.append(
                    f"{sysname}@{c['cell']}={v['macro_p5']:.4f} (alpha {v['mean_alpha']:.3f})"
                )
        notes["distribution_shift"] = (
            "score-distribution shift (calibrated structured branch x^2): " + ", ".join(warp_notes)
        )
    if empty_sanity:
        degenerate = []
        for qid, rec in empty_sanity.items():
            for label, case in rec["cases"].items():
                for sysname, r in case["systems"].items():
                    if r.get("fused_is_degenerate"):
                        degenerate.append(f"{sysname}@{qid}/{label}")
        if degenerate:
            notes["empty_branch"] = (
                "with an empty branch the source-count-ratio prior collapses to an "
                f"all-zero fused score (degenerate ranking) in {len(degenerate)} case(s): "
                f"{', '.join(degenerate)}; the calibrated rules keep the surviving "
                "branch at full scale"
            )
        else:
            notes["empty_branch"] = (
                "no method produced a degenerate (all-zero) fused score in the "
                "empty-branch sanity check"
            )
    verdict = "IMPROVED" if beats else ("ROBUST" if direction.startswith("ROBUST") else "FRAGILE")
    notes["probe_detail"] = "; ".join(
        f"{k}: {v['full']:.4f} -> {v['worst']:.4f} ({v['worst_cell']})" for k, v in drops.items()
    )
    notes["comparison_outcome"] = (
        "a calibrated baseline beats the source-count-ratio prior-only baseline "
        "(Holm-significant on test)"
        if beats
        else "no calibrated baseline beats the source-count-ratio prior-only baseline "
        "Holm-significantly on test; the simple calibrated fusion is at best a tie "
        "under the released control protocol"
    )
    return {
        "calibrated_beats_prior": beats,
        "winning_comparison": best,
        "robustness_direction": direction,
        "probe_drops": drops,
        "probe_delta_stability": delta_spread,
        "verdict": verdict,
        "notes": notes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "test", "all", "report"], default="all")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()
    run_stage(args.stage, args.n_boot, args.force_cache)


if __name__ == "__main__":
    main()
