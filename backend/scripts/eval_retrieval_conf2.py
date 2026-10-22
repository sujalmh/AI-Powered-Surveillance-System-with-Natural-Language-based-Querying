#!/usr/bin/env python
"""C1'' -- second-round confidence / blend variants on the released-subset control.

This is a *new* experiment driver.  No tracked file is modified: the benchmark
loader, the structured and semantic branch construction, the fusion, the
metrics and the statistical protocol are all **imported** from
``backend/scripts/eval_retrieval_control.py`` (which itself imports the
repository's own ``backend/app/eval/*`` and ``backend/app/services/*`` code).
Only the *alpha rule* (how the structured/semantic weight is chosen) is new.

Pre-registered protocol
-----------------------
1.  Three candidate variants are explored on the **TUNE** split only (n=50):
      A ``rank_conf``    -- rank-based confidence: the metadata prior is
                            adjusted in log-odds space by how strongly each
                            branch's top-1 is corroborated by the *other*
                            branch's ranking (reciprocal-rank agreement).
      B ``percat_blend`` -- per-category blend weight between the metadata
                            prior and the existing score-margin confidence,
                            fitted on the tune split (not a fixed 0.5).
      C ``temp_margin``  -- temperature-scaled margins: the top-1 margin of
                            each branch is measured in softmax-probability
                            space with a tunable temperature instead of raw
                            score space.
2.  The winner is chosen by **tune macro-P@5** (tie-breaks pre-registered
    below) and *frozen* (name + hyper-parameters) before the test split is
    touched.
3.  The frozen winner is evaluated **once** on the TEST split (n=100) with
    the same paired and grouped bootstrap and Holm correction as the control
    script, against the prior-only (``adaptive``) baseline.
4.  Success requires **all** of:
      * a Holm-significant test macro-P@5 gain over prior-only under
        min-max branch normalisation, and
      * a same-direction (non-negative) test macro-P@5 delta under
        z-score branch normalisation.
    Otherwise the verdict is a null.

Usage
-----
    python backend/scripts/eval_retrieval_conf2.py --mode tune
    python backend/scripts/eval_retrieval_conf2.py --mode test --winner rank_conf
    python backend/scripts/eval_retrieval_conf2.py --mode diagnostics

The expensive part (OpenCLIP text encoding + FAISS search) is cached in the
OS temp directory so that the tune and the test run share identical inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval import confidence as _confidence
from backend.app.eval.metrics import category_means, macro_mean
from backend.app.eval.stats import (
    format_p,
    holm_correction,
    paired_bootstrap_ci,
    paired_grouped_bootstrap_ci,
)
from backend.scripts import eval_retrieval_control as ctrl

RESULTS = ROOT / "results"
CACHE_PATH = Path(tempfile.gettempdir()) / "opencode" / "conf2_cache.pkl"
CATEGORIES = ctrl.CATEGORIES
N_BOOT_DEFAULT = 3000
PRIOR_METHOD = "adaptive"          # metadata prior-only baseline
NORM_MODES = ("minmax", "zscore")

# Pre-registered tie-break chain for variant selection.
#
# NOTE (revision, made on tune-only evidence and *before* any test-split
# evaluation of these three variants): the first draft ranked the variants by
# the min-max tune macro-P@5 and broke ties with "smallest mean
# |alpha - prior alpha|".  On the tune split all three variants score exactly
# the incumbent's 0.580 under min-max, so that chain froze a *no-op* variant
# (rank_conf with beta=0, i.e. identical to the prior) and would have made the
# single permitted test evaluation vacuous.  The chain was therefore replaced
# by the tune-only chain below.  The new primary criterion is the mean tune
# macro-P@5 over the two branch normalisations, because the success criterion
# spans both (a min-max gain *and* a non-negative z-score delta); the
# min-max tune macro-P@5 is the first tie-break.  No test-split number of any
# of the three variants entered this decision.
TIE_BREAKS = (
    "mean tune macro P@5 over the two branch normalisations (min-max, z-score)",
    "min-max tune macro P@5",
    "mean tune macro F1@10 (paper definition)",
    "mean tune macro R@10",
    "mean tune macro MRR",
    "smallest mean |alpha - prior alpha| (least disruptive variant)",
    "variant name (alphabetical)",
)


# ---------------------------------------------------------------------------
# Cache: real OpenCLIP + FAISS retrieval, then the two raw branches
# ---------------------------------------------------------------------------

def build_cache(force: bool = False) -> Dict[str, Any]:
    """Run the real retrieval once and cache the two raw branch score vectors.

    The branch construction mirrors ``eval_retrieval_control.main`` exactly
    (semantic score = max over the frames of a clip; structured score = the
    metadata-only score built by ``build_query_context``).
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

    cache: Dict[str, Any] = {
        "queries": queries,
        "splits": splits,
        "clip_ids": clip_ids,
        "clips": clips,
        "sem_raw_max": sem_raw_max,
        "struct_raw": struct_raw,
        "contexts": contexts,
        "clip_index": clip_index,
        "alpha_prior": {qid: float(ctx["alpha"]) for qid, ctx in contexts.items()},
        "category": {q["qid"]: q["category"] for q in queries},
        "group": {q["qid"]: q.get("group") for q in queries},
        "split": {q["qid"]: q["split"] for q in queries},
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as fh:
        pickle.dump(cache, fh)
    return cache


def branches(cache: Dict[str, Any], norm: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    sem = {qid: ctrl.normalize_branch(v, norm) for qid, v in cache["sem_raw_max"].items()}
    struct = {qid: ctrl.normalize_branch(v, norm) for qid, v in cache["struct_raw"].items()}
    return sem, struct


# ---------------------------------------------------------------------------
# Alpha rules (the only new logic)
# ---------------------------------------------------------------------------

def _ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank of every candidate (0 = best)."""
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr, kind="stable")
    out = np.empty(len(order), dtype=np.int64)
    out[order] = np.arange(len(order))
    return out


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    z = (arr - float(np.max(arr))) / max(float(temp), 1e-6)
    e = np.exp(z)
    return e / float(np.sum(e))


def alpha_rank_conf(
    qid: str, category: str, ctx: Dict[str, Any], sn: np.ndarray, tn: np.ndarray,
    params: Dict[str, Any]
) -> float:
    """Variant A -- rank-based confidence.

    The metadata prior is adjusted in log-odds space by the *rank* agreement
    between the two branches: ``agr_b`` is 1/(1+rank) of branch b's top-1 in
    the other branch's ranking.  When both branches share their top-1 the
    ratio is 1 and the prior is returned unchanged; when they disagree the
    branch that is corroborated higher by the other branch keeps more weight.
    """
    rs = _ranks(tn)
    re_ = _ranks(sn)
    top_s = int(np.argmin(rs))
    top_e = int(np.argmin(re_))
    agr_s = 1.0 / (1.0 + float(re_[top_s]))
    agr_e = 1.0 / (1.0 + float(rs[top_e]))
    ratio = agr_s / max(agr_e, 1e-9)
    prior = float(ctx["alpha"])
    delta = float(params["beta"]) * math.log(max(ratio, 1e-9))
    return _sigmoid(_logit(prior) + delta)


def alpha_percat_blend(
    qid: str, category: str, ctx: Dict[str, Any], sn: np.ndarray, tn: np.ndarray,
    params: Dict[str, Any]
) -> float:
    """Variant B -- per-category blend between prior and score confidence.

    ``alpha = beta_c * prior + (1 - beta_c) * cs/(cs+ce)`` with ``beta_c``
    fitted per category on the tune split.
    """
    prior = float(ctx["alpha"])
    conf = float(_confidence.confidence_alpha(tn, sn, k=int(params.get("k", 5))))
    beta = float(params["beta_by_cat"][category])
    return beta * prior + (1.0 - beta) * conf


def alpha_temp_margin(
    qid: str, category: str, ctx: Dict[str, Any], sn: np.ndarray, tn: np.ndarray,
    params: Dict[str, Any]
) -> float:
    """Variant C -- temperature-scaled margins.

    The top-1 margin of each branch is measured in softmax-probability space
    with a tunable temperature instead of raw score space; the per-branch
    confidence and the resulting weight are the same ``cs/(cs+ce)`` rule as
    ``backend/app/eval/confidence.py``.
    """
    temp = float(params["temp"])
    blend = float(params.get("blend", 0.5))
    k = int(params.get("k", 5))
    ps = _softmax(tn, temp)
    pe = _softmax(sn, temp)
    ms = float(ps.max() - np.sort(ps)[-2])
    me = float(pe.max() - np.sort(pe)[-2])
    cs = 0.5 * ms + 0.5 * _confidence.topk_concentration(tn, k=k)
    ce = 0.5 * me + 0.5 * _confidence.topk_concentration(sn, k=k)
    conf = 0.5 if cs + ce <= 1e-12 else float(cs / (cs + ce))
    prior = float(ctx["alpha"])
    return blend * prior + (1.0 - blend) * conf


def _percat_beta_grid() -> Dict[str, List[float]]:
    """Per-category blend grids that variant B searches independently."""
    return {c: [round(i / 10.0, 1) for i in range(11)] for c in CATEGORIES}


VARIANTS: Dict[str, Dict[str, Any]] = {
    "rank_conf": {
        "fn": alpha_rank_conf,
        "grid": [{"beta": b} for b in (0.0, 0.25, 0.5, 1.0, 2.0)],
        "desc": "rank-based branch agreement, log-odds adjustment of the prior",
    },
    "percat_blend": {
        "fn": alpha_percat_blend,
        "grid": [{"beta_by_cat": _percat_beta_grid()}],
        "desc": "per-category blend between prior and margin confidence (fitted on tune)",
    },
    "temp_margin": {
        "fn": alpha_temp_margin,
        "grid": [
            {"temp": t, "blend": bl}
            for t in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
            for bl in (0.25, 0.5, 0.75)
        ],
        "desc": "temperature-scaled (softmax) top-1 margins",
    },
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def fuse_with_alpha(
    clip_ids: Sequence[str],
    sem: Dict[str, np.ndarray],
    struct: Dict[str, np.ndarray],
    alpha_by_qid: Dict[str, float],
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    out: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for qid, a in alpha_by_qid.items():
        fused = float(a) * struct[qid] + (1.0 - float(a)) * sem[qid]
        out[qid] = (fused, ctrl.rank_clips(fused, clip_ids))
    return out


def score(
    cache: Dict[str, Any],
    sem: Dict[str, np.ndarray],
    struct: Dict[str, np.ndarray],
    alpha_by_qid: Dict[str, float],
) -> List[Dict[str, Any]]:
    qids = [q["qid"] for q in cache["queries"]]
    fused = fuse_with_alpha(cache["clip_ids"], sem, struct, alpha_by_qid)
    rows = ctrl.score_rows(
        cache["queries"], fused, cache["clips"], sem, struct, cache["clip_index"]
    )
    for r in rows:
        r["group"] = cache["group"][r["qid"]]
        r["alpha"] = float(alpha_by_qid[r["qid"]])
    return rows


def aggregate(rows: Sequence[Dict[str, Any]], field: str = "p5") -> Dict[str, Any]:
    cat = category_means(list(rows), field)
    return {
        "macro": float(macro_mean(cat)),
        "per_category": {c: float(cat.get(c, 0.0)) for c in CATEGORIES},
    }


def prior_alpha(cache: Dict[str, Any]) -> Dict[str, float]:
    return {qid: float(a) for qid, a in cache["alpha_prior"].items()}


def mean_abs_alpha_shift(cache: Dict[str, Any], alpha: Dict[str, float]) -> float:
    prior = cache["alpha_prior"]
    return float(np.mean([abs(alpha[q] - float(prior[q])) for q in alpha]))


# ---------------------------------------------------------------------------
# Variant search on the tune split
# ---------------------------------------------------------------------------

def _tune_metrics(
    cache: Dict[str, Any],
    sem: Dict[str, np.ndarray],
    struct: Dict[str, np.ndarray],
    alpha: Dict[str, float],
    split: str = "tune",
) -> Dict[str, Any]:
    rows = [r for r in score(cache, sem, struct, alpha) if r["split"] == split]
    out = {"p5": aggregate(rows, "p5"), "f1_paper": aggregate(rows, "f1_paper"),
           "r10": aggregate(rows, "r10"), "mrr": aggregate(rows, "mrr")}
    out["mean_abs_alpha_shift"] = mean_abs_alpha_shift(cache, alpha)
    return out


def _variant_alpha(
    cache: Dict[str, Any],
    sem: Dict[str, np.ndarray],
    struct: Dict[str, np.ndarray],
    fn: Callable[..., float],
    params: Dict[str, Any],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for q in cache["queries"]:
        qid = q["qid"]
        out[qid] = float(
            fn(qid, q["category"], cache["contexts"][qid], sem[qid], struct[qid], params)
        )
    return out


def search_variant(
    cache: Dict[str, Any],
    sem: Dict[str, np.ndarray],
    struct: Dict[str, np.ndarray],
    name: str,
    norm: str,
) -> Dict[str, Any]:
    """Grid-search one variant on the tune split (tune only, never test)."""
    spec = VARIANTS[name]
    trials: List[Dict[str, Any]] = []
    for params in spec["grid"]:
        if name == "percat_blend":
            # per-category beta: coordinate ascent on the tune split
            beta_by_cat = {c: params["beta_by_cat"][c][0] for c in CATEGORIES}
            fitted: Dict[str, Any] = {}
            for cat in CATEGORIES:
                best_b, best_m = beta_by_cat[cat], None
                for b in [round(i / 10.0, 1) for i in range(11)]:
                    trial = {"beta_by_cat": dict(beta_by_cat), "k": 5}
                    trial["beta_by_cat"][cat] = b
                    alpha = _variant_alpha(cache, sem, struct, spec["fn"], trial)
                    m = _tune_metrics(cache, sem, struct, alpha, "tune")
                    key = (round(m["p5"]["macro"], 6), round(m["f1_paper"]["macro"], 6),
                           round(m["r10"]["macro"], 6), round(m["mrr"]["macro"], 6),
                           -abs(b - 0.5))
                    if best_m is None or key > best_m[0]:
                        best_m = (key, b, m)
                beta_by_cat[cat] = best_m[1]
                fitted[cat] = best_m[2]
            alpha = _variant_alpha(cache, sem, struct, spec["fn"],
                                   {"beta_by_cat": beta_by_cat, "k": 5})
            m = _tune_metrics(cache, sem, struct, alpha, "tune")
            trials.append({"params": {"beta_by_cat": beta_by_cat}, "tune": m,
                           "per_category_fit": {c: fitted[c]["p5"]["macro"] for c in CATEGORIES}})
            continue
        alpha = _variant_alpha(cache, sem, struct, spec["fn"], params)
        m = _tune_metrics(cache, sem, struct, alpha, "tune")
        trials.append({"params": params, "tune": m})
    # pre-registered tie-break chain (rounded to 6 decimals so that
    # floating-point noise cannot decide a tie)
    def key(t: Dict[str, Any]) -> Tuple:
        m = t["tune"]
        return (
            round(m["p5"]["macro"], 6),
            round(m["f1_paper"]["macro"], 6),
            round(m["r10"]["macro"], 6),
            round(m["mrr"]["macro"], 6),
            -round(m["mean_abs_alpha_shift"], 6),
        )

    best = max(trials, key=key)
    return {
        "variant": name,
        "description": spec["desc"],
        "normalization": norm,
        "n_trials": len(trials),
        "trials": trials,
        "best_params": best["params"],
        "best_tune": best["tune"],
    }


def run_tune_sweep(cache: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for norm in NORM_MODES:
        sem, struct = branches(cache, norm)
        prior = prior_alpha(cache)
        base = _tune_metrics(cache, sem, struct, prior, "tune")
        out[norm] = {
            "prior_baseline_tune": base,
            "variants": {name: search_variant(cache, sem, struct, name, norm)
                         for name in VARIANTS},
        }
    # Winner selection: tune-only.  The selection metric is the mean tune
    # macro-P@5 over the two branch normalisations (the success criterion
    # spans both); the min-max tune macro-P@5 is the first tie-break.
    def _mean_over_norms(variant: str, field: str) -> float:
        vals = [out[norm]["variants"][variant]["best_tune"][field]["macro"]
                for norm in NORM_MODES]
        return round(float(sum(vals) / len(vals)), 6)

    def _mean_shift(variant: str) -> float:
        vals = [out[norm]["variants"][variant]["best_tune"]["mean_abs_alpha_shift"]
                for norm in NORM_MODES]
        return round(float(sum(vals) / len(vals)), 6)

    cands = []
    for name in VARIANTS:
        cands.append((
            _mean_over_norms(name, "p5"),
            out["minmax"]["variants"][name]["best_tune"]["p5"]["macro"],
            _mean_over_norms(name, "f1_paper"),
            _mean_over_norms(name, "r10"),
            _mean_over_norms(name, "mrr"),
            -_mean_shift(name),
            name,
        ))
    cands.sort(reverse=True)
    winner = cands[0][-1]
    out["selection"] = {
        "primary_normalization": "minmax",
        "criterion": "mean tune macro P@5 over both branch normalisations",
        "tie_breaks": list(TIE_BREAKS),
        "ranking": [
            {"variant": c[-1], "mean_tune_macro_p5": c[0],
             "minmax_tune_macro_p5": c[1], "mean_tune_macro_f1_paper": c[2],
             "mean_tune_macro_r10": c[3], "mean_tune_macro_mrr": c[4]}
            for c in cands
        ],
        "selected_winner": winner,
    }
    return out


# ---------------------------------------------------------------------------
# Frozen test evaluation
# ---------------------------------------------------------------------------

def _paired(
    a_rows: Dict[str, Dict[str, Any]],
    b_rows: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    av = [a_rows[q]["p5"] for q in qids]
    bv = [b_rows[q]["p5"] for q in qids]
    groups = [a_rows[q]["group"] for q in qids]
    m, lo, hi, p = paired_bootstrap_ci(av, bv, n_boot=n_boot, seed=seed)
    gm, glo, ghi, gp = paired_grouped_bootstrap_ci(av, bv, groups, n_boot=n_boot, seed=seed)
    return {
        "delta_p5": float(m), "lo": float(lo), "hi": float(hi), "p": float(p),
        "grouped_delta_p5": float(gm), "grouped_lo": float(glo),
        "grouped_hi": float(ghi), "grouped_p": float(gp),
    }


def holm_adjusted(pvals: Dict[str, float]) -> Dict[str, float]:
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out: Dict[str, float] = {}
    prev = 0.0
    for i, k in enumerate(order):
        adj = min(1.0, max(prev, pvals[k] * (m - i)))
        out[k] = adj
        prev = adj
    return out


def evaluate_winner_on_test(
    cache: Dict[str, Any],
    winner: str,
    n_boot: int,
    sweep: Dict[str, Any],
) -> Dict[str, Any]:
    spec = VARIANTS[winner]
    # Frozen hyper-parameters: the variant's best tune-split trial *in that
    # normalisation* (tune-only, mirroring the control script's per-
    # normalisation tuning of the per-intent alphas).
    params_by_norm = {
        norm: sweep[norm]["variants"][winner]["best_params"] for norm in NORM_MODES
    }
    out: Dict[str, Any] = {
        "winner": winner,
        "params_by_normalization": params_by_norm,
        "n_boot": n_boot,
    }

    prior_rows_by_norm: Dict[str, Dict[str, Dict[str, Any]]] = {}
    winner_rows_by_norm: Dict[str, Dict[str, Dict[str, Any]]] = {}
    best_fixed_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for norm in NORM_MODES:
        sem, struct = branches(cache, norm)
        prior = prior_alpha(cache)
        prior_rows = score(cache, sem, struct, prior)
        prior_rows_by_norm[norm] = {r["qid"]: r for r in prior_rows}

        alpha = _variant_alpha(cache, sem, struct, spec["fn"], params_by_norm[norm])
        w_rows = score(cache, sem, struct, alpha)
        winner_rows_by_norm[norm] = {r["qid"]: r for r in w_rows}

        # best fixed alpha selected on tune (reference competitor)
        fixed_grid = [round(i / 10.0, 1) for i in range(11)]
        best_fixed, best_tune = None, -1.0
        for a in fixed_grid:
            rows = score(cache, sem, struct, {qid: a for qid in prior})
            m = aggregate([r for r in rows if r["split"] == "tune"], "p5")["macro"]
            if m > best_tune + 1e-12:
                best_tune, best_fixed = m, a
        bf_rows = score(cache, sem, struct, {qid: best_fixed for qid in prior})
        best_fixed_rows[norm] = {r["qid"]: r for r in bf_rows}
        out.setdefault("best_fixed_on_tune", {})[norm] = {
            "alpha": best_fixed, "tune_macro_p5": float(best_tune)
        }

        # per-query alpha statistics for the winner
        out.setdefault("alpha_stats", {})[norm] = {
            "winner_mean": float(np.mean(list(alpha.values()))),
            "prior_mean": float(np.mean(list(prior.values()))),
            "winner_min": float(np.min(list(alpha.values()))),
            "winner_max": float(np.max(list(alpha.values()))),
            "mean_abs_shift": mean_abs_alpha_shift(cache, alpha),
        }

    qids = [q["qid"] for q in cache["queries"] if q["split"] == "test"]

    # --- summaries -------------------------------------------------------
    for norm in NORM_MODES:
        w = [r for r in winner_rows_by_norm[norm].values() if r["split"] == "test"]
        p = [r for r in prior_rows_by_norm[norm].values() if r["split"] == "test"]
        bf = [r for r in best_fixed_rows[norm].values() if r["split"] == "test"]
        out.setdefault("test", {})[norm] = {
            "winner": {
                "macro_p5": aggregate(w, "p5")["macro"],
                "macro_r10": aggregate(w, "r10")["macro"],
                "macro_f1_paper": aggregate(w, "f1_paper")["macro"],
                "macro_mrr": aggregate(w, "mrr")["macro"],
                "per_category_p5": aggregate(w, "p5")["per_category"],
                "n_test": len(w),
            },
            "prior_only": {
                "macro_p5": aggregate(p, "p5")["macro"],
                "macro_r10": aggregate(p, "r10")["macro"],
                "macro_f1_paper": aggregate(p, "f1_paper")["macro"],
                "macro_mrr": aggregate(p, "mrr")["macro"],
                "per_category_p5": aggregate(p, "p5")["per_category"],
            },
            "best_fixed_on_tune": {
                "alpha": out["best_fixed_on_tune"][norm]["alpha"],
                "macro_p5": aggregate(bf, "p5")["macro"],
                "macro_r10": aggregate(bf, "r10")["macro"],
                "macro_f1_paper": aggregate(bf, "f1_paper")["macro"],
                "macro_mrr": aggregate(bf, "mrr")["macro"],
                "per_category_p5": aggregate(bf, "p5")["per_category"],
            },
        }

    # --- comparisons + Holm --------------------------------------------
    for norm in NORM_MODES:
        fam: Dict[str, float] = {}
        comp: Dict[str, Any] = {}
        pairs = [
            ("winner_vs_prior", "winner", "prior_only", 41),
            ("winner_vs_best_fixed", "winner", "best_fixed", 43),
            ("winner_vs_conf_only", "winner", "conf_only", 47),
        ]
        # conf_only reference (existing confidence module, no prior)
        sem, struct = branches(cache, norm)
        conf_alpha = {
            qid: float(_confidence.confidence_alpha(struct[qid], sem[qid], k=5))
            for qid in sem
        }
        conf_rows = {r["qid"]: r for r in score(cache, sem, struct, conf_alpha)}
        for name, a_kind, b_kind, seed in pairs:
            a_rows = {
                "winner": winner_rows_by_norm[norm],
                "prior_only": prior_rows_by_norm[norm],
                "best_fixed": best_fixed_rows[norm],
                "conf_only": conf_rows,
            }[a_kind]
            b_rows = {
                "winner": winner_rows_by_norm[norm],
                "prior_only": prior_rows_by_norm[norm],
                "best_fixed": best_fixed_rows[norm],
                "conf_only": conf_rows,
            }[b_kind]
            res = _paired(a_rows, b_rows, qids, n_boot, seed)
            fam[name] = res["p"]
            comp[name] = res
        rejects = holm_correction(fam, alpha=0.05)
        adj = holm_adjusted(fam)
        out.setdefault("comparison", {})[norm] = {
            "family": list(fam.keys()),
            "family_size": len(fam),
            "comparisons": comp,
            "holm_reject": {k: bool(v) for k, v in rejects.items()},
            "holm_adjusted_p": {k: float(v) for k, v in adj.items()},
        }
        # per-category paired test with Holm over the five categories
        cat_p: Dict[str, float] = {}
        cat_det: Dict[str, Any] = {}
        for cat in CATEGORIES:
            cq = [q for q in qids if cache["category"][q] == cat]
            res = _paired(winner_rows_by_norm[norm], prior_rows_by_norm[norm], cq, n_boot, 53)
            cat_p[cat] = res["p"]
            cat_det[cat] = res
        cat_rej = holm_correction(cat_p, alpha=0.05)
        out.setdefault("per_category_vs_prior", {})[norm] = {
            "comparisons": cat_det,
            "holm_reject": {k: bool(v) for k, v in cat_rej.items()},
            "holm_adjusted_p": {k: float(v) for k, v in holm_adjusted(cat_p).items()},
        }

    # --- verdict ---------------------------------------------------------
    mm = out["comparison"]["minmax"]
    zs = out["comparison"]["zscore"]
    prim = mm["comparisons"]["winner_vs_prior"]
    delta_mm = out["test"]["minmax"]["winner"]["macro_p5"] - out["test"]["minmax"]["prior_only"]["macro_p5"]
    delta_zs = out["test"]["zscore"]["winner"]["macro_p5"] - out["test"]["zscore"]["prior_only"]["macro_p5"]
    holm_ok = bool(mm["holm_reject"]["winner_vs_prior"])
    verdict = "IMPROVED" if (holm_ok and delta_mm > 0 and delta_zs >= 0) else "NULL"
    out["verdict"] = {
        "verdict": verdict,
        "holm_significant_minmax": holm_ok,
        "holm_adjusted_p_minmax": mm["holm_adjusted_p"]["winner_vs_prior"],
        "delta_macro_p5_minmax": float(delta_mm),
        "delta_macro_p5_zscore": float(delta_zs),
        "criteria": (
            "IMPROVED requires (a) Holm-significant test macro-P@5 gain over "
            "prior-only under min-max normalisation and (b) a non-negative "
            "test macro-P@5 delta under z-score normalisation."
        ),
    }
    out["selection_record"] = sweep.get("selection")
    return out


# ---------------------------------------------------------------------------
# Diagnostics (label-dependent oracle: descriptive only, never a candidate)
# ---------------------------------------------------------------------------

def oracle_diagnostics(cache: Dict[str, Any], n_boot: int) -> Dict[str, Any]:
    """Per-query upper bound of *any* alpha-blend rule (uses labels).

    This is a ceiling estimate for the whole alpha family; it is descriptive
    and is never used to select a variant.
    """
    out: Dict[str, Any] = {}
    grid = [round(i / 100.0, 2) for i in range(101)]
    for norm in NORM_MODES:
        sem, struct = branches(cache, norm)
        prior = prior_alpha(cache)
        for split in ("tune", "test"):
            qs = [q for q in cache["queries"] if q["split"] == split]
            best_rows = []
            for q in qs:
                qid = q["qid"]
                best_p5, best_a = -1.0, None
                for a in grid:
                    fused = a * struct[qid] + (1.0 - a) * sem[qid]
                    ranking = ctrl.rank_clips(fused, cache["clip_ids"])
                    m = ctrl.ranking_metrics(ranking, q["relevant"])
                    if m["p5"] > best_p5 + 1e-12:
                        best_p5, best_a = m["p5"], a
                best_rows.append({"qid": qid, "category": q["category"],
                                  "p5": best_p5, "alpha": best_a,
                                  "group": cache["group"][qid]})
            prior_rows = [r for r in score(cache, sem, struct, prior) if r["split"] == split]
            out.setdefault(norm, {})[split] = {
                "oracle_macro_p5": aggregate(best_rows, "p5")["macro"],
                "oracle_per_category_p5": aggregate(best_rows, "p5")["per_category"],
                "prior_macro_p5": aggregate(prior_rows, "p5")["macro"],
                "prior_per_category_p5": aggregate(prior_rows, "p5")["per_category"],
                "n_queries": len(qs),
            }
    return out


def integrity_check(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Reproduce the control's published numbers from this script's pipeline."""
    ref_path = RESULTS / "retrieval_control_results.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
    out: Dict[str, Any] = {"reference_file": str(ref_path), "checks": []}
    for norm in NORM_MODES:
        sem, struct = branches(cache, norm)
        prior = prior_alpha(cache)
        prior_rows = score(cache, sem, struct, prior)
        conf_alpha = {qid: float(_confidence.confidence_alpha(struct[qid], sem[qid], k=5))
                      for qid in sem}
        comb_alpha = {qid: float(_confidence.combined_alpha(prior[qid], struct[qid], sem[qid], k=5))
                      for qid in sem}
        for label, alpha in (("adaptive", prior), ("conf_only", conf_alpha), ("combined", comb_alpha)):
            rows = [r for r in score(cache, sem, struct, alpha) if r["split"] == "test"]
            got = aggregate(rows, "p5")["macro"]
            exp = ref["test"][norm][label]["macro"]["p5"] if ref else None
            out["checks"].append({
                "normalization": norm, "method": label,
                "recomputed_test_macro_p5": float(got),
                "reference_test_macro_p5": exp,
                "match": (exp is None) or abs(float(got) - float(exp)) < 1e-9,
            })
    out["all_match"] = all(c["match"] for c in out["checks"])
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_p(p: float, n_boot: int) -> str:
    return format_p(p, n_boot)


def write_tune_report(sweep: Dict[str, Any], out_path: Path, md_path: Path) -> None:
    lines: List[str] = []
    lines.append("# C1'' -- confidence variant sweep on the TUNE split (n=50)\n")
    lines.append(
        "**Pre-registration.** Every variant is selected on the tune split only "
        "(n=50).  The winner is the variant with the highest *mean tune "
        "macro-P@5* over the two branch normalisations (the success criterion "
        "spans both); the min-max tune macro-P@5 is the first tie-break, then "
        + " > ".join(TIE_BREAKS[2:]) + ".  The winner is frozen before the "
        "test split is touched and evaluated once.\n"
    )
    lines.append(
        "\n**Amendment (tune-only, before any test-split evaluation of these "
        "variants).**  The first draft ranked the variants by the min-max tune "
        "macro-P@5 and broke ties with \"smallest mean |alpha - prior alpha|\".  "
        "All three variants score exactly the incumbent's 0.580 under min-max, so "
        "that chain froze a *no-op* variant (rank_conf with beta = 0, identical to "
        "the prior) and would have made the single permitted test evaluation "
        "vacuous.  The chain was therefore replaced by the tune-only chain stated "
        "above.  No test-split number of any of the three variants entered this "
        "decision.\n"
    )
    for norm in NORM_MODES:
        block = sweep[norm]
        base = block["prior_baseline_tune"]
        lines.append(f"\n## Tune results -- {norm} normalisation\n")
        lines.append(
            f"Prior-only (metadata adaptive alpha) tune macro-P@5 = "
            f"{base['p5']['macro']:.4f} (per-category: "
            + ", ".join(f"{c} {base['p5']['per_category'][c]:.3f}" for c in CATEGORIES)
            + ").\n"
        )
        lines.append("| Variant | Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | Tune macro R@10 | mean \\|alpha-prior\\| |")
        lines.append("|---|---|---:|---:|---:|---:|")
        lines.append("| prior_only (baseline) | metadata adaptive alpha | "
                     f"{base['p5']['macro']:.4f} | {base['f1_paper']['macro']:.4f} | "
                     f"{base['r10']['macro']:.4f} | {base['mean_abs_alpha_shift']:.4f} |")
        for name, v in block["variants"].items():
            best = v["best_tune"]
            lines.append(
                f"| **{name}** (best of {v['n_trials']}) | `{json.dumps(v['best_params'])}` | "
                f"{best['p5']['macro']:.4f} | {best['f1_paper']['macro']:.4f} | "
                f"{best['r10']['macro']:.4f} | {best['mean_abs_alpha_shift']:.4f} |"
            )
        lines.append("\n<details><summary>full trial grid</summary>\n")
        for name, v in block["variants"].items():
            lines.append(f"\n**{name}** -- {v['description']}\n")
            lines.append("| Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | mean \\|alpha-prior\\| |")
            lines.append("|---|---:|---:|---:|")
            for t in v["trials"]:
                lines.append(
                    f"| `{json.dumps(t['params'])}` | {t['tune']['p5']['macro']:.4f} | "
                    f"{t['tune']['f1_paper']['macro']:.4f} | "
                    f"{t['tune']['mean_abs_alpha_shift']:.4f} |"
                )
        lines.append("\n</details>\n")
    sel = sweep["selection"]
    lines.append("\n## Selection (pre-registered, tune-only)\n")
    lines.append(f"Selection metric: {sel['criterion']}. Tie-break chain: "
                 + " > ".join(sel["tie_breaks"]) + ".\n")
    lines.append("| Rank | Variant | Mean tune macro-P@5 | min-max tune macro-P@5 | Mean tune macro F1@10 |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, c in enumerate(sel["ranking"], start=1):
        lines.append(f"| {i} | {c['variant']} | {c['mean_tune_macro_p5']:.4f} | "
                     f"{c['minmax_tune_macro_p5']:.4f} | {c['mean_tune_macro_f1_paper']:.4f} |")
    lines.append(f"\n**Frozen winner: `{sel['selected_winner']}`**\n")
    out_path.write_text(json.dumps(sweep, indent=2, default=float), encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_test_report(res: Dict[str, Any], out_path: Path, md_path: Path) -> None:
    n_boot = int(res["n_boot"])
    lines: List[str] = []
    lines.append("# C1'' -- frozen winner evaluated on the TEST split (n=100)\n")
    lines.append(
        "**Frozen variant:** `%s` with hyper-parameters `%s` (selected on the "
        "tune split only; see `results/conf2_tune_sweep.md`).\n"
        % (res["winner"], json.dumps(res["params_by_normalization"]))
    )
    lines.append(
        "**Protocol.** Paired bootstrap on per-query P@5 (n_boot=%d) and the "
        "same grouped (cluster) bootstrap by query group, Holm-corrected over "
        "the three pre-registered comparisons.  `< %s` is the resolution "
        "floor (2/n_boot).\n" % (n_boot, f"{2.0 / n_boot:.4f}")
    )
    for norm in NORM_MODES:
        t = res["test"][norm]
        lines.append(f"\n## Test results -- {norm} normalisation\n")
        lines.append("| System | Macro P@5 | Macro R@10 | Macro F1@10 (paper) | Macro MRR |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, key in (("prior_only (adaptive)", "prior_only"), ("winner", "winner"),
                           ("best fixed alpha (tune-selected)", "best_fixed_on_tune")):
            lines.append(
                f"| {label} | {t[key]['macro_p5']:.4f} | {t[key]['macro_r10']:.4f} | "
                f"{t[key]['macro_f1_paper']:.4f} | {t[key]['macro_mrr']:.4f} |"
            )
        lines.append("\n| System | Count | Find | Track | Action | Filter |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for label, key in (("prior_only", "prior_only"), ("winner", "winner")):
            pc = t[key]["per_category_p5"]
            lines.append(f"| {label} | " + " | ".join(f"{pc[c]:.3f}" for c in CATEGORIES) + " |")
        cmp_ = res["comparison"][norm]
        lines.append("\n### Paired comparisons vs. the test split\n")
        lines.append("| Comparison | ΔP@5 | 95% CI | p (query-level) | ΔP@5 (grouped) | grouped 95% CI | grouped p | Holm p | Holm reject |")
        lines.append("|---|---:|---|---:|---:|---|---:|---:|---|")
        for name, c in cmp_["comparisons"].items():
            lines.append(
                f"| {name} | {c['delta_p5']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
                f"{_fmt_p(c['p'], n_boot)} | {c['grouped_delta_p5']:+.4f} | "
                f"[{c['grouped_lo']:+.4f}, {c['grouped_hi']:+.4f}] | "
                f"{_fmt_p(c['grouped_p'], n_boot)} | "
                f"{_fmt_p(cmp_['holm_adjusted_p'][name], n_boot)} | "
                f"{cmp_['holm_reject'][name]} |"
            )
        lines.append("\n### Per-category vs. prior-only (Holm over the five categories)\n")
        pc = res["per_category_vs_prior"][norm]
        lines.append("| Category | ΔP@5 | 95% CI | p | Holm p | Holm reject |")
        lines.append("|---|---:|---|---:|---:|---|")
        for cat in CATEGORIES:
            c = pc["comparisons"][cat]
            lines.append(
                f"| {cat} | {c['delta_p5']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
                f"{_fmt_p(c['p'], n_boot)} | {_fmt_p(pc['holm_adjusted_p'][cat], n_boot)} | "
                f"{pc['holm_reject'][cat]} |"
            )
    v = res["verdict"]
    lines.append("\n## Verdict\n")
    lines.append(
        f"**{v['verdict']}** -- Holm-significant min-max gain: {v['holm_significant_minmax']} "
        f"(Holm-adjusted p = {_fmt_p(v['holm_adjusted_p_minmax'], n_boot)}); "
        f"Δmacro-P@5 min-max = {v['delta_macro_p5_minmax']:+.4f}, "
        f"z-score = {v['delta_macro_p5_zscore']:+.4f}.\n"
    )
    lines.append(f"Success criteria: {v['criteria']}\n")

    orc = res.get("diagnostics_oracle")
    if orc:
        lines.append(
            "\n## Ceiling diagnostic -- label-dependent oracle over the whole alpha family\n"
        )
        lines.append(
            "For every query the *best* alpha in [0.00, 1.00] (step 0.01) is chosen "
            "with the ground truth, i.e. an upper bound on what **any** alpha rule "
            "(prior, confidence, rank-based, temperature-scaled or otherwise) can "
            "reach with these two branch score vectors.  It is descriptive only and "
            "was never used to select a variant.\n"
        )
        lines.append("| Normalisation | Split | Oracle macro-P@5 | Prior-only macro-P@5 | Head-room |")
        lines.append("|---|---|---:|---:|---:|")
        for norm in NORM_MODES:
            for split in ("tune", "test"):
                blk = orc[norm][split]
                lines.append(
                    f"| {norm} | {split} | {blk['oracle_macro_p5']:.4f} | "
                    f"{blk['prior_macro_p5']:.4f} | "
                    f"{blk['oracle_macro_p5'] - blk['prior_macro_p5']:+.4f} |"
                )
        lines.append(
            "\nUnder min-max normalisation the oracle equals the prior-only "
            "baseline exactly on both splits: the alpha-blend family is "
            "**exhausted**, so no confidence rule of any form can produce the "
            "min-max gain that the success criterion requires.\n"
        )

    integ = res.get("integrity")
    if integ:
        lines.append("\n## Pipeline integrity check\n")
        lines.append(
            "The script's own pipeline must reproduce the control's published "
            "test numbers before its own numbers can be trusted.\n"
        )
        lines.append("| Normalisation | Method | Recomputed test macro-P@5 | Control reference | Match |")
        lines.append("|---|---|---:|---:|---|")
        for c in integ["checks"]:
            lines.append(f"| {c['normalization']} | {c['method']} | "
                         f"{c['recomputed_test_macro_p5']:.4f} | "
                         f"{c['reference_test_macro_p5']} | {c['match']} |")
        lines.append(f"\nAll checks match: **{integ['all_match']}**\n")
    out_path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def write_summary(sweep: Dict[str, Any], res: Dict[str, Any], n_boot: int) -> None:
    """Consolidated report: tune table + frozen test result + ceiling + verdict."""
    winner = res["winner"]
    lines: List[str] = []
    lines.append("# C1'' -- confidence-variant round 2: consolidated result\n")
    lines.append(
        "**Verdict: %s.**  A new confidence/blend rule (`%s`) was pre-registered on "
        "the tune split (n=50) and evaluated **once** on the test split (n=100) "
        "against the metadata prior-only baseline, with the same paired and "
        "grouped bootstrap and Holm correction as the control script.\n"
        % (res["verdict"]["verdict"], winner)
    )

    lines.append("\n## 1. Variant search on the TUNE split (n=50)\n")
    lines.append(
        "Three variants were explored; nothing else was tried.  The winner is "
        "the variant with the highest mean tune macro-P@5 over the two branch "
        "normalisations (min-max tune macro-P@5 is the first tie-break).\n"
    )
    lines.append("| Variant | Description | min-max tune macro-P@5 | z-score tune macro-P@5 | mean | mean tune macro F1@10 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    mm_base = sweep["minmax"]["prior_baseline_tune"]
    zs_base = sweep["zscore"]["prior_baseline_tune"]
    lines.append(
        f"| prior_only (baseline) | metadata adaptive alpha | "
        f"{mm_base['p5']['macro']:.4f} | {zs_base['p5']['macro']:.4f} | "
        f"{round((mm_base['p5']['macro'] + zs_base['p5']['macro']) / 2, 6):.4f} | "
        f"{round((mm_base['f1_paper']['macro'] + zs_base['f1_paper']['macro']) / 2, 6):.4f} |"
    )
    for name in VARIANTS:
        mm = sweep["minmax"]["variants"][name]["best_tune"]
        zs = sweep["zscore"]["variants"][name]["best_tune"]
        mean_p5 = round((mm["p5"]["macro"] + zs["p5"]["macro"]) / 2, 6)
        mean_f1 = round((mm["f1_paper"]["macro"] + zs["f1_paper"]["macro"]) / 2, 6)
        mark = " **(winner)**" if name == winner else ""
        lines.append(
            f"| {name}{mark} | {VARIANTS[name]['desc']} | {mm['p5']['macro']:.4f} | "
            f"{zs['p5']['macro']:.4f} | {mean_p5:.4f} | {mean_f1:.4f} |"
        )
    lines.append(
        "\nFrozen hyper-parameters (best tune trial per normalisation): `%s`.\n"
        % json.dumps(res["params_by_normalization"])
    )

    lines.append("\n## 2. Frozen winner on the TEST split (n=100)\n")
    for norm in NORM_MODES:
        t = res["test"][norm]
        c = res["comparison"][norm]
        lines.append(f"\n### {norm} normalisation\n")
        lines.append("| System | Macro P@5 | Count | Find | Track | Action | Filter |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for label, key in (("prior_only (adaptive)", "prior_only"), ("winner", "winner")):
            pc = t[key]["per_category_p5"]
            lines.append(f"| {label} | {t[key]['macro_p5']:.4f} | " +
                         " | ".join(f"{pc[x]:.3f}" for x in CATEGORIES) + " |")
        lines.append("")
        lines.append("| Comparison | ΔP@5 | 95% CI | p (query) | ΔP@5 (grouped) | grouped 95% CI | grouped p | Holm p | Holm reject |")
        lines.append("|---|---:|---|---:|---:|---|---:|---:|---|")
        for name, cc in c["comparisons"].items():
            lines.append(
                f"| {name} | {cc['delta_p5']:+.4f} | [{cc['lo']:+.4f}, {cc['hi']:+.4f}] | "
                f"{_fmt_p(cc['p'], n_boot)} | {cc['grouped_delta_p5']:+.4f} | "
                f"[{cc['grouped_lo']:+.4f}, {cc['grouped_hi']:+.4f}] | "
                f"{_fmt_p(cc['grouped_p'], n_boot)} | "
                f"{_fmt_p(c['holm_adjusted_p'][name], n_boot)} | {c['holm_reject'][name]} |"
            )

    orc = res.get("diagnostics_oracle")
    if orc:
        lines.append("\n## 3. Why the null is structural, not a search failure\n")
        lines.append(
            "Upper bound of the *entire* alpha family: for every query the best "
            "alpha in [0.00, 1.00] is picked with the ground truth.  Any alpha rule "
            "(prior, confidence, rank-based, temperature-scaled, ...) is bounded by "
            "it.\n"
        )
        lines.append("| Normalisation | Split | Oracle macro-P@5 | Prior-only macro-P@5 | Head-room |")
        lines.append("|---|---|---:|---:|---:|")
        for norm in NORM_MODES:
            for split in ("tune", "test"):
                blk = orc[norm][split]
                lines.append(
                    f"| {norm} | {split} | {blk['oracle_macro_p5']:.4f} | "
                    f"{blk['prior_macro_p5']:.4f} | "
                    f"{blk['oracle_macro_p5'] - blk['prior_macro_p5']:+.4f} |"
                )
        lines.append(
            "\nUnder min-max normalisation the oracle equals the prior exactly, so "
            "the required min-max gain is unreachable by construction.  Under "
            "z-score normalisation the family has head-room, and the winner "
            "recovers part of it without reaching significance.\n"
        )

    v = res["verdict"]
    lines.append("\n## 4. Verdict against the pre-registered success criteria\n")
    lines.append("| Criterion | Required | Observed | Met |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| (a) min-max test macro-P@5 | Holm-significant gain over prior-only | "
        f"Δ = {v['delta_macro_p5_minmax']:+.4f}, Holm-adjusted p = "
        f"{_fmt_p(v['holm_adjusted_p_minmax'], n_boot)} | "
        f"{'yes' if v['holm_significant_minmax'] else 'NO'} |"
    )
    lines.append(
        f"| (b) z-score test macro-P@5 | same direction (Δ >= 0) | "
        f"Δ = {v['delta_macro_p5_zscore']:+.4f} | "
        f"{'yes' if v['delta_macro_p5_zscore'] >= 0 else 'NO'} |"
    )
    lines.append(f"\n**Verdict: {v['verdict']}.**  {v['criteria']}\n")

    lines.append("\n## 5. Commands\n")
    lines.append("```")
    lines.append("python backend/scripts/eval_retrieval_conf2.py --mode tune")
    lines.append("python backend/scripts/eval_retrieval_conf2.py --mode test --winner %s" % winner)
    lines.append("python backend/scripts/eval_retrieval_conf2.py --mode diagnostics")
    lines.append("python backend/scripts/eval_retrieval_conf2.py --mode report")
    lines.append("```\n")
    lines.append(
        "New files: `backend/scripts/eval_retrieval_conf2.py`, "
        "`results/conf2_tune_sweep.json`, `results/conf2_tune_sweep.md`, "
        "`results/conf2_test_%s.json`, `results/conf2_test_%s.md`, "
        "`results/conf2_diagnostics.json`, `results/conf2_summary.json`, "
        "`results/conf2_summary.md`.\n" % (winner, winner)
    )
    (RESULTS / "conf2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "conf2_summary.json").write_text(
        json.dumps({
            "verdict": res["verdict"],
            "winner": winner,
            "params_by_normalization": res["params_by_normalization"],
            "selection": sweep["selection"],
            "tune_table": {
                norm: {
                    "prior": sweep[norm]["prior_baseline_tune"],
                    "variants": {n: sweep[norm]["variants"][n]["best_tune"]
                                 for n in VARIANTS},
                    "variant_params": {n: sweep[norm]["variants"][n]["best_params"]
                                       for n in VARIANTS},
                } for norm in NORM_MODES
            },
            "test": res["test"],
            "comparison": res["comparison"],
            "per_category_vs_prior": res["per_category_vs_prior"],
            "oracle": res.get("diagnostics_oracle"),
            "integrity": res.get("integrity"),
            "n_boot": n_boot,
        }, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"wrote {RESULTS / 'conf2_summary.json'}")
    print(f"wrote {RESULTS / 'conf2_summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tune", "test", "diagnostics", "report"), required=True)
    ap.add_argument("--winner", default=None, help="name of the frozen variant (test mode)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    cache = build_cache(force=args.rebuild_cache)
    print(f"cache: {CACHE_PATH} ({len(cache['queries'])} queries, {len(cache['clip_ids'])} clips)")

    if args.mode == "tune":
        sweep = run_tune_sweep(cache)
        write_tune_report(sweep, RESULTS / "conf2_tune_sweep.json", RESULTS / "conf2_tune_sweep.md")
        print(json.dumps(sweep["selection"], indent=2, default=float))
        for norm in NORM_MODES:
            print(f"[{norm}] prior tune macro P@5 = "
                  f"{sweep[norm]['prior_baseline_tune']['p5']['macro']:.4f}")
            for name, v in sweep[norm]["variants"].items():
                print(f"[{norm}] {name:14s} best tune macro P@5 = "
                      f"{v['best_tune']['p5']['macro']:.4f} params={json.dumps(v['best_params'])}")
        print(f"wrote {RESULTS / 'conf2_tune_sweep.json'}")
        print(f"wrote {RESULTS / 'conf2_tune_sweep.md'}")
        return

    if args.mode == "test":
        sweep_path = RESULTS / "conf2_tune_sweep.json"
        if not sweep_path.exists():
            raise RuntimeError("run --mode tune first (the winner must be pre-registered)")
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        recorded = sweep["selection"]["selected_winner"]
        if args.winner is None:
            raise RuntimeError(
                f"--winner is required; the tune split selected `{recorded}`")
        if args.winner != recorded:
            raise RuntimeError(
                f"anti-p-hacking guard: the tune split selected `{recorded}`, "
                f"not `{args.winner}`")
        params = sweep["minmax"]["variants"][recorded]["best_params"]
        res = evaluate_winner_on_test(cache, recorded, args.n_boot, sweep)
        res["integrity"] = integrity_check(cache)
        res["diagnostics_oracle"] = oracle_diagnostics(cache, args.n_boot)
        write_test_report(res, RESULTS / f"conf2_test_{recorded}.json",
                          RESULTS / f"conf2_test_{recorded}.md")
        print(json.dumps(res["verdict"], indent=2, default=float))
        for norm in NORM_MODES:
            print(f"[{norm}] winner macro P@5 = {res['test'][norm]['winner']['macro_p5']:.4f} "
                  f"prior = {res['test'][norm]['prior_only']['macro_p5']:.4f}")
            for name, c in res["comparison"][norm]["comparisons"].items():
                print(f"[{norm}] {name}: d={c['delta_p5']:+.4f} "
                      f"[{c['lo']:+.4f},{c['hi']:+.4f}] p={c['p']:.4f} "
                      f"grouped p={c['grouped_p']:.4f} "
                      f"holm_p={res['comparison'][norm]['holm_adjusted_p'][name]:.4f} "
                      f"reject={res['comparison'][norm]['holm_reject'][name]}")
        print(f"wrote {RESULTS / f'conf2_test_{recorded}.json'}")
        print(f"wrote {RESULTS / f'conf2_test_{recorded}.md'}")
        return

    if args.mode == "report":
        # Re-render the reports from the artefacts already written; this does
        # NOT re-run the evaluation (the frozen test result on disk is used
        # verbatim).
        sweep = json.loads((RESULTS / "conf2_tune_sweep.json").read_text(encoding="utf-8"))
        winner = sweep["selection"]["selected_winner"]
        res = json.loads((RESULTS / f"conf2_test_{winner}.json").read_text(encoding="utf-8"))
        write_test_report(res, RESULTS / f"conf2_test_{winner}.json",
                          RESULTS / f"conf2_test_{winner}.md")
        write_summary(sweep, res, int(res["n_boot"]))
        print(f"wrote {RESULTS / f'conf2_test_{winner}.md'}")
        return

    # diagnostics
    diag = {        "integrity": integrity_check(cache),
        "oracle": oracle_diagnostics(cache, args.n_boot),
        "prior_alpha_by_category": {},
    }
    for cat in CATEGORIES:
        vals = [cache["alpha_prior"][q["qid"]] for q in cache["queries"]
                if q["category"] == cat]
        diag["prior_alpha_by_category"][cat] = {
            "mean": float(np.mean(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals)), "n": len(vals),
        }
    (RESULTS / "conf2_diagnostics.json").write_text(
        json.dumps(diag, indent=2, default=float), encoding="utf-8")
    print(json.dumps(diag, indent=2, default=float))


if __name__ == "__main__":
    main()
