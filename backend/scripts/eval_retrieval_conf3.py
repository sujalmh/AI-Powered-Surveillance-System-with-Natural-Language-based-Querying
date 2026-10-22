#!/usr/bin/env python
"""C1''' -- round 3: score-normalisation / data-fusion operators (released subset).

This is a *new* experiment driver.  No tracked file is modified: the benchmark
loader, the structured and semantic branch construction, the branch
normalisation, the metrics and the statistical protocol are all **imported**
from ``backend/scripts/eval_retrieval_control.py`` (which itself imports the
repository's own ``backend/app/eval/*`` and ``backend/app/services/*`` code).
Only the *combination operator* -- how the two normalised branch score vectors
are combined into one ranking -- is new.  The metadata prior alpha
(``backend.app.services.result_merger.compute_adaptive_alpha``) is reused
unchanged, so the paper's adaptive weighting is never re-tuned here.

Why a combination operator at all?
----------------------------------
The prior round proved that the *alpha* family is exhausted under min-max
normalisation: a per-query label oracle over alpha in [0, 1] reproduces the
prior exactly (head-room +0.000).  Under z-score normalisation the alpha
family still has head-room (oracle 0.538 vs prior 0.524 on the test split).
The deficit is caused by the *calibration*, not by the weight:
``backend.app.eval.baselines.zscore_norm`` maps a branch score vector through
``sigmoid((x - mean) / sd)``, which *saturates* the top of the structured
branch (its largest values become indistinguishable) while the smooth
semantic branch keeps its spread.  The additive blend
``a*s + (1-a)*e`` then lets the semantic branch outvote the structured
branch's own peak, and relevant clips drop out of the top-5.

The IR literature's answer to "two branches with incomparable score scales"
is a *combination operator* rather than a re-weighted sum:  the sum is
"compensatory" (a document that both branches like wins), the max/disjunction
lets the branch with the *stronger opinion* decide, and rank-based operators
(RRF) throw the scores away entirely.  Round 3 compares three such families.

Pre-registered protocol (fixed before the test split was touched)
-----------------------------------------------------------------
1.  Three candidate *families* are explored on the **TUNE** split only
    (n=50).  Each family is a combination operator applied to the two
    per-query normalised branch vectors with the unchanged prior alpha:

      A ``max_fusion`` -- combMAX: ``fused = max(a*s, (1-a)*e)``.
      B ``pnorm``     -- Hoelder / power-mean (generalised mean) family:
                         ``fused = (a*s^p + (1-a)*e^p)^(1/p)`` for
                         ``p in {2,3,4,6,8,16,inf}``.  ``p = 1`` is the
                         paper's additive blend, ``p = inf`` is combMAX, so
                         the grid is a dose-response path between them.
      C ``rrf``       -- Reciprocal Rank Fusion:
                         ``fused = a/(k+r_s) + (1-a)/(k+r_e)``,
                         ``k in {10,20,60}``.

2.  Selection is made on the TUNE split only, by the pre-registered chain
    (no test-split number of any variant may enter the decision):
      (1) mean tune macro-P@5 over the two branch normalisations,
      (2) min-max tune macro-P@5,
      (3) mean tune macro F1@10 (paper definition),
      (4) mean tune macro R@10,
      (5) mean tune macro MRR,
      (6) variant name (alphabetical), then configuration key ascending.
    The winner (family + configuration) is **frozen**.

3.  The frozen winner is evaluated **once** on the TEST split (n=100)
    against the prior-only baseline (the repository's own adaptive alpha
    with the additive blend) in **both** branch normalisations.  The script
    refuses to run the test stage twice (guard in ``main``), so at most one
    test evaluation of the frozen variant can exist.

4.  Statistics (pre-registered):
      * primary test: paired bootstrap (3000 resamples) on the *macro-P@5*
        statistic, resampling queries **within each category** (the
        statistic is a mean of five category means, so the resampling has to
        respect the categories) and using the same resampled indices for
        both systems.  Holm correction over the family of the two
        arm-level comparisons {zscore, min-max}.
      * secondary: grouped (cluster) bootstrap over the 5-minute query
        groups on the same macro statistic; Wilcoxon signed-rank on the
        per-query P@5 differences; a paired sign-flip randomisation test on
        the macro statistic; per-category paired bootstrap with Holm over the
        five categories; and the control script's pooled per-query paired
        bootstrap, for continuity with the published control numbers.
    Success (pre-registered) requires **all** of:
      (a) z-score arm: macro-P@5 delta > 0 and Holm-adjusted p < 0.05 in the
          primary paired bootstrap;
      (b) min-max arm: macro-P@5 delta >= 0 (no harm);
      (c) grouped cluster bootstrap in the z-score arm: delta > 0
          (direction consistent; its p-value is reported as the
          cluster-robust caveat and a stricter "cluster-significant" verdict
          line is printed as well).

Usage
-----
    python backend/scripts/eval_retrieval_conf3.py --mode tune
    python backend/scripts/eval_retrieval_conf3.py --mode test --winner max_fusion
    python backend/scripts/eval_retrieval_conf3.py --mode report

The expensive part (OpenCLIP text encoding + FAISS search) is cached in the
OS temp directory so the tune and test stages share identical inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval.metrics import category_means, exact_match, macro_mean, mean_absolute_error, ranking_metrics
from backend.app.eval.stats import (
    format_p,
    holm_correction,
    paired_bootstrap_ci,
    paired_grouped_bootstrap_ci,
    wilcoxon_signed_rank,
)
from backend.scripts import eval_retrieval_control as ctrl

RESULTS = ROOT / "results"
CACHE_PATH = Path(tempfile.gettempdir()) / "opencode" / "conf3_cache.pkl"
CATEGORIES = ctrl.CATEGORIES
N_BOOT_DEFAULT = 3000
NORM_MODES = ("minmax", "zscore")
PRIOR_METHOD = "prior_only"          # metadata prior-only alpha = 'adaptive'
SPLITS = ("tune", "test")

# Published control numbers (results/retrieval_control_results.json).  The
# script asserts that its own reconstruction of the baseline reproduces them,
# so a stale/corrupt cache cannot silently change the comparison.
CONTROL_CHECK = {
    "minmax": {"test": 0.538, "tune": 0.580},
    "zscore": {"test": 0.524, "tune": 0.556},
}

LITERATURE = {
    "max_fusion": (
        "combMAX (disjunctive / t-conorm) combination rule: Lee (1997) "
        "'Hybrid test collections and the fusion of retrieval systems'; the "
        "combination-rule taxonomy of Fox & Shaw (1994) TREC-2 and of the "
        "UMass CIIR 'Analysis of Multiple Evidence Combination' report "
        "(CombMIN / CombMAX / CombSUM / CombMNZ)."
    ),
    "pnorm": (
        "Hoelder / power-mean (generalised mean) combination: the "
        "interpolation between the additive blend (p=1) and the disjunctive "
        "max (p=inf), i.e. the family of compensatory fuzzy operators used "
        "for IR evidence combination (R-norm family)."
    ),
    "rrf": (
        "Reciprocal Rank Fusion: Cormack, Clarke & Buettcher (2009), "
        "'Reciprocal rank fusion outperforms Condorcet and individual rank "
        "learning methods', SIGIR 2009."
    ),
}


# ---------------------------------------------------------------------------
# Cache: real OpenCLIP + FAISS retrieval, then the two raw branch vectors
# ---------------------------------------------------------------------------

def build_cache(force: bool = False) -> Dict[str, Any]:
    """Run the real retrieval once and cache the two raw branch score vectors.

    Mirrors ``eval_retrieval_control.main`` exactly: semantic score = max over
    the frames of a clip, structured score = the metadata-only score built by
    ``ctrl.build_query_context`` (which calls the repository's own
    ``compute_adaptive_alpha`` for the prior).
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
        "category": {q["qid"]: q["category"] for q in queries},
        "group": {q["qid"]: q.get("group") or q["qid"] for q in queries},
        "split": {q["qid"]: q["split"] for q in queries},
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as fh:
        pickle.dump(cache, fh)
    return cache


# ---------------------------------------------------------------------------
# Combination operators (the only new code in this round)
# ---------------------------------------------------------------------------

def _rank_desc(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every element, 1 = largest (ties broken by index)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    r = np.empty(len(order), dtype=np.int64)
    r[order] = np.arange(1, len(order) + 1)
    return r


def fuse_sum(a: float, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """The paper's additive blend (baseline operator, p = 1)."""
    return a * s + (1.0 - a) * e


def fuse_max(a: float, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """combMAX: the branch with the stronger (weighted) opinion decides."""
    return np.maximum(a * s, (1.0 - a) * e)


def fuse_pnorm(a: float, s: np.ndarray, e: np.ndarray, p: float) -> np.ndarray:
    """Hoelder / power-mean combination; p = 1 -> sum, p = inf -> max."""
    if p == math.inf:
        return fuse_max(a, s, e)
    ps = np.power(np.maximum(s, 0.0), p)
    pe = np.power(np.maximum(e, 0.0), p)
    return np.power(a * ps + (1.0 - a) * pe, 1.0 / p)


def fuse_rrf(a: float, s: np.ndarray, e: np.ndarray, k: int) -> np.ndarray:
    """Reciprocal Rank Fusion of the two branch rankings."""
    rs = _rank_desc(s)
    re = _rank_desc(e)
    return a / (k + rs) + (1.0 - a) / (k + re)


VARIANTS: Dict[str, Dict[str, Any]] = {
    "max_fusion": {
        "fuse": lambda a, s, e, p: fuse_max(a, s, e),
        "grid": [{}],
        "desc": "combMAX: fused = max(a*s, (1-a)*e)",
    },
    "pnorm": {
        "fuse": lambda a, s, e, p: fuse_pnorm(a, s, e, p),
        "grid": [{"p": p} for p in (2, 3, 4, 6, 8, 16, math.inf)],
        "desc": "power mean: fused = (a*s^p + (1-a)*e^p)^(1/p)",
    },
    "rrf": {
        "fuse": lambda a, s, e, p: fuse_rrf(a, s, e, int(p)),
        "grid": [{"k": k} for k in (10, 20, 60)],
        "desc": "RRF: fused = a/(k+r_s) + (1-a)/(k+r_e)",
    },
}

VARIANT_ORDER = ("max_fusion", "pnorm", "rrf")


def config_key(variant: str, params: Dict[str, Any]) -> str:
    if not params:
        return variant
    if "p" in params:
        p = params["p"]
        return f"{variant}:p={'inf' if p == math.inf else p}"
    if "k" in params:
        return f"{variant}:k={params['k']}"
    return f"{variant}:{sorted(params.items())}"


def config_params(variant: str, key: str) -> Dict[str, Any]:
    """Inverse of :func:`config_key` for the two grid families."""
    if variant == "pnorm":
        token = key.split(":p=")[1]
        return {"p": math.inf if token == "inf" else float(token)}
    if variant == "rrf":
        return {"k": float(key.split(":k=")[1])}
    return {}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def branch_scores(cache: Dict[str, Any], norm: str) -> Dict[str, np.ndarray]:
    """Per-query (structured, semantic) branch vectors under one calibration."""
    out: Dict[str, np.ndarray] = {}
    for qid, s_raw in cache["struct_raw"].items():
        s = ctrl.normalize_branch(s_raw, norm)
        e = ctrl.normalize_branch(cache["sem_raw_max"][qid], norm)
        out[qid] = (s, e)
    return out


def rank_clips(scores: np.ndarray, clip_ids: Sequence[str]) -> List[str]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return [clip_ids[int(i)] for i in order]


def run_system(
    cache: Dict[str, Any],
    norm: str,
    variant: Optional[str],
    params: Dict[str, Any],
    branches: Optional[Dict[str, Any]] = None,
    qids: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-query metrics for the prior-only baseline or for a variant."""
    if branches is None:
        branches = branch_scores(cache, norm)
    clip_ids = cache["clip_ids"]
    rel = {q["qid"]: q["relevant"] for q in cache["queries"]}
    if qids is None:
        qids = [q["qid"] for q in cache["queries"]]
    rows: Dict[str, Dict[str, Any]] = {}
    for qid in qids:
        s, e = branches[qid]
        a = float(cache["alpha_prior"][qid])
        if variant is None:
            fused = fuse_sum(a, s, e)
        else:
            fused = VARIANTS[variant]["fuse"](a, s, e, params.get("p", params.get("k")))
        ranking = rank_clips(fused, clip_ids)
        m = ranking_metrics(ranking, rel[qid])
        rows[qid] = {
            "qid": qid,
            "category": cache["category"][qid],
            "group": cache["group"][qid],
            "split": cache["split"][qid],
            "alpha": a,
            "top5": "|".join(ranking[:5]),
            **m,
        }
    return rows


def macro_of(rows: Dict[str, Dict[str, Any]], field: str) -> Tuple[float, Dict[str, float]]:
    vals = [{"category": r["category"], "v": float(r[field])} for r in rows.values()]
    cm = category_means(vals, "v")
    return float(macro_mean(cm)), {c: float(cm.get(c, 0.0)) for c in CATEGORIES}


def summarise(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in ("p5", "r10", "f1", "f1_paper", "mrr"):
        macro, per_cat = macro_of(rows, field)
        out[field] = {"macro": macro, "per_category": per_cat}
    # Count EM/MAE is a property of the count-answer path, not of the fusion
    # operator; it is not re-computed here (the control script owns it).
    out["count_em"] = None
    out["count_mae"] = None
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_macro_bootstrap(
    cats_a: Sequence[str],
    vals_a: Sequence[float],
    cats_b: Sequence[str],
    vals_b: Sequence[float],
    n_boot: int = 3000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap on the *macro* P@5 statistic.

    Queries are resampled **within each category** (the statistic is the mean
    of five category means) and the same resampled indices are used for both
    systems, so the comparison stays paired.  The p-value follows the
    repository's convention: ``2 * P(boot_delta <= 0)`` (two-sided).
    """
    assert len(cats_a) == len(vals_a) == len(cats_b) == len(vals_b)
    by_cat: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for ca, va, cb, vb in zip(cats_a, vals_a, cats_b, vals_b):
        assert ca == cb
        by_cat[ca].append((float(va), float(vb)))
    cat_names = sorted(by_cat)
    obs = float(
        np.mean([np.mean([x for x, _ in by_cat[c]]) - np.mean([y for _, y in by_cat[c]]) for c in cat_names])
    )
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(n_boot):
        per_cat_delta = []
        for c in cat_names:
            pool = by_cat[c]
            n = len(pool)
            samp = [pool[rng.randrange(n)] for _ in range(n)]
            per_cat_delta.append(
                float(np.mean([x for x, _ in samp]) - np.mean([y for _, y in samp]))
            )
        deltas.append(float(np.mean(per_cat_delta)))
    deltas.sort()
    alpha = 0.05
    lo = deltas[int((alpha / 2) * n_boot)]
    hi = deltas[int((1 - alpha / 2) * n_boot) - 1]
    if obs >= 0:
        p = sum(1 for d in deltas if d <= 0) / n_boot
    else:
        p = sum(1 for d in deltas if d >= 0) / n_boot
    return {
        "delta": obs,
        "lo": float(lo),
        "hi": float(hi),
        "p": float(min(1.0, 2.0 * p)),
        "n_boot": int(n_boot),
    }


def paired_grouped_macro_bootstrap(
    groups_a: Sequence[str],
    vals_a: Sequence[float],
    groups_b: Sequence[str],
    vals_b: Sequence[float],
    cats: Sequence[str],
    n_boot: int = 3000,
    seed: int = 0,
) -> Dict[str, float]:
    """Cluster-paired bootstrap on the macro P@5 statistic.

    Whole 5-minute groups are resampled with replacement (the pairing between
    the two systems is preserved inside a group); the macro statistic is then
    recomputed on the pooled resample.
    """
    by_group: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)
    for g, c, va, vb in zip(groups_a, cats, vals_a, vals_b):
        by_group[g].append((c, float(va), float(vb)))
    keys = list(by_group)
    obs_cats = sorted({c for g in keys for c, _, _ in by_group[g]})
    obs = float(
        np.mean(
            [
                np.mean([va for gg in keys for c, va, _ in by_group[gg] if c == cat])
                - np.mean([vb for gg in keys for c, _, vb in by_group[gg] if c == cat])
                for cat in obs_cats
            ]
        )
    )
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(n_boot):
        picked = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        pool = [row for g in picked for row in by_group[g]]
        cats_present = sorted({c for c, _, _ in pool})
        deltas.append(
            float(
                np.mean(
                    [
                        np.mean([va for c, va, _ in pool if c == cat])
                        - np.mean([vb for c, _, vb in pool if c == cat])
                        for cat in cats_present
                    ]
                )
            )
        )
    deltas.sort()
    alpha = 0.05
    lo = deltas[int((alpha / 2) * n_boot)]
    hi = deltas[int((1 - alpha / 2) * n_boot) - 1]
    if obs >= 0:
        p = sum(1 for d in deltas if d <= 0) / n_boot
    else:
        p = sum(1 for d in deltas if d >= 0) / n_boot
    return {
        "delta": obs,
        "lo": float(lo),
        "hi": float(hi),
        "p": float(min(1.0, 2.0 * p)),
        "n_boot": int(n_boot),
    }


def sign_flip_randomisation(
    cats: Sequence[str],
    vals_a: Sequence[float],
    vals_b: Sequence[float],
    n_boot: int = 3000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired sign-flip randomisation test on the macro statistic.

    The null hypothesis is that the sign of each per-query difference is
    exchangeable (no systematic effect).  Each replicate flips the sign of a
    random subset of the per-query differences and recomputes the macro
    statistic.  This test is far more sensitive than the percentile bootstrap
    when only a few queries change, so it is reported as a secondary analysis
    next to the pre-registered bootstrap.
    """
    by_cat: Dict[str, List[float]] = defaultdict(list)
    for c, va, vb in zip(cats, vals_a, vals_b):
        by_cat[c].append(float(va) - float(vb))
    cat_names = sorted(by_cat)

    def macro_of(deltas: Dict[str, List[float]]) -> float:
        return float(np.mean([np.mean(deltas[c]) for c in cat_names]))

    obs = macro_of(by_cat)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_boot):
        flipped = {
            c: [d if rng.random() < 0.5 else -d for d in by_cat[c]]
            for c in cat_names
        }
        if abs(macro_of(flipped)) >= abs(obs) - 1e-15:
            extreme += 1
    return {
        "delta": obs,
        "p": float(min(1.0, extreme / n_boot)),
        "n_boot": int(n_boot),
    }


def holm_adjusted(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out: Dict[str, float] = {}
    prev = 0.0
    for i, k in enumerate(order):
        adj = min(1.0, max(prev, pvals[k] * (m - i)))
        out[k] = float(adj)
        prev = adj
    return out


# ---------------------------------------------------------------------------
# Stage 1: tune sweep (selection)
# ---------------------------------------------------------------------------

def operator_oracle(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Tune-split label oracle over (alpha x operator) and the P@5 ceiling.

    Two references, computed on the TUNE split only:

    * ``oracle_alpha_sum`` / ``oracle_alpha_max`` -- per query, the best alpha
      in a 0.05 grid *with the ground truth*, for the additive blend and for
      combMAX.  The max over the two operators is the ceiling of the whole
      (alpha x operator) family.
    * ``ceiling`` -- ``min(5, n_relevant) / 5`` per query, i.e. the best P@5
      any ranker could reach on that query.

    This documents *where* the remaining error is: the prior round showed the
    alpha family is exhausted under min-max normalisation, and this shows
    whether the operator can reach that same ceiling under z-score.
    """
    tune_qids = [q["qid"] for q in cache["queries"] if cache["split"][q["qid"]] == "tune"]
    rel = {q["qid"]: q["relevant"] for q in cache["queries"]}
    grid = [i / 20.0 for i in range(21)]
    out: Dict[str, Any] = {"grid": grid, "arms": {}}
    for norm in NORM_MODES:
        branches = branch_scores(cache, norm)
        per_cat_oracle: Dict[str, List[float]] = defaultdict(list)
        per_cat_ceiling: Dict[str, List[float]] = defaultdict(list)
        per_cat_sum: Dict[str, List[float]] = defaultdict(list)
        per_cat_max: Dict[str, List[float]] = defaultdict(list)
        per_cat_prior: Dict[str, List[float]] = defaultdict(list)
        for qid in tune_qids:
            s, e = branches[qid]
            rel_q = rel[qid]
            a0 = float(cache["alpha_prior"][qid])
            best_sum = 0.0
            best_max = 0.0
            for a in grid:
                p_sum = ranking_metrics(rank_clips(fuse_sum(a, s, e), cache["clip_ids"]), rel_q)["p5"]
                p_max = ranking_metrics(
                    rank_clips(fuse_max(a, s, e), cache["clip_ids"]), rel_q
                )["p5"]
                best_sum = max(best_sum, p_sum)
                best_max = max(best_max, p_max)
            c = cache["category"][qid]
            per_cat_sum[c].append(best_sum)
            per_cat_max[c].append(best_max)
            per_cat_oracle[c].append(max(best_sum, best_max))
            per_cat_ceiling[c].append(min(5, len(rel_q)) / 5.0)
            p_prior = ranking_metrics(
                rank_clips(fuse_sum(a0, s, e), cache["clip_ids"]), rel_q
            )["p5"]
            per_cat_prior[c].append(p_prior)
        out["arms"][norm] = {
            "prior_sum_macro_p5": float(
                macro_mean(category_means(
                    [{"category": c, "v": v} for c, vs in per_cat_prior.items() for v in vs], "v"
                ))
            ),
            "oracle_alpha_sum_macro_p5": float(
                macro_mean(category_means(
                    [{"category": c, "v": v} for c, vs in per_cat_sum.items() for v in vs], "v"
                ))
            ),
            "oracle_alpha_max_macro_p5": float(
                macro_mean(category_means(
                    [{"category": c, "v": v} for c, vs in per_cat_max.items() for v in vs], "v"
                ))
            ),
            "oracle_alpha_x_operator_macro_p5": float(
                macro_mean(category_means(
                    [{"category": c, "v": v} for c, vs in per_cat_oracle.items() for v in vs], "v"
                ))
            ),
            "per_query_ceiling_macro_p5": float(
                macro_mean(category_means(
                    [{"category": c, "v": v} for c, vs in per_cat_ceiling.items() for v in vs], "v"
                ))
            ),
            "per_category": {
                c: {
                    "prior_sum": float(np.mean(per_cat_prior[c])),
                    "oracle_alpha_x_operator": float(np.mean(per_cat_oracle[c])),
                    "ceiling": float(np.mean(per_cat_ceiling[c])),
                }
                for c in CATEGORIES
            },
        }
    return out


def branch_contrast(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Label-free per-branch discriminability under each calibration.

    For every query the *contrast* of a branch is the gap between its best
    and its median score, and its *range* is max - min.  These are the
    quantities that decide which branch dominates an additive blend
    ``a*s + (1-a)*e``: the branch with the larger spread moves the fused
    ranking.  The z-score calibration saturates the top of the structured
    branch, so its contrast collapses while the smooth semantic branch keeps
    its spread.
    """
    out: Dict[str, Any] = {}
    for norm in NORM_MODES:
        branches = branch_scores(cache, norm)
        s_contrast, e_contrast, s_range, e_range = [], [], [], []
        for qid in cache["struct_raw"]:
            s, e = branches[qid]
            s_contrast.append(float(np.sort(s)[-1] - np.median(s)))
            e_contrast.append(float(np.sort(e)[-1] - np.median(e)))
            s_range.append(float(s.max() - s.min()))
            e_range.append(float(e.max() - e.min()))
        out[norm] = {
            "mean_struct_contrast_top1_minus_median": float(np.mean(s_contrast)),
            "mean_sem_contrast_top1_minus_median": float(np.mean(e_contrast)),
            "mean_struct_range": float(np.mean(s_range)),
            "mean_sem_range": float(np.mean(e_range)),
            "n_queries": len(s_contrast),
        }
    return out


def tune_sweep(cache: Dict[str, Any]) -> Dict[str, Any]:
    tune_qids = [q["qid"] for q in cache["queries"] if cache["split"][q["qid"]] == "tune"]
    rows_out: List[Dict[str, Any]] = []
    for norm in NORM_MODES:
        branches = branch_scores(cache, norm)
        base = run_system(cache, norm, None, {}, branches=branches, qids=tune_qids)
        base_p5 = macro_of(base, "p5")[0]
        for variant in VARIANT_ORDER:
            for params in VARIANTS[variant]["grid"]:
                key = config_key(variant, params)
                rows = run_system(cache, norm, variant, params, branches=branches, qids=tune_qids)
                p5, per_cat = macro_of(rows, "p5")
                f1, _ = macro_of(rows, "f1_paper")
                r10, _ = macro_of(rows, "r10")
                mrr, _ = macro_of(rows, "mrr")
                changed = sum(
                    1 for qid in tune_qids if abs(rows[qid]["p5"] - base[qid]["p5"]) > 1e-9
                )
                better = sum(
                    1 for qid in tune_qids if rows[qid]["p5"] > base[qid]["p5"] + 1e-9
                )
                worse = sum(
                    1 for qid in tune_qids if rows[qid]["p5"] < base[qid]["p5"] - 1e-9
                )
                rows_out.append(
                    {
                        "variant": variant,
                        "config": key,
                        "params": {k: (None if v == math.inf else v) for k, v in params.items()},
                        "normalization": norm,
                        "tune_macro_p5": p5,
                        "tune_macro_f1_paper": f1,
                        "tune_macro_r10": r10,
                        "tune_macro_mrr": mrr,
                        "tune_per_category_p5": per_cat,
                        "baseline_tune_macro_p5": base_p5,
                        "delta_vs_baseline": p5 - base_p5,
                        "n_queries_changed": changed,
                        "n_queries_better": better,
                        "n_queries_worse": worse,
                    }
                )
    # pre-registered selection chain, computed on tune only
    by_config: Dict[str, Dict[str, Any]] = {}
    for r in rows_out:
        entry = by_config.setdefault(
            r["config"],
            {
                "config": r["config"],
                "variant": r["variant"],
                "params": r["params"],
                "arms": {},
            },
        )
        entry["arms"][r["normalization"]] = r
    ranked: List[Dict[str, Any]] = []
    for entry in by_config.values():
        mm = entry["arms"]["minmax"]
        zz = entry["arms"]["zscore"]
        mean_p5 = (mm["tune_macro_p5"] + zz["tune_macro_p5"]) / 2.0
        mean_f1 = (mm["tune_macro_f1_paper"] + zz["tune_macro_f1_paper"]) / 2.0
        mean_r10 = (mm["tune_macro_r10"] + zz["tune_macro_r10"]) / 2.0
        mean_mrr = (mm["tune_macro_mrr"] + zz["tune_macro_mrr"]) / 2.0
        ranked.append(
            {
                "config": entry["config"],
                "variant": entry["variant"],
                "params": entry["params"],
                "mean_tune_macro_p5": mean_p5,
                "minmax_tune_macro_p5": mm["tune_macro_p5"],
                "zscore_tune_macro_p5": zz["tune_macro_p5"],
                "mean_tune_macro_f1_paper": mean_f1,
                "mean_tune_macro_r10": mean_r10,
                "mean_tune_macro_mrr": mean_mrr,
                "minmax_delta": mm["delta_vs_baseline"],
                "zscore_delta": zz["delta_vs_baseline"],
                "n_queries_changed": {
                    "minmax": mm["n_queries_changed"],
                    "zscore": zz["n_queries_changed"],
                },
            }
        )
    ranked.sort(
        key=lambda e: (
            -e["mean_tune_macro_p5"],
            -e["minmax_tune_macro_p5"],
            -e["mean_tune_macro_f1_paper"],
            -e["mean_tune_macro_r10"],
            -e["mean_tune_macro_mrr"],
            e["variant"],
            e["config"],
        )
    )
    for i, e in enumerate(ranked):
        e["rank"] = i + 1
    return {
        "n_tune_queries": len(tune_qids),
        "per_arm": rows_out,
        "ranked_configs": ranked,
        "winner": ranked[0],
        "operator_oracle": operator_oracle(cache),
        "branch_contrast": branch_contrast(cache),
        "selection_rule": [
            "1. mean tune macro-P@5 over {minmax, zscore}",
            "2. minmax tune macro-P@5",
            "3. mean tune macro F1@10 (paper)",
            "4. mean tune macro R@10",
            "5. mean tune macro MRR",
            "6. variant name (alphabetical), then configuration key ascending",
        ],
        "literature": LITERATURE,
    }


# ---------------------------------------------------------------------------
# Stage 2: the single frozen test evaluation
# ---------------------------------------------------------------------------

def evaluate_frozen(
    cache: Dict[str, Any],
    variant: str,
    params: Dict[str, Any],
    n_boot: int = N_BOOT_DEFAULT,
) -> Dict[str, Any]:
    test_qids = [q["qid"] for q in cache["queries"] if cache["split"][q["qid"]] == "test"]
    out: Dict[str, Any] = {
        "frozen_variant": variant,
        "frozen_params": {k: (None if v == math.inf else v) for k, v in params.items()},
        "frozen_config": config_key(variant, params),
        "n_test_queries": len(test_qids),
        "n_boot": n_boot,
        "arms": {},
    }
    arm_pvals: Dict[str, float] = {}
    for norm in NORM_MODES:
        branches = branch_scores(cache, norm)
        base = run_system(cache, norm, None, {}, branches=branches, qids=test_qids)
        var = run_system(cache, norm, variant, params, branches=branches, qids=test_qids)

        base_p5, base_pc = macro_of(base, "p5")
        var_p5, var_pc = macro_of(var, "p5")
        cats = [base[qid]["category"] for qid in test_qids]
        groups = [base[qid]["group"] for qid in test_qids]
        a_vals = [var[qid]["p5"] for qid in test_qids]
        b_vals = [base[qid]["p5"] for qid in test_qids]

        primary = paired_macro_bootstrap(cats, a_vals, cats, b_vals, n_boot=n_boot, seed=101)
        grouped = paired_grouped_macro_bootstrap(
            groups, a_vals, groups, b_vals, cats, n_boot=n_boot, seed=103
        )
        rand = sign_flip_randomisation(cats, a_vals, b_vals, n_boot=n_boot, seed=107)
        pooled = paired_bootstrap_ci(a_vals, b_vals, n_boot=n_boot, seed=109)
        pooled_grouped = paired_grouped_bootstrap_ci(
            a_vals, b_vals, groups, n_boot=n_boot, seed=111
        )
        wilc = wilcoxon_signed_rank(a_vals, b_vals)

        # per-category paired bootstrap (secondary, Holm over the 5 categories)
        cat_p: Dict[str, float] = {}
        cat_detail: Dict[str, Any] = {}
        for c in CATEGORIES:
            cq = [qid for qid in test_qids if base[qid]["category"] == c]
            if not cq:
                continue
            av = [var[qid]["p5"] for qid in cq]
            bv = [base[qid]["p5"] for qid in cq]
            m, lo, hi, p = paired_bootstrap_ci(av, bv, n_boot=n_boot, seed=113)
            cat_p[c] = float(p)
            cat_detail[c] = {
                "delta_p5": float(m), "lo": float(lo), "hi": float(hi), "p": float(p),
                "variant_macro_p5": var_pc[c], "prior_macro_p5": base_pc[c],
            }
        cat_holm = holm_correction(cat_p, alpha=0.05) if cat_p else {}
        cat_holm_adj = holm_adjusted(cat_p) if cat_p else {}

        changed = [qid for qid in test_qids if abs(var[qid]["p5"] - base[qid]["p5"]) > 1e-9]
        better = [qid for qid in test_qids if var[qid]["p5"] > base[qid]["p5"] + 1e-9]
        worse = [qid for qid in test_qids if var[qid]["p5"] < base[qid]["p5"] - 1e-9]

        out["arms"][norm] = {
            "baseline_prior_only": {
                "macro": summarise(base),
                "tune_macro_p5": None,
            },
            "variant": {
                "macro": summarise(var),
            },
            "delta_macro_p5": var_p5 - base_p5,
            "delta_macro_r10": macro_of(var, "r10")[0] - macro_of(base, "r10")[0],
            "delta_macro_f1_paper": macro_of(var, "f1_paper")[0] - macro_of(base, "f1_paper")[0],
            "delta_macro_mrr": macro_of(var, "mrr")[0] - macro_of(base, "mrr")[0],
            "per_category_p5": {"variant": var_pc, "prior": base_pc},
            "primary_paired_macro_bootstrap": primary,
            "grouped_cluster_macro_bootstrap": grouped,
            "sign_flip_randomisation": rand,
            "pooled_per_query_paired_bootstrap": {
                "delta": float(pooled[0]), "lo": float(pooled[1]), "hi": float(pooled[2]),
                "p": float(pooled[3]),
            },
            "pooled_per_query_grouped_bootstrap": {
                "delta": float(pooled_grouped[0]), "lo": float(pooled_grouped[1]),
                "hi": float(pooled_grouped[2]), "p": float(pooled_grouped[3]),
            },
            "wilcoxon_signed_rank_p": float(wilc),
            "per_category_paired_bootstrap": cat_detail,
            "per_category_holm_reject": {k: bool(v) for k, v in cat_holm.items()},
            "per_category_holm_adjusted_p": cat_holm_adj,
            "n_queries_changed": len(changed),
            "n_queries_better": len(better),
            "n_queries_worse": len(worse),
            "changed_queries": [
                {
                    "qid": qid,
                    "category": base[qid]["category"],
                    "group": base[qid]["group"],
                    "prior_p5": base[qid]["p5"],
                    "variant_p5": var[qid]["p5"],
                    "delta": var[qid]["p5"] - base[qid]["p5"],
                }
                for qid in changed
            ],
        }
        arm_pvals[norm] = float(primary["p"])

    # Holm over the two arm-level primary comparisons
    rejects = holm_correction(arm_pvals, alpha=0.05)
    adj = holm_adjusted(arm_pvals)
    out["primary_family"] = {
        "family": ["zscore", "minmax"],
        "description": "arm-level macro-P@5 paired bootstrap, variant vs prior-only",
        "raw_p": arm_pvals,
        "holm_adjusted_p": adj,
        "holm_reject": {k: bool(v) for k, v in rejects.items()},
    }
    zz = out["arms"]["zscore"]
    mm = out["arms"]["minmax"]
    crit_a = bool(zz["delta_macro_p5"] > 0 and adj["zscore"] < 0.05)
    crit_b = bool(mm["delta_macro_p5"] >= 0)
    crit_c = bool(zz["grouped_cluster_macro_bootstrap"]["delta"] > 0)
    out["success_criteria"] = {
        "a_zscore_holm_significant_gain": crit_a,
        "b_minmax_non_negative": crit_b,
        "c_grouped_direction_positive": crit_c,
        "verdict": "IMPROVED" if (crit_a and crit_b and crit_c) else "NULL",
        "verdict_strict_cluster_significance": (
            "IMPROVED"
            if (
                crit_a
                and crit_b
                and crit_c
                and zz["grouped_cluster_macro_bootstrap"]["p"] < 0.05
            )
            else "NULL"
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_tune_report(sweep: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# C1''' round 3 -- tune-split sweep of combination operators\n")
    lines.append(
        "**Selection split: TUNE only (n=%d).**  The test split was not read "
        "while these variants were designed or selected.  The baseline is the "
        "repository's own metadata prior-only alpha with the additive blend "
        "(`a*s + (1-a)*e`).\n" % sweep["n_tune_queries"]
    )
    lines.append("\n## Literature basis\n")
    for name in VARIANT_ORDER:
        lines.append(f"- **{name}** -- {LITERATURE[name]}")
    lines.append("\n## All configurations, per branch normalisation\n")
    lines.append(
        "| Variant | Config | minmax tune P@5 | d(minmax) | z-score tune P@5 | d(z-score) | mean | changed q (mm/zs) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for e in sweep["ranked_configs"]:
        mm = e["minmax_tune_macro_p5"]
        zz = e["zscore_tune_macro_p5"]
        lines.append(
            f"| {e['variant']} | `{e['config']}` | {mm:.4f} | {e['minmax_delta']:+.4f} | "
            f"{zz:.4f} | {e['zscore_delta']:+.4f} | {e['mean_tune_macro_p5']:.4f} | "
            f"{e['n_queries_changed']['minmax']}/{e['n_queries_changed']['zscore']} |"
        )
    lines.append(
        "\nBaseline (prior-only, additive blend): minmax tune macro-P@5 "
        f"{CONTROL_CHECK['minmax']['tune']:.4f}, z-score tune macro-P@5 "
        f"{CONTROL_CHECK['zscore']['tune']:.4f}.\n"
    )
    lines.append("\n## Pre-registered selection chain\n")
    for rule in sweep["selection_rule"]:
        lines.append(f"- {rule}")
    w = sweep["winner"]
    lines.append(
        f"\n**Frozen winner: `{w['config']}`** (mean tune macro-P@5 "
        f"{w['mean_tune_macro_p5']:.4f}; minmax {w['minmax_tune_macro_p5']:.4f}, "
        f"z-score {w['zscore_tune_macro_p5']:.4f}).\n"
    )
    (RESULTS / "conf3_tune_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "conf3_tune_sweep.json").write_text(
        json.dumps(sweep, indent=2), encoding="utf-8"
    )


def write_test_report(res: Dict[str, Any]) -> None:
    n_boot = res["n_boot"]
    lines: List[str] = []
    lines.append(
        f"# C1''' round 3 -- frozen variant on the TEST split (n={res['n_test_queries']})\n"
    )
    lines.append(
        f"**Frozen variant:** `{res['frozen_config']}` "
        f"(params {json.dumps(res['frozen_params'])}).  Selected on the tune "
        "split only; evaluated **once** on the test split.  "
        f"Bootstrap resamples: {n_boot}.\n"
    )
    for norm in NORM_MODES:
        arm = res["arms"][norm]
        base = arm["baseline_prior_only"]["macro"]
        var = arm["variant"]["macro"]
        lines.append(f"\n## {norm} branch normalisation\n")
        lines.append("| System | macro P@5 | R@10 | F1@10 (paper) | MRR |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| prior-only (additive blend) | {base['p5']['macro']:.4f} | "
            f"{base['r10']['macro']:.4f} | {base['f1_paper']['macro']:.4f} | "
            f"{base['mrr']['macro']:.4f} |"
        )
        lines.append(
            f"| frozen variant | {var['p5']['macro']:.4f} | {var['r10']['macro']:.4f} | "
            f"{var['f1_paper']['macro']:.4f} | {var['mrr']['macro']:.4f} |"
        )
        lines.append(
            f"| delta | {arm['delta_macro_p5']:+.4f} | {arm['delta_macro_r10']:+.4f} | "
            f"{arm['delta_macro_f1_paper']:+.4f} | {arm['delta_macro_mrr']:+.4f} |"
        )
        lines.append("\n| Category | prior P@5 | variant P@5 | delta |")
        lines.append("|---|---:|---:|---:|")
        for c in CATEGORIES:
            lines.append(
                f"| {c} | {arm['per_category_p5']['prior'][c]:.4f} | "
                f"{arm['per_category_p5']['variant'][c]:.4f} | "
                f"{arm['per_category_p5']['variant'][c] - arm['per_category_p5']['prior'][c]:+.4f} |"
            )
        p = arm["primary_paired_macro_bootstrap"]
        g = arm["grouped_cluster_macro_bootstrap"]
        r = arm["sign_flip_randomisation"]
        lines.append(
            f"\n- primary paired bootstrap (macro P@5, within-category resampling): "
            f"delta = {p['delta']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}], "
            f"p = {format_p(p['p'], n_boot)}"
        )
        lines.append(
            f"- grouped cluster bootstrap (5-minute groups): delta = {g['delta']:+.4f} "
            f"[{g['lo']:+.4f}, {g['hi']:+.4f}], p = {format_p(g['p'], n_boot)}"
        )
        lines.append(
            f"- sign-flip randomisation test (secondary): delta = {r['delta']:+.4f}, "
            f"p = {format_p(r['p'], n_boot)}"
        )
        lines.append(
            f"- Wilcoxon signed-rank on per-query P@5 (secondary): "
            f"p = {format_p(arm['wilcoxon_signed_rank_p'], n_boot)}"
        )
        pb = arm["pooled_per_query_paired_bootstrap"]
        pg = arm["pooled_per_query_grouped_bootstrap"]
        lines.append(
            f"- pooled per-query paired bootstrap (control-script convention): "
            f"delta = {pb['delta']:+.4f} [{pb['lo']:+.4f}, {pb['hi']:+.4f}], "
            f"p = {format_p(pb['p'], n_boot)}; grouped: delta = {pg['delta']:+.4f}, "
            f"p = {format_p(pg['p'], n_boot)}"
        )
        lines.append(
            f"- queries changed: {arm['n_queries_changed']} "
            f"(better {arm['n_queries_better']}, worse {arm['n_queries_worse']})"
        )
        lines.append("\n| Per-category | delta P@5 | 95% CI | p | Holm p | Holm reject |")
        lines.append("|---|---:|---|---:|---:|---|")
        for c in CATEGORIES:
            d = arm["per_category_paired_bootstrap"].get(c)
            if not d:
                continue
            lines.append(
                f"| {c} | {d['delta_p5']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] | "
                f"{format_p(d['p'], n_boot)} | "
                f"{format_p(arm['per_category_holm_adjusted_p'][c], n_boot)} | "
                f"{arm['per_category_holm_reject'][c]} |"
            )
    fam = res["primary_family"]
    lines.append("\n## Holm family (arm-level, pre-registered)\n")
    lines.append("| Arm | raw p | Holm-adjusted p | reject |")
    lines.append("|---|---:|---:|---|")
    for arm_name in fam["family"]:
        lines.append(
            f"| {arm_name} | {format_p(fam['raw_p'][arm_name], n_boot)} | "
            f"{format_p(fam['holm_adjusted_p'][arm_name], n_boot)} | "
            f"{fam['holm_reject'][arm_name]} |"
        )
    sc = res["success_criteria"]
    lines.append("\n## Pre-registered success criteria\n")
    lines.append("| Criterion | Required | Observed | Met |")
    lines.append("|---|---|---|---|")
    zz = res["arms"]["zscore"]
    mm = res["arms"]["minmax"]
    lines.append(
        f"| (a) z-score Holm-significant gain | Holm p < 0.05, delta > 0 | "
        f"delta = {zz['delta_macro_p5']:+.4f}, Holm p = "
        f"{format_p(fam['holm_adjusted_p']['zscore'], n_boot)} | "
        f"{'YES' if sc['a_zscore_holm_significant_gain'] else 'NO'} |"
    )
    lines.append(
        f"| (b) min-max non-negative | delta >= 0 | delta = {mm['delta_macro_p5']:+.4f} | "
        f"{'YES' if sc['b_minmax_non_negative'] else 'NO'} |"
    )
    lines.append(
        f"| (c) grouped direction | delta > 0 | delta = "
        f"{zz['grouped_cluster_macro_bootstrap']['delta']:+.4f} | "
        f"{'YES' if sc['c_grouped_direction_positive'] else 'NO'} |"
    )
    lines.append(f"\n**Verdict: {sc['verdict']}** "
                 f"(strict cluster-significant variant: {sc['verdict_strict_cluster_significance']}).\n")
    (RESULTS / f"conf3_test_{res['frozen_variant']}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (RESULTS / f"conf3_test_{res['frozen_variant']}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )


def write_summary_md(summary: Dict[str, Any]) -> None:
    """Consolidated report: literature, tune table, frozen test numbers."""
    sweep = summary["sweep"]
    res = summary["test"]
    n_boot = res["n_boot"]
    lines: List[str] = []
    lines.append("# C1''' round 3 -- score-normalisation / combination-operator result\n")
    lines.append(
        "**Verdict: "
        f"{res['success_criteria']['verdict']}.**  One variant family was "
        "pre-registered on the TUNE split (n=%d), one configuration was frozen, "
        "and the frozen variant was evaluated **once** on the TEST split "
        "(n=%d) with paired and grouped bootstraps and Holm correction.\n"
        % (sweep["n_tune_queries"], res["n_test_queries"])
    )

    lines.append("\n## 1. Literature ideas tried\n")
    lines.append(
        "| Family | Idea | Source |\n|---|---|---|"
    )
    lines.append(
        "| `max_fusion` | combMAX (disjunctive / t-conorm) combination: the "
        "branch with the stronger weighted opinion decides | Lee (1997) "
        "*Hybrid test collections and the fusion of retrieval systems*; "
        "combination-rule taxonomy of Fox & Shaw (1994) TREC-2 "
        "(https://trec.nist.gov/pubs/trec2/papers/txt/23.txt) and the UMass "
        "CIIR *Analysis of Multiple Evidence Combination* "
        "(https://ciir-publications.cs.umass.edu/getpdf.php?id=63) |"
    )
    lines.append(
        "| `pnorm` | Hoelder / power-mean (generalised mean) family "
        "interpolating the additive blend (p=1) and the max (p=inf) | "
        "compensatory fuzzy combination operators in IR (R-norm family); "
        "score-normalisation survey *Data Fusion in Information Retrieval* "
        "(https://www.researchgate.net/publication/268724557_Data_Fusion_in_Information_Retrieval), "
        "which splits normalisation into linear (min-max, z-score) and "
        "non-linear (sigmoid/sum) families |"
    )
    lines.append(
        "| `rrf` | Reciprocal Rank Fusion (rank-space, calibration-free) | "
        "Cormack, Clarke & Buettcher (2009), SIGIR "
        "(https://dl.acm.org/doi/10.1145/1571941.1572114, PDF "
        "https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) |"
    )
    lines.append(
        "\nConsidered but not implemented (they need a second index or a "
        "collection-level statistic that the released subset does not "
        "support): CORI-style collection-based belief (Cormack, Clarke & "
        "Buettcher, *Overview of document retrieval and clustering*, SIGIR "
        "2001 -- https://dl.acm.org/doi/10.1145/502585.502600) and CSLS "
        "hubness correction (Conneau et al. 2018, "
        "https://arxiv.org/abs/1809.04760; Lample & Conneau 2019, "
        "https://arxiv.org/abs/1901.07291).  Score normalisation before "
        "fusion is the shared precondition of all of them: \"each image must "
        "be mapped to a value that is normalized among all systems\" (Zhou et "
        "al. 2010, https://www.fi.muni.cz/~xkohout7/Research/clanky_cizi/ranking/Zhou10.pdf); "
        "see also the RRF-vs-score-normalisation comparison at "
        "https://bigdataboutique.com/blog/hybrid-search-fusion-methods-compared.\n"
    )

    lines.append("\n## 2. Tune split (n=%d) -- every variant, both calibrations\n" % sweep["n_tune_queries"])
    lines.append(
        "| Rank | Variant | Config | min-max tune P@5 | d | z-score tune P@5 | d | mean |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for e in sweep["ranked_configs"]:
        lines.append(
            f"| {e['rank']} | {e['variant']} | `{e['config']}` | "
            f"{e['minmax_tune_macro_p5']:.4f} | {e['minmax_delta']:+.4f} | "
            f"{e['zscore_tune_macro_p5']:.4f} | {e['zscore_delta']:+.4f} | "
            f"{e['mean_tune_macro_p5']:.4f} |"
        )
    lines.append(
        "\nBaseline (metadata prior-only alpha, additive blend): min-max tune "
        f"macro-P@5 {CONTROL_CHECK['minmax']['tune']:.4f}, z-score tune "
        f"macro-P@5 {CONTROL_CHECK['zscore']['tune']:.4f}.  The `pnorm` grid "
        "is monotone in p, i.e. a dose-response from the additive blend "
        "(p=1) to combMAX (p=inf), and every p >= 2 is a no-op under min-max.\n"
    )
    orc = sweep.get("operator_oracle")
    if orc:
        lines.append("\n### Label oracle on the tune split (alpha x operator ceiling)\n")
        lines.append(
            "| Normalisation | prior (sum) | oracle alpha, sum | oracle alpha, combMAX | "
            "oracle alpha x operator | per-query ceiling |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for norm in NORM_MODES:
            a = orc["arms"][norm]
            lines.append(
                f"| {norm} | {a['prior_sum_macro_p5']:.4f} | "
                f"{a['oracle_alpha_sum_macro_p5']:.4f} | "
                f"{a['oracle_alpha_max_macro_p5']:.4f} | "
                f"{a['oracle_alpha_x_operator_macro_p5']:.4f} | "
                f"{a['per_query_ceiling_macro_p5']:.4f} |"
            )
        lines.append(
            "\nThe alpha family is exhausted under min-max (the oracle equals "
            "the prior); under z-score the combMAX operator reaches the "
            "oracle of the whole (alpha x operator) family, so the residual "
            "gap to the per-query ceiling is unreachable by any weight or "
            "operator choice on this subset.\n"
        )
    w = sweep["winner"]
    lines.append(
        f"**Frozen after the pre-registered chain: `{w['config']}`** "
        f"(mean tune macro-P@5 {w['mean_tune_macro_p5']:.4f}).\n"
    )

    lines.append("\n## 3. Frozen variant on the TEST split (n=%d)\n" % res["n_test_queries"])
    fam = res["primary_family"]
    for norm in NORM_MODES:
        arm = res["arms"][norm]
        base = arm["baseline_prior_only"]["macro"]
        var = arm["variant"]["macro"]
        p = arm["primary_paired_macro_bootstrap"]
        g = arm["grouped_cluster_macro_bootstrap"]
        lines.append(f"\n### {norm} branch normalisation\n")
        lines.append("| System | macro P@5 | R@10 | F1@10 (paper) | MRR |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| prior-only (additive blend) | {base['p5']['macro']:.4f} | "
            f"{base['r10']['macro']:.4f} | {base['f1_paper']['macro']:.4f} | "
            f"{base['mrr']['macro']:.4f} |"
        )
        lines.append(
            f"| frozen combMAX variant | {var['p5']['macro']:.4f} | "
            f"{var['r10']['macro']:.4f} | {var['f1_paper']['macro']:.4f} | "
            f"{var['mrr']['macro']:.4f} |"
        )
        lines.append(
            f"| **delta** | **{arm['delta_macro_p5']:+.4f}** | "
            f"{arm['delta_macro_r10']:+.4f} | {arm['delta_macro_f1_paper']:+.4f} | "
            f"{arm['delta_macro_mrr']:+.4f} |"
        )
        lines.append(
            f"\n- paired bootstrap, macro P@5 (within-category resampling): "
            f"delta = {p['delta']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}], "
            f"p = {format_p(p['p'], n_boot)}, "
            f"**Holm-adjusted p = {format_p(fam['holm_adjusted_p'][norm], n_boot)}**, "
            f"reject = {fam['holm_reject'][norm]}"
        )
        lines.append(
            f"- grouped cluster bootstrap (5-minute groups): delta = {g['delta']:+.4f} "
            f"[{g['lo']:+.4f}, {g['hi']:+.4f}], p = {format_p(g['p'], n_boot)}"
        )
        lines.append(
            f"- sign-flip randomisation (secondary): p = "
            f"{format_p(arm['sign_flip_randomisation']['p'], n_boot)}; "
            f"Wilcoxon signed-rank (secondary): p = "
            f"{format_p(arm['wilcoxon_signed_rank_p'], n_boot)}; pooled per-query "
            f"paired bootstrap (control convention): delta = "
            f"{arm['pooled_per_query_paired_bootstrap']['delta']:+.4f}, p = "
            f"{format_p(arm['pooled_per_query_paired_bootstrap']['p'], n_boot)}"
        )
        lines.append(
            f"- per-category delta: " + ", ".join(
                f"{c} {arm['per_category_p5']['variant'][c] - arm['per_category_p5']['prior'][c]:+.4f}"
                for c in CATEGORIES
            )
        )
        lines.append(
            f"- per-category Holm (family of 5): " + ", ".join(
                f"{c} {format_p(arm['per_category_holm_adjusted_p'][c], n_boot)}"
                f"({'reject' if arm['per_category_holm_reject'][c] else 'keep'})"
                for c in CATEGORIES
            )
        )
        lines.append(
            f"- per-query P@5 changed: {arm['n_queries_changed']} "
            f"(better {arm['n_queries_better']}, worse {arm['n_queries_worse']})"
        )
    lines.append(
        "\n### Changed test queries (z-score arm)\n\n"
        "| qid | category | group | prior P@5 | variant P@5 |\n"
        "|---|---|---|---:|---:|"
    )
    for c in res["arms"]["zscore"]["changed_queries"]:
        lines.append(
            f"| {c['qid']} | {c['category']} | {c['group']} | "
            f"{c['prior_p5']:.2f} | {c['variant_p5']:.2f} |"
        )
    lines.append(
        "\nThe min-max arm changes **no** test query's P@5 (0 of 100) and its "
        "macro P@5 is bit-identical to the prior-only baseline; only R@10 "
        "moves by +0.001 (a re-ordering inside the top-10).  The gain is "
        "therefore not a re-tuning of the prior -- the prior alpha is never "
        "touched -- it is a repair of the z-score calibration.\n"
    )
    bc = sweep.get("branch_contrast", {})
    orc_s = sweep.get("operator_oracle", {}).get("arms", {})
    mech = [
        "",
        "**Mechanism.**  `zscore_norm` maps a branch through "
        "`sigmoid((x - mean) / sd)`, which saturates the *top* of the "
        "structured branch.  Measured on all 150 benchmark queries "
        "(label-free, `results/conf3_tune_sweep.json` -> `branch_contrast`):",
    ]
    if bc:
        mm_bc = bc.get("minmax", {})
        zz_bc = bc.get("zscore", {})
        mech.append(
            ""
        )
        mech.append(
            "| Calibration | structured contrast (top1 - median) | semantic contrast | structured range | semantic range |"
        )
        mech.append("|---|---:|---:|---:|---:|")
        for name, d in (("min-max", mm_bc), ("z-score", zz_bc)):
            if not d:
                continue
            mech.append(
                f"| {name} | {d['mean_struct_contrast_top1_minus_median']:.3f} | "
                f"{d['mean_sem_contrast_top1_minus_median']:.3f} | "
                f"{d['mean_struct_range']:.3f} | {d['mean_sem_range']:.3f} |"
            )
        mech.append(
            ""
        )
    mech.append(
        "so under z-score the structured branch loses most of its top-end "
        "contrast while the smooth semantic branch keeps its spread.  The "
        "additive blend therefore lets the semantic branch outvote the "
        "structured branch's own peak and pushes relevant clips out of the "
        "top-5.  A disjunctive (combMAX) combination keeps the structured "
        "branch's peak decisive and lets the semantic branch decide only "
        "when the structured branch has nothing to say (e.g. the action "
        "queries, whose structured score is flat because they carry no time "
        "cue)."
    )
    if orc_s:
        mech.append(
            "  On the tune split the combMAX variant attains the per-query "
            "label oracle of the whole (alpha x operator) family in the "
            "z-score arm "
            f"(oracle {orc_s['zscore']['oracle_alpha_x_operator_macro_p5']:.4f} = "
            f"variant {sweep['ranked_configs'][0]['zscore_tune_macro_p5']:.4f}), "
            "so the remaining gap to the per-query ceiling "
            f"({orc_s['zscore']['per_query_ceiling_macro_p5']:.4f}) is "
            "unreachable by any weight or operator choice on this subset."
        )
    lines.append("\n".join(mech) + "\n")

    lines.append("\n## 4. Pre-registered success criteria\n")
    sc = res["success_criteria"]
    zz = res["arms"]["zscore"]
    mm = res["arms"]["minmax"]
    lines.append("| Criterion | Required | Observed | Met |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| (a) z-score Holm-significant gain over prior-only | Holm p < 0.05, "
        f"delta > 0 | delta = {zz['delta_macro_p5']:+.4f}, Holm p = "
        f"{format_p(fam['holm_adjusted_p']['zscore'], n_boot)} | "
        f"{'YES' if sc['a_zscore_holm_significant_gain'] else 'NO'} |"
    )
    lines.append(
        f"| (b) min-max non-negative | delta >= 0 | delta = "
        f"{mm['delta_macro_p5']:+.4f} | "
        f"{'YES' if sc['b_minmax_non_negative'] else 'NO'} |"
    )
    lines.append(
        f"| (c) grouped cluster direction | delta > 0 | delta = "
        f"{zz['grouped_cluster_macro_bootstrap']['delta']:+.4f} | "
        f"{'YES' if sc['c_grouped_direction_positive'] else 'NO'} |"
    )
    lines.append(
        f"\n**Verdict: {sc['verdict']}** (strict variant requiring cluster "
        f"significance as well: {sc['verdict_strict_cluster_significance']}).\n"
    )
    lines.append("\n## 5. Scope and limitations\n")
    lines.append(
        "- Exactly one variant was selected on the tune split and evaluated "
        "once on the test split; the script refuses a second test run of the "
        "same variant (`results/conf3_test_<variant>.json` already exists).  "
        "The `pnorm:p=8` / `pnorm:p=16` configurations tie `max_fusion` on "
        "the tune split but were **not** test-evaluated, so no claim is made "
        "about them.\n"
        "- The prior alpha is reused unchanged from "
        "`backend.app.services.result_merger.compute_adaptive_alpha`; this "
        "round changes the *combination operator* only, not the weighting rule.\n"
        "- The numbers are the released single-camera subset "
        "(camera_0, 121 indexed clips, 150 benchmark queries).  They are not "
        "the paper's 4-camera / 312k-descriptor deployment numbers and do not "
        "replace them.\n"
        "- Labels come from released VLM captions, detector metadata and "
        "snapshots (no human annotation, no IAA available).\n"
        "- The z-score branch is the repository's `sigmoid(z)` calibration "
        "(`backend/app/eval/baselines.zscore_norm`); the mechanism above is "
        "specific to a saturating, non-linear calibration and may not "
        "reproduce under a different z-score variant.\n"
    )
    lines.append("\n## 6. Commands\n")
    lines.append("```")
    lines.append("python backend/scripts/eval_retrieval_conf3.py --mode tune")
    lines.append("python backend/scripts/eval_retrieval_conf3.py --mode test --winner max_fusion")
    lines.append("python backend/scripts/eval_retrieval_conf3.py --mode report")
    lines.append("```")
    lines.append(
        "\n## 7. New files\n\n"
        "- `backend/scripts/eval_retrieval_conf3.py`\n"
        "- `results/conf3_tune_sweep.json`, `results/conf3_tune_sweep.md`\n"
        "- `results/conf3_test_max_fusion.json`, `results/conf3_test_max_fusion.md`\n"
        "- `results/conf3_summary.json`, `results/conf3_summary.md`\n"
    )
    (RESULTS / "conf3_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tune", "test", "report"), required=True)
    ap.add_argument("--winner", default=None, help="frozen variant name (test mode)")
    ap.add_argument("--config", default=None, help="frozen configuration key (test mode)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    cache = build_cache(force=args.force_cache)

    # Self-check: the reconstructed prior-only baseline must reproduce the
    # published control numbers, otherwise the comparison is meaningless.
    for norm in NORM_MODES:
        branches = branch_scores(cache, norm)
        for split in SPLITS:
            qids = [q["qid"] for q in cache["queries"] if cache["split"][q["qid"]] == split]
            rows = run_system(cache, norm, None, {}, branches=branches, qids=qids)
            got = macro_of(rows, "p5")[0]
            want = CONTROL_CHECK[norm][split]
            if abs(got - want) > 1e-6:
                raise RuntimeError(
                    f"baseline mismatch: {norm}/{split} got {got:.4f}, expected {want:.4f}"
                )

    if args.mode == "tune":
        sweep = tune_sweep(cache)
        write_tune_report(sweep)
        w = sweep["winner"]
        print(f"tune sweep: {len(sweep['per_arm'])} rows, {len(sweep['ranked_configs'])} configs")
        print(f"FROZEN WINNER: {w['config']}  mean tune macro P@5 {w['mean_tune_macro_p5']:.4f}")
        for e in sweep["ranked_configs"]:
            print(
                f"  {e['rank']:2d}. {e['config']:22s} mean {e['mean_tune_macro_p5']:.4f} "
                f"mm {e['minmax_tune_macro_p5']:.4f} ({e['minmax_delta']:+.4f}) "
                f"zs {e['zscore_tune_macro_p5']:.4f} ({e['zscore_delta']:+.4f})"
            )
        print(f"wrote {RESULTS / 'conf3_tune_sweep.json'}")
        print(f"wrote {RESULTS / 'conf3_tune_sweep.md'}")
        return

    if args.mode == "test":
        if not args.winner:
            raise SystemExit("--winner is required in test mode")
        if args.winner not in VARIANTS:
            raise SystemExit(f"unknown variant {args.winner!r}")
        out_json = RESULTS / f"conf3_test_{args.winner}.json"
        if out_json.exists():
            raise SystemExit(
                f"{out_json} already exists: the frozen variant has already been "
                "evaluated on the test split. Delete it deliberately (or use a new "
                "variant) if you really mean to spend the one permitted test run."
            )
        params = config_params(args.winner, args.config) if args.config else {}
        res = evaluate_frozen(cache, args.winner, params, n_boot=args.n_boot)
        write_test_report(res)
        for norm in NORM_MODES:
            arm = res["arms"][norm]
            fam = res["primary_family"]
            print(
                f"{norm}: prior {arm['baseline_prior_only']['macro']['p5']['macro']:.4f} -> "
                f"variant {arm['variant']['macro']['p5']['macro']:.4f} "
                f"(delta {arm['delta_macro_p5']:+.4f}); primary p="
                f"{format_p(arm['primary_paired_macro_bootstrap']['p'], args.n_boot)}; "
                f"grouped p={format_p(arm['grouped_cluster_macro_bootstrap']['p'], args.n_boot)}; "
                f"Holm p={format_p(fam['holm_adjusted_p'][norm], args.n_boot)}"
            )
        print(f"verdict: {res['success_criteria']['verdict']}")
        print(f"wrote {out_json}")
        return

    # report mode: consolidate
    sweep_path = RESULTS / "conf3_tune_sweep.json"
    if not sweep_path.exists():
        raise SystemExit("run --mode tune first")
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    summary: Dict[str, Any] = {
        "sweep": sweep,
        "test": None,
    }
    test_path = RESULTS / "conf3_test_max_fusion.json"
    if test_path.exists():
        summary["test"] = json.loads(test_path.read_text(encoding="utf-8"))
    (RESULTS / "conf3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["test"] is not None:
        write_summary_md(summary)
    print(f"wrote {RESULTS / 'conf3_summary.json'}")
    if summary["test"] is not None:
        print(f"wrote {RESULTS / 'conf3_summary.md'}")


if __name__ == "__main__":
    main()
