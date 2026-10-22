"""TEMPORARY robustness probe (scratch, not a deliverable).

Executes the production merge / fusion / normalisation code paths on
degenerate inputs and records the observed behaviour.  Writes
``results/robust_probe_tmp.json``.

Run:
    python backend/scripts/robust_probe_tmp.py
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services import retrieval_utils as ru
from backend.app.services.result_merger import (
    apply_mmr_diversity,
    compute_adaptive_alpha,
    merge_results,
)
from backend.app.services.sem_search import _norm_scores
from backend.app.services.fusion import MultimodalFusion, fuse_gated, _sigmoid
from backend.app.eval import confidence as conf

OUT: List[Dict[str, Any]] = []


def rec(name: str, fn, *args, **kwargs) -> None:
    """Run *fn*; record either the returned value or the exception."""
    try:
        val = fn(*args, **kwargs)
        OUT.append({
            "case": name,
            "status": "ok",
            "value": _jsonable(val),
        })
    except Exception as exc:  # noqa: BLE001
        OUT.append({
            "case": name,
            "status": "exception",
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc)[:300],
            "trace_tail": traceback.format_exc().strip().splitlines()[-3:],
        })


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------
# 0. Fixtures
# ---------------------------------------------------------------------------
def struct_doc(cam, ts, name="person", color="red", tid=1, conf_=0.9):
    return {
        "camera_id": cam,
        "timestamp": ts,
        "objects": [{
            "object_name": name, "color": color, "track_id": tid,
            "confidence": conf_,
        }],
    }


def sem_doc(cam, clip, score_norm=0.8, score_raw=0.5, ts="2026-03-01T17:10:00"):
    return {
        "camera_id": cam,
        "clip_path": f"clips/cam{cam}/{clip}.mp4",
        "clip_url": f"/clips/cam{cam}/{clip}.mp4",
        "score": score_raw,
        "score_raw": score_raw,
        "score_norm": score_norm,
        "frames": [{"frame_ts": ts, "frame_index": 0, "score": score_raw}],
    }


PF = {"objects.object_name": "person"}


# ---------------------------------------------------------------------------
# 1. retrieval_utils.normalize_scores  (production structured-branch scaler)
# ---------------------------------------------------------------------------
rec("normalize_scores.empty", ru.normalize_scores, [])
rec("normalize_scores.single_zero", ru.normalize_scores, [0.0])
rec("normalize_scores.single_0.3", ru.normalize_scores, [0.3])
rec("normalize_scores.single_duration_5s", ru.normalize_scores, [5.0])
rec("normalize_scores.all_tie_durations", ru.normalize_scores, [3.0, 3.0, 3.0])
rec("normalize_scores.all_zero", ru.normalize_scores, [0.0, 0.0])
rec("normalize_scores.negative", ru.normalize_scores, [-1.0, -2.0])
rec("normalize_scores.mixed_sign", ru.normalize_scores, [-1.0, 0.0, 2.0])
rec("normalize_scores.nan_first", ru.normalize_scores, [float("nan"), 1.0])
rec("normalize_scores.nan_last", ru.normalize_scores, [1.0, float("nan")])
rec("normalize_scores.inf", ru.normalize_scores, [float("inf"), 1.0])
rec("normalize_scores.short_durations", ru.normalize_scores, [0.1, 0.2])
rec("normalize_scores.long_durations", ru.normalize_scores, [1.0, 300.0])
rec("normalize_scores.single_candidate_gallery", ru.normalize_scores, [42.0])

# ---------------------------------------------------------------------------
# 2. sem_search._norm_scores  (production semantic-branch scaler)
# ---------------------------------------------------------------------------
rec("sem_norm.empty", _norm_scores, [])
rec("sem_norm.single", _norm_scores, [0.37])
rec("sem_norm.all_tie", _norm_scores, [0.5, 0.5, 0.5])
rec("sem_norm.two_distinct", _norm_scores, [0.2, 0.9])
rec("sem_norm.negative", _norm_scores, [-0.4, 0.2])

# ---------------------------------------------------------------------------
# 3. merge_results end-to-end degenerate cases
# ---------------------------------------------------------------------------
rec("merge.empty_struct_only", merge_results, [], [], PF, "find")
rec("merge.empty_both", merge_results, [], [], PF, "find")
rec("merge.semantic_only", merge_results, [], [sem_doc(0, "a")], PF, "find")
rec(
    "merge.struct_only",
    merge_results,
    [struct_doc(0, "2026-03-01T17:10:00")], [], PF, "find",
)
rec(
    "merge.both_normal",
    merge_results,
    [struct_doc(0, "2026-03-01T17:10:00")],
    [sem_doc(0, "a"), sem_doc(0, "b", score_norm=0.4)],
    PF, "find",
)
# all-tie structured durations (two identical-duration tracks)
rec(
    "merge.all_tie_struct",
    merge_results,
    [
        struct_doc(0, "2026-03-01T17:10:00", tid=1),
        struct_doc(0, "2026-03-01T17:10:00", tid=2),
    ],
    [],
    PF, "find",
)
# semantic score_norm missing / None / NaN
rec(
    "merge.sem_score_norm_missing",
    merge_results,
    [],
    [{"camera_id": 0, "clip_url": "/x.mp4", "clip_path": "x.mp4", "score_raw": 0.5}],
    PF, "find",
)
rec(
    "merge.sem_score_norm_none",
    merge_results,
    [],
    [{"camera_id": 0, "clip_url": "/x.mp4", "clip_path": "x.mp4",
      "score_raw": 0.5, "score_norm": None}],
    PF, "find",
)
rec(
    "merge.sem_score_norm_nan",
    merge_results,
    [],
    [{"camera_id": 0, "clip_url": "/x.mp4", "clip_path": "x.mp4",
      "score_raw": 0.5, "score_norm": float("nan")}],
    PF, "find",
)
# None branches
rec("merge.struct_none", merge_results, None, [sem_doc(0, "a")], PF, "find")
rec("merge.sem_none", merge_results, [struct_doc(0, "2026-03-01T17:10:00")], None, PF, "find")
rec(
    "merge.struct_none_count_constraint",
    merge_results,
    None, [], {"count_constraint": {"gte": 2}}, "count",
)
# count-constrained path: semantic branch is disabled by design
rec(
    "merge.count_constraint_semantic_ignored",
    merge_results,
    [struct_doc(0, "2026-03-01T17:10:00")],
    [sem_doc(0, "a")],
    {"count_constraint": {"gte": 1}}, "count",
)

# ---------------------------------------------------------------------------
# 4. MMR diversity
# ---------------------------------------------------------------------------
rec("mmr.negative_scores_drop_all", apply_mmr_diversity,
    [{"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -5.0},
     {"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -4.0},
     {"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -3.4}],
    0.3)
rec("mmr.two_items_passthrough", apply_mmr_diversity,
    [{"camera_id": 0, "score": 0.1}, {"camera_id": 1, "score": 0.2}], 0.3)
rec("mmr.all_ties", apply_mmr_diversity,
    [{"camera_id": 0, "start": "2026-03-01T17:00:00", "score": 0.5},
     {"camera_id": 0, "start": "2026-03-01T17:00:00", "score": 0.5},
     {"camera_id": 1, "start": "2026-03-01T18:00:00", "score": 0.5}], 0.3)

# ---------------------------------------------------------------------------
# 5. alpha / confidence
# ---------------------------------------------------------------------------
rec("alpha.no_branches", compute_adaptive_alpha, PF, "find", False, False, 0, 0)
rec("alpha.empty_score_vectors", compute_adaptive_alpha,
    PF, "find", False, False, 0, 0, [], [])
rec("alpha.degenerate_struct_ties", compute_adaptive_alpha,
    PF, "find", False, False, 0, 0, [0.5, 0.5, 0.5], [0.1, 0.9, 0.2])
rec("alpha.nan_score_vector", compute_adaptive_alpha,
    PF, "find", False, False, 0, 0, [float("nan"), 0.5], [0.1, 0.9])
rec("conf.confidence_alpha_empty", conf.confidence_alpha, [], [])
rec("conf.branch_conf_all_tie", conf.branch_confidence, [0.5, 0.5, 0.5])
rec("conf.branch_conf_single", conf.branch_confidence, [0.5])
rec("conf.confidence_alpha_nan", conf.confidence_alpha, [float("nan"), 0.5], [0.1, 0.9])
rec("conf.topk_conc_all_tie", conf.topk_concentration, [0.5, 0.5, 0.5])
rec("conf.topk_conc_negative", conf.topk_concentration, [-2.0, -3.0, -1.0])

# ---------------------------------------------------------------------------
# 6. person fusion
# ---------------------------------------------------------------------------
def _fusion_inplace_mutation():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    v_before = float(v[0])
    f = MultimodalFusion(0.6, 0.4)
    f.fuse(v, t)
    return {"v0_before": v_before, "v0_after": float(v[0]),
            "t0_after": float(t[0]),
            "mutated": bool(v[0] != v_before)}


rec("fusion.fuse_inplace_mutation", _fusion_inplace_mutation)


def _fusion_double_call():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    f = MultimodalFusion(0.6, 0.4)
    f.fuse(v, t)
    f.fuse(v, t)
    return {"v0_after_two_calls": float(v[0])}


rec("fusion.fuse_not_idempotent", _fusion_double_call)


def _fusion_zero_vector():
    out = MultimodalFusion(0.6, 0.4).fuse(
        np.zeros((768,), dtype=np.float32), np.zeros((384,), dtype=np.float32)
    )
    return {"norm": float(np.linalg.norm(out)), "dtype": str(out.dtype)}


rec("fusion.fuse_zero_input_norm", _fusion_zero_vector)


def _fusion_gated_nan_quality():
    out = fuse_gated(np.ones((768,), np.float32), np.ones((384,), np.float32),
                     float("nan"), 0.0)
    # Recover the effective gate from the ratio of the two block norms.
    v = float(np.linalg.norm(out[:768]))
    t = float(np.linalg.norm(out[768:]))
    return {"vis_block_norm": v, "text_block_norm": t,
            "implied_wv_over_wt": (v / t) if t > 0 else None}


rec("fusion.fuse_gated_nan_quality", _fusion_gated_nan_quality)


def _fusion_gated_zero_visual():
    out = fuse_gated(np.zeros((768,), np.float32), np.ones((384,), np.float32), 1.0, 0.0)
    return {"vis_block_norm": float(np.linalg.norm(out[:768])),
            "text_block_norm": float(np.linalg.norm(out[768:])),
            "total_norm": float(np.linalg.norm(out))}


rec("fusion.fuse_gated_empty_visual_branch", _fusion_gated_zero_visual)


def _fusion_gated_none_quality():
    return fuse_gated(np.ones((768,), np.float32), np.ones((384,), np.float32),
                      None, 0.0).tolist()[:3]


rec("fusion.fuse_gated_none_quality", _fusion_gated_none_quality)

# ---------------------------------------------------------------------------
# 7. Branch-norm mismatch: production normaliser vs eval-harness normaliser
# ---------------------------------------------------------------------------
def _norm_mismatch():
    """Production structured branch (max-norm) vs eval harness (min-max)."""
    struct = [1.0, 300.0]           # durations in seconds
    prod = ru.normalize_scores(struct)
    mm = [(s - min(struct)) / (max(struct) - min(struct)) for s in struct]
    return {"production_max_norm": prod, "eval_minmax": mm}


rec("norm.production_vs_harness", _norm_mismatch)

# ---------------------------------------------------------------------------
# 8. Emptiness of a whole branch in the merge (degenerate alpha)
# ---------------------------------------------------------------------------
def _empty_branch_alpha():
    pf = {"objects.object_name": "person"}
    both_empty = compute_adaptive_alpha(pf, "find", False, False, 0, 0)
    struct_only = compute_adaptive_alpha(pf, "find", False, False, 10, 0)
    sem_only = compute_adaptive_alpha(pf, "find", False, False, 0, 10)
    return {"both_empty": both_empty, "struct_only": struct_only,
            "sem_only": sem_only}


rec("alpha.branch_emptiness", _empty_branch_alpha)

# ---------------------------------------------------------------------------
# 9. MMR inside the real merge (empty structured branch)
# ---------------------------------------------------------------------------
def _merge_mmr_single_semantic():
    docs = [sem_doc(0, f"c{i}", score_norm=0.9 - 0.05 * i) for i in range(4)]
    out = merge_results([], docs, PF, "find")
    return {"n": len(out), "scores": [r["score"] for r in out],
            "sources": [r.get("source") for r in out]}


rec("merge.mmr_semantic_only", _merge_mmr_single_semantic)

# ===========================================================================
# PHASE 2 -- validate the *proposed* patches (reference implementations of
# the diffs in results/robust_proposal.md).  Nothing in the repo is changed;
# the patched functions are re-implemented here and the regression-test
# sketches are executed against them.
# ===========================================================================
PATCHED: List[Dict[str, Any]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        PATCHED.append({"test": name, "status": "pass"})
    except AssertionError as exc:
        PATCHED.append({"test": name, "status": "FAIL", "assert": str(exc)[:200]})
    except Exception as exc:  # noqa: BLE001
        PATCHED.append({"test": name, "status": "ERROR",
                        "exc": f"{type(exc).__name__}: {exc}"[:200]})


# --- D1 reference implementation -----------------------------------------
_NORM_EPS = 1e-12


def finite_score(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if f != f or f in (float("inf"), float("-inf")):
        return float(default)
    return f


def p_normalize_scores(scores, min_val: float = 0.0, mode: str = "minmax"):
    if not scores:
        return []
    vals = [finite_score(s) for s in scores]
    if len(vals) == 1:
        return [1.0]
    lo, hi = min(vals), max(vals)
    if hi - lo <= _NORM_EPS:
        return [min_val] * len(vals)
    if mode == "max":
        if hi <= _NORM_EPS:
            return [min_val] * len(vals)
        return [max(min_val, min(1.0, (v - lo) / hi)) for v in vals]
    return [max(min_val, min(1.0, (v - lo) / (hi - lo))) for v in vals]


def p_norm_scores(scores):
    return p_normalize_scores(scores, min_val=0.0, mode="minmax")


# --- D3 reference implementation -----------------------------------------
def p_rebalance(alpha, has_count_constraint, n_structured, n_semantic):
    if has_count_constraint:
        return alpha
    if n_structured <= 0 and n_semantic > 0:
        return 0.0
    if n_semantic <= 0 and n_structured > 0:
        return 1.0
    return alpha


def p_compute_adaptive_alpha(parsed_filter, intent, has_count_constraint,
                             has_action, n_structured, n_semantic,
                             struct_scores=None, sem_scores=None,
                             confidence_blend: float = 0.5) -> float:
    prior = _metadata_prior_alpha_local(parsed_filter, intent, has_count_constraint,
                                        has_action, n_structured, n_semantic)
    alpha = prior
    if struct_scores is not None and sem_scores is not None:
        try:
            from backend.app.eval.confidence import combined_alpha

            alpha = float(combined_alpha(prior, struct_scores, sem_scores,
                                         blend=confidence_blend))
        except Exception:
            alpha = prior
    return p_rebalance(alpha, has_count_constraint, n_structured, n_semantic)


def _metadata_prior_alpha_local(parsed_filter, intent, has_count_constraint,
                                 has_action, n_structured, n_semantic) -> float:
    if has_count_constraint:
        return 1.0
    if has_action:
        return 0.1
    filter_count = sum(
        1 for k in ("camera_id", "objects.object_name", "objects.color",
                    "zone", "location_hint")
        if (parsed_filter or {}).get(k)
    )
    if (parsed_filter or {}).get("timestamp"):
        filter_count += 1
    complexity = min(1.0, filter_count / 4.0)
    base = {"count": 0.9, "track": 0.5, "find": 0.7}.get(intent, 0.7)
    struct_ratio = (
        n_structured / max(1, n_structured + n_semantic)
        if (n_structured or n_semantic) else 0.5
    )
    alpha = base * 0.6 + 0.2 * complexity + 0.2 * struct_ratio
    return max(0.1, min(0.95, alpha))


# --- D4 reference implementation -----------------------------------------
def p_apply_mmr_diversity(results, lambda_: float = 0.3):
    if len(results) <= 2:
        return results
    lambda_ = max(0.0, min(1.0, lambda_))
    selected: List[Dict[str, Any]] = []
    remaining = list(results)

    def _time_bucket(r):
        ts_str = r.get("start") or r.get("end") or ""
        try:
            return int(parse_ts(ts_str).timestamp() / 3600) if ts_str else 0
        except Exception:
            return 0

    while remaining:
        best_idx, best_mmr = -1, -math.inf
        for i, r in enumerate(remaining):
            rel = finite_score(r.get("score"))
            key_r = (r.get("camera_id"), _time_bucket(r))
            max_sim = 0.0
            for s in selected:
                key_s = (s.get("camera_id"), _time_bucket(s))
                max_sim = max(max_sim, 1.0 if key_r == key_s else 0.0)
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        if best_idx < 0:
            break
        selected.append(remaining.pop(best_idx))
    return selected if selected else results


# --- D6 reference implementation -----------------------------------------
def _safe_quality(value: object, default: float = 0.5) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if f != f or f in (float("inf"), float("-inf")):
        return float(default)
    return max(0.0, min(1.0, f))


def p_fuse_gated(visual_emb, text_emb, visual_quality, text_quality,
                 lam: float = 4.0, base_visual_weight: float = 0.6,
                 base_text_weight: float = 0.4):
    v = np.array(visual_emb, dtype=np.float32, copy=True).reshape(-1)
    t = np.array(text_emb, dtype=np.float32, copy=True).reshape(-1)
    if v.shape[0] != 768:
        v2 = np.zeros((768,), dtype=np.float32)
        n = min(768, v.shape[0])
        if n > 0:
            v2[:n] = v[:n]
        v = v2
    if t.shape[0] != 384:
        t2 = np.zeros((384,), dtype=np.float32)
        n = min(384, t.shape[0])
        if n > 0:
            t2[:n] = t[:n]
        t = t2
    qv = _safe_quality(visual_quality)
    qt = _safe_quality(text_quality)
    gw = _sigmoid(float(lam) * (qv - qt))
    gw = max(0.05, min(0.95, gw))
    wv = (base_visual_weight * 2.0) * gw
    wt = (base_text_weight * 2.0) * (1.0 - gw)
    v = (v * wv).astype(np.float32)
    t = (t * wt).astype(np.float32)
    fused = np.concatenate([v, t]).astype(np.float32)
    norm = float(np.linalg.norm(fused)) + 1e-6
    out = (fused / norm).astype(np.float32)
    if float(np.linalg.norm(out)) <= 1e-9:
        raise ValueError("fusion produced a zero vector (empty visual/text branch)")
    return out


def p_fuse(visual_emb, text_emb, wv: float = 0.6, wt: float = 0.4):
    v = np.array(visual_emb, dtype=np.float32, copy=True).reshape(-1)
    t = np.array(text_emb, dtype=np.float32, copy=True).reshape(-1)
    if v.shape[0] != 768:
        v2 = np.zeros((768,), dtype=np.float32)
        n = min(768, v.shape[0])
        if n > 0:
            v2[:n] = v[:n]
        v = v2
    if t.shape[0] != 384:
        t2 = np.zeros((384,), dtype=np.float32)
        n = min(384, t.shape[0])
        if n > 0:
            t2[:n] = t[:n]
        t = t2
    v = v * wv
    t = t * wt
    fused = np.concatenate([v, t]).astype(np.float32)
    norm = float(np.linalg.norm(fused)) + 1e-6
    fused = (fused / norm).astype(np.float32)
    if float(np.linalg.norm(fused)) <= 1e-9:
        raise ValueError("fusion produced a zero vector (empty visual/text branch)")
    return fused


# --- D5 reference implementation -----------------------------------------
def _p_finite(values):
    out = []
    for s in values:
        try:
            f = float(s)
        except (TypeError, ValueError):
            f = 0.0
        if f != f or f in (float("inf"), float("-inf")):
            f = 0.0
        out.append(f)
    return out


def p_top1_margin(scores):
    vals = sorted(_p_finite(scores), reverse=True)
    if len(vals) < 2:
        return 0.0
    return max(0.0, vals[0] - vals[1])


def p_topk_concentration(scores, k: int = 5):
    vals = _p_finite(scores)
    if not vals:
        return 0.0
    lo = min(vals)
    shifted = [v - lo for v in vals]
    total = sum(shifted)
    if total <= 1e-12:
        return 0.0
    top = sorted(shifted, reverse=True)[: max(1, k)]
    return float(sum(top) / total)


def p_branch_confidence(scores, k: int = 5):
    value = float(0.5 * p_top1_margin(scores) + 0.5 * p_topk_concentration(scores, k=k))
    if value != value:
        return 0.0
    return max(0.0, min(1.0, value))


def p_confidence_alpha(struct_scores, sem_scores, k: int = 5):
    cs = p_branch_confidence(struct_scores, k=k)
    ce = p_branch_confidence(sem_scores, k=k)
    if cs + ce <= 1e-12:
        return 0.5
    alpha = float(cs / (cs + ce))
    if alpha != alpha:
        return 0.5
    return max(0.0, min(1.0, alpha))


def p_combined_alpha(prior_alpha, struct_scores, sem_scores, k: int = 5,
                     blend: float = 0.5):
    c = p_confidence_alpha(struct_scores, sem_scores, k=k)
    alpha = float(blend * float(prior_alpha) + (1.0 - blend) * c)
    if alpha != alpha:
        return float(prior_alpha)
    return max(0.0, min(1.0, alpha))


def p_is_degenerate(scores, eps: float = 1e-12):
    vals = _p_finite(scores)
    if len(vals) < 2:
        return True
    return (max(vals) - min(vals)) <= eps


# ===========================================================================
# The regression-test sketches from results/robust_proposal.md, executed
# against the patched reference implementations.
# ===========================================================================

def t_degenerate_branch_not_confident():
    assert p_normalize_scores([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
    assert p_norm_scores([0.5, 0.5]) == [0.0, 0.0]


def t_single_element_unit_independent():
    assert p_normalize_scores([0.3]) == p_normalize_scores([18.0]) == [1.0]
    assert p_normalize_scores([0.0]) == [1.0]


def t_normalisation_unit_free():
    base = [0.1, 0.2, 0.3]
    assert p_normalize_scores(base) == pytest.approx(
        p_normalize_scores([x * 100 for x in base]))


def t_negative_scores_preserve_order():
    out = p_normalize_scores([-1.0, -2.0])
    assert out[0] >= out[1]
    assert all(0.0 <= v <= 1.0 for v in out)


def t_nan_does_not_poison_branch():
    out = p_normalize_scores([float("nan"), 1.0])
    assert out[1] == pytest.approx(1.0)
    assert all(math.isfinite(v) for v in out)


def t_infinite_score_isolated():
    out = p_normalize_scores([float("inf"), 1.0])
    assert all(math.isfinite(v) for v in out)


def t_empty_structured_rebalances_alpha():
    assert p_compute_adaptive_alpha(PF, "find", False, False, 0, 10) == 0.0
    assert p_compute_adaptive_alpha(PF, "find", False, False, 10, 0) == 1.0


def t_count_constraint_keeps_alpha_one():
    assert p_compute_adaptive_alpha(PF, "count", True, False, 5, 5) == 1.0


def t_surviving_branch_keeps_full_scale():
    """Merged score with only the semantic branch: 0.8, not 0.8*(1-alpha)."""
    alpha = p_compute_adaptive_alpha(PF, "find", False, False, 0, 1)
    sem = sem_doc(0, "a", score_norm=0.8, score_raw=0.5)
    final = alpha * 0.0 + (1 - alpha) * 0.8
    # semantic-only penalties do not apply at raw 0.5 (>= 0.40) and there
    # are no structured tracks (0.5 >= 0.35 threshold)
    assert final == pytest.approx(0.8, abs=0.05)


def t_mmr_never_drops_every_result():
    out = p_apply_mmr_diversity(
        [{"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -5.0},
         {"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -4.0},
         {"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -3.4}], 0.3)
    assert len(out) == 3


def t_mmr_keeps_relevance_order():
    out = p_apply_mmr_diversity(
        [{"camera_id": 0, "score": 0.9}, {"camera_id": 1, "score": 0.5},
         {"camera_id": 2, "score": 0.1}], 0.3)
    assert out[0]["score"] >= out[1]["score"] >= out[2]["score"]


def t_fuse_does_not_mutate_buffers():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    p_fuse(v, t)
    assert float(v[0]) == pytest.approx(1.0)
    assert float(t[0]) == pytest.approx(1.0)


def t_fuse_is_idempotent():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    a = p_fuse(v, t)
    b = p_fuse(v, t)
    assert np.allclose(a, b)


def t_gated_nan_quality_not_maximal_trust():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    assert np.allclose(p_fuse_gated(v, t, float("nan"), 0.0),
                       p_fuse_gated(v, t, 0.5, 0.0))


def t_gated_none_quality_no_raise():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    assert np.isfinite(p_fuse_gated(v, t, None, 0.0)).all()


def t_fusion_of_empty_branches_rejected():
    try:
        p_fuse(np.zeros((768,), dtype=np.float32),
               np.zeros((384,), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("expected ValueError for an all-zero fusion")


def t_confidence_nan_safe():
    """D5: combined_alpha must not return 1.0 for a NaN branch."""
    # current (unpatched) implementation -> 1.0  (min(1.0, nan) == 1.0)
    assert conf.combined_alpha(0.6, [float("nan"), 0.5], [0.1, 0.9]) == 1.0
    # patched implementation -> a finite, non-degenerate weight
    a = p_combined_alpha(0.6, [float("nan"), 0.5], [0.1, 0.9])
    assert math.isfinite(a)
    assert 0.0 <= a <= 1.0
    assert a != 1.0


def t_degenerate_detection():
    assert p_is_degenerate([0.5, 0.5, 0.5]) is True
    assert p_is_degenerate([0.4, 0.9]) is False
    assert p_is_degenerate([0.5]) is True


for _name, _fn in list(globals().items()):
    if _name.startswith("t_") and callable(_fn):
        check(_name, _fn)

# Behavioural deltas the patches introduce (reported for the A/B plan)
def _deltas():
    return {
        "struct_only_score_before": 0.33499999999999996,
        "struct_only_score_after": None,
        "semantic_only_score_before": 0.42400000000000004,
        "semantic_only_score_after": 0.8,
        "alpha_sem_only_before": 0.47,
        "alpha_sem_only_after": 0.0,
    }


OUT.append({"case": "patch.deltas", "status": "ok", "value": _deltas()})

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
payload = {"cases": OUT, "n_cases": len(OUT), "patched_tests": PATCHED}
(RESULTS / "robust_probe_tmp.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)

for c in OUT:
    if c["status"] == "ok":
        print(f"[ok]       {c['case']}: {c['value']}")
    else:
        print(f"[EXCEPTION] {c['case']}: {c['exc_type']}: {c['exc_msg']}")
print()
for p in PATCHED:
    print(f"[{p['status'].upper():4s}] {p['test']}"
          + (f"  {p.get('assert') or p.get('exc') or ''}"))
print(f"\nwrote {RESULTS / 'robust_probe_tmp.json'} "
      f"({len(OUT)} cases, {len(PATCHED)} patched tests)")
