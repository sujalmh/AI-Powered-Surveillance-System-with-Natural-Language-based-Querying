"""Robustness regression tests for the merge / fusion paths.

Each test maps 1:1 to a row of the edge-case audit in results/robust_proposal.md.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.app.eval import confidence as conf
from backend.app.services.fusion import MultimodalFusion, fuse_gated
from backend.app.services.result_merger import (
    apply_mmr_diversity,
    compute_adaptive_alpha,
    merge_results,
)
from backend.app.services.retrieval_utils import normalize_scores
from backend.app.services.sem_search import _norm_scores


PF = {"objects.object_name": "person"}


def _sem(clip="a", score_norm=0.8, score_raw=0.5):
    return {
        "camera_id": 0,
        "clip_path": f"clips/{clip}.mp4",
        "clip_url": f"/clips/{clip}.mp4",
        "score": score_raw,
        "score_raw": score_raw,
        "score_norm": score_norm,
        "frames": [{"frame_ts": "2026-03-01T17:10:00"}],
    }


def _struct(ts="2026-03-01T17:10:00", tid=1):
    return {
        "camera_id": 0,
        "timestamp": ts,
        "objects": [{"object_name": "person", "color": "red",
                     "track_id": tid, "confidence": 0.9}],
    }


# --- D1: normalisation -------------------------------------------------

def test_degenerate_branch_is_not_confident():
    """B1/B15: an all-tie branch carries no evidence -> 0.0, not 1.0."""
    assert normalize_scores([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
    assert _norm_scores([0.5, 0.5]) == [0.0, 0.0]


def test_single_element_branch_is_unit_independent():
    """B2: the value must not depend on the unit of the score."""
    assert normalize_scores([0.3]) == normalize_scores([18.0]) == [1.0]
    assert normalize_scores([0.0]) == [1.0]


def test_normalisation_is_unit_free():
    """B2: scaling the input must not change the output (no 0.25/0.40 rule)."""
    base = [0.1, 0.2, 0.3]
    assert normalize_scores(base) == pytest.approx(normalize_scores([x * 100 for x in base]))


def test_negative_scores_preserve_order():
    """B3: the worse candidate must not score higher."""
    out = normalize_scores([-1.0, -2.0])
    assert out[0] >= out[1]
    assert all(0.0 <= v <= 1.0 for v in out)


def test_nan_does_not_poison_the_branch():
    """B4: one NaN must not zero out the whole branch."""
    out = normalize_scores([float("nan"), 1.0])
    assert out[1] == pytest.approx(1.0)
    assert all(math.isfinite(v) for v in out)


def test_infinite_score_is_isolated():
    """B4: +inf must not outrank everything."""
    out = normalize_scores([float("inf"), 1.0])
    assert all(math.isfinite(v) for v in out)


# --- D3: merge --------------------------------------------------------

def test_semantic_score_norm_none_does_not_crash():
    """B5: a None score must not raise out of the merge."""
    sem = _sem()
    sem["score_norm"] = None
    out = merge_results([], [sem], PF, "find")
    assert len(out) == 1
    assert math.isfinite(out[0]["score"])


def test_nan_semantic_score_is_sanitised():
    """B4: NaN must not reach the ranking."""
    sem = _sem()
    sem["score_norm"] = float("nan")
    out = merge_results([], [sem], PF, "find")
    assert math.isfinite(out[0]["score"])


def test_none_branches_are_tolerated():
    """B6: None structured/semantic, with and without a count constraint."""
    assert merge_results(None, None, PF, "find") == []
    assert merge_results(None, [], {"count_constraint": {"gte": 2}}, "count") == []


def test_empty_structured_branch_rebalances_alpha():
    """B8: CombANZ-style -- a branch that returned nothing gives up its weight."""
    only_sem = compute_adaptive_alpha(PF, "find", False, False, 0, 10)
    assert only_sem == 0.0
    only_struct = compute_adaptive_alpha(PF, "find", False, False, 10, 0)
    assert only_struct == 1.0


def test_empty_branch_does_not_deflate_the_surviving_branch():
    """B8: the surviving branch keeps the full [0, 1] scale."""
    sem = _sem(score_norm=0.8)
    out = merge_results([], [sem], PF, "find")
    assert out[0]["score"] == pytest.approx(0.8, abs=0.05)


def test_count_constraint_keeps_alpha_one():
    """B8: the count constraint deliberately disables the semantic branch."""
    assert compute_adaptive_alpha(PF, "count", True, False, 5, 5) == 1.0


# --- D4: MMR ----------------------------------------------------------

def test_mmr_never_drops_every_result():
    """B7: fail open."""
    out = apply_mmr_diversity(
        [{"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -5.0},
         {"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -4.0},
         {"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -3.4}], 0.3)
    assert len(out) == 3


def test_mmr_keeps_the_relevance_order_for_positive_scores():
    out = apply_mmr_diversity(
        [{"camera_id": 0, "score": 0.9}, {"camera_id": 1, "score": 0.5},
         {"camera_id": 2, "score": 0.1}], 0.3)
    assert out[0]["score"] >= out[1]["score"] >= out[2]["score"]


# --- D5: confidence ----------------------------------------------------

def test_confidence_is_nan_safe():
    """B9: NaN must not become alpha=1.0 (min(1.0, nan) == 1.0)."""
    a = conf.combined_alpha(0.6, [float("nan"), 0.5], [0.1, 0.9])
    assert math.isfinite(a)
    assert 0.0 <= a <= 1.0
    assert a != 1.0          # the NaN must not collapse to full structured trust
    assert conf.confidence_alpha([float("nan")], [0.1, 0.9]) == 0.0


def test_degenerate_branch_is_detected():
    assert conf.is_degenerate([0.5, 0.5, 0.5]) is True
    assert conf.is_degenerate([0.4, 0.9]) is False
    assert conf.is_degenerate([0.5]) is True


# --- D6: fusion -------------------------------------------------------

def test_fuse_does_not_mutate_the_caller_buffers():
    """B12: object_detection.py passes float32 batch rows."""
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    MultimodalFusion(0.6, 0.4).fuse(v, t)
    assert float(v[0]) == pytest.approx(1.0)
    assert float(t[0]) == pytest.approx(1.0)


def test_fuse_is_idempotent():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    f = MultimodalFusion(0.6, 0.4)
    a = f.fuse(v.copy(), t.copy())
    b = f.fuse(v.copy(), t.copy())
    assert np.allclose(a, b)


def test_gated_fusion_with_nan_quality_is_not_maximal_trust():
    """B13: NaN must not read as quality 1.0."""
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    out_nan = fuse_gated(v, t, float("nan"), 0.0)
    out_half = fuse_gated(v, t, 0.5, 0.0)
    assert np.allclose(out_nan, out_half)


def test_gated_fusion_with_none_quality_does_not_raise():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    out = fuse_gated(v, t, None, 0.0)
    assert np.isfinite(out).all()


def test_fusion_of_empty_branches_is_rejected():
    """B14: a zero vector must not be indexed."""
    with pytest.raises(ValueError):
        MultimodalFusion(0.6, 0.4).fuse(
            np.zeros((768,), dtype=np.float32), np.zeros((384,), dtype=np.float32))
