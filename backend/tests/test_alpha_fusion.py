import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.eval.baselines import (
    fixed_alpha_score,
    norm_matched_add,
    per_intent_lookup_alpha,
    random_proj_add,
    rrf_fuse,
    zero_pad_add,
    zscore_norm,
)
from backend.app.services.fusion import MultimodalFusion
from backend.app.services.result_merger import compute_adaptive_alpha


def test_novel_alpha_overrides_preserved():
    assert compute_adaptive_alpha({}, "find", True, False, 5, 5) == 1.0
    assert compute_adaptive_alpha({}, "find", False, True, 5, 5) == 0.1
    a = compute_adaptive_alpha(
        {"camera_id": 1, "zone": "A"}, "find", False, False, 3, 5)
    assert 0.10 <= a <= 0.95


def test_lookup_and_fixed():
    assert per_intent_lookup_alpha("count") == 1.0
    assert per_intent_lookup_alpha("action") == 0.0
    assert fixed_alpha_score(1.0, 0.0, 0.3) == 0.3
    assert len(zscore_norm([1.0, 2.0, 3.0])) == 3


def test_rrf_prefers_top():
    out = rrf_fuse(["a", "b"], ["b", "a"])
    assert set(out) == {"a", "b"}


def test_concat_vs_additives_rank():
    rng = np.random.RandomState(0)
    v = rng.randn(768).astype(np.float32)
    t = rng.randn(384).astype(np.float32)
    c = MultimodalFusion(0.6, 0.4).fuse(v, t)
    assert c.shape == (1152,)
    assert abs(float(np.linalg.norm(c)) - 1.0) < 1e-4
    assert zero_pad_add(v, t).shape == (768,)
    assert norm_matched_add(v, t).shape == (768,)
    assert random_proj_add(v, t, seed=0).shape == (768,)
