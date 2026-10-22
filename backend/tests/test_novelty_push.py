"""Tests for the C1'/C2'/C3' novelty push (all CPU-only, no network)."""
from __future__ import annotations

import numpy as np

from backend.app.eval import confidence as conf
from backend.app.services.fusion import MultimodalFusion, fuse_gated
from backend.app.services.result_merger import compute_adaptive_alpha
from backend.app.services.alert_validator import validate_rule
from backend.app.services.clarification_policy import minimal_questions


def test_confidence_margin_and_concentration():
    assert conf.top1_margin([0.9, 0.1, 0.05]) == 0.8
    assert conf.top1_margin([0.5]) == 0.0
    assert conf.top1_margin([]) == 0.0
    assert conf.topk_concentration([0.5, 0.5, 0.5]) == 0.0  # flat -> no mass
    c = conf.topk_concentration([0.9, 0.1, 0.05])
    assert 0.0 < c <= 1.0
    assert conf.overlap_at_k(["a", "b", "c"], ["b", "c", "d"], 2) == 0.5
    assert conf.confidence_alpha([0.9, 0.1], [0.5, 0.49]) > 0.5
    assert conf.confidence_alpha([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_adaptive_alpha_score_path_backward_compatible():
    pf = {"camera_id": 0, "timestamp": True, "objects.object_name": "person"}
    prior = compute_adaptive_alpha(pf, "find", False, False, 10, 20)
    assert prior == compute_adaptive_alpha(
        pf, "find", False, False, 10, 20,
        struct_scores=None, sem_scores=None,
    )
    blended = compute_adaptive_alpha(
        pf, "find", False, False, 10, 20,
        struct_scores=[0.9, 0.1, 0.05], sem_scores=[0.4, 0.38, 0.37],
    )
    assert 0.0 <= blended <= 1.0
    # Overrides still dominate through the prior.
    assert compute_adaptive_alpha({}, "find", True, False, 5, 5) == 1.0
    assert compute_adaptive_alpha({}, "find", False, True, 5, 5) == 0.1


def test_gated_fusion_tie_reproduces_static():
    rng = np.random.RandomState(0)
    v = rng.randn(768).astype(np.float32)
    t = rng.randn(384).astype(np.float32)
    static = MultimodalFusion(0.6, 0.4).fuse(v.copy(), t.copy())
    gated = fuse_gated(v.copy(), t.copy(), 0.5, 0.5)
    assert gated.shape == (1152,)
    assert gated.dtype == np.float32
    assert np.allclose(gated, static, atol=1e-6)
    # Higher visual quality shifts mass to the visual branch.
    g_hi = fuse_gated(v.copy(), t.copy(), 1.0, 0.0)
    assert float(np.linalg.norm(g_hi[:768])) > float(np.linalg.norm(gated[:768]))
    assert abs(float(np.linalg.norm(gated)) - 1.0) < 1e-4


def test_validator_hardening_rejects_degenerate_rules():
    base = {
        "event": "crowd",
        "objects": [{"name": "person"}],
        "area": {"zone_id": "Zone A"},
        "threshold": {"op": ">=", "value": 5, "unit": "persons"},
        "time_of_day": {},
        "severity": "info",
        "cooldown_sec": 60,
        "side_effects": ["snapshot", "push_ws"],
        "cadence_hz": 1.0,
    }
    assert validate_rule(dict(base))[0] is True

    bad = dict(base, threshold={"op": "between", "value": [10, 3], "unit": "persons"})
    ok, errors, _ = validate_rule(bad)
    assert ok is False and any("lower bound" in e for e in errors)

    bad = dict(base, threshold={"op": ">=", "value": -1, "unit": "persons"})
    assert validate_rule(bad)[0] is False

    bad = dict(base, side_effects=[])
    ok, errors, _ = validate_rule(bad)
    assert ok is False and any("side_effects" in e for e in errors)

    bad = dict(base, cadence_hz=100.0)
    assert validate_rule(bad)[0] is False

    bad = dict(base, time_of_day={"start": "20:00", "end": "20:00"})
    ok, errors, _ = validate_rule(bad)
    assert ok is False and any("degenerate" in e for e in errors)


def test_minimal_questions_policy():
    assert minimal_questions("alert me when more than five people gather in Zone A") == []
    qs = minimal_questions("alert me when people gather")
    slots = [q["slot"] for q in qs]
    assert "zone" in slots  # missing zone is asked; crowd threshold present? no number -> threshold asked too
    assert "threshold" in slots
    qs = minimal_questions("alert me if a few people gather in Zone A")
    assert len(qs) == 1 and qs[0]["slot"] == "ambiguous"
    qs = minimal_questions("follow him home and alert me")
    assert len(qs) == 1 and qs[0]["slot"] == "unsupported"


def test_conjunctive_bounds_combine():
    from backend.app.services.alert_materializer import materialize_alert_rule

    r = materialize_alert_rule(
        "alert me when more than five fewer than ten people gather in Zone A"
    )
    assert r["needs_clarification"] is False
    assert r["threshold"] == {"op": "between", "value": [6, 9], "unit": "persons"}
    r = materialize_alert_rule(
        "notify me if someone loiters in Zone B for more than five minutes less than ten minutes"
    )
    assert r["needs_clarification"] is False
    assert r["threshold"] == {"op": "between", "value": [301, 599], "unit": "seconds"}
    assert validate_rule(r)[0] is True


def test_inverted_between_and_ceiling_clarify():
    from backend.app.services.alert_materializer import materialize_alert_rule

    r = materialize_alert_rule("alert when between ten and three people gather in Zone A")
    assert r["needs_clarification"] is True
    assert r["clarification_kind"] == "conflicting"
    r = materialize_alert_rule("alert me 20 hz more than five people gather in Zone A")
    assert r["needs_clarification"] is True
    assert r["clarification_kind"] == "conflicting"


def test_zone_alias_does_not_leak_object():
    from backend.app.services.alert_materializer import materialize_alert_rule

    r = materialize_alert_rule("alert me during the day car park")
    assert r["objects"] == [{"name": "person"}]
    assert validate_rule(r)[0] is True


def test_clean_materializations_pass_validator():
    from backend.app.services.alert_materializer import materialize_alert_rule

    corpus = [
        "alert me when more than five people gather in Zone A after 10 PM",
        "alert me when more than five fewer than ten people gather in Zone A",
        "warn me on a fight near the Entrance",
        "notify me if someone loiters in Zone B",
        "alert me if a person loiters in Zone A for at least one minute, cooldown 5 minutes",
        "alert me when at least nine people gather in Zone A, notification only",
    ]
    for text in corpus:
        rule = materialize_alert_rule(text)
        if not rule.get("needs_clarification"):
            assert validate_rule(rule)[0] is True, text


def test_two_zone_conjunction_clarifies():
    from backend.app.services.alert_materializer import materialize_alert_rule

    r = materialize_alert_rule("more than 5 people in Zone A and Zone B")
    assert r["needs_clarification"] is True
    assert "Zone A" in r["clarification"] and "Zone B" in r["clarification"]


def test_until_single_materializes_window():
    from backend.app.services.alert_materializer import materialize_alert_rule

    r = materialize_alert_rule(
        "alert me when more than five people gather in Zone A until 10 PM"
    )
    assert r["needs_clarification"] is False
    assert r["time_of_day"] == {"start": "00:00", "end": "22:00"}
    assert validate_rule(r)[0] is True


def test_invalid_clock_clarifies():
    from backend.app.services.alert_materializer import materialize_alert_rule

    r = materialize_alert_rule(
        "alert me when more than five people gather in Zone A between 10:70 and 12 AM"
    )
    assert r["needs_clarification"] is True
