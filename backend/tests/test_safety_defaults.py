"""Responsible-deployment defaults: gender off, approval gate, MMR off (E4-E6).

Paper evidence lives in the released JSONs; this module checks the code
defaults and the gate behavior without re-running long eval scripts:

* ``INCLUDE_GENDER_ATTRIBUTES`` defaults to False (fresh Settings, no env).
* ``ENABLE_RESULT_DIVERSITY`` (MMR) defaults to False; the released
  ablation (``results/ablation_results.json``) records base 0.538 vs MMR
  0.400, so the default matches the 0.538 control.
* ``requires_approval`` schema: default True for consequential families,
  opt-out without ``approval_authorized`` is rejected, explicit opt-out
  with the flag validates.
* ``alert_engine`` stages ``pending_approval`` with no normal alert log
  (no side effects) until ``approve_alert``; reject path covered too.
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.services.alert_validator import (  # noqa: E402
    APPROVAL_AUTH_FLAG,
    CONSEQUENTIAL_EVENTS,
    default_requires_approval,
    validate_rule,
)


def _fresh_settings(monkeypatch):
    monkeypatch.delenv("INCLUDE_GENDER_ATTRIBUTES", raising=False)
    monkeypatch.delenv("ENABLE_RESULT_DIVERSITY", raising=False)
    monkeypatch.delenv("MMR_DIVERSITY_LAMBDA", raising=False)
    import backend.app.config as cfg

    importlib.reload(cfg)
    try:
        return cfg.Settings()
    finally:
        importlib.reload(cfg)


def _rule(**overrides):
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
    base.update(overrides)
    return base


# --- defaults -----------------------------------------------------------

def test_gender_attributes_default_off(monkeypatch):
    settings = _fresh_settings(monkeypatch)
    assert settings.INCLUDE_GENDER_ATTRIBUTES is False


def test_mmr_diversity_default_off_but_option_kept(monkeypatch):
    settings = _fresh_settings(monkeypatch)
    assert settings.ENABLE_RESULT_DIVERSITY is False
    assert float(settings.MMR_DIVERSITY_LAMBDA) == pytest.approx(0.3)


def test_released_ablation_numbers_unchanged():
    """MMR off (0.538) vs MMR on (0.400): the released evidence table."""
    data = json.loads((ROOT / "results" / "ablation_results.json").read_text(encoding="utf-8"))
    assert data["summary"]["base (no recency, no MMR)"]["macro"] == pytest.approx(0.538)
    assert data["summary"]["MMR only"]["macro"] == pytest.approx(0.400)
    assert data["significance_vs_base"]["MMR only"]["delta_p5"] == pytest.approx(-0.138)


# --- approval schema ----------------------------------------------------

def test_consequential_events_default_to_approval():
    for event in ("crowd", "loiter", "fight", "enter", "zone"):
        assert event in CONSEQUENTIAL_EVENTS
        assert default_requires_approval(event) is True
    assert default_requires_approval("count") is False
    assert default_requires_approval("time") is False


def test_validator_accepts_missing_flag_and_rejects_silent_opt_out():
    ok, errors, _ = validate_rule(_rule())
    assert ok is True and errors == []

    ok, errors, _ = validate_rule(_rule(requires_approval=True))
    assert ok is True and errors == []

    ok, errors, _ = validate_rule(_rule(requires_approval=False))
    assert ok is False
    assert any("approval_authorized" in e for e in errors)

    ok, errors, _ = validate_rule(
        _rule(requires_approval=False, approval_authorized=True)
    )
    assert ok is True and errors == []

    ok, errors, _ = validate_rule(_rule(requires_approval="yes"))
    assert ok is False
    assert any("boolean" in e for e in errors)


# --- approval engine ----------------------------------------------------

class _FakeCol:
    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(doc)
        return SimpleNamespace(inserted_id=len(self.docs))


class _FakeAlerts:
    def __init__(self, docs):
        self.docs = {d["_id"]: dict(d) for d in docs}

    def update_one(self, filt, update):
        _id = filt.get("_id")
        if _id not in self.docs:
            return SimpleNamespace(matched_count=0)
        self.docs[_id].update(update.get("$set", {}))
        return SimpleNamespace(matched_count=1)


def _engine(monkeypatch, rule_docs):
    import backend.app.services.alert_engine as eng

    logs = _FakeCol()
    monkeypatch.setattr(eng, "alert_logs_col", logs)
    monkeypatch.setattr(eng, "_RULE_CACHE", eng.RuleCache(rules=list(rule_docs), last_refresh=1e12))
    monkeypatch.setattr(eng, "_LAST_FIRED", {})
    from collections import defaultdict as _dd

    monkeypatch.setattr(eng, "_PER_CAMERA", _dd(dict))
    return eng, logs


def _realtime_rule(requires_approval=True, **kw):
    doc = {
        "_id": "r-gate-1",
        "name": "NL: gate test",
        "enabled": True,
        "cooldown_sec": 0,
        "actions": ["snapshot", "push_ws"],
        "severity": "info",
        "rule": {
            "event": "crowd",
            "objects": [{"name": "person"}],
            "requires_approval": requires_approval,
        },
    }
    doc.update(kw)
    return doc


def test_no_side_effect_before_approval_then_fires_after(monkeypatch):
    eng, logs = _engine(monkeypatch, [_realtime_rule(requires_approval=True)])
    ts = datetime.now(timezone.utc).isoformat()
    objs = [{"object_name": "person"}]

    eng.evaluate_realtime(999, ts, objs)
    pending = [d for d in logs.docs if d.get("status") == "pending_approval"]
    normal = [d for d in logs.docs if d.get("status") != "pending_approval"]
    assert len(pending) == 1, "one staged pending_approval log"
    assert normal == [], "no normal alert log (no side effects) before approval"

    # Operator approves; the same trigger must now produce a normal log.
    eng._RULE_CACHE.rules[0]["approved"] = True
    eng.evaluate_realtime(999, ts, objs)
    normal = [d for d in logs.docs if d.get("status") != "pending_approval"]
    assert len(normal) == 1, "normal alert log written after approval"


def test_explicit_opt_out_with_auth_fires_without_approval(monkeypatch):
    doc = _realtime_rule(requires_approval=False)
    doc["rule"][APPROVAL_AUTH_FLAG] = True
    eng, logs = _engine(monkeypatch, [doc])
    ts = datetime.now(timezone.utc).isoformat()
    eng.evaluate_realtime(999, ts, [{"object_name": "person"}])
    assert len(logs.docs) == 1
    assert logs.docs[0].get("status") != "pending_approval"


def test_approve_and_reject_paths(monkeypatch):
    import backend.app.services.alert_engine as eng

    fake_alerts = _FakeAlerts([{"_id": "a1", "name": "x"}])
    logs = _FakeCol()
    monkeypatch.setattr(eng, "alerts_col", fake_alerts)
    monkeypatch.setattr(eng, "alert_logs_col", logs)

    out = eng.approve_alert("a1", actor="tester")
    assert out == {"ok": True, "alert_id": "a1", "status": "approved"}
    assert fake_alerts.docs["a1"]["approval_status"] == "approved"
    assert logs.docs[-1]["status"] == "approved"

    out = eng.reject_alert("a1", actor="tester")
    assert out["status"] == "rejected"
    assert fake_alerts.docs["a1"]["approval_status"] == "rejected"

    out = eng.approve_alert("missing", actor="tester")
    assert out["ok"] is False


# --- rebuilt gender-free descriptor -------------------------------------

_GENDER_RE = re.compile(
    r"\b(male|female|man|woman|men|women|boy|girl|masculine|feminine|gender)\b",
    re.IGNORECASE,
)


def test_rebuilt_person_index_has_no_gender():
    meta_path = ROOT / "data" / "index" / "faiss_person_nogender" / "meta.json"
    if not meta_path.exists():
        pytest.skip("nogender index not built in this checkout")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(meta) == 1778
    hits = [m["faiss_id"] for m in meta if _GENDER_RE.search(m.get("attribute_text", ""))]
    assert hits == []
