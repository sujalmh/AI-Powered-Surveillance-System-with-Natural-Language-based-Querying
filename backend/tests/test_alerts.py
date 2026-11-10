"""Tests for the C3 alert grammar, benchmark, validator, and chat wiring."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.services.alert_materializer import (  # noqa: E402
    CANONICAL_FIELDS,
    materialize_alert_rule,
)
from backend.app.services.alert_validator import (  # noqa: E402
    classify_stress,
    needs_clarification,
    validate_rule,
)

BENCH = ROOT / "backend" / "benchmarks" / "alert_utterances.jsonl"
SCHEMA = ROOT / "backend" / "benchmarks" / "alert_schema.json"
PAPER_UTTERANCE = "alert me when more than five people gather in Zone A after 10 PM"


def _rows():
    return [
        json.loads(line)
        for line in BENCH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_benchmark_is_120_unique_rows_with_stratified_split():
    rows = _rows()
    assert len(rows) == 120
    assert len({r["uid"] for r in rows}) == 120
    assert len({r["text"].lower() for r in rows}) == 120
    assert sum(r["stress"] == "clean" for r in rows) == 60
    assert sum(r["stress"] != "clean" for r in rows) == 60
    for kind in ("ambiguous", "conflicting", "missing", "unsupported"):
        assert sum(r["stress"] == kind for r in rows) == 15
    for split in ("dev", "test"):
        assert sum(r["stress"] == "clean" and r["split"] == split for r in rows) == 30
        assert sum(r["stress"] != "clean" and r["split"] == split for r in rows) == 30
    for row in rows:
        assert row["split"] in {"dev", "test"}
        assert isinstance(row["should_clarify"], bool)
        if row["stress"] == "clean":
            assert set(CANONICAL_FIELDS) <= set(row["expected"])
        else:
            assert row["should_clarify"] is True
            assert row["expected"] is None
            assert row["reason"]


def test_schema_matches_canonical_nine_fields():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert set(schema["required"]) == set(CANONICAL_FIELDS)
    assert "count" in schema["properties"]
    assert schema["properties"]["clarification_kind"]["enum"] == [
        "clean", "ambiguous", "conflicting", "missing", "unsupported"
    ]


def test_paper_example_materializes_crowd_window():
    rule = materialize_alert_rule(PAPER_UTTERANCE)
    assert rule["needs_clarification"] is False
    assert rule["clarification"] is None
    assert rule["event"] == "crowd"
    assert rule["objects"] == [{"name": "person"}]
    assert rule["area"] == {"zone_id": "Zone A"}
    assert rule["count"] == {">": 5}
    assert rule["threshold"] == {"op": ">", "value": 5, "unit": "persons"}
    assert rule["time_of_day"] == {"start": "22:00", "end": "06:00"}
    assert rule["severity"] == "info"
    assert rule["cooldown_sec"] == 60
    assert rule["side_effects"] == ["snapshot", "push_ws"]
    assert rule["cadence_hz"] == 1.0
    assert validate_rule(rule)[0] is True


@pytest.mark.parametrize(
    ("text", "event"),
    [
        ("alert me when more than five people gather in Zone A after 10 PM", "crowd"),
        ("notify me if someone loiters in Zone B for over a minute", "loiter"),
        ("warn me on a fight near the Entrance", "fight"),
        ("alert me when a car enters the Gate", "enter"),
        ("alert me if anyone is inside the Gate", "zone"),
        ("notify me if the number of cars in the Parking lot is at least 5", "count"),
        ("alert me every night at the Gate", "time"),
    ],
)
def test_each_clean_event_family_materializes(text, event):
    rule = materialize_alert_rule(text)
    assert rule["needs_clarification"] is False
    assert rule["event"] == event
    assert validate_rule(rule)[0] is True


def test_event_specific_threshold_defaults():
    fight = materialize_alert_rule("warn me on a fight near the Entrance")
    assert fight["threshold"] == {"op": ">=", "value": 2, "unit": "persons"}
    loiter = materialize_alert_rule("notify me if someone loiters in Zone B")
    assert loiter["threshold"] == {"op": ">=", "value": 60, "unit": "seconds"}


@pytest.mark.parametrize(
    ("text", "window"),
    [
        ("notify me if someone enters Zone B after 11 PM", {"start": "23:00", "end": "06:00"}),
        ("alert me between 9 and 11 PM in Zone A", {"start": "21:00", "end": "23:00"}),
        ("alert me during the day in Parking", {"start": "06:00", "end": "20:00"}),
        ("alert me every night at the Gate", {"start": "20:00", "end": "06:00"}),
    ],
)
def test_time_windows_and_cross_midnight(text, window):
    rule = materialize_alert_rule(text)
    assert rule["time_of_day"] == window
    assert rule["needs_clarification"] is False


def test_quantifiers_cadence_cooldown_and_side_effects():
    rule = materialize_alert_rule(
        "notify me when five or more people gather in the Parking lot every 30 seconds"
    )
    assert rule["threshold"] == {"op": ">=", "value": 5, "unit": "persons"}
    assert rule["cadence_hz"] == 1 / 30

    rule = materialize_alert_rule(
        "alert me if a person loiters in Zone A for at least one minute, cooldown 5 minutes"
    )
    assert rule["threshold"] == {"op": ">=", "value": 60, "unit": "seconds"}
    assert rule["cooldown_sec"] == 300

    rule = materialize_alert_rule(
        "alert me when at least nine people gather in Zone A, notification only"
    )
    assert rule["side_effects"] == ["push_ws"]

    rule = materialize_alert_rule(
        "notify me when the vehicle count at the Gate is between six and ten "
        "from 8 PM to 10 PM"
    )
    assert rule["threshold"] == {"op": "between", "value": [6, 10], "unit": "vehicles"}
    assert rule["count"] == {">=": 6, "<=": 10}
    assert rule["time_of_day"] == {"start": "20:00", "end": "22:00"}


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("alert me if a few people gather in Zone A", "ambiguous"),
        ("notify me if people gather over there", "ambiguous"),
        ("notify me if people gather in Zone A or Zone B", "conflicting"),
        ("alert me if more than five but fewer than three people gather in Zone A", "conflicting"),
        ("alert me when people gather", "missing"),
        ("notify me if a crowd gathers in Zone A", "missing"),
        ("notify me to follow him home", "unsupported"),
        ("alert me if you can identify the person by face", "unsupported"),
        ("alert me and delete the footage after an hour", "unsupported"),
        ("notify me if someone tries to disable the camera", "unsupported"),
    ],
)
def test_stress_kinds_abstain(text, kind):
    rule = materialize_alert_rule(text)
    assert rule["needs_clarification"] is True
    assert rule["clarification"]
    assert rule["clarification_kind"] == kind
    assert classify_stress(text) == kind
    assert needs_clarification(text) is not None


def test_legacy_count_only_rule_still_validates_for_alert_rule_spec():
    legacy = {
        "event": "crowd",
        "objects": [{"name": "person"}],
        "area": {"zone_id": "Zone A"},
        "count": {">=": 5},
    }
    ok, errors, warnings = validate_rule(legacy)
    assert ok is True
    assert errors == []
    assert any("legacy" in warning.lower() for warning in warnings)


class _StubAlerts:
    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(doc)
        return SimpleNamespace(inserted_id="stub-alert-id")


def test_chat_helper_creates_alert_and_clarifies_without_insert(monkeypatch):
    import backend.app.routers.chat as chat

    stub = _StubAlerts()
    monkeypatch.setattr(chat, "alerts_col", stub)

    created = chat._maybe_create_alert_from_nl(PAPER_UTTERANCE, {})
    assert created is not None
    assert created["created"] is True
    assert created["needs_clarification"] is False
    assert len(stub.docs) == 1
    stored = stub.docs[0]
    assert stored["rule"]["event"] == "crowd"
    assert stored["rule"]["area"] == {"zone_id": "Zone A"}
    assert stored["rule"]["time_of_day"] == {"start": "22:00", "end": "06:00"}
    assert stored["cooldown_sec"] == 60.0
    assert stored["actions"] == ["snapshot", "push_ws"]

    clarified = chat._maybe_create_alert_from_nl(
        "alert me if a few people gather in Zone A", {}
    )
    assert clarified is not None
    assert clarified["created"] is False
    assert clarified["needs_clarification"] is True
    assert clarified["clarification"]["kind"] == "ambiguous"
    assert clarified["clarification"]["message"]
    assert len(stub.docs) == 1, "clarification must not insert an alert"

    assert chat._maybe_create_alert_from_nl("show me people in Zone A", {}) is None
    assert len(stub.docs) == 1


def test_chat_rejects_materialized_rule_that_fails_validator(monkeypatch):
    import backend.app.routers.chat as chat

    stub = _StubAlerts()
    monkeypatch.setattr(chat, "alerts_col", stub)
    bad_rule = {
        "event": "crowd",
        "objects": [{"name": "person"}],
        "area": {},
        "count": {">=": 5},
        "threshold": {"op": ">=", "value": 5, "unit": "persons"},
        "time_of_day": {},
        "severity": "info",
        "cooldown_sec": 60,
        "side_effects": ["snapshot", "push_ws"],
        "cadence_hz": 1.0,
        "needs_clarification": False,
        "clarification": None,
        "clarification_kind": "clean",
    }
    monkeypatch.setattr(chat, "materialize_alert_rule", lambda *args, **kwargs: dict(bad_rule))

    result = chat._maybe_create_alert_from_nl(PAPER_UTTERANCE, {})
    assert result is not None
    assert result["created"] is False
    assert result["needs_clarification"] is True
    assert result["clarification"]["kind"] == "invalid"
    assert "area.zone_id is required" in result["clarification"]["message"]
    assert stub.docs == []


def test_chat_response_exposes_alert_clarification_field():
    from backend.app.routers.chat import ChatSendResponse

    response = ChatSendResponse(
        session_id="s1", parsed_filter={}, results=[], answer="need detail"
    )
    assert response.alert_clarification is None
