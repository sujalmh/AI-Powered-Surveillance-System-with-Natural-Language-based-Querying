import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.services.alert_validator import (
    classify_stress,
    needs_clarification,
    validate_rule,
)


def test_clean_rule_materializes():
    rule = {"event": "crowd", "area": {"zone_id": "Zone A"},
            "count": {">=": 5}, "cooldown_sec": 60}
    ok, err, _ = validate_rule(rule)
    assert ok and not err
    assert needs_clarification(
        "alert me when more than five people gather in Zone A after 10 PM", rule) is None


def test_stress_taxonomy_and_guard():
    assert classify_stress("notify me to follow him home") == "unsupported"
    assert classify_stress("alert if a few people gather over there") == "ambiguous"
    bad = {"event": "crowd", "area": {"zone_id": "Zone X"},
           "count": {">=": 5}, "cooldown_sec": 60}
    assert needs_clarification("alert if a few people gather over there", bad) is not None
    assert needs_clarification("notify me to follow him home", bad) is not None
