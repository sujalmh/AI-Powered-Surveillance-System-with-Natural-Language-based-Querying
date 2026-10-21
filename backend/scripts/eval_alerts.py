"""Eval: alert authoring incl. stress + validator (R1-10, R2-4).

Scores backend/benchmarks/alert_utterances.jsonl with the real
classify_stress/validate_rule/needs_clarification (no LLM needed).
Reports clean vs stress accuracy + clarify/abstain rate.

Run: python backend/scripts/eval_alerts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.services.alert_validator import (
    classify_stress,
    needs_clarification,
    validate_rule,
)

BENCH = ROOT / "backend" / "benchmarks" / "alert_utterances.jsonl"


def fake_rule(u: dict) -> dict:
    return {
        "event": u.get("expected_event") or "crowd",
        "area": {"zone_id": u.get("expected_zone") or "Zone X"},
        "count": {">=": u.get("expected_count") or 1},
        "time_of_day": {"start": "22:00", "end": "06:00"} if u.get("expected_tod") else {},
        "cooldown_sec": 60,
    }


def main() -> None:
    rows = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines()]
    clean = [r for r in rows if r["stress"] == "clean"]
    stress = [r for r in rows if r["stress"] != "clean"]
    ok_clean = sum(1 for r in clean if needs_clarification(r["text"], fake_rule(r)) is None)
    clarified = sum(1 for r in stress if needs_clarification(r["text"], fake_rule(r)) is not None)
    print(f"clean materialize: {ok_clean}/{len(clean)}")
    print(f"stress clarified/abstained: {clarified}/{len(stress)}")
    by_kind: dict = {}
    for r in stress:
        k = classify_stress(r["text"])
        by_kind[k] = by_kind.get(k, 0) + 1
    print("stress kinds:", by_kind)
    ok, err, warn = validate_rule({"event": "crowd", "area": {"zone_id": "Zone A"},
                                   "count": {">=": 5}, "cooldown_sec": 60})
    print("validator smoke:", ok, err, warn)


if __name__ == "__main__":
    main()
