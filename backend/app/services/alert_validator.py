"""Alert-rule validation safeguards (R1-10, R2-4).

Novel grammar (routers/alerts.py AlertRuleSpec, chat.py _maybe_create_alert_from_nl)
is untouched. This module ADDS a validator + clarify/abstain path:

- validate_rule(rule) -> (ok, errors, warnings)
- needs_clarification(utterance, rule) -> reason | None
- classify_stress(utterance) -> clean|ambiguous|conflicting|missing|unsupported

Unknown zones, missing thresholds/times, and unsupported verbs must request
clarification instead of materializing a rule (paper §C3 guard).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

KNOWN_ZONES = {"Zone A", "Zone B", "Zone C", "Gate", "Entrance", "Parking"}
KNOWN_EVENTS = {"crowd", "loiter", "fight", "zone", "count", "time", "enter"}
SUPPORTED_VERBS = {
    "alert", "notify", "warn", "detect", "count", "track", "find",
    "gather", "enter", "loiter", "crowd", "fight",
}
UNSUPPORTED_HINTS = {
    "follow him", "follow her", "track him home", "identify the person",
    "recognize face", "unlock", "delete footage", "disable camera",
}


def classify_stress(utterance: str) -> str:
    u = (utterance or "").lower()
    if any(h in u for h in UNSUPPORTED_HINTS):
        return "unsupported"
    if " or " in u and (" and " in u or " vs " in u):
        return "conflicting"
    if re.search(r"\b(a few|several|some|many|back entrance|over there)\b", u):
        return "ambiguous"
    if not re.search(r"\d+", u) or not re.search(
        r"(zone|gate|entrance|parking|camera|am|pm|:)", u
    ):
        return "missing"
    return "clean"


def validate_rule(rule: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Check a 9-tuple rule dict. Returns (ok, errors, warnings)."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(rule, dict):
        return False, ["rule must be a dict"], []
    event = rule.get("event")
    if event not in KNOWN_EVENTS:
        errors.append(f"unknown event: {event!r}")
    zone = (rule.get("area") or {}).get("zone_id", rule.get("zone"))
    if zone and zone not in KNOWN_ZONES:
        warnings.append(f"fuzzy zone {zone!r} needs canonicalization")
    count = rule.get("count") or {}
    if isinstance(count, dict):
        for op in ("==", ">=", "<=", ">", "<", "eq", "gt", "gte", "lt", "lte"):
            if op in count and not isinstance(count[op], (int, float)):
                errors.append(f"count {op} must be numeric")
    else:
        warnings.append("count missing; default >=1 applies")
    tod = rule.get("time_of_day") or {}
    if tod and ("start" not in tod or "end" not in tod):
        warnings.append("partial time window; cross-midnight default may apply")
    cd = rule.get("cooldown_sec", rule.get("cooldown", 60))
    try:
        if int(cd) < 0:
            errors.append("cooldown must be >= 0")
    except Exception:
        errors.append("cooldown must be numeric")
    return (len(errors) == 0, errors, warnings)


def needs_clarification(utterance: str, rule: Dict[str, Any]) -> Optional[str]:
    """Return a clarification prompt, or None if the rule can be materialized."""
    kind = classify_stress(utterance or "")
    if kind == "unsupported":
        return "Unsupported request: I can only create crowd/loiter/fight/zone/count/time rules on fixed zones."
    ok, errors, warnings = validate_rule(rule or {})
    if not ok:
        return "Cannot create rule: " + "; ".join(errors) + ". Please rephrase."
    if kind in ("ambiguous", "missing", "conflicting") or warnings:
        detail = "; ".join(warnings) if warnings else f"{kind} specification"
        return f"Clarify before creating: {detail}."
    return None
