"""Alert-rule validation safeguards (R1-10, R1-14, R2-4).

This module is the validation layer used by the live chat alert path and by
``backend/scripts/eval_alerts.py``.  The canonical rule representation is the
paper's 9-tuple::

    (event, objects, area/zone, threshold, time_of_day,
     severity, cooldown_sec, side_effects, cadence_hz)

``count`` is a derived compatibility field for
``backend/app/routers/alerts.py::AlertRuleSpec``.  It is *not* a tenth field.

Public API
----------
- ``validate_rule(rule) -> (ok, errors, warnings)``
- ``needs_clarification(utterance, rule=None) -> reason | None``
- ``classify_stress(utterance) -> clean|ambiguous|conflicting|missing|unsupported``

The stress classifier is deterministic and intentionally conservative: an
utterance that cannot be turned into a complete rule without guessing is
abstained on instead of being materialized.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Canonical closed vocabularies (keep in sync with alert_schema.json).
KNOWN_ZONES = {"Zone A", "Zone B", "Zone C", "Gate", "Entrance", "Parking"}
KNOWN_EVENTS = {"crowd", "loiter", "fight", "zone", "count", "time", "enter"}
SEVERITIES = {"info", "warning", "high", "critical"}
SIDE_EFFECTS = {"store_clip", "snapshot", "push_ws"}
THRESHOLD_OPS = {">", ">=", "<", "<=", "==", "between"}
THRESHOLD_UNITS = {"persons", "vehicles", "objects", "seconds"}

SUPPORTED_VERBS = {
    "alert", "notify", "warn", "detect", "count", "find",
    "gather", "enter", "loiter", "crowd", "fight",
}

# Phrases that are outside the typed grammar.  Kept as a set for compatibility
# with the original validator and as a cheap pre-filter in the materializer.
UNSUPPORTED_HINTS = {
    "follow him home", "follow her home", "follow them home",
    "track him home", "identify the person", "identify a person",
    "recognize face", "recognise face", "face recognition",
    "delete footage", "delete the footage", "erase footage",
    "disable camera", "disable the camera", "turn off the camera",
    "unlock the door", "license plate", "number plate",
    "gender", "criminal record", "weapon",
}


def _zone_of(rule: Dict[str, Any]) -> Optional[str]:
    area = rule.get("area")
    if isinstance(area, dict):
        z = area.get("zone_id")
        if z:
            return str(z)
    z = rule.get("zone")
    return str(z) if z else None


def _objects_of(rule: Dict[str, Any]) -> List[str]:
    objects = rule.get("objects")
    if not isinstance(objects, list):
        return []
    out: List[str] = []
    for obj in objects:
        if isinstance(obj, dict) and obj.get("name"):
            out.append(str(obj["name"]))
        elif isinstance(obj, str):
            out.append(obj)
    return out


def _threshold_of(rule: Dict[str, Any]) -> Dict[str, Any]:
    threshold = rule.get("threshold")
    if isinstance(threshold, dict):
        return threshold
    # Backwards compatibility with AlertRuleSpec's count-only rules.
    count = rule.get("count")
    if isinstance(count, dict) and count:
        op_map = {
            ">": ">", ">=": ">=", "<": "<", "<=": "<=",
            "==": "==", "eq": "==", "gt": ">", "gte": ">=",
            "lt": "<", "lte": "<=",
        }
        for key, value in count.items():
            if key in op_map and isinstance(value, (int, float)):
                unit = "persons"
                names = _objects_of(rule)
                if names and names[0] in {"car", "truck", "bus", "van", "vehicle"}:
                    unit = "vehicles"
                elif names and names[0] in {"bag", "backpack"}:
                    unit = "objects"
                return {"op": op_map[key], "value": value, "unit": unit}
    return {}


def validate_rule(rule: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Check a complete 9-field rule.  Returns ``(ok, errors, warnings)``.

    Legacy rules that carry only ``count`` (AlertRuleSpec) are accepted with a
    warning so existing API callers do not break.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(rule, dict):
        return False, ["rule must be a dict"], []

    event = rule.get("event")
    if event not in KNOWN_EVENTS:
        errors.append(f"unknown event: {event!r}")

    names = _objects_of(rule)
    if not names:
        errors.append("objects must contain at least one {'name': ...} entry")

    zone = _zone_of(rule)
    if not zone:
        errors.append("area.zone_id is required")
    elif zone not in KNOWN_ZONES:
        errors.append(f"unknown zone: {zone!r}")

    threshold = _threshold_of(rule)
    if not threshold:
        errors.append("threshold is required (or a legacy count dict)")
    else:
        op = threshold.get("op")
        value = threshold.get("value")
        unit = threshold.get("unit")
        if op not in THRESHOLD_OPS:
            errors.append(f"invalid threshold op: {op!r}")
        if op == "between":
            if not (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(isinstance(v, (int, float)) for v in value)
            ):
                errors.append("between threshold value must be [low, high]")
        elif not isinstance(value, (int, float)):
            errors.append("threshold value must be numeric")
        if unit not in THRESHOLD_UNITS:
            errors.append(f"invalid threshold unit: {unit!r}")

    tod = rule.get("time_of_day")
    if tod is not None and not isinstance(tod, dict):
        errors.append("time_of_day must be an object")
    elif isinstance(tod, dict) and tod:
        if "start" not in tod or "end" not in tod:
            errors.append("time_of_day needs both start and end")
        else:
            for key in ("start", "end"):
                if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", str(tod[key])):
                    errors.append(f"time_of_day.{key} must be HH:MM")

    severity = rule.get("severity", "info")
    if severity not in SEVERITIES:
        errors.append(f"invalid severity: {severity!r}")

    try:
        if float(rule.get("cooldown_sec", 60)) < 0:
            errors.append("cooldown_sec must be >= 0")
    except (TypeError, ValueError):
        errors.append("cooldown_sec must be numeric")

    effects = rule.get("side_effects", ["snapshot", "push_ws"])
    if not isinstance(effects, list) or any(e not in SIDE_EFFECTS for e in effects):
        errors.append(f"invalid side_effects: {effects!r}")

    try:
        if float(rule.get("cadence_hz", 1.0)) <= 0:
            errors.append("cadence_hz must be > 0")
    except (TypeError, ValueError):
        errors.append("cadence_hz must be numeric")

    if "threshold" not in rule and "count" in rule:
        warnings.append("legacy count-only rule; threshold inferred for validation")

    return (len(errors) == 0, errors, warnings)


def classify_stress(utterance: str) -> str:
    """Classify an utterance as ``clean`` or one of the four stress kinds.

    Delegates to the materializer so the benchmark taxonomy and the live
    clarification path cannot drift apart.
    """
    from backend.app.services.alert_materializer import classify_utterance

    return classify_utterance(utterance or "")


def needs_clarification(
    utterance: str, rule: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Return a clarification prompt, or ``None`` if the rule can be created.

    The materializer is the source of truth for the utterance.  If a caller
    also supplies a pre-built rule (e.g. from an LLM or the REST API), the rule
    is validated too; an invalid rule is never inserted.
    """
    from backend.app.services.alert_materializer import materialize_alert_rule

    materialized = materialize_alert_rule(utterance or "")
    if materialized.get("needs_clarification"):
        return materialized.get("clarification")

    if rule is not None:
        ok, errors, warnings = validate_rule(rule)
        if not ok:
            return "Cannot create rule: " + "; ".join(errors) + ". Please rephrase."
        blocking = [w for w in warnings if "legacy" not in w.lower()]
        if blocking:
            return "Clarify before creating: " + "; ".join(blocking) + "."
    return None
