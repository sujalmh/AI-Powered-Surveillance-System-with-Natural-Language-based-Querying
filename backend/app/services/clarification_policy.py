"""Minimal-question clarification policy (C3').

Given an utterance (and optionally its materialized rule), return the
minimal ordered question list an oracle user must answer to yield a
complete, valid 9-tuple rule.

Minimality rule: slots with documented defaults (objects, severity,
cooldown_sec, side_effects, cadence_hz, all-day time window) are never
asked about.  Only *required* slots without a usable value produce a
question, in dependency order:

1. ``event`` -- the denotation to watch for;
2. ``zone`` -- the canonical area the predicate ranges over;
3. ``threshold`` -- the numeric bound (crowd/count require an explicit one;
   other events fall back to documented defaults and are not asked about).

Ambiguous / conflicting / unsupported utterances produce a single targeted
rephrase question (one turn).  Unsupported requests cannot be completed
within the grammar; the question says what to change instead.

Public API:
- ``minimal_questions(utterance, rule=None) -> List[Dict]``
  (each item: ``{"slot": ..., "question": ...}``)
- ``NAIVE_BASELINE_FIELDS``: the 9 canonical fields asked one-per-turn by
  the naive per-field baseline used in ``eval_alerts.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.app.services.alert_materializer import materialize_alert_rule
from backend.app.services.alert_validator import KNOWN_EVENTS, KNOWN_ZONES

NAIVE_BASELINE_FIELDS = [
    "event",
    "objects",
    "area",
    "threshold",
    "time_of_day",
    "severity",
    "cooldown_sec",
    "side_effects",
    "cadence_hz",
]

_QUESTIONS = {
    "event": (
        "What should I detect? Choose one: crowd, loiter, fight, enter, "
        "zone occupancy, count, or time-of-day watch."
    ),
    "zone": (
        "Which canonical zone? Choose one: Zone A, Zone B, Zone C, Gate, "
        "Entrance, or Parking."
    ),
    "threshold": (
        "What numeric threshold? For example 'more than five people' or "
        "'at least two minutes'."
    ),
}

_REPHRASE = {
    "ambiguous": (
        "Your request is ambiguous ({detail}). Please restate it with one "
        "explicit canonical zone and one numeric threshold or time."
    ),
    "conflicting": (
        "Your conditions conflict ({detail}). Please restate them with one "
        "consistent zone, object set, numeric threshold, and time window."
    ),
    "unsupported": (
        "I can only create rule-based detections for crowd, loiter, fight, "
        "enter, zone occupancy, count, and time-of-day events on fixed zones "
        "({detail} is outside the grammar). Please restate your request "
        "within those capabilities."
    ),
}


def _needs_threshold(event: Optional[str], threshold: Optional[Dict[str, Any]]) -> bool:
    if threshold is not None:
        return False
    # Events with documented default thresholds never trigger a question;
    # crowd/count require an explicit numeric bound.
    return event in {"crowd", "count"}


def minimal_questions(
    utterance: str, rule: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """Ordered minimal questions that complete the rule.  Empty when complete."""
    materialized = rule if rule is not None else materialize_alert_rule(utterance or "")
    kind = materialized.get("clarification_kind", "clean")
    if kind == "clean":
        return []
    if kind in _REPHRASE:
        detail = materialized.get("clarification") or kind
        return [{"slot": kind, "question": _REPHRASE[kind].format(detail=detail)}]
    if kind == "missing":
        questions: List[Dict[str, str]] = []
        event = materialized.get("event")
        area = materialized.get("area") or {}
        zone = area.get("zone_id")
        threshold = materialized.get("threshold")
        if event not in KNOWN_EVENTS:
            questions.append({"slot": "event", "question": _QUESTIONS["event"]})
        if not zone or zone not in KNOWN_ZONES:
            questions.append({"slot": "zone", "question": _QUESTIONS["zone"]})
        if _needs_threshold(event if event in KNOWN_EVENTS else None, threshold):
            questions.append({"slot": "threshold", "question": _QUESTIONS["threshold"]})
        if not questions:
            # Defensive: materializer flagged missing but every required slot
            # parses -- ask one generic question rather than stalling.
            questions.append(
                {"slot": "event", "question": _QUESTIONS["event"]}
            )
        return questions
    # Unknown kinds: one generic question (never stall the dialogue).
    return [{"slot": "other", "question": _QUESTIONS["event"]}]
