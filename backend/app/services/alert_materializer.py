"""Deterministic natural-language -> typed alert-rule materializer (C3).

The paper defines a conversational alert rule as the 9-tuple
``(event, objects, zone, threshold, time_of_day, severity, cooldown_sec,
side_effects, cadence_hz)``.  This module implements the grammar with
keyword/regex rules only -- no LLM and no network access -- so the benchmark
in ``backend/benchmarks/alert_utterances.jsonl`` is reproducible offline.

Design rules
------------
* Clean utterances materialize a complete rule using documented defaults.
* Missing required details (zone; crowd/count threshold) and unsupported,
  ambiguous, or conflicting requests set ``needs_clarification`` and are not
  inserted into MongoDB by the live chat path.
* ``count`` is a derived compatibility field for ``AlertRuleSpec``; the
  canonical scored fields are ``CANONICAL_FIELDS``.

Public API::

    materialize_alert_rule(utterance, parsed=None) -> dict
    classify_utterance(utterance) -> clean|ambiguous|conflicting|missing|unsupported
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.app.services.alert_validator import (
    KNOWN_EVENTS,
    KNOWN_ZONES,
    UNSUPPORTED_HINTS,
    validate_rule,
)

CANONICAL_FIELDS = [
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

DEFAULT_OBJECT = "person"
DEFAULT_SEVERITY = "info"
DEFAULT_COOLDOWN_SEC = 60
DEFAULT_SIDE_EFFECTS = ["snapshot", "push_ws"]
DEFAULT_CADENCE_HZ = 1.0
LOITER_DEFAULT_SEC = 60
FIGHT_DEFAULT_PERSONS = 2

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
_WORD_RE = "|".join(
    sorted(_WORD_NUMBERS, key=len, reverse=True)
)
_NUM_RE = rf"(?:\d{{1,4}}(?:\.\d+)?|{_WORD_RE})"

_TIME_UNIT_SEC = {
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
    "s": 1,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "m": 60,
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
    "h": 3600,
}
_TIME_UNIT_RE = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|s|m|h)"

_OBJECT_PATTERNS: List[Tuple[str, str]] = [
    ("backpack", r"\bbackpacks?\b"),
    ("bag", r"\bbags?\b"),
    ("truck", r"\btrucks?\b"),
    ("bus", r"\bbuses?\b|\bbus\b"),
    ("van", r"\bvans?\b"),
    ("car", r"\bcars?\b"),
    ("vehicle", r"\bvehicles?\b"),
    ("person", r"\b(?:persons?|people|someone|somebody|anyone|individuals?|intruders?|pedestrians?)\b"),
]

# Zone aliases are masked before the object scan so that a zone phrase such
# as "car park" does not contribute a spurious object ("car").
_ZONE_SPAN_RE = re.compile(
    r"\b(?:zone\s*[- ]?[abc]|gate|entrance|"
    r"parking\s+lot|car\s+park|parking\s+area|parking)\b",
    re.IGNORECASE,
)

_VEHICLE_OBJECTS = {"car", "truck", "bus", "van", "vehicle", "motorcycle"}

_ZONE_PATTERNS: List[Tuple[str, str]] = [
    ("Zone A", r"\bzone\s*[- ]?a\b"),
    ("Zone B", r"\bzone\s*[- ]?b\b"),
    ("Zone C", r"\bzone\s*[- ]?c\b"),
    ("Entrance", r"\b(?:front|main)?\s*entrance\b"),
    ("Gate", r"\b(?:main|front)?\s*gate\b"),
    ("Parking", r"\b(?:parking\s+lot|car\s+park|parking\s+area|parking)\b"),
]

_FUZZY_ZONE_RE = re.compile(
    r"\b(?:back\s+entrance|rear\s+entrance|back\s+gate|rear\s+gate|"
    r"over\s+there|nearby|around\s+the\s+corner|somewhere|the\s+area|"
    r"the\s+building|lobby|corridor)\b",
    re.IGNORECASE,
)

_AMBIGUOUS_QUANTITY_RE = re.compile(
    r"\b(?:a\s+few|several|some|many|a\s+bunch(?:\s+of)?|a\s+handful(?:\s+of)?|"
    r"a\s+lot\s+of|lots\s+of|too\s+many|a\s+couple\s+of)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_GROUP_RE = re.compile(
    rf"\ba\s+group\b(?!\s+of\s+(?:\d+|(?:{_WORD_RE})\b))",
    re.IGNORECASE,
)
_AMBIGUOUS_OCCUPANCY_RE = re.compile(
    r"\b(?:gets?\s+busy|is\s+busy|gets?\s+crowded|is\s+crowded|crowded)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_TIME_RE = re.compile(
    r"\b(?:sometime|soon|later|around\s+\d|about\s+\d)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_DURATION_RE = re.compile(
    r"\b(?:for|over)\s+(?:a\s+while|a\s+long\s+time|some\s+time)\b",
    re.IGNORECASE,
)

_COMPARATOR_RE = (
    r"(?:no\s+more\s+than|no\s+fewer\s+than|more\s+than|greater\s+than|"
    r"less\s+than|fewer\s+than|at\s+least|at\s+most|minimum\s+of|"
    r"equal\s+to|in\s+excess\s+of|exactly|over|above|under|below|up\s+to)"
)

_QUANTIFIER_RE = re.compile(
    rf"\b(?P<cmp>{_COMPARATOR_RE})\s+(?P<num>{_NUM_RE})"
    rf"(?:\s*(?P<unit>{_TIME_UNIT_RE}|people|persons?|cars?|vehicles?|"
    rf"trucks?|vans?|buses?|bus|backpacks?|bags?|objects?))?\b",
    re.IGNORECASE,
)
_OR_MORE_RE = re.compile(
    rf"\b(?P<num>{_NUM_RE})\s*(?:or\s+more|or\s+greater|and\s+above)\b"
    rf"(?:\s*(?P<unit>{_TIME_UNIT_RE}|people|persons?|cars?|vehicles?|"
    rf"trucks?|vans?|buses?|bus|backpacks?|bags?|objects?))?",
    re.IGNORECASE,
)
_OR_LESS_RE = re.compile(
    rf"\b(?P<num>{_NUM_RE})\s*(?:or\s+fewer|or\s+less|and\s+below)\b"
    rf"(?:\s*(?P<unit>{_TIME_UNIT_RE}|people|persons?|cars?|vehicles?|"
    rf"trucks?|vans?|buses?|bus|backpacks?|bags?|objects?))?",
    re.IGNORECASE,
)
_BETWEEN_COUNT_RE = re.compile(
    rf"\bbetween\s+(?P<lo>{_NUM_RE})\s+and\s+(?P<hi>{_NUM_RE})\s+"
    rf"(?P<unit>people|persons?|cars?|vehicles?|trucks?|vans?|buses?|bus|"
    rf"backpacks?|bags?|objects?)\b",
    re.IGNORECASE,
)

# Duration bounds with their own unit, so that minutes and seconds are not
# compared as raw numbers.
_SECONDS_BOUND_RE = re.compile(
    rf"\b(?P<cmp>{_COMPARATOR_RE})\s+(?P<num>{_NUM_RE})\s*(?P<unit>{_TIME_UNIT_RE})\b",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    rf"\bfor\s+(?:(?P<cmp>{_COMPARATOR_RE})\s+)?"
    rf"(?P<num>{_NUM_RE}|a|an)\s*(?P<unit>{_TIME_UNIT_RE})\b",
    re.IGNORECASE,
)

_UNSUPPORTED_PATTERNS: List[Tuple[str, str]] = [
    (r"\bfollow\w*\b.*\b(?:home|outside|off[- ]premises)\b", "following a person off-premises"),
    (r"\btrack\w*\b.*\bhome\b", "tracking a person off-premises"),
    (r"\bidentify\b.*\b(?:person|face|individual)\b", "identifying a person"),
    (r"\brecogni[sz]e\b.*\bface\b|\bface\s+recognition\b", "face recognition"),
    (r"\b(?:delete|erase)\b.*\b(?:footage|recording|video|clip)\b", "deleting footage"),
    (r"\b(?:disable|turn\s+off)\b.*\bcamera\b", "disabling a camera"),
    (r"\bunlock\b.*\bdoor\b|\bopen\b.*\bdoor\b", "actuating a door lock"),
    (r"\bweapons?\b|\bguns?\b|\bknifes?\b|\bknives\b", "weapon detection"),
    (r"\blicen[cs]e\s+plate\b|\bnumber\s+plate\b|\banpr\b", "license-plate recognition"),
    (r"\bgender\b|\bsex\b|\brace\b|\bethnicity\b", "protected-attribute inference"),
    (r"\bemotion\b|\bangry\b|\bsmil(?:e|es|ing)\b", "emotion inference"),
    (r"\bcriminal\s+record\b", "criminal-record lookup"),
    (r"\bhear\b|\blisten\b|\bshout(?:ing)?\b|\baudio\b", "audio analytics"),
    (r"\bdangerous\b|\bsuspicious\b", "subjective dangerousness inference"),
]


# ---------------------------------------------------------------------------
# Normalisation and primitive parsers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    text = (text or "").replace("\u2019", "'").replace("\u2013", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def _num(token: str) -> Optional[float]:
    token = (token or "").strip().lower()
    if not token:
        return None
    try:
        return float(token) if "." in token else int(token)
    except ValueError:
        return _WORD_NUMBERS.get(token)


def _fmt_time(hour: int, minute: int = 0) -> Optional[str]:
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    return None


def _apply_meridiem(hour: int, meridiem: Optional[str], inherited: Optional[str] = None) -> int:
    mer = (meridiem or inherited or "").lower()
    h = hour
    if mer == "pm" and h < 12:
        h += 12
    elif mer == "am" and h == 12:
        h = 0
    return h


def _parse_clock(hour_s: str, minute_s: Optional[str], mer: Optional[str]) -> Optional[Tuple[int, int]]:
    try:
        h = int(hour_s)
        m = int(minute_s) if minute_s else 0
    except ValueError:
        return None
    if h > 23 or m > 59:
        return None
    return h, m


def _range(start: Optional[Tuple[int, int]], end: Optional[Tuple[int, int]]) -> Optional[Dict[str, str]]:
    if not start or not end:
        return None
    s = _fmt_time(*start)
    e = _fmt_time(*end)
    if not s or not e:
        return None
    return {"start": s, "end": e}


def _time_window(text: str) -> Optional[Dict[str, str]]:
    """Parse one time-of-day window; returns ``None`` when none is stated."""
    # between H[:MM] [am/pm] and H[:MM] [am/pm]
    m = re.search(
        r"\bbetween\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+and\s+"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        text,
    )
    if m:
        h1, m1, mer1, h2, m2, mer2 = m.groups()
        end_pair = _parse_clock(h2, m2, mer2)
        if end_pair:
            end_pair = (_apply_meridiem(end_pair[0], mer2), end_pair[1])
        start_pair = _parse_clock(h1, m1, mer1)
        if start_pair:
            start_pair = (_apply_meridiem(start_pair[0], mer1, mer2), start_pair[1])
        result = _range(start_pair, end_pair)
        if result:
            return result

    # from H[:MM] [am/pm] to H[:MM] [am/pm]
    m = re.search(
        r"\bfrom\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+to\s+"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        text,
    )
    if m:
        h1, m1, mer1, h2, m2, mer2 = m.groups()
        end_pair = _parse_clock(h2, m2, mer2)
        if end_pair:
            end_pair = (_apply_meridiem(end_pair[0], mer2), end_pair[1])
        start_pair = _parse_clock(h1, m1, mer1)
        if start_pair:
            start_pair = (_apply_meridiem(start_pair[0], mer1, mer2), start_pair[1])
        result = _range(start_pair, end_pair)
        if result:
            return result

    # H[:MM] am/pm to/until/till H[:MM] am/pm
    m = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*(?:to|until|till)\s+"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        text,
    )
    if m:
        h1, m1, mer1, h2, m2, mer2 = m.groups()
        start_pair = _parse_clock(h1, m1, mer1)
        if start_pair:
            start_pair = (_apply_meridiem(start_pair[0], mer1), start_pair[1])
        end_pair = _parse_clock(h2, m2, mer2)
        if end_pair:
            end_pair = (_apply_meridiem(end_pair[0], mer2), end_pair[1])
        result = _range(start_pair, end_pair)
        if result:
            return result

    # after H[:MM] am/pm -> start .. 06:00 (night policy)
    m = re.search(r"\bafter\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if m:
        h, minute, mer = m.groups()
        pair = _parse_clock(h, minute, mer)
        if pair:
            start = (_apply_meridiem(pair[0], mer), pair[1])
            result = _range(start, (6, 0))
            if result:
                return result

    # before H[:MM] am/pm -> 00:00 .. H
    m = re.search(r"\bbefore\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if m:
        h, minute, mer = m.groups()
        pair = _parse_clock(h, minute, mer)
        if pair:
            end = (_apply_meridiem(pair[0], mer), pair[1])
            result = _range((0, 0), end)
            if result:
                return result

    # night / day / evening / morning / afternoon, in priority order.
    if re.search(r"\b(?:at\s+night|overnight|every\s+night|nighttime|night)\b", text):
        return {"start": "20:00", "end": "06:00"}
    if re.search(r"\bduring\s+the\s+day\b|\bdaytime\b|\bin\s+the\s+day\b", text):
        return {"start": "06:00", "end": "20:00"}
    if re.search(r"\bin\s+the\s+evening\b|\bevening\b", text):
        return {"start": "18:00", "end": "22:00"}
    if re.search(r"\bin\s+the\s+morning\b|\bmorning\b", text):
        return {"start": "06:00", "end": "12:00"}
    if re.search(r"\bin\s+the\s+afternoon\b|\bafternoon\b", text):
        return {"start": "12:00", "end": "18:00"}

    # at H[:MM] am/pm (point time -> a two-hour window, clipped through 06:00)
    m = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if m:
        h, minute, mer = m.groups()
        pair = _parse_clock(h, minute, mer)
        if pair:
            start_h = _apply_meridiem(pair[0], mer)
            end_h = (start_h + 2) % 24
            if start_h >= 18 or start_h < 6:
                end_h = 6
            result = _range((start_h, pair[1]), (end_h, 0))
            if result:
                return result

    # HH:MM-HH:MM (24-hour clock range)
    m = re.search(
        r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\s*-\s*(\d{1,2}):(\d{2})\s*(am|pm)?\b",
        text,
    )
    if m:
        h1, m1, mer1, h2, m2, mer2 = m.groups()
        start_pair = _parse_clock(h1, m1, mer1)
        if start_pair:
            start_pair = (_apply_meridiem(start_pair[0], mer1), start_pair[1])
        end_pair = _parse_clock(h2, m2, mer2)
        if end_pair:
            end_pair = (_apply_meridiem(end_pair[0], mer2), end_pair[1])
        result = _range(start_pair, end_pair)
        if result:
            return result

    return None


_TIME_WINDOW_COUNT_RE = re.compile(
    r"\bbetween\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+and\s+"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b"
    r"|\bfrom\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+to\s+"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b"
    r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*(?:to|until|till)\s+"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"
    r"|\bafter\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"
    r"|\bbefore\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    re.IGNORECASE,
)


def _time_window_count(text: str) -> int:
    """Count distinct explicit time windows (used for conflict detection).

    One combined regex is used so "from 8 PM to 10 PM" is not counted twice
    (once as a ``from ... to`` window and once as a bare ``8 PM to 10 PM``
    window).
    """
    count = len(list(_TIME_WINDOW_COUNT_RE.finditer(text)))
    if count == 0 and re.search(r"\b(?:at\s+night|overnight|night)\b", text):
        count = 1
    return count


def _objects(text: str) -> List[str]:
    hits: List[Tuple[int, str]] = []
    scan = _ZONE_SPAN_RE.sub(" ", text)
    for name, pattern in _OBJECT_PATTERNS:
        for match in re.finditer(pattern, scan, re.IGNORECASE):
            hits.append((match.start(), name))
    hits.sort(key=lambda item: item[0])
    out: List[str] = []
    for _, name in hits:
        if name not in out:
            out.append(name)
    return out


def _zones(text: str) -> List[str]:
    hits: List[Tuple[int, str]] = []
    for name, pattern in _ZONE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            hits.append((match.start(), name))
    hits.sort(key=lambda item: item[0])
    out: List[str] = []
    for _, name in hits:
        if name not in out:
            out.append(name)
    return out


def _event(text: str) -> Optional[str]:
    if re.search(r"\b(?:fight|fights|fighting|brawl|scuffle|altercation|assault)\b", text):
        return "fight"
    if re.search(
        r"\b(?:loiter|loiters|loitering|linger|lingers|lingering)\b|"
        r"\bhanging\s+around\b|\bstanding\s+around\b",
        text,
    ):
        return "loiter"
    if re.search(r"\b(?:count|counts|counting|tally|number\s+of|how\s+many)\b", text):
        return "count"
    if re.search(
        r"\b(?:gather|gathers|gathering|crowd|crowds|crowded|congregate|"
        r"congregates|congregating|assemble|assembles|assembling|cluster|"
        r"clusters|clustering)\b|\bgroup\s+up\b",
        text,
    ):
        return "crowd"
    if re.search(
        r"\b(?:enter|enters|entering|entered)\b|\bcross(?:es|ing)?\s+into\b|"
        r"\bcomes?\s+into\b|\bwalks?\s+into\b|\bsteps?\s+into\b",
        text,
    ):
        return "enter"
    if re.search(
        r"\b(?:occupied|occupancy|inside|present\s+in|is\s+in|are\s+in|"
        r"remains?\s+in|stays?\s+in)\b",
        text,
    ):
        return "zone"
    if _time_window(text) is not None:
        return "time"
    return None


def _unit_for_objects(objects: List[str]) -> str:
    if any(name in _VEHICLE_OBJECTS for name in objects):
        return "vehicles"
    if objects and all(name in {"bag", "backpack"} for name in objects):
        return "objects"
    return "persons"


def _unit_from_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    token = token.lower()
    if token in _TIME_UNIT_SEC:
        return "seconds"
    if token.rstrip("s") in {"person", "people"} or token in {"people", "persons"}:
        return "persons"
    if token.rstrip("s") in _VEHICLE_OBJECTS or token in _VEHICLE_OBJECTS:
        return "vehicles"
    if token.rstrip("s") in {"object", "bag", "backpack"}:
        return "objects"
    return None


def _op_from_comparator(cmp_token: str) -> str:
    c = re.sub(r"\s+", " ", cmp_token.lower())
    if c in {"more than", "greater than", "over", "above", "in excess of"}:
        return ">"
    if c in {"at least", "minimum of", "no fewer than"}:
        return ">="
    if c in {"fewer than", "less than", "under", "below"}:
        return "<"
    if c in {"at most", "no more than", "up to"}:
        return "<="
    if c in {"exactly", "equal to"}:
        return "=="
    return ">="


def _combine_bounds(bounds: List[Dict[str, Any]], unit: str) -> Optional[Dict[str, Any]]:
    """Combine conjunctive bounds of one unit family into a single threshold.

    A request may state several bounds of the same unit ("more than five but
    fewer than ten people").  Keeping only one of them silently widens the
    rule: the created alert then fires in configurations the user excluded.
    This helper returns the conjunction of ``bounds`` as one threshold
    (``between`` for a two-sided range, the strictest bound otherwise) and
    ``None`` when the bounds cannot be combined.
    """
    lowers = [b for b in bounds if b["op"] in {">", ">="}]
    uppers = [b for b in bounds if b["op"] in {"<", "<="}]
    exact = [b for b in bounds if b["op"] == "=="]
    if exact:
        return {"op": "==", "value": exact[0]["value"], "unit": unit}
    if lowers and uppers:
        lo_b = max(lowers, key=lambda b: (b["value"], b["op"] == ">"))
        hi_b = min(uppers, key=lambda b: (b["value"], b["op"] == "<"))
        lo, hi = lo_b["value"], hi_b["value"]
        # Counts and durations are integers: tighten strict bounds into
        # inclusive integer bounds so the range is represented exactly.
        if lo_b["op"] == ">" and isinstance(lo, int) and not isinstance(lo, bool):
            lo += 1
        if hi_b["op"] == "<" and isinstance(hi, int) and not isinstance(hi, bool):
            hi -= 1
        if lo > hi:
            return None
        return {"op": "between", "value": [lo, hi], "unit": unit}
    if lowers:
        b = max(lowers, key=lambda b: (b["value"], b["op"] == ">"))
        return {"op": b["op"], "value": b["value"], "unit": unit}
    if uppers:
        b = min(uppers, key=lambda b: (b["value"], b["op"] == "<"))
        return {"op": b["op"], "value": b["value"], "unit": unit}
    return None


def _quantifiers(text: str) -> List[Dict[str, Any]]:
    """All comparator+number mentions, for conflict detection and extraction."""
    out: List[Dict[str, Any]] = []
    for match in _QUANTIFIER_RE.finditer(text):
        value = _num(match.group("num"))
        if value is None:
            continue
        out.append(
            {
                "op": _op_from_comparator(match.group("cmp")),
                "value": value,
                "unit": _unit_from_token(match.group("unit")),
                "start": match.start(),
            }
        )
    for regex, op in ((_OR_MORE_RE, ">="), (_OR_LESS_RE, "<=")):
        for match in regex.finditer(text):
            value = _num(match.group("num"))
            if value is None:
                continue
            out.append(
                {
                    "op": op,
                    "value": value,
                    "unit": _unit_from_token(match.group("unit")),
                    "start": match.start(),
                }
            )
    out.sort(key=lambda item: item["start"])
    return out


def _extract_count(text: str, objects: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a numeric count threshold, if one is stated."""
    default_unit = _unit_for_objects(objects)

    # "between six and ten people" / "between two and five" after a count cue.
    match = _BETWEEN_COUNT_RE.search(text)
    if match:
        low = _num(match.group("lo"))
        high = _num(match.group("hi"))
        if low is not None and high is not None:
            unit = _unit_from_token(match.group("unit")) or default_unit
            return {"op": "between", "value": [low, high], "unit": unit}
    if re.search(r"\b(?:count|number\s+of|vehicle\s+count)\b", text):
        match = re.search(
            rf"\bbetween\s+(?P<lo>{_NUM_RE})\s+and\s+(?P<hi>{_NUM_RE})\b",
            text,
        )
        if match and not re.search(
            rf"\bbetween\s+{match.group('lo')}\s+and\s+{match.group('hi')}\s*(?:am|pm)\b",
            text,
        ):
            low = _num(match.group("lo"))
            high = _num(match.group("hi"))
            if low is not None and high is not None:
                return {"op": "between", "value": [low, high], "unit": default_unit}

    # "exceeds ten" / "is above ten" without a comparator keyword.
    match = re.search(
        rf"\b(?:exceeds?|is\s+above|is\s+over)\s+(?P<num>{_NUM_RE})\b",
        text,
    )
    if match:
        value = _num(match.group("num"))
        if value is not None:
            return {"op": ">", "value": value, "unit": default_unit}

    quantifiers = _quantifiers(text)
    if not quantifiers:
        return None

    # Prefer a quantifier whose unit matches the parsed object family.
    chosen = quantifiers[0]
    for q in quantifiers:
        if q["unit"] == default_unit:
            chosen = q
            break

    unit = chosen["unit"] or default_unit
    value = chosen["value"]
    if unit == "seconds":
        # A duration used as a count is not meaningful for crowd/count events;
        # keep it as seconds and let the caller reject/clarify.
        return {"op": chosen["op"], "value": value, "unit": "seconds"}

    # Keep every bound stated for the same unit family: a two-sided request
    # ("more than five but fewer than ten people") must not lose one side.
    family = [q for q in quantifiers if (q["unit"] or unit) == unit]
    if len(family) > 1:
        combined = _combine_bounds(family, unit)
        if combined is not None:
            return combined
    return {"op": chosen["op"], "value": value, "unit": unit}


def _extract_duration(text: str) -> Optional[Dict[str, Any]]:
    match = _DURATION_RE.search(text)
    if not match:
        return None
    token = match.group("num")
    value = 1.0 if token in {"a", "an"} else _num(token)
    if value is None:
        return None
    unit = _TIME_UNIT_SEC.get(match.group("unit").lower())
    if not unit:
        return None
    seconds = value * unit
    if float(seconds).is_integer():
        seconds = int(seconds)
    cmp_token = match.group("cmp")
    op = _op_from_comparator(cmp_token) if cmp_token else ">="
    bounds = [{"op": op, "value": seconds}]
    for other in _SECONDS_BOUND_RE.finditer(text):
        val = _num(other.group("num"))
        if val is None:
            continue
        factor = _TIME_UNIT_SEC.get(other.group("unit").lower())
        if not factor:
            continue
        secs = val * factor
        if float(secs).is_integer():
            secs = int(secs)
        other_op = _op_from_comparator(other.group("cmp"))
        if (other_op, secs) == (op, seconds):
            continue
        bounds.append({"op": other_op, "value": secs})
    if len(bounds) > 1:
        combined = _combine_bounds(bounds, "seconds")
        if combined is not None:
            return combined
        lowers = [b for b in bounds if b["op"] in {">", ">="}]
        if lowers:
            strictest = max(lowers, key=lambda b: (b["value"], b["op"] == ">"))
            return {"op": strictest["op"], "value": strictest["value"], "unit": "seconds"}
    return {"op": op, "value": seconds, "unit": "seconds"}


def _threshold_for_event(
    text: str, event: Optional[str], objects: List[str]
) -> Optional[Dict[str, Any]]:
    default_unit = _unit_for_objects(objects)

    if event == "loiter":
        duration = _extract_duration(text)
        if duration is not None:
            return duration
        return {"op": ">=", "value": LOITER_DEFAULT_SEC, "unit": "seconds"}

    if event == "fight":
        count = _extract_count(text, objects)
        if count is not None and count["unit"] != "seconds":
            return count
        return {"op": ">=", "value": FIGHT_DEFAULT_PERSONS, "unit": "persons"}

    count = _extract_count(text, objects)
    if count is not None:
        return count

    if event in {"zone", "enter", "time"}:
        return {"op": ">=", "value": 1, "unit": default_unit}

    # crowd/count require an explicit numeric threshold.
    return None


def _count_from_threshold(
    threshold: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not threshold:
        return None
    if threshold.get("unit") == "seconds":
        return {">=": 1}
    op = threshold.get("op")
    value = threshold.get("value")
    if op == "between" and isinstance(value, (list, tuple)) and len(value) == 2:
        # Use the operators understood by alert_engine.matches_count_condition.
        return {">=": value[0], "<=": value[1]}
    if op in {">", ">=", "<", "<=", "=="}:
        return {op: value}
    return None


def _severity(text: str) -> str:
    if re.search(r"\b(?:critical|severe|urgent|emergency)\b", text):
        return "critical"
    if re.search(r"\b(?:high\s+priority|high\s+severity|important)\b", text):
        return "high"
    if re.search(
        r"\b(?:warning\s+severity|warning\s+priority|moderate|medium\s+priority|"
        r"medium\s+severity)\b",
        text,
    ):
        return "warning"
    if re.search(r"\b(?:low\s+priority|low\s+severity)\b", text):
        return "info"
    return DEFAULT_SEVERITY


def _cooldown(text: str) -> int:
    match = re.search(
        rf"\bcooldown\s+(?:of\s+)?(?P<num>\d+(?:\.\d+)?)\s*"
        rf"(?P<unit>{_TIME_UNIT_RE})?\b",
        text,
    )
    if not match:
        match = re.search(
            rf"\b(?:no\s+more\s+than\s+one\s+alert|at\s+most\s+one\s+alert)"
            rf"\s+every\s+(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>{_TIME_UNIT_RE})\b",
            text,
        )
    if not match:
        return DEFAULT_COOLDOWN_SEC
    value = float(match.group("num"))
    unit = match.group("unit")
    factor = _TIME_UNIT_SEC.get(unit.lower(), 1) if unit else 1
    seconds = value * factor
    return int(seconds) if seconds.is_integer() else seconds


def _cadence(text: str) -> float:
    match = re.search(rf"(?<!alert )\bevery\s+(\d+(?:\.\d+)?)\s*({_TIME_UNIT_RE})\b", text)
    if match:
        seconds = float(match.group(1)) * _TIME_UNIT_SEC[match.group(2).lower()]
        if seconds > 0:
            return 1.0 / seconds
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*hz\b", text)
    if match:
        value = float(match.group(1))
        if value > 0:
            return value
    if re.search(r"\btwice\s+(?:per|a)\s+second\b", text):
        return 2.0
    if re.search(r"\bonce\s+(?:per|a)\s+second\b|\bevery\s+second\b", text):
        return 1.0
    if re.search(r"\bevery\s+minute\b", text):
        return 1.0 / 60.0
    return DEFAULT_CADENCE_HZ


def _side_effects(text: str) -> List[str]:
    if re.search(r"\b(?:notification|notify|push)\s+only\b|\bjust\s+notify\b|\bpush\s+only\b", text):
        return ["push_ws"]
    if re.search(r"\bsnapshot\s+only\b|\bonly\s+(?:a\s+)?snapshot\b|\bonly\s+take\s+a\s+snapshot\b", text):
        return ["snapshot"]
    if re.search(r"\bclip\s+only\b|\bonly\s+(?:save|store)\s+(?:a\s+)?clip\b", text):
        return ["store_clip"]
    effects = list(DEFAULT_SIDE_EFFECTS)
    if re.search(r"\b(?:save|store)\b.*\bclip\b", text):
        effects.insert(0, "store_clip")
    return effects


# ---------------------------------------------------------------------------
# Clarify / abstain taxonomy
# ---------------------------------------------------------------------------

def _unsupported(text: str) -> Optional[str]:
    for hint in UNSUPPORTED_HINTS:
        if hint in text:
            return hint
    for pattern, label in _UNSUPPORTED_PATTERNS:
        if re.search(pattern, text):
            return label
    return None


def _conflict(text: str, objects: List[str], zones: List[str]) -> Optional[str]:
    # Contradictory quantifier bounds (same unit family).
    groups: Dict[str, Dict[str, List[Any]]] = {}
    for q in _quantifiers(text):
        # ``None`` means the unit was elided ("more than five but ..."); group
        # it with object counts, while durations stay in their own family.
        unit = "seconds" if q["unit"] == "seconds" else "count"
        group = groups.setdefault(unit, {"lower": [], "upper": [], "eq": []})
        if q["op"] in {">", ">="}:
            group["lower"].append((q["op"], q["value"]))
        elif q["op"] in {"<", "<="}:
            group["upper"].append((q["op"], q["value"]))
        elif q["op"] == "==":
            group["eq"].append(q["value"])
    for group in groups.values():
        eq_values = set(group["eq"])
        if len(eq_values) > 1:
            return "multiple exact thresholds"
        lower = max((v for _, v in group["lower"]), default=None)
        upper = min((v for _, v in group["upper"]), default=None)
        if lower is not None and upper is not None:
            if lower > upper or (lower == upper and (
                any(op == ">" for op, v in group["lower"] if v == lower)
                or any(op == "<" for op, v in group["upper"] if v == upper)
            )):
                return "lower threshold exceeds upper threshold"
        if eq_values and (lower is not None and lower > next(iter(eq_values))):
            return "exact threshold below lower bound"
        if eq_values and (upper is not None and upper < next(iter(eq_values))):
            return "exact threshold above upper bound"

    # An inverted "between X and Y <count-unit>" range contradicts itself.
    match = _BETWEEN_COUNT_RE.search(text)
    if match:
        lo = _num(match.group("lo"))
        hi = _num(match.group("hi"))
        if lo is not None and hi is not None and lo > hi:
            return "between bounds are inverted (lower exceeds upper)"
    if re.search(r"\b(?:count|number\s+of|vehicle\s+count)\b", text):
        match = re.search(
            rf"\bbetween\s+(?P<lo>{_NUM_RE})\s+and\s+(?P<hi>{_NUM_RE})\b", text
        )
        if match and not re.search(
            rf"\bbetween\s+{match.group('lo')}\s+and\s+{match.group('hi')}\s*(?:am|pm)\b",
            text,
        ):
            lo = _num(match.group("lo"))
            hi = _num(match.group("hi"))
            if lo is not None and hi is not None and lo > hi:
                return "between bounds are inverted (lower exceeds upper)"

    # Conflicting zones / object sets expressed as disjunctions.
    if re.search(r"\bor\b", text):
        if len(set(zones)) > 1:
            return f"ambiguous zone disjunction ({', '.join(zones)})"
        object_set = {name for name in objects}
        if len(object_set) > 1:
            return f"ambiguous object disjunction ({', '.join(sorted(object_set))})"

    # Direct contradictions.
    if re.search(r"\b(?:does\s+not|doesn't|not)\s+enter\b", text) and re.search(r"\benters?\b", text):
        return "enters and does not enter"
    if re.search(r"\binside\b", text) and re.search(r"\boutside\b", text):
        return "inside and outside the same zone"
    if re.search(r"\b(?:crowded|occupied|full)\b", text) and re.search(r"\b(?:empty|vacant)\b", text):
        return "occupied and empty"
    if re.search(r"\blarge\b", text) and re.search(r"\bsmall\b", text):
        return "large and small"

    # More than one explicit time window cannot be represented by one T.
    if _time_window_count(text) > 1:
        return "multiple time windows"
    return None


def _ambiguous(text: str) -> Optional[str]:
    if _AMBIGUOUS_QUANTITY_RE.search(text):
        return "implicit quantity (a few/several/some/many/...)"
    if _AMBIGUOUS_GROUP_RE.search(text):
        return "implicit group size"
    if _AMBIGUOUS_OCCUPANCY_RE.search(text):
        return "implicit occupancy threshold"
    if _AMBIGUOUS_TIME_RE.search(text):
        return "fuzzy time reference"
    if _FUZZY_ZONE_RE.search(text):
        return "fuzzy zone reference"
    if _AMBIGUOUS_DURATION_RE.search(text):
        return "fuzzy duration"
    return None


def _missing(
    event: Optional[str],
    zone: Optional[str],
    threshold: Optional[Dict[str, Any]],
) -> Optional[str]:
    if event not in KNOWN_EVENTS:
        return "event"
    if not zone or zone not in KNOWN_ZONES:
        return "zone"
    if event in {"crowd", "count"} and threshold is None:
        return "threshold"
    return None


def _clarification_reason(kind: str, detail: str) -> str:
    if kind == "unsupported":
        return (
            "Unsupported request: I can only create rule-based detections for "
            "crowd, loiter, fight, enter, zone occupancy, count, and "
            "time-of-day events on fixed zones. I cannot follow people "
            "off-premises, identify faces, delete footage, disable cameras, "
            "unlock doors, or infer protected attributes."
        )
    if kind == "conflicting":
        return (
            f"Conflicting conditions ({detail}). Please give one consistent "
            "zone, object set, numeric threshold, and time window."
        )
    if kind == "ambiguous":
        return (
            f"Ambiguous specification ({detail}). Please give an explicit "
            "canonical zone (Zone A/B/C, Gate, Entrance, or Parking) and a "
            "numeric threshold/time."
        )
    if detail == "zone":
        return (
            "Missing required detail: zone. Please say which canonical zone "
            "(Zone A/B/C, Gate, Entrance, or Parking)."
        )
    if detail == "threshold":
        return (
            "Missing required detail: numeric threshold. Please say how many "
            "people/objects (e.g., 'more than five people')."
        )
    if detail == "event":
        return (
            "Missing required detail: event. Please say what to detect "
            "(crowd, loiter, fight, enter, zone occupancy, count, or time)."
        )
    return f"Missing required detail: {detail}. Please be more specific."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def materialize_alert_rule(
    utterance: str, parsed: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Materialize an utterance into a complete or partial 9-field rule.

    ``parsed`` is accepted for live-path compatibility.  Grammar fields are
    parsed only from the utterance so that an LLM cannot silently invent a
    missing zone/threshold.  ``parsed`` is used only for non-grammar metadata
    (camera id and colour) by the chat router.

    The returned dict always contains the canonical keys plus ``count``,
    ``needs_clarification``, ``clarification``, and ``clarification_kind``.
    """
    text = _norm(utterance)
    objects = _objects(text)
    zones = _zones(text)
    zone = zones[0] if zones else None
    event = _event(text)
    time_of_day = _time_window(text) or {}

    threshold = _threshold_for_event(text, event, objects)
    if threshold is not None and threshold.get("unit") == "seconds" and event not in {"loiter"}:
        # A duration is not a count threshold for non-loiter events.
        threshold = None

    if not objects:
        objects = [DEFAULT_OBJECT]

    severity = _severity(text)
    cooldown_sec = _cooldown(text)
    side_effects = _side_effects(text)
    cadence_hz = _cadence(text)

    unsupported = _unsupported(text)
    conflict = _conflict(text, objects, zones) if not unsupported else None
    ambiguous = _ambiguous(text) if not unsupported and not conflict else None
    missing = (
        _missing(event, zone, threshold)
        if not unsupported and not conflict and not ambiguous
        else None
    )

    if unsupported:
        kind, detail = "unsupported", unsupported
    elif conflict:
        kind, detail = "conflicting", conflict
    elif ambiguous:
        kind, detail = "ambiguous", ambiguous
    elif missing:
        kind, detail = "missing", missing
    else:
        kind, detail = "clean", ""

    rule: Dict[str, Any] = {
        "event": event,
        "objects": [{"name": name} for name in objects],
        "area": {"zone_id": zone} if zone else {},
        "count": _count_from_threshold(threshold),
        "threshold": threshold,
        "time_of_day": time_of_day,
        "severity": severity,
        "cooldown_sec": cooldown_sec,
        "side_effects": side_effects,
        "cadence_hz": cadence_hz,
        "needs_clarification": kind != "clean",
        "clarification": _clarification_reason(kind, detail) if kind != "clean" else None,
        "clarification_kind": kind,
    }

    # Safety gate: a rule the validator rejects must never be reported as
    # "clean" (the live chat path refuses to insert it and the dialogue would
    # assume the rule is complete).  Such a request is contradictory for the
    # typed grammar, so it is clarified instead of materialized.
    if kind == "clean":
        ok, errors, _warnings = validate_rule(rule)
        if not ok:
            kind = "conflicting"
            detail = "the requested rule is not representable (" + "; ".join(errors) + ")"
            rule["needs_clarification"] = True
            rule["clarification"] = _clarification_reason(kind, detail)
            rule["clarification_kind"] = kind
    return rule


def classify_utterance(utterance: str) -> str:
    """Return ``clean`` or one of the four stress kinds for an utterance."""
    return str(materialize_alert_rule(utterance).get("clarification_kind", "clean"))
