"""
Shared utilities for the retrieval pipeline.
Consolidates timestamp parsing, score normalization, count constraint logic,
and object-matching helpers used across structured search, result merging, and
the unified retrieval orchestrator.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic score thresholds & penalty factors
# ---------------------------------------------------------------------------
SEM_RAW_THRESH_LOW = 0.22
SEM_RAW_THRESH_MED = 0.30
SEM_RAW_THRESH_HIGH = 0.40
SEM_NO_STRUCTURED_THRESH = 0.35

SEM_PENALTY_STRONG = 0.3
SEM_PENALTY_MED = 0.6
SEM_PENALTY_LIGHT = 0.8
SEM_PENALTY_NO_STRUCTURED = 0.7


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------
_TZ_SUFFIX_RE = re.compile(r"(Z|[+-]\d{2}:\d{2})$")


def parse_ts(ts: str) -> datetime:
    """
    Parse an ISO-ish timestamp string into a *naive* datetime.

    Strips trailing Z / timezone offsets so that stored detection timestamps
    (which are naive-local or naive-UTC) can be compared consistently.
    Falls back to ``datetime.now(timezone.utc)`` when parsing fails entirely.
    """
    try:
        safe = _TZ_SUFFIX_RE.sub("", ts)
        return datetime.fromisoformat(safe)
    except Exception:
        try:
            return datetime.strptime(ts.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------
def normalize_scores(scores: List[float], min_val: float = 0.0) -> List[float]:
    """
    Normalize *scores* to [0, 1], scaling down when the absolute maximum is
    weak so that final scores honestly reflect low confidence.
    """
    if not scores:
        return []
    if len(scores) == 1:
        s = scores[0]
        if s < 0.25:
            return [max(min_val, 0.5)]
        if s < 0.4:
            return [max(min_val, 0.8)]
        return [max(min_val, 1.0)]

    max_score = max(scores)
    if max_score == 0:
        return [min_val] * len(scores)

    scale = 1.0
    if max_score < 0.25:
        scale = 0.5
    elif max_score < 0.4:
        scale = 0.8

    return [max(min_val, (s / max_score) * scale) for s in scores]


# ---------------------------------------------------------------------------
# Count-constraint helpers
# ---------------------------------------------------------------------------
_COUNT_KEY_MAP = {
    "eq": "eq", "=": "eq", "==": "eq",
    "gte": "gte", ">=": "gte",
    "gt": "gt", ">": "gt",
    "lte": "lte", "<=": "lte",
    "lt": "lt", "<": "lt",
}


def normalize_count_constraint(raw: Any) -> Optional[Dict[str, int]]:
    """Normalize a parsed count constraint dict into ``{op: value}`` form."""
    if not isinstance(raw, dict):
        return None
    out: Dict[str, int] = {}
    for k, v in raw.items():
        norm_k = _COUNT_KEY_MAP.get(str(k).strip().lower())
        if not norm_k:
            continue
        try:
            out[norm_k] = int(v)
        except Exception:
            continue
    return out or None


def count_matches_constraint(count: int, constraint: Dict[str, int]) -> bool:
    """Return *True* if *count* satisfies every operator in *constraint*."""
    if "eq" in constraint and count != constraint["eq"]:
        return False
    if "gte" in constraint and count < constraint["gte"]:
        return False
    if "gt" in constraint and count <= constraint["gt"]:
        return False
    if "lte" in constraint and count > constraint["lte"]:
        return False
    if "lt" in constraint and count >= constraint["lt"]:
        return False
    return True


# ---------------------------------------------------------------------------
# Object matching
# ---------------------------------------------------------------------------
def object_matches(obj: Dict[str, Any], parsed_filter: Dict[str, Any]) -> bool:
    """Return *True* if a single detection object matches the filter criteria."""
    if not isinstance(obj, dict):
        return False
    name = parsed_filter.get("objects.object_name")
    color = parsed_filter.get("objects.color")
    if name and str(obj.get("object_name", "")).lower() != str(name).lower():
        return False
    if color and str(obj.get("color", "")).lower() != str(color).lower():
        return False
    return True


def count_matching_objects_in_doc(
    doc: Dict[str, Any],
    parsed_filter: Dict[str, Any],
) -> int:
    """
    Count how many objects in *doc* satisfy the filter, using pre-computed
    fast-path fields (``person_count``, ``object_counts``, ``zone_counts``)
    where available.
    """
    obj_name = parsed_filter.get("objects.object_name")
    color = parsed_filter.get("objects.color")
    zone_ids = parsed_filter.get("__zone_ids") or []

    # Zone-aware fast path
    if zone_ids and obj_name and str(obj_name).lower() == "person" and not color:
        zc = doc.get("zone_counts") or {}
        if isinstance(zc, dict):
            counts = [int(zc.get(zid, 0)) for zid in zone_ids]
            constraint = normalize_count_constraint(parsed_filter.get("count_constraint"))
            if constraint:
                for c in counts:
                    if count_matches_constraint(c, constraint):
                        return c
                return 0
            return max(counts) if counts else 0

    # person_count fast path
    if obj_name and str(obj_name).lower() == "person" and not color:
        pc = doc.get("person_count")
        if pc is not None:
            return int(pc)

    # object_counts fast path
    if obj_name and not color:
        oc = doc.get("object_counts")
        if isinstance(oc, dict):
            cnt = oc.get(str(obj_name).lower())
            if cnt is not None:
                return int(cnt)

    # Fallback: iterate objects array
    objs = doc.get("objects", [])
    if not isinstance(objs, list):
        return 0
    return sum(1 for o in objs if isinstance(o, dict) and object_matches(o, parsed_filter))


def filter_docs_by_count_constraint(
    docs: List[Dict[str, Any]],
    parsed_filter: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Keep only docs whose matching-object count satisfies the constraint."""
    constraint = normalize_count_constraint(parsed_filter.get("count_constraint"))
    if not constraint:
        return docs
    out: List[Dict[str, Any]] = []
    for d in docs:
        cnt = count_matching_objects_in_doc(d, parsed_filter)
        if count_matches_constraint(cnt, constraint):
            dd = dict(d)
            dd["matched_object_count"] = cnt
            out.append(dd)
    return out


# ---------------------------------------------------------------------------
# Object aggregation
# ---------------------------------------------------------------------------
def aggregate_objects(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate multiple detection object dicts into a representative list
    grouped by ``object_name`` with averaged confidence and collected colors.
    """
    if not objects:
        return []

    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        by_name[obj.get("object_name", "unknown")].append(obj)

    aggregated: List[Dict[str, Any]] = []
    for _name, instances in by_name.items():
        representative = instances[0].copy()

        confidences = [
            o["confidence"] for o in instances
            if "confidence" in o and o.get("confidence") is not None
        ]
        if confidences:
            representative["confidence"] = sum(confidences) / len(confidences)
        elif "confidence" in representative:
            del representative["confidence"]

        colors: List[str] = [str(o.get("color")) for o in instances if o.get("color")]
        colors = list(dict.fromkeys(colors))  # dedupe preserving order
        if colors:
            representative["color"] = colors[0] if len(colors) == 1 else ", ".join(colors)
            representative["colors"] = colors

        aggregated.append(representative)
    return aggregated


# ---------------------------------------------------------------------------
# VLM caption helpers
# ---------------------------------------------------------------------------
def flatten_object_captions(payload: Any) -> List[str]:
    """Flatten a nested list of VLM object captions into a lowercase list."""
    out: List[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, list):
                for x in item:
                    if isinstance(x, str) and x.strip():
                        out.append(x.strip().lower())
            elif isinstance(item, str) and item.strip():
                out.append(item.strip().lower())
    return out


def extract_object_tokens(flat_caps: List[str]) -> List[str]:
    """Extract unique object-name tokens from flattened captions."""
    out: set[str] = set()
    for cap in flat_caps:
        low = cap.lower()
        if "person" in low:
            out.add("person")
        if "vehicle" in low or "car" in low:
            out.add("car")
        tok = low.split(" ")[0].strip()
        if tok:
            out.add(tok)
    return sorted(out)


def vlm_matches_object_filter(
    flat_caps: List[str],
    parsed_filter: Dict[str, Any],
) -> bool:
    """Return True if at least one VLM caption matches the object/color filter."""
    obj = parsed_filter.get("objects.object_name")
    color = parsed_filter.get("objects.color")
    if not obj and not color:
        return True
    if not flat_caps:
        return False

    obj_ok = True
    if obj:
        obj_low = str(obj).lower()
        aliases = {obj_low}
        if obj_low == "person":
            aliases.update({"people", "person"})
        elif obj_low == "car":
            aliases.update({"car", "vehicle", "cars", "vehicles"})
        obj_ok = any(any(a in c for a in aliases) for c in flat_caps)

    color_ok = True
    if color:
        color_low = str(color).lower()
        color_ok = any(color_low in c for c in flat_caps)

    return obj_ok and color_ok
