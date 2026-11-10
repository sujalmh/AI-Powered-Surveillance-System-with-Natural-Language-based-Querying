"""
Shared utilities for the retrieval pipeline.
Consolidates timestamp parsing, score normalization, count constraint logic,
and object-matching helpers used across structured search, result merging, and
the unified retrieval orchestrator.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

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

    Returns ``datetime.min`` (epoch sentinel) when parsing fails entirely so
    that malformed timestamps sort to the *bottom* of any result list instead
    of silently appearing at the top (which ``datetime.now()`` would cause).
    """
    try:
        safe = _TZ_SUFFIX_RE.sub("", ts)
        return datetime.fromisoformat(safe)
    except Exception:
        try:
            return datetime.strptime(ts.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            logger.warning("parse_ts: could not parse %r – using datetime.min sentinel", ts)
            return datetime.min


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------
_NORM_EPS = 1e-12


def finite_score(value: Any, default: float = 0.0) -> float:
    """Coerce *value* to a finite float (NaN/inf/None -> *default*).

    A single corrupt score (NaN from a broken FAISS index, ``None`` from a
    partially-populated record) must never be able to zero out or NaN-poison
    an entire branch.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if f != f or f in (float("inf"), float("-inf")):
        return float(default)
    return f


def normalize_scores(
    scores: List[float],
    min_val: float = 0.0,
    mode: str = "minmax",
) -> List[float]:
    """
    Normalize one query's *scores* into [min_val, 1].

    Robustness contract (see results/robust_proposal.md):

    * non-finite inputs are coerced to 0.0 (see :func:`finite_score`);
    * a *degenerate* branch (two or more candidates, all scores equal)
      carries no discrimination information and is mapped to ``min_val``
      (0.0).  The scikit-learn convention for a constant feature is the
      same; reporting it as 1.0 would claim confidence the branch does
      not have.
    * a *single-candidate* branch is mapped to 1.0 -- that candidate is
      trivially the best of its branch -- but the value must not depend on
      the unit of the score.  The "this branch cannot discriminate" signal
      is carried by the branch *confidence* (margin/concentration are 0 for
      a 1-item branch), not by the score.
    * negative scores are *shifted*, never divided by a negative maximum,
      so the within-branch order is preserved and the output stays in
      [min_val, 1].
    * the historical "absolute maximum" rescaling (0.25 / 0.40) is removed:
      those thresholds are only meaningful for scores in [0, 1] and are
      meaningless for the track durations (seconds) they are applied to.

    ``mode="minmax"`` (default) matches the normalisation used by the
    evaluation harness; ``mode="max"`` reproduces max-normalisation
    *without* the unit-dependent rescaling, for A/B against old numbers.
    """
    if not scores:
        return []
    vals = [finite_score(s) for s in scores]
    if len(vals) == 1:
        return [1.0]
    lo, hi = min(vals), max(vals)
    if hi - lo <= _NORM_EPS:
        # Degenerate (constant) branch: no information, no confidence.
        return [min_val] * len(vals)
    if mode == "max":
        if hi <= _NORM_EPS:
            return [min_val] * len(vals)
        return [max(min_val, min(1.0, (v - lo) / hi)) for v in vals]
    return [max(min_val, min(1.0, (v - lo) / (hi - lo))) for v in vals]


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
# Color family mapping — backward compat for old CSS3-granular data & fuzzy user queries.
# Maps each basic colour to a set of names that belong to the same perceptual family.
# ---------------------------------------------------------------------------
COLOR_FAMILIES: Dict[str, frozenset] = {
    "red":    frozenset({"red", "crimson", "maroon", "scarlet", "burgundy", "firebrick", "darkred", "dark red"}),
    "blue":   frozenset({"blue", "navy", "azure", "cobalt", "royal blue", "royalblue", "steelblue",
                          "steel blue", "dodgerblue", "dodger blue", "cornflowerblue", "darkblue",
                          "cyan", "teal", "turquoise", "aqua", "cadetblue"}),
    "green":  frozenset({"green", "lime", "olive", "emerald", "forestgreen", "forest green",
                          "limegreen", "lime green", "darkgreen", "dark green", "seagreen",
                          "sea green", "springgreen", "spring green", "cyan", "teal"}),
    "yellow": frozenset({"yellow", "gold", "amber", "khaki", "lemon", "lightyellow", "light yellow"}),
    "orange": frozenset({"orange", "coral", "tangerine", "peach", "darkorange", "dark orange",
                          "orangered", "orange red", "tomato"}),
    "pink":   frozenset({"pink", "magenta", "fuchsia", "rose", "salmon", "hotpink", "hot pink",
                          "lightpink", "light pink", "deeppink", "deep pink"}),
    "purple": frozenset({"purple", "violet", "indigo", "lavender", "plum", "orchid",
                          "darkviolet", "dark violet", "blueviolet", "blue violet",
                          "mediumpurple", "medium purple"}),
    "black":  frozenset({"black", "dark"}),
    "white":  frozenset({"white", "ivory", "cream", "snow", "linen", "floralwhite", "floral white",
                          "ghostwhite", "ghost white", "beige"}),
    "gray":   frozenset({"gray", "grey", "silver", "charcoal", "slate", "darkgray", "dark gray",
                          "lightgray", "light gray", "dimgray", "dim gray", "gainsboro"}),
    "brown":  frozenset({"brown", "tan", "chocolate", "sienna", "saddlebrown", "saddle brown",
                          "peru", "burlywood", "sandy brown", "sandybrown", "rosybrown", "rosy brown"}),
    "cyan":   frozenset({"cyan", "teal", "turquoise", "aqua", "aquamarine"}),
}

# Invert for O(1) lookup: detected_name → set of families it belongs to
_COLOR_TO_FAMILIES: Dict[str, set] = {}
for _family, _members in COLOR_FAMILIES.items():
    for _m in _members:
        _COLOR_TO_FAMILIES.setdefault(_m, set()).add(_family)
    _COLOR_TO_FAMILIES.setdefault(_family, set()).add(_family)


def _color_family_match(requested: str, detected: str) -> bool:
    """Return True if *detected* colour belongs to the same family as *requested*.
    
    A colour like 'cyan' may belong to both the 'blue' and 'cyan' families,
    so we check bidirectionally: is the detected colour a member of the
    requested colour's family, OR vice-versa?
    """
    req = requested.strip().lower()
    det = detected.strip().lower()
    if req == det:
        return True
    # Check if detected is in the requested colour's family members
    family = COLOR_FAMILIES.get(req)
    if family and det in family:
        return True
    # Check if requested is in the detected colour's family members
    det_family = COLOR_FAMILIES.get(det)
    if det_family and req in det_family:
        return True
    return False


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
    if color:
        # Match against top-3 colors array + upper/lower body colors + legacy single color
        all_obj_colors: set = set()
        for c in (obj.get("colors") or []):
            all_obj_colors.add(str(c).strip().lower())
        for c in (obj.get("upper_body_colors") or []):
            all_obj_colors.add(str(c).strip().lower())
        for c in (obj.get("lower_body_colors") or []):
            all_obj_colors.add(str(c).strip().lower())
        legacy = str(obj.get("color", "")).strip().lower()
        if legacy:
            all_obj_colors.add(legacy)
        all_obj_colors.discard("")
        all_obj_colors.discard("unknown")
        color_lower = str(color).lower()
        # Exact match first, then family match for backward compat with old CSS3 names
        if color_lower not in all_obj_colors:
            if not any(_color_family_match(color_lower, c) for c in all_obj_colors):
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
    Aggregate repeated observations of the *same* object into one entry.

    Grouping is by ``(object_name, track_id)``, not by class alone: a clip
    routinely contains several people, and collapsing them into a single
    ``person`` entry fuses distinct individuals -- their colours, confidences
    and track ids bleed together, so a "brown" match displays as "black".
    Objects without a usable track id fall back to class grouping.
    """
    if not objects:
        return []

    by_identity: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        track_id = obj.get("track_id")
        usable_tid = track_id if isinstance(track_id, (int, float)) and track_id >= 0 else None
        key = (obj.get("object_name", "unknown"), usable_tid)
        by_identity[key].append(obj)

    aggregated: List[Dict[str, Any]] = []
    for _key, instances in by_identity.items():
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
            # Per-frame colour extraction is noisy (lighting, occlusion), so a
            # track usually reports several colours.  ``color`` must stay a
            # single representative value -- clients filter and compare on it,
            # and a comma-joined string ("black, brown, blue") matches nothing.
            # Keep the dominant colour (first-seen wins ties) and expose the
            # full set separately in ``colors``.
            counts = Counter(str(o.get("color")) for o in instances if o.get("color"))
            first_seen = {c: i for i, c in enumerate(colors)}
            representative["color"] = max(
                counts.items(), key=lambda kv: (kv[1], -first_seen.get(kv[0], 0))
            )[0]
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
            aliases.update({"people", "person", "man", "woman", "child",
                           "boy", "girl", "pedestrian", "individual"})
        elif obj_low == "car":
            aliases.update({"car", "vehicle", "cars", "vehicles",
                           "automobile", "sedan", "suv"})
        obj_ok = any(any(a in c for a in aliases) for c in flat_caps)

    color_ok = True
    if color:
        color_low = str(color).lower()
        color_ok = any(color_low in c for c in flat_caps)

    return obj_ok and color_ok
