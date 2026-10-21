"""Parser-quality evaluation for the surveillance NL parser (INDICON 2026).

Why this script exists
----------------------
Claim C1 of the paper states that a closed-form, training-free mixing weight
``alpha(F, iota, n_s, n_v)`` -- i.e. an alpha that depends on the *parsed
filter* F -- beats every fixed alpha.  The filter therefore has to be right:
a dropped colour or a mis-resolved time window destroys the structured branch
and with it the whole retrieval story.  This script measures the parser
itself, so the claim can be defended with a number instead of an assertion.

What is measured
----------------
Every query of the benchmark file (the protocol-matched control queries
by default) is parsed through the
*production* path (``backend.app.services.nl_parser.parse_nl_with_llm``) and
the resulting filter is compared field by field against the reference parse
stored in the same file.

Honest framing (also reproduced in the report itself)
-----------------------------------------------------
* The reference parse is **metadata-derived**, not human-annotated: it was
  reconstructed from the archive metadata that generated each query (camera,
  colour/zone/action label, anchor timestamps).  It is an *approximation* of
  the true query intent, so a disagreement may be a wrong reference rather
  than a wrong parse.
* Therefore the numbers reported here are **agreement rates**, not accuracy.
* The released benchmark is single-camera (camera 0) and does not represent
  the 4-camera deployment quoted in the paper.
* The free tiers of the opencode providers have different daily quotas, so a
  re-run on another day can be answered by a different fallback model.

Run
---
    python backend/scripts/eval_parser.py --limit 150
    python backend/scripts/eval_parser.py --model opencode-go/minimax-m3.1-flash-free
    python backend/scripts/eval_parser.py --refresh          # ignore the cache

The parses are cached under ``backend/benchmarks/parser_cache/<model>.json``
so the script can be repeated without calling the model again.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
# The parser track shares the end-to-end track's cache and query file: the
# same parses are then measured twice (fidelity here, retrieval there) and
# the run is reproducible from the cache without calling the model again.
CACHE_DIR = ROOT / "backend" / "benchmarks" / "parsed_cache"
DEFAULT_QUERIES = ROOT / "backend" / "benchmarks" / "queries_control.jsonl"

#: Fields compared between the parser output and the reference parse.
FIELDS: Tuple[str, ...] = (
    "camera_id",
    "time_window",
    "object_name",
    "color",
    "zone",
    "action",
    "count_constraint",
    "intent",
)

#: Tolerance (seconds) used for the secondary, forgiving window-overlap
#: metric.  The reference window is a *metadata context window* around the
#: anchor, so a parse that pins the exact instant of the query legitimately
#: falls just outside it.
WINDOW_TOLERANCE_S = 60.0

#: Canonical object classes.  The reference always says "person"; a parser
#: that answers "people" / "pedestrian" names the same class and must not be
#: counted as a mismatch.  Mirrors the synonym table of the parser.
_OBJECT_CANONICAL: Dict[str, str] = {
    "person": "person", "people": "person", "persons": "person",
    "pedestrian": "person", "human": "person",
    "car": "car", "cars": "car", "vehicle": "car", "vehicles": "car",
    "automobile": "car",
    "bicycle": "bicycle", "bicycles": "bicycle", "bike": "bicycle",
    "bikes": "bicycle", "cyclist": "bicycle",
    "truck": "truck", "trucks": "truck", "lorry": "truck",
    "dog": "dog", "dogs": "dog", "canine": "dog",
}

#: Regex used to detect that a query names a camera explicitly.
_CAMERA_IN_TEXT = re.compile(r"\bcameras?\s*(\d+)\b")

#: Precision used when a rate is printed in the markdown report.
RATE_DECIMALS = 1
SECONDS_DECIMALS = 2


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _norm(value: Any) -> Optional[str]:
    """Lower-cased, stripped string form of a scalar (``None`` stays ``None``)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lower()


def _present(value: Any) -> bool:
    """True when a field value was actually stated (not ``None``/empty)."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _as_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; return ``None`` when unparseable."""
    if not _present(value):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _align_zone(a: datetime, b: datetime) -> Tuple[datetime, datetime]:
    """Make two datetimes comparable (naive vs aware)."""
    if (a.tzinfo is None) == (b.tzinfo is None):
        return a, b
    if a.tzinfo is not None:
        a = a.replace(tzinfo=None)
    if b.tzinfo is not None:
        b = b.replace(tzinfo=None)
    return a, b


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def git_state() -> Dict[str, Any]:
    """Best-effort record of the code the report was produced with."""
    import subprocess

    state: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=15,
        )
        if commit.returncode == 0:
            state["commit"] = commit.stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=15,
        )
        if status.returncode == 0:
            state["dirty"] = bool(status.stdout.strip())
    except Exception:  # noqa: BLE001 - the metadata is optional
        pass
    return state


def _round(value: Optional[float], digits: int) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _fmt(value: Optional[float], digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Optional[float]) -> str:
    """Rates are printed as percentages (one decimal), ``n/a`` when undefined."""
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.{RATE_DECIMALS}f}%"


# ---------------------------------------------------------------------------
# Field extraction (reference side / parser side)
# ---------------------------------------------------------------------------

def reference_field(name: str, ref_parsed: Dict[str, Any], category: str) -> Any:
    """Read one field out of the metadata-derived reference parse."""
    if name == "camera_id":
        return ref_parsed.get("camera_id")
    if name == "time_window":
        return (ref_parsed.get("time_start"), ref_parsed.get("time_end"))
    if name == "object_name":
        return ref_parsed.get("objects.object_name")
    if name == "color":
        return ref_parsed.get("objects.color")
    if name == "zone":
        return ref_parsed.get("zone")
    if name == "action":
        # The reference uses ``action`` for behaviour-like events and
        # ``event`` for the same labels in the event subset.
        return ref_parsed.get("action") or ref_parsed.get("event")
    if name == "count_constraint":
        return ref_parsed.get("count_target_in_text")
    if name == "intent":
        return ref_parsed.get("intent") or category
    raise KeyError(f"unknown field {name!r}")


def extract_window(parsed: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return the absolute window of a parsed filter as ``(start, end)``."""
    stamp = parsed.get("timestamp")
    if isinstance(stamp, dict):
        start = stamp.get("$gte") or stamp.get("gte") or stamp.get("from_iso") or stamp.get("$from")
        end = stamp.get("$lte") or stamp.get("lte") or stamp.get("to_iso") or stamp.get("$to")
    elif isinstance(stamp, str):
        start = end = stamp
    else:
        start = end = None
    return (str(start) if start else None, str(end) if end else None)


def parsed_field(name: str, parsed: Dict[str, Any], query_text: str = "") -> Any:
    """Read one field out of the filter produced by the parser.

    ``query_text`` is needed for the intent: the intent is the parser's
    deterministic routing rule, not a model output, and the cached filter
    does not carry the internal ``__intent`` key.
    """
    if name == "camera_id":
        return parsed.get("camera_id")
    if name == "time_window":
        return extract_window(parsed)
    if name == "object_name":
        return parsed.get("objects.object_name")
    if name == "color":
        return parsed.get("objects.color")
    if name == "zone":
        return parsed.get("zone")
    if name == "action":
        return parsed.get("action")
    if name == "count_constraint":
        return parsed.get("count_constraint")
    if name == "intent":
        from backend.app.services.nl_parser import infer_intent

        return parsed.get("__intent") or infer_intent(str(query_text or ""))
    raise KeyError(f"unknown field {name!r}")


# ---------------------------------------------------------------------------
# Per-field comparators
# ---------------------------------------------------------------------------

def compare_camera_id(reference: Any, parsed: Any) -> bool:
    """Exact camera-id match (numeric)."""
    ref_int, par_int = _as_int(reference), _as_int(parsed)
    if ref_int is not None and par_int is not None:
        return ref_int == par_int
    return _norm(reference) == _norm(parsed) and _norm(reference) is not None


def canonical_object_classes(value: Any) -> List[str]:
    """Normalise an object description into a set of canonical classes."""
    if not _present(value):
        return []
    if isinstance(value, (list, tuple, set)):
        tokens: List[str] = []
        for item in value:
            tokens.extend(str(item).split(","))
    else:
        tokens = str(value).split(",")
    out: List[str] = []
    for token in tokens:
        key = _norm(token)
        if not key:
            continue
        out.append(_OBJECT_CANONICAL.get(key, key))
    return out


def compare_object_class(reference: Any, parsed: Any) -> bool:
    """Set agreement: do the two descriptions name the same object class?"""
    ref_classes = set(canonical_object_classes(reference))
    par_classes = set(canonical_object_classes(parsed))
    if not ref_classes or not par_classes:
        return False
    return bool(ref_classes & par_classes)


def compare_ci_exact(reference: Any, parsed: Any) -> bool:
    """Case-insensitive exact match."""
    ref_n, par_n = _norm(reference), _norm(parsed)
    if ref_n is None or par_n is None:
        return False
    return ref_n == par_n


def count_constraint_values(value: Any) -> List[int]:
    """Numeric targets of a count constraint (``{"gte": 3}`` -> ``[3]``)."""
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        digits = re.sub(r"[^0-9-]", "", value)
        return [int(digits)] if digits else []
    if isinstance(value, dict):
        out: List[int] = []
        for key in ("eq", "gte", "lte", "gt", "lt"):
            if key in value:
                number = _as_int(value.get(key))
                if number is not None:
                    out.append(number)
        return out
    return []


def compare_count_constraint(reference: Any, parsed: Any) -> bool:
    """Agreement of the numeric count target.

    The reference stores a bare number (``count_target_in_text``) and no
    operator, so only the numeric target is comparable; the comparison of the
    operator (``gt`` vs ``eq``) is impossible against this reference.
    """
    ref_numbers = count_constraint_values(reference)
    par_numbers = count_constraint_values(parsed)
    if not ref_numbers or not par_numbers:
        return False
    return bool(set(ref_numbers) & set(par_numbers))


def compare_intent(reference: Any, parsed: Any) -> bool:
    """Intent agreement (case-insensitive)."""
    return compare_ci_exact(reference, parsed)


def compare_time_window(
    reference: Any, parsed: Any
) -> Dict[str, Any]:
    """Compare two time windows.

    ``reference`` and ``parsed`` are ``(start, end)`` ISO strings.  The
    comparison never requires exact equality: the parsed window only has to
    *overlap* the reference window.  The endpoint errors are reported
    separately so a reviewer can see how far off the bounds are.
    """
    ref_start_raw, ref_end_raw = reference if reference else (None, None)
    par_start_raw, par_end_raw = parsed if parsed else (None, None)

    ref_start, ref_end = _parse_iso(ref_start_raw), _parse_iso(ref_end_raw)
    par_start, par_end = _parse_iso(par_start_raw), _parse_iso(par_end_raw)

    if ref_start and ref_end:
        ref_start, ref_end = _align_zone(ref_start, ref_end)
        if ref_start > ref_end:
            ref_start, ref_end = ref_end, ref_start
    if par_start and par_end:
        par_start, par_end = _align_zone(par_start, par_end)
        if par_start > par_end:
            par_start, par_end = par_end, par_start

    result: Dict[str, Any] = {
        "reference_start": ref_start_raw,
        "reference_end": ref_end_raw,
        "parsed_start": par_start_raw,
        "parsed_end": par_end_raw,
        "overlap": None,
        "overlap_with_tolerance": None,
        "abs_error_start_s": None,
        "abs_error_end_s": None,
        "iou": None,
        "partial_parsed_window": bool(par_start or par_end) and not (par_start and par_end),
    }
    if not (ref_start and ref_end) or not (par_start or par_end):
        return result

    # Interval overlap (closed intervals).
    p_lo = par_start
    p_hi = par_end or par_start
    r_lo, r_hi = ref_start, ref_end
    result["overlap"] = bool(max(p_lo, r_lo) <= min(p_hi, r_hi))

    tol = timedelta(seconds=WINDOW_TOLERANCE_S)
    result["overlap_with_tolerance"] = bool(
        max(p_lo - tol, r_lo - tol) <= min(p_hi + tol, r_hi + tol)
    )

    if par_start and ref_start:
        a, b = _align_zone(par_start, ref_start)
        result["abs_error_start_s"] = abs((a - b).total_seconds())
    if par_end and ref_end:
        a, b = _align_zone(par_end, ref_end)
        result["abs_error_end_s"] = abs((a - b).total_seconds())
    if par_start and par_end and ref_start and ref_end:
        inter = (min(p_hi, r_hi) - max(p_lo, r_lo)).total_seconds()
        inter = max(0.0, inter)
        union = (p_hi - p_lo).total_seconds() + (r_hi - r_lo).total_seconds() - inter
        result["iou"] = (inter / union) if union > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# Field comparison / aggregation
# ---------------------------------------------------------------------------

def _window_present(value: Any) -> bool:
    """A time window is stated when at least one bound is present."""
    if isinstance(value, (tuple, list)):
        return any(_present(bound) for bound in value)
    return _present(value)


def compare_field(name: str, reference: Any, parsed: Any) -> Dict[str, Any]:
    """Compare one field and describe the outcome.

    Outcomes
    --------
    ``agree``             both sides state the field and they match
    ``mismatch``          both sides state the field but differ
    ``missing_in_parse``  the reference states it, the parser did not
    ``spurious_in_parse`` the parser states it, the reference did not
    ``absent``            neither side states it
    """
    if name == "time_window":
        ref_present = _window_present(reference)
        par_present = _window_present(parsed)
    else:
        ref_present = _present(reference)
        par_present = _present(parsed)

    if name == "time_window":
        detail = compare_time_window(reference, parsed)
        if not ref_present:
            status = "absent" if not par_present else "spurious_in_parse"
        elif not par_present:
            status = "missing_in_parse"
        else:
            status = "agree" if detail["overlap"] else "mismatch"
        agree: Optional[bool]
        if status == "agree":
            agree = True
        elif status == "mismatch":
            agree = False
        elif status == "missing_in_parse":
            agree = False
        else:
            agree = None
        return {
            "field": name,
            "reference": list(reference) if isinstance(reference, tuple) else reference,
            "parsed": list(parsed) if isinstance(parsed, tuple) else parsed,
            "reference_present": ref_present,
            "parsed_present": par_present,
            "both": ref_present and par_present,
            "agree": agree,
            "status": status,
            "detail": detail,
        }

    comparator = {
        "camera_id": compare_camera_id,
        "object_name": compare_object_class,
        "color": compare_ci_exact,
        "zone": compare_ci_exact,
        "action": compare_ci_exact,
        "count_constraint": compare_count_constraint,
        "intent": compare_intent,
    }[name]

    if ref_present and par_present:
        agree = bool(comparator(reference, parsed))
        status = "agree" if agree else "mismatch"
    elif ref_present:
        agree = False
        status = "missing_in_parse"
    elif par_present:
        agree = None
        status = "spurious_in_parse"
    else:
        agree = None
        status = "absent"
    return {
        "field": name,
        "reference": reference,
        "parsed": parsed,
        "reference_present": ref_present,
        "parsed_present": par_present,
        "both": ref_present and par_present,
        "agree": agree,
        "status": status,
        "detail": {},
    }


def aggregate_field(outcomes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the outcomes of one field over a set of queries."""
    n_ref = sum(1 for o in outcomes if o["reference_present"])
    n_par = sum(1 for o in outcomes if o["parsed_present"])
    n_both = sum(1 for o in outcomes if o["both"])
    n_agree = sum(1 for o in outcomes if o["agree"] is True)
    n_mismatch = sum(1 for o in outcomes if o["status"] == "mismatch")
    n_missing = sum(1 for o in outcomes if o["status"] == "missing_in_parse")
    n_spurious = sum(1 for o in outcomes if o["status"] == "spurious_in_parse")
    rate = (n_agree / n_ref) if n_ref else None
    rate_both = (n_agree / n_both) if n_both else None
    return {
        "n_evaluated": len(outcomes),
        "n_reference": n_ref,
        "n_parsed": n_par,
        "n_both": n_both,
        "n_agree": n_agree,
        "n_mismatch": n_mismatch,
        "n_missing_in_parse": n_missing,
        "n_spurious_in_parse": n_spurious,
        "agreement_rate": rate,
        "agreement_rate_pct": _round(None if rate is None else rate * 100.0, RATE_DECIMALS),
        "agreement_rate_among_both": rate_both,
        "agreement_rate_among_both_pct": _round(
            None if rate_both is None else rate_both * 100.0, RATE_DECIMALS
        ),
    }


def aggregate_fields(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-field table over a list of per-query records."""
    table: Dict[str, Dict[str, Any]] = {}
    for name in FIELDS:
        outcomes = [rec["fields"][name] for rec in records if name in rec["fields"]]
        table[name] = aggregate_field(outcomes)
    return table


def window_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Median endpoint errors and overlap statistics of the time windows."""
    details = []
    for rec in records:
        field = rec["fields"].get("time_window")
        if not field:
            continue
        detail = field.get("detail") or {}
        if detail.get("reference_start") and detail.get("parsed_start"):
            details.append(detail)

    start_errors = [d["abs_error_start_s"] for d in details if d.get("abs_error_start_s") is not None]
    end_errors = [d["abs_error_end_s"] for d in details if d.get("abs_error_end_s") is not None]
    pooled = start_errors + end_errors
    overlaps = [d for d in details if d.get("overlap") is not None]
    ious = [d["iou"] for d in details if d.get("iou") is not None]

    n_evaluated = len(details)
    median_start = statistics.median(start_errors) if start_errors else None
    median_end = statistics.median(end_errors) if end_errors else None
    median_pooled = statistics.median(pooled) if pooled else None
    overlap_rate = (sum(1 for d in overlaps if d["overlap"]) / len(overlaps)) if overlaps else None
    overlap_tol = (
        sum(1 for d in overlaps if d.get("overlap_with_tolerance")) / len(overlaps)
        if overlaps else None
    )
    median_iou = statistics.median(ious) if ious else None
    mean_pooled = (sum(pooled) / len(pooled)) if pooled else None
    n_partial = sum(1 for rec in records if (rec["fields"].get("time_window") or {}).get(
        "detail", {}).get("partial_parsed_window"))
    n_no_window = sum(
        1 for rec in records
        if not (rec["fields"].get("time_window") or {}).get("parsed_present")
    )
    return {
        "n_evaluated": n_evaluated,
        "n_without_parsed_window": n_no_window,
        "n_partial_parsed_window": n_partial,
        "n_start_errors": len(start_errors),
        "n_end_errors": len(end_errors),
        "median_abs_error_start_s": median_start,
        "median_abs_error_start_s_r": _round(median_start, SECONDS_DECIMALS),
        "median_abs_error_end_s": median_end,
        "median_abs_error_end_s_r": _round(median_end, SECONDS_DECIMALS),
        "median_abs_error_s": median_pooled,
        "median_abs_error_s_r": _round(median_pooled, SECONDS_DECIMALS),
        "mean_abs_error_s": mean_pooled,
        "mean_abs_error_s_r": _round(mean_pooled, SECONDS_DECIMALS),
        "window_overlap_rate": overlap_rate,
        "window_overlap_rate_pct": _round(None if overlap_rate is None else overlap_rate * 100.0, RATE_DECIMALS),
        "window_overlap_rate_tolerance": overlap_tol,
        "window_overlap_rate_tolerance_pct": _round(
            None if overlap_tol is None else overlap_tol * 100.0, RATE_DECIMALS
        ),
        "tolerance_seconds": WINDOW_TOLERANCE_S,
        "median_iou": median_iou,
        "median_iou_r": _round(median_iou, 4),
    }


def latency_stats(seconds: Sequence[float]) -> Dict[str, Any]:
    """Mean / median / p95 / max of the per-query wall-clock parse time."""
    values = [float(s) for s in seconds if s is not None]
    if not values:
        return {
            "n": 0, "mean_seconds": None, "median_seconds": None, "p95_seconds": None,
            "max_seconds": None, "min_seconds": None,
        }

    def percentile(values_: List[float], q: float) -> float:
        ordered = sorted(values_)
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        low = int(pos)
        high = min(low + 1, len(ordered) - 1)
        frac = pos - low
        return ordered[low] + (ordered[high] - ordered[low]) * frac

    mean = sum(values) / len(values)
    median = statistics.median(values)
    p95 = percentile(values, 0.95)
    return {
        "n": len(values),
        "mean_seconds": mean,
        "mean_seconds_r": _round(mean, SECONDS_DECIMALS),
        "median_seconds": median,
        "median_seconds_r": _round(median, SECONDS_DECIMALS),
        "p95_seconds": p95,
        "p95_seconds_r": _round(p95, SECONDS_DECIMALS),
        "max_seconds": max(values),
        "max_seconds_r": _round(max(values), SECONDS_DECIMALS),
        "min_seconds": min(values),
        "min_seconds_r": _round(min(values), SECONDS_DECIMALS),
    }


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(query: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Compare one parsed filter against the reference parse of one query."""
    ref_parsed = query.get("parsed") or {}
    category = query.get("category", "")
    fields: Dict[str, Dict[str, Any]] = {}
    for name in FIELDS:
        reference = reference_field(name, ref_parsed, category)
        produced = parsed_field(name, parsed, str(query.get("text", "")))
        fields[name] = compare_field(name, reference, produced)
    return {
        "qid": query.get("qid"),
        "category": category,
        "split": query.get("split"),
        "group": query.get("group"),
        "text": query.get("text"),
        "fields": fields,
    }


def camera_explicit_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Camera-id agreement restricted to queries that name a camera.

    The reference camera (0) is an archive property; it is only *expressible*
    in the query when the text mentions it.  This block therefore reports the
    agreement on the subset where the camera is stated in the text.
    """
    n_named = 0
    n_agree = 0
    n_parsed_camera = 0
    qids: List[str] = []
    for rec in records:
        match = _CAMERA_IN_TEXT.search(str(rec.get("text") or ""))
        if not match:
            continue
        n_named += 1
        field = rec["fields"]["camera_id"]
        if field["parsed_present"]:
            n_parsed_camera += 1
        if field["both"] and compare_camera_id(field["reference"], field["parsed"]):
            n_agree += 1
        else:
            qids.append(str(rec.get("qid")))
    rate = (n_agree / n_named) if n_named else None
    return {
        "definition": "queries whose text names a camera explicitly (regex 'camera <n>')",
        "n_named_in_text": n_named,
        "n_parsed_camera": n_parsed_camera,
        "n_agree": n_agree,
        "agreement_rate": rate,
        "agreement_rate_pct": _round(None if rate is None else rate * 100.0, RATE_DECIMALS),
        "disagreeing_qids": qids,
    }


def intent_explicit_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Intent agreement restricted to references that state an intent."""
    n_ref = 0
    n_agree = 0
    qids: List[str] = []
    for rec in records:
        ref_intent = (rec.get("reference_parsed") or {}).get("intent")
        if not ref_intent:
            continue
        n_ref += 1
        field = rec["fields"]["intent"]
        if field["both"] and compare_intent(field["reference"], field["parsed"]):
            n_agree += 1
        else:
            qids.append(str(rec.get("qid")))
    rate = (n_agree / n_ref) if n_ref else None
    return {
        "definition": "queries whose reference parse states an explicit intent",
        "n_reference_with_intent": n_ref,
        "n_agree": n_agree,
        "agreement_rate": rate,
        "agreement_rate_pct": _round(None if rate is None else rate * 100.0, RATE_DECIMALS),
        "disagreeing_qids": qids,
    }


# ---------------------------------------------------------------------------
# Parsing (production path) with an on-disk cache
# ---------------------------------------------------------------------------

class _BackendCapture:
    """Record the LLM backend actually used by the production parser.

    The opencode transport falls back to a secondary free model when the
    primary model's daily quota is exhausted, so the model that answered is
    recorded per query instead of assuming the requested one.
    """

    def __init__(self) -> None:
        self._original = None
        self.backends: List[Any] = []

    def __enter__(self) -> "_BackendCapture":
        from backend.app.services import llm_client

        self._original = llm_client.build_backend

        def wrapper(cfg: Optional[Dict[str, Any]] = None) -> Any:
            backend = self._original(cfg)
            self.backends.append(backend)
            return backend

        llm_client.build_backend = wrapper  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        from backend.app.services import llm_client

        if self._original is not None:
            llm_client.build_backend = self._original  # type: ignore[assignment]

    def last_model(self) -> Optional[str]:
        if not self.backends:
            return None
        model = getattr(self.backends[-1], "model", None)
        return str(model) if model else None


def parse_queries(
    queries: Sequence[Dict[str, Any]],
    model: str,
    refresh: bool,
    cache_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse every query with the production parser and cache the results."""
    from backend.app.services import llm_client, nl_parser

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_safe_name(model)}.json"
    cache = llm_client.ParseCache(cache_path)

    rows: List[Dict[str, Any]] = []
    models_used: Dict[str, int] = {}
    n_cached = 0
    n_called = 0

    for i, query in enumerate(queries, 1):
        qid = str(query.get("qid"))
        # Same content-addressed key as the end-to-end track, so a parse is
        # never reused for a different query text or a different model.
        from backend.scripts.eval_end_to_end import cache_key

        key = cache_key(qid, str(query.get("text", "")), model)
        entry = None if refresh else cache.get(key)
        if entry is None:
            n_called += 1
            with _BackendCapture() as capture:
                started = time.perf_counter()
                try:
                    parsed = nl_parser.parse_nl_with_llm(query.get("text", ""))
                    error = None
                    ok = True
                except Exception as exc:  # noqa: BLE001 - the failure must be reported
                    parsed = {}
                    error = f"{type(exc).__name__}: {exc}"[:300]
                    ok = False
                elapsed = time.perf_counter() - started
            used_model = capture.last_model() or model
            entry = {
                "ok": ok,
                "parsed": parsed,
                "error": error,
                "seconds": round(elapsed, 3),
                "model": used_model,
            }
            cache.put(key, entry)
            cache.save()  # keep the run resumable
        else:
            n_cached += 1
            used_model = entry.get("model") or model
        models_used[used_model] = models_used.get(used_model, 0) + 1
        rows.append({"query": query, "entry": entry})
        print(
            f"  [{i}/{len(queries)}] {qid}: "
            f"{'llm' if entry.get('ok') else 'FAILED'} ({entry.get('seconds')}s)",
            flush=True,
        )
    cache.save()
    try:
        cache_rel = str(cache_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        cache_rel = str(cache_path)
    meta = {
        "cache_path": cache_rel,
        "n_calls": n_called,
        "n_from_cache": n_cached,
        "models_used": models_used,
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(
    rows: Sequence[Dict[str, Any]],
    model: str,
    provider: Optional[str],
    meta: Dict[str, Any],
    queries_path: Path,
    reference_date: Optional[str],
    n_total_queries: int,
) -> Dict[str, Any]:
    """Assemble the machine-readable evaluation report."""
    records: List[Dict[str, Any]] = []
    per_query: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    seconds: List[float] = []

    for row in rows:
        query = row["query"]
        entry = row["entry"]
        parsed = entry.get("parsed") or {}
        ok = bool(entry.get("ok"))
        record = evaluate_query(query, parsed)
        record["reference_parsed"] = query.get("parsed") or {}
        record["ok"] = ok
        record["seconds"] = entry.get("seconds")
        record["parsed"] = parsed
        record["error"] = entry.get("error")
        records.append(record)

        if not ok:
            failed.append({
                "qid": record["qid"],
                "category": record["category"],
                "split": record["split"],
                "text": record["text"],
                "error": entry.get("error"),
            })
        conflict_fields = [
            name for name in FIELDS if record["fields"][name]["status"] == "mismatch"
        ]
        dropped_fields = [
            name for name in FIELDS if record["fields"][name]["status"] == "missing_in_parse"
        ]
        if conflict_fields:
            conflicts.append({
                "qid": record["qid"],
                "category": record["category"],
                "split": record["split"],
                "text": record["text"],
                "conflicting_fields": conflict_fields,
            })
        if dropped_fields:
            dropped.append({
                "qid": record["qid"],
                "category": record["category"],
                "split": record["split"],
                "text": record["text"],
                "dropped_fields": dropped_fields,
            })
        mismatched = conflict_fields + dropped_fields
        if mismatched:
            ambiguous.append({
                "qid": record["qid"],
                "category": record["category"],
                "split": record["split"],
                "text": record["text"],
                "mismatched_fields": mismatched,
                "conflicting_fields": conflict_fields,
                "dropped_fields": dropped_fields,
            })
        if record["seconds"] is not None:
            seconds.append(float(record["seconds"]))
        per_query.append({
            "qid": record["qid"],
            "category": record["category"],
            "split": record["split"],
            "group": record["group"],
            "text": record["text"],
            "ok": ok,
            "error": entry.get("error"),
            "seconds": record["seconds"],
            "model": entry.get("model"),
            "parsed": parsed,
            "fields": {
                name: {
                    "reference": record["fields"][name]["reference"],
                    "parsed": record["fields"][name]["parsed"],
                    "status": record["fields"][name]["status"],
                    "agree": record["fields"][name]["agree"],
                }
                for name in FIELDS
            },
            "time_window": record["fields"]["time_window"].get("detail", {}),
        })

    categories = sorted({str(r["category"]) for r in records})
    splits: Dict[str, int] = {}
    for rec in records:
        splits[str(rec["split"])] = splits.get(str(rec["split"]), 0) + 1

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in categories:
        subset = [r for r in records if r["category"] == category]
        by_category[category] = {
            "n": len(subset),
            "fields": aggregate_fields(subset),
            "time_window": window_stats(subset),
            "latency": latency_stats([r["seconds"] for r in subset if r["seconds"] is not None]),
        }

    overall_fields = aggregate_fields(records)
    camera_explicit = camera_explicit_stats(records)
    intent_explicit = intent_explicit_stats(records)

    # Facts used by the limitation text (all computed, never hard-coded).
    facts = {
        "n_queries_in_file": n_total_queries,
        "n_reference_with_count_target": sum(
            1 for r in records
            if _present(reference_field("count_constraint", r["reference_parsed"], r["category"]))
        ),
        "n_reference_with_action": sum(
            1 for r in records
            if _present(reference_field("action", r["reference_parsed"], r["category"]))
        ),
        "n_reference_action_motion_energy": sum(
            1 for r in records
            if _norm(reference_field("action", r["reference_parsed"], r["category"])) == "motion_energy"
        ),
        "n_reference_with_zone": sum(
            1 for r in records
            if _present(reference_field("zone", r["reference_parsed"], r["category"]))
        ),
        "n_queries_naming_camera": sum(
            1 for r in records if _CAMERA_IN_TEXT.search(str(r.get("text") or ""))
        ),
    }

    # Interpretation counters: they separate "the reference is odd" from
    # "the parse is wrong".  Every number is computed from the records.
    intent_mismatch_action_filter = sum(
        1 for r in records
        if r["category"] in {"action", "filter"}
        and r["fields"]["intent"]["status"] == "mismatch"
    )
    interpretation = {
        "camera_id_dropped_but_not_expressible": overall_fields["camera_id"]["n_missing_in_parse"],
        "camera_id_agreement_when_named": camera_explicit["agreement_rate"],
        "object_name_dropped": overall_fields["object_name"]["n_missing_in_parse"],
        "object_name_spurious": overall_fields["object_name"]["n_spurious_in_parse"],
        "action_dropped": overall_fields["action"]["n_missing_in_parse"],
        "action_conflicting": overall_fields["action"]["n_mismatch"],
        "action_reference_motion_energy": facts["n_reference_action_motion_energy"],
        "action_reference_natural_language": (
            facts["n_reference_with_action"] - facts["n_reference_action_motion_energy"]
        ),
        "intent_mismatch_in_action_and_filter": intent_mismatch_action_filter,
        "intent_mismatch_with_explicit_reference": (
            intent_explicit["n_reference_with_intent"] - intent_explicit["n_agree"]
        ),
    }

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "backend/scripts/eval_parser.py",
        "scope": {
            "track": "natural-language parser quality (parsed filter vs reference parse)",
            "model_requested": model,
            "provider": provider,
            "models_actually_used": meta.get("models_used", {}),
            "reference_date": reference_date,
            "queries_file": str(queries_path.relative_to(ROOT)).replace("\\", "/"),
            "repo": git_state(),
            "note": (
                "Single-camera released subset (camera 0); the paper's "
                "4-camera deployment benchmark is not represented here."
            ),
        },
        "dataset": {
            "queries_file": str(queries_path.relative_to(ROOT)).replace("\\", "/"),
            "n_queries_in_file": n_total_queries,
            "n_queries_evaluated": len(records),
            "splits": splits,
            "categories": {
                category: sum(1 for r in records if r["category"] == category)
                for category in categories
            },
        },
        "reference": {
            "source": "the 'parsed' field of the query file used for the run",
            "provenance": "metadata-derived (archive metadata), NOT human-annotated",
            "warning": (
                "The reference is an approximation of the true query intent; a "
                "disagreement may be a wrong reference rather than a wrong parse. "
                "The rates below are agreement rates, not accuracy."
            ),
        },
        "totals": {
            "n": len(records),
            "n_parse_ok": sum(1 for r in records if r["ok"]),
            "n_parse_failed": len(failed),
            "parse_failure_rate": (len(failed) / len(records)) if records else None,
            "parse_failure_rate_pct": _round(
                None if not records else (len(failed) / len(records)) * 100.0, RATE_DECIMALS
            ),
        },
        "field_agreement_overall": overall_fields,
        "field_agreement_by_category": {
            category: by_category[category]["fields"] for category in categories
        },
        "time_window": window_stats(records),
        "camera_id_explicit_text": camera_explicit,
        "intent_explicit_reference": intent_explicit,
        "interpretation": interpretation,
        "latency": latency_stats(seconds),
        "latency_by_category": {
            category: by_category[category]["latency"] for category in categories
        },
        "failed_queries": failed,
        "failed_query_ids": [f["qid"] for f in failed],
        "n_failed_queries": len(failed),
        "ambiguous_queries": ambiguous,
        "ambiguous_query_ids": [a["qid"] for a in ambiguous],
        "n_ambiguous_queries": len(ambiguous),
        "conflict_queries": conflicts,
        "conflict_query_ids": [c["qid"] for c in conflicts],
        "n_conflict_queries": len(conflicts),
        "dropped_field_queries": dropped,
        "dropped_field_query_ids": [d["qid"] for d in dropped],
        "n_dropped_field_queries": len(dropped),
        "dropped_field_breakdown": {
            name: sum(1 for d in dropped if name in d["dropped_fields"])
            for name in FIELDS
        },
        "cache": {
            "path": meta.get("cache_path"),
            "n_calls": meta.get("n_calls"),
            "n_from_cache": meta.get("n_from_cache"),
        },
        "per_query": per_query,
        "limitations": build_limitations(facts),
    }
    return report


def build_limitations(facts: Dict[str, Any]) -> List[str]:
    """Known limitations of this measurement.

    Every number quoted here is taken from ``facts`` (i.e. computed from the
    data), never hard-coded.
    """
    n_total = facts["n_queries_in_file"]
    n_ref_count = facts["n_reference_with_count_target"]
    n_motion = facts["n_reference_action_motion_energy"]
    n_action_ref = facts["n_reference_with_action"]
    n_camera_named = facts["n_queries_naming_camera"]
    n_zone_ref = facts["n_reference_with_zone"]
    return [
        "The reference parse is metadata-derived, not human-annotated: it was "
        "reconstructed from the archive metadata that generated each query "
        "(camera, colour/zone/action label, anchor timestamps). It approximates "
        "the true query intent, so a disagreement may be a wrong reference "
        "rather than a wrong parse.",
        "The reported numbers are agreement rates with a metadata-derived "
        "reference; they are NOT accuracy against a human-annotated gold standard.",
        "The released benchmark subset is single-camera (camera 0) and does not "
        "represent the 4-camera deployment used in the paper.",
        "Free-tier quotas differ between providers: the opencode transport falls "
        "back to a secondary free model when the primary model's daily quota is "
        "exhausted, so a re-run on another day may be answered by a different "
        "model (the model that answered each query is recorded in the JSON).",
        f"The count-constraint agreement rate is undefined for this benchmark: "
        f"{n_ref_count} of the {n_total} reference parses state a numeric count "
        f"target (count_target_in_text is null everywhere), because the count "
        f"queries carry no number in their text, so there is nothing to agree "
        f"on for most of them.",
        f"The action reference of the 'highest-motion' queries is the internal "
        f"token 'motion_energy' ({n_motion} of the {n_action_ref} reference "
        f"action values), which no natural-language parser can emit; those "
        f"queries are structurally unmatchable.",
        "The intent reference falls back to the query category (count / find / "
        "track / action / filter), a wider vocabulary than the parser's "
        "__intent (find / count / track); the action and filter categories are "
        "therefore structurally unmatched in the headline intent rate.",
        f"The reference camera id is 0 for all {n_total} queries although only "
        f"{n_camera_named} of the texts names a camera; the strict camera-id "
        f"rate is therefore dominated by queries where the camera is not "
        f"expressible. The camera_id_explicit_text block reports the restricted rate.",
        f"The reference zone is stated for only {n_zone_ref} queries, and the "
        f"reference window is a metadata context window around the anchor "
        f"rather than the window written in the query text; endpoint errors "
        f"therefore mix parser error with reference/window-definition mismatch.",
    ]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def write_json_report(report: Dict[str, Any], path: Path) -> Path:
    """Write the machine-readable report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_markdown_report(report: Dict[str, Any], path: Path) -> Path:
    """Write the human-readable report (same numbers as the JSON)."""
    lines: List[str] = []
    scope = report["scope"]
    totals = report["totals"]
    overall = report["field_agreement_overall"]
    window = report["time_window"]
    latency = report["latency"]
    interp = report.get("interpretation") or {}

    lines += [
        "# NL parser evaluation (agreement with the metadata-derived reference)",
        "",
        f"- Model requested: `{scope['model_requested']}` "
        f"(provider `{scope['provider']}`)",
        f"- Models that answered: "
        + ", ".join(
            f"`{k}` x{v}" for k, v in sorted(scope.get("models_actually_used", {}).items())
        ),
        f"- Generated at: {report['generated_at']}",
        f"- Repo commit: `{scope.get('repo', {}).get('commit')}` "
        f"(working tree dirty: {scope.get('repo', {}).get('dirty')})",
        f"- Queries: {report['dataset']['n_queries_evaluated']} of "
        f"{report['dataset']['n_queries_in_file']} in "
        f"`{report['dataset']['queries_file']}` "
        f"(splits: "
        + ", ".join(f"{k} {v}" for k, v in sorted(report["dataset"]["splits"].items()))
        + ")",
        f"- Reference: {report['reference']['source']}",
        "",
        "> **Read this before quoting a number.** "
        + report["reference"]["warning"],
        "",
        "## Headline",
        "",
        f"- Parse failures (LLM error / unusable JSON): **{totals['n_parse_failed']}** of "
        f"{totals['n']} queries "
        f"({_fmt_pct(totals['parse_failure_rate'])}).",
        f"- Parse latency: mean **{_fmt(latency.get('mean_seconds'), SECONDS_DECIMALS)} s**, "
        f"median {_fmt(latency.get('median_seconds'), SECONDS_DECIMALS)} s, "
        f"p95 **{_fmt(latency.get('p95_seconds'), SECONDS_DECIMALS)} s**, "
        f"max {_fmt(latency.get('max_seconds'), SECONDS_DECIMALS)} s.",
    ]
    for name in FIELDS:
        stat = overall[name]
        if not stat["n_reference"]:
            lines.append(
                f"- Agreement `{name}`: n/a "
                f"(no reference parse states this field for the evaluated queries)."
            )
            continue
        lines.append(
            f"- Agreement `{name}`: **{_fmt_pct(stat['agreement_rate'])}** "
            f"({stat['n_agree']}/{stat['n_reference']} queries whose reference states the field)."
        )
    lines += [
        "",
        "## Per-field agreement (all queries)",
        "",
        "Agreement is computed over the queries whose *reference* states the field; "
        "a field the parser does not state counts as a disagreement. "
        "`rate among both` restricts the denominator to the queries where both "
        "sides state the field.",
        "",
    ]
    rows = []
    for name in FIELDS:
        stat = overall[name]
        rows.append([
            f"`{name}`",
            stat["n_reference"],
            stat["n_parsed"],
            stat["n_both"],
            stat["n_agree"],
            _fmt_pct(stat["agreement_rate"]),
            _fmt_pct(stat["agreement_rate_among_both"]),
            stat["n_missing_in_parse"],
            stat["n_spurious_in_parse"],
        ])
    lines += _md_table(
        ["field", "reference states", "parser states", "both", "agree",
         "agreement %", "rate among both %", "missing in parse", "spurious in parse"],
        rows,
    )
    lines += [
        "",
        "`time_window` agreement means *interval overlap* with the reference "
        "window, not exact equality of the two bounds.",
        "",
        "## Per-field agreement by category",
        "",
    ]
    rows = []
    for category, table in report["field_agreement_by_category"].items():
        row = [category]
        for name in FIELDS:
            stat = table[name]
            row.append(_fmt_pct(stat["agreement_rate"]))
        rows.append(row)
    lines += _md_table(["category"] + [f"`{name}`" for name in FIELDS], rows)
    lines += [
        "",
        "A cell is `n/a` when the reference never states that field for the "
        "category (e.g. there is no reference zone for the count queries).",
        "",
        "## Time window",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| queries evaluated | {window['n_evaluated']} |",
        f"| queries without a parsed absolute window | {window['n_without_parsed_window']} |",
        f"| median absolute error, start bound (s) | {_fmt(window['median_abs_error_start_s'], SECONDS_DECIMALS)} |",
        f"| median absolute error, end bound (s) | {_fmt(window['median_abs_error_end_s'], SECONDS_DECIMALS)} |",
        f"| median absolute error, both bounds (s) | {_fmt(window['median_abs_error_s'], SECONDS_DECIMALS)} |",
        f"| mean absolute error, both bounds (s) | {_fmt(window['mean_abs_error_s'], SECONDS_DECIMALS)} |",
        f"| window overlap rate | {_fmt_pct(window['window_overlap_rate'])} |",
        f"| window overlap rate (+/- {window['tolerance_seconds']:.0f} s tolerance) | "
        f"{_fmt_pct(window['window_overlap_rate_tolerance'])} |",
        f"| median IoU | {_fmt(window['median_iou'], 4)} |",
        "",
        "The reference window is a metadata *context window* around the anchor "
        "of the query, while the parser emits the window written in the text "
        "(often a single instant for \"around HH:MM:SS\"). The endpoint errors "
        "therefore mix parser error with a difference in the definition of the "
        "window; the overlap rate with a tolerance is given as the more "
        "forgiving diagnostic.",
        "",
        "## Camera id and intent (restricted views)",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    cam = report["camera_id_explicit_text"]
    intent = report["intent_explicit_reference"]
    lines += [
        f"| queries whose text names a camera | {cam['n_named_in_text']} |",
        f"| camera-id agreement on those queries | {_fmt_pct(cam['agreement_rate'])} "
        f"({cam['n_agree']}/{cam['n_named_in_text']}) |",
        f"| queries whose reference states an intent | {intent['n_reference_with_intent']} |",
        f"| intent agreement on those queries | {_fmt_pct(intent['agreement_rate'])} "
        f"({intent['n_agree']}/{intent['n_reference_with_intent']}) |",
        "",
        "## Latency",
        "",
    ]
    rows = [["all queries", latency["n"], _fmt(latency.get("mean_seconds"), SECONDS_DECIMALS),
             _fmt(latency.get("median_seconds"), SECONDS_DECIMALS),
             _fmt(latency.get("p95_seconds"), SECONDS_DECIMALS),
             _fmt(latency.get("max_seconds"), SECONDS_DECIMALS)]]
    for category, stat in report["latency_by_category"].items():
        rows.append([
            category, stat["n"],
            _fmt(stat.get("mean_seconds"), SECONDS_DECIMALS),
            _fmt(stat.get("median_seconds"), SECONDS_DECIMALS),
            _fmt(stat.get("p95_seconds"), SECONDS_DECIMALS),
            _fmt(stat.get("max_seconds"), SECONDS_DECIMALS),
        ])
    lines += _md_table(["scope", "n", "mean (s)", "median (s)", "p95 (s)", "max (s)"], rows)
    lines += [
        "",
        "## Parse failures",
        "",
    ]
    if report["failed_queries"]:
        rows = [[f["qid"], f["category"], f["split"], (f["error"] or "")[:120]]
                for f in report["failed_queries"]]
        lines += _md_table(["qid", "category", "split", "error"], rows)
    else:
        lines.append(f"No parse failures: all {totals['n']} queries produced a "
                     "usable filter.")
    lines += [
        "",
        "## Queries with a value conflict",
        "",
        "Both sides state the field but with different values. Each of these is "
        "a case where the parse and the metadata-derived reference disagree; it "
        "may be a wrong reference rather than a wrong parse.",
        "",
    ]
    if report["conflict_queries"]:
        rows = [[c["qid"], c["category"], c["split"], ", ".join(c["conflicting_fields"])]
                for c in report["conflict_queries"]]
        lines += _md_table(["qid", "category", "split", "conflicting fields"], rows)
    else:
        lines.append("None.")
    lines += [
        "",
        "## Queries where the parser dropped a field",
        "",
        "The reference states the field, the parser does not. The breakdown "
        "shows how often each field was dropped; a dropped `camera_id` is "
        "expected for the queries whose text does not name a camera (the "
        "camera is an archive property, not something the query can state).",
        "",
    ]
    breakdown = report.get("dropped_field_breakdown") or {}
    rows = [[f"`{name}`", count] for name, count in breakdown.items()]
    if rows:
        lines += _md_table(["dropped field", "queries"], rows)
    else:
        lines.append("None.")
    lines += [
        "",
        f"Queries with at least one disagreement of any kind: "
        f"{report['n_ambiguous_queries']}.",
        "",
        "## Reading the disagreements",
        "",
        "The counters below separate the cases where the *reference* is the "
        "odd one from the cases where the *parse* is the odd one. They are "
        "computed from the same records as the tables above.",
        "",
        f"- `camera_id`: {interp['camera_id_dropped_but_not_expressible']} queries state no "
        f"camera because their text does not name one; on the queries that do "
        f"name a camera the agreement is "
        f"{_fmt_pct(interp['camera_id_agreement_when_named'])}.",
        f"- `object_name`: {interp['object_name_dropped']} queries state no object "
        f"(the text names clothing or a track, not a person), while "
        f"{interp['object_name_spurious']} queries state an object the reference "
        f"does not (the parser says 'person' for the count queries, whose "
        f"reference states no object at all) - the parse is more specific there, "
        f"not wrong.",
        f"- `action`: {interp['action_reference_natural_language']} reference action "
        f"values are natural-language labels and "
        f"{interp['action_reference_motion_energy']} are the internal token "
        f"'motion_energy'; the parser dropped the field on "
        f"{interp['action_dropped']} queries and produced a conflicting value on "
        f"{interp['action_conflicting']}.",
        f"- `intent`: {interp['intent_mismatch_in_action_and_filter']} mismatches "
        f"come from the action and filter categories, whose reference falls back "
        f"to the category name (a value the parser's intent vocabulary does not "
        f"contain); {interp['intent_mismatch_with_explicit_reference']} mismatches "
        f"remain for the queries whose reference states an explicit intent.",
        "",
        "## Known limitations",
        "",
    ]
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$env:PYTHONIOENCODING='utf-8'",
        f"python backend/scripts/eval_parser.py --limit {report['dataset']['n_queries_evaluated']} "
        f"--model {report['scope']['model_requested']}",
        "```",
        "",
        "Outputs: `results/parser_results.json` and `results/parser_results.md`. "
        f"The parses are cached in `{report['cache']['path']}` "
        f"({report['cache']['n_from_cache']} from cache, {report['cache']['n_calls']} "
        "model calls in this run). Use `--refresh` to ignore the cache and "
        "`--model <id>` to evaluate another model.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_summary(report: Dict[str, Any], results_dir: Path = RESULTS_DIR) -> None:
    """Compact stdout summary."""
    totals = report["totals"]
    overall = report["field_agreement_overall"]
    window = report["time_window"]
    latency = report["latency"]
    print()
    print(f"model           : {report['scope']['model_requested']} "
          f"({report['scope']['provider']})")
    print(f"queries         : {totals['n']}  "
          f"failures: {totals['n_parse_failed']}")
    print(f"latency (s)     : mean {latency.get('mean_seconds', 0):.2f}  "
          f"p95 {latency.get('p95_seconds', 0):.2f}")
    for name in FIELDS:
        stat = overall[name]
        rate = stat["agreement_rate"]
        print(f"  {name:<16s}: "
              f"{'n/a' if rate is None else f'{rate * 100:5.1f}%'}  "
              f"({stat['n_agree']}/{stat['n_reference']})")
    med_err = window["median_abs_error_s"]
    overlap = window["window_overlap_rate"]
    med_txt = "n/a" if med_err is None else "%.1fs" % med_err
    overlap_txt = "n/a" if overlap is None else "%.1f%%" % (overlap * 100)
    print("  time window    : median abs err %s, overlap %s" % (med_txt, overlap_txt))
    print(f"wrote {results_dir / 'parser_results.json'}")
    print(f"wrote {results_dir / 'parser_results.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", default=str(DEFAULT_QUERIES),
                    help="benchmark file with the reference parses")
    ap.add_argument("--model", default=None,
                    help="model id override (default: the active configuration)")
    ap.add_argument("--provider", default=None, help="provider override")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N queries")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the parse cache and call the model again")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR),
                    help="directory of the parse cache")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="directory of the report files")
    args = ap.parse_args(argv)

    from backend.app.config import settings

    if args.provider:
        settings.LLM_PROVIDER = args.provider
    if args.model:
        settings.NL_DEFAULT_MODEL = args.model
        settings.OPENCODE_MODEL = args.model
    if args.model or args.provider:
        # Force the configuration used by the parser so the requested model is
        # honoured even when a MongoDB llm_config override exists.
        settings._llm_config_cache = {
            "provider": settings.LLM_PROVIDER,
            "model": args.model or settings.NL_DEFAULT_MODEL,
            "api_key": settings.OPENAI_API_KEY,
        }
        settings._llm_config_checked = True
    cfg = settings.get_active_llm_config()
    model = args.model or cfg.get("model") or settings.OPENCODE_MODEL
    provider = cfg.get("provider") or settings.LLM_PROVIDER

    from backend.app.services.nl_parser import resolve_reference_date

    reference_date = resolve_reference_date()

    queries_path = Path(args.queries)
    queries = load_queries(queries_path)
    n_total = len(queries)
    if args.limit is not None:
        queries = queries[: max(0, args.limit)]

    print(f"provider={provider} model={model} reference_date={reference_date}")
    print(f"parsing {len(queries)} queries with the production parser "
          f"(cache: {Path(args.cache_dir) / (_safe_name(model) + '.json')})...")

    rows, meta = parse_queries(queries, model, args.refresh, Path(args.cache_dir))

    report = build_report(
        rows=rows,
        model=model,
        provider=provider,
        meta=meta,
        queries_path=queries_path,
        reference_date=reference_date,
        n_total_queries=n_total,
    )

    results_dir = Path(args.results_dir)
    write_json_report(report, results_dir / "parser_results.json")
    write_markdown_report(report, results_dir / "parser_results.md")
    print_summary(report, results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
