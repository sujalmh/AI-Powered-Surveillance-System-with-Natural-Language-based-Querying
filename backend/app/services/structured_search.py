"""
Structured (MongoDB) search for detections.

Handles all the complexity of building MongoDB filter pipelines including:
- Location → camera resolution
- Zone → zone_id resolution
- Count-constraint server-side filters (person_count, zone_counts)
- Timestamp normalization & local-time fallback
- VLM-frames fallback for uploaded videos
"""
from __future__ import annotations

from loguru import logger
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import settings
from backend.app.db.mongo import (
    cameras as cameras_col,
    detections as detections_col,
    vlm_frames as vlm_frames_col,
    zones as zones_col,
)
from backend.app.services.retrieval_utils import (
    filter_docs_by_count_constraint,
    flatten_object_captions,
    extract_object_tokens,
    normalize_count_constraint,
    vlm_matches_object_filter,
)

# ---------------------------------------------------------------------------
# Detection quality constants
# ---------------------------------------------------------------------------

#: Only include detections whose confidence meets or exceeds this threshold.
MIN_DETECTION_CONFIDENCE: float = 0.6

# ---------------------------------------------------------------------------
# Location / zone resolution
# ---------------------------------------------------------------------------

def resolve_camera_from_location(location_hint: str) -> Optional[List[int]]:
    """
    Resolve camera IDs from a location name hint via case-insensitive regex
    search on the ``cameras`` collection.
    """
    try:
        cameras = list(cameras_col.find(
            {"location": {"$regex": location_hint, "$options": "i"}},
            {"camera_id": 1, "location": 1, "_id": 0},
        ))
        if cameras:
            camera_ids = [cam["camera_id"] for cam in cameras]
            locations = [cam["location"] for cam in cameras]
            logger.opt(exception=True).info("Location '{}' resolved to cameras: {} ({})", location_hint, camera_ids, locations)
            return camera_ids
        logger.warning("No cameras found for location: {}", location_hint)
        return None
    except Exception:
        logger.error("Error resolving location '{}'", location_hint)
        return None


def resolve_zone_to_ids(
    zone_text: Optional[str],
    camera_id: Any,
) -> List[str]:
    """Resolve a zone name/text to a list of ``zone_id`` values."""
    if not zone_text or not str(zone_text).strip():
        return []
    zone_text = str(zone_text).strip().lower()
    try:
        q: Dict[str, Any] = {
            "$or": [
                {"name": {"$regex": re.escape(zone_text), "$options": "i"}},
                {"zone_id": {"$regex": re.escape(zone_text), "$options": "i"}},
            ]
        }
        if camera_id is not None:
            if isinstance(camera_id, (list, tuple)):
                q["camera_id"] = {"$in": list(camera_id)}
            else:
                q["camera_id"] = int(camera_id)
        zone_docs = list(zones_col.find(q, {"zone_id": 1}))
        zone_ids = [str(z["zone_id"]) for z in zone_docs if z.get("zone_id")]
        if zone_ids:
            logger.opt(exception=True).info("Resolved zone '{}' -> zone_ids: {}", zone_text, zone_ids)
        return zone_ids
    except Exception:
        logger.debug("Zone resolution for '{}' failed", zone_text)
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_mongo_filter(
    parsed_filter: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]], bool, List[str]]:
    """
    Build the MongoDB query filter from *parsed_filter*.

    Returns:
        ``(mongo_filter, count_constraint, is_zero_count, zone_ids)``
    """
    metadata_fields = {"query_type", "action", "zone", "count_constraint"}
    mongo_filter: Dict[str, Any] = {
        k: v for k, v in parsed_filter.items()
        if not k.startswith("__") and k not in metadata_fields
    }

    count_constraint = normalize_count_constraint(parsed_filter.get("count_constraint"))
    is_zero_count = bool(count_constraint and count_constraint.get("eq") == 0)

    # Zone resolution
    zone_text = parsed_filter.get("zone")
    camera_id_filter = mongo_filter.get("camera_id")
    zone_ids: List[str] = []
    if zone_text:
        zone_ids = resolve_zone_to_ids(zone_text, camera_id_filter)
        parsed_filter["__zone_ids"] = zone_ids

    # For zero-count queries, broaden the object filter so docs aren't excluded
    if is_zero_count:
        mongo_filter.pop("objects.object_name", None)
        mongo_filter.pop("objects.color", None)

    # Apply server-side count pre-filters
    _apply_count_prefilter(mongo_filter, parsed_filter, count_constraint, is_zero_count, zone_ids)

    # Location hint → camera IDs
    location_hint = parsed_filter.get("__location_hint")
    if location_hint:
        camera_ids = resolve_camera_from_location(location_hint)
        if camera_ids:
            cam_val = camera_ids[0] if len(camera_ids) == 1 else {"$in": camera_ids}
            mongo_filter["camera_id"] = cam_val
            parsed_filter["camera_id"] = cam_val  # share with semantic search
        # If no cameras found, query will naturally return empty.

    # Normalize object name to lowercase (YOLO stores lowercase)
    if "objects.object_name" in mongo_filter:
        original = mongo_filter["objects.object_name"]
        mongo_filter["objects.object_name"] = str(original).lower()
        if str(original) != mongo_filter["objects.object_name"]:
            logger.opt(exception=True).debug("Normalized object_name: '{}' -> '{}'", original, mongo_filter["objects.object_name"])

    # Normalize timestamp values (strip tz suffixes)
    _normalize_timestamp_filter(mongo_filter)

    return mongo_filter, count_constraint, is_zero_count, zone_ids


def _apply_count_prefilter(
    mongo_filter: Dict[str, Any],
    parsed_filter: Dict[str, Any],
    count_constraint: Optional[Dict[str, int]],
    is_zero_count: bool,
    zone_ids: List[str],
) -> None:
    """Mutate *mongo_filter* with server-side count index conditions."""
    if not count_constraint:
        return

    obj_name = parsed_filter.get("objects.object_name", "")
    obj_color = parsed_filter.get("objects.color")
    is_person = obj_name and str(obj_name).lower() == "person"
    use_zone_counts = bool(zone_ids and is_person and not obj_color and not is_zero_count)

    if is_zero_count and zone_ids:
        or_zero: List[Dict[str, Any]] = []
        for zid in zone_ids:
            or_zero.append({"zone_counts." + zid: 0})
            or_zero.append({"zone_counts." + zid: {"$exists": False}})
        mongo_filter["$or"] = or_zero
    elif is_person and not obj_color and not use_zone_counts:
        _add_person_count_filter(mongo_filter, count_constraint)
    elif use_zone_counts:
        _add_zone_count_filter(mongo_filter, count_constraint, zone_ids)


def _add_person_count_filter(
    mongo_filter: Dict[str, Any],
    constraint: Dict[str, int],
) -> None:
    pc: Dict[str, Any] = {}
    if "eq" in constraint:
        pc = {"person_count": int(constraint["eq"])}
    elif "gt" in constraint:
        pc = {"person_count": {"$gt": int(constraint["gt"])}}
    elif "gte" in constraint:
        pc = {"person_count": {"$gte": int(constraint["gte"])}}
    elif "lt" in constraint:
        pc = {"person_count": {"$lt": int(constraint["lt"])}}
    elif "lte" in constraint:
        pc = {"person_count": {"$lte": int(constraint["lte"])}}
    if pc:
        mongo_filter.update(pc)
        logger.debug("Using person_count index: {}", pc)


def _add_zone_count_filter(
    mongo_filter: Dict[str, Any],
    constraint: Dict[str, int],
    zone_ids: List[str],
) -> None:
    or_conditions: List[Dict[str, Any]] = []
    for zid in zone_ids:
        key = f"zone_counts.{zid}"
        if "eq" in constraint:
            or_conditions.append({key: int(constraint["eq"])})
        elif "gt" in constraint:
            or_conditions.append({key: {"$gt": int(constraint["gt"])}})
        elif "gte" in constraint:
            or_conditions.append({key: {"$gte": int(constraint["gte"])}})
        elif "lt" in constraint:
            or_conditions.append({key: {"$lt": int(constraint["lt"])}})
        elif "lte" in constraint:
            or_conditions.append({key: {"$lte": int(constraint["lte"])}})
    if or_conditions:
        mongo_filter["$or"] = or_conditions
        logger.debug("Using zone_counts filter: $or over {}", zone_ids)


_TZ_RE = re.compile(r"(Z|[+-]\d{2}:\d{2})$")


def _normalize_timestamp_filter(mongo_filter: Dict[str, Any]) -> None:
    """Strip timezone suffixes from timestamp filter values in-place."""
    ts_f = mongo_filter.get("timestamp")
    if not isinstance(ts_f, dict):
        return
    for op in ("$gte", "$lte", "$gt", "$lt"):
        if op in ts_f and isinstance(ts_f[op], str):
            original = ts_f[op]
            ts_f[op] = _TZ_RE.sub("", ts_f[op])
            if original != ts_f[op]:
                logger.debug("Normalized timestamp {}: '{}' -> '{}'", op, original, ts_f[op])
    if ts_f.get("$gte") or ts_f.get("$lte"):
        logger.debug("Timestamp range: {} to {}", ts_f.get("$gte", "N/A"), ts_f.get("$lte", "N/A"))


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def _run_local_time_fallback(
    mongo_filter: Dict[str, Any],
    query_limit: int,
) -> List[Dict[str, Any]]:
    """
    Retry with local-time converted timestamps when the UTC-based query
    returned no results (handles detections stored with local timestamps).
    """
    ts_f = mongo_filter.get("timestamp")
    if not isinstance(ts_f, dict) or "$gte" not in ts_f or "$lte" not in ts_f:
        return []
    try:
        gte_utc = datetime.fromisoformat(ts_f["$gte"])
        lte_utc = datetime.fromisoformat(ts_f["$lte"])
        if gte_utc.tzinfo is None:
            gte_utc = gte_utc.replace(tzinfo=timezone.utc)
        if lte_utc.tzinfo is None:
            lte_utc = lte_utc.replace(tzinfo=timezone.utc)
        local_gte = gte_utc.astimezone().isoformat()
        local_lte = lte_utc.astimezone().isoformat()

        fallback_filter = {k: v for k, v in mongo_filter.items() if k != "timestamp"}
        fallback_filter["timestamp"] = {"$gte": local_gte, "$lte": local_lte}

        logger.info("Local-time fallback: converting UTC to local [{}, {}]", local_gte, local_lte)
        results = list(
            detections_col.find(fallback_filter, {"_id": 0})
            .sort("timestamp", -1)
            .limit(query_limit)
        )
        if results:
            logger.warning(
                "Local-time fallback returned {} detection(s) from a DIFFERENT time window "
                "({} – {}). Results are tagged __local_time_fallback=True.",
                len(results), local_gte, local_lte,
            )
            # Tag every result so callers / the answer generator can signal that
            # the requested time window had no data and these come from a local-
            # time reinterpretation of the same wall-clock range.
            for r in results:
                r["__local_time_fallback"] = True
        return results
    except Exception:
        logger.debug("Local-time fallback failed")
        return []


def _run_diagnostic_query(mongo_filter: Dict[str, Any]) -> None:
    """Best-effort debug diagnostic when structured query returns 0 results."""
    diag_filter: Dict[str, Any] = {}
    if "timestamp" in mongo_filter:
        diag_filter["timestamp"] = mongo_filter["timestamp"]
    if "camera_id" in mongo_filter:
        diag_filter["camera_id"] = mongo_filter["camera_id"]
    if not diag_filter:
        return
    try:
        diag_results = list(
            detections_col.find(
                diag_filter,
                {"_id": 0, "timestamp": 1, "camera_id": 1, "objects.object_name": 1},
            )
            .sort("timestamp", -1)
            .limit(5)
        )
        if diag_results:
            logger.opt(exception=True).warning("DIAGNOSTIC: {} detection(s) without object filter", len(diag_results))
            obj_names = {
                str(obj["object_name"])
                for d in diag_results
                for obj in d.get("objects", [])
                if isinstance(obj, dict) and obj.get("object_name")
            }
            if obj_names:
                logger.warning("DIAGNOSTIC: Object names in DB: {}", sorted(obj_names))
        else:
            logger.warning("DIAGNOSTIC: No detections with time/camera filter only")
    except Exception:
        logger.error("DIAGNOSTIC query failed")


def _vlm_frames_fallback(
    mongo_filter: Dict[str, Any],
    parsed_filter: Dict[str, Any],
    query_limit: int,
) -> List[Dict[str, Any]]:
    """
    Query ``vlm_frames`` for uploaded videos that were indexed but have no
    matching detections.
    """
    vlm_filter: Dict[str, Any] = {}
    if "camera_id" in mongo_filter:
        vlm_filter["camera_id"] = mongo_filter["camera_id"]

    ts_filter = mongo_filter.get("timestamp")
    if isinstance(ts_filter, dict):
        vlm_filter["frame_ts"] = {
            op: ts_filter[op] for op in ("$gte", "$lte") if op in ts_filter
        } or ts_filter
    elif ts_filter is not None:
        vlm_filter["frame_ts"] = ts_filter

    try:
        pipeline = [
            {"$match": vlm_filter},
            {"$sort": {"frame_ts": -1}},
            {"$group": {
                "_id": "$clip_path",
                "camera_id": {"$first": "$camera_id"},
                "clip_path": {"$first": "$clip_path"},
                "clip_url": {"$first": "$clip_url"},
                "first_frame_ts": {"$first": "$frame_ts"},
                "last_frame_ts": {"$last": "$frame_ts"},
                "frame_count": {"$sum": 1},
                "captions": {"$push": "$caption"},
                "object_captions": {"$push": "$object_captions"},
            }},
            {"$limit": query_limit},
        ]
        vlm_results = list(vlm_frames_col.aggregate(pipeline))

        extras: List[Dict[str, Any]] = []
        for vlm_doc in vlm_results:
            flat_caps = flatten_object_captions(vlm_doc.get("object_captions", []))
            if not vlm_matches_object_filter(flat_caps, parsed_filter):
                continue
            objects_set = set(extract_object_tokens(flat_caps))

            # Compute real duration from first/last frame timestamps.
            # The old code stored frame_count (an integer frame count) in
            # duration_seconds, which was wrong and inflated the relevance score.
            first_ts_str = vlm_doc.get("first_frame_ts")
            last_ts_str = vlm_doc.get("last_frame_ts")
            try:
                from backend.app.services.retrieval_utils import parse_ts as _pts
                duration_sec = max(0, int(
                    (_pts(last_ts_str) - _pts(first_ts_str)).total_seconds()
                )) if first_ts_str and last_ts_str else 0
            except Exception:
                duration_sec = 0

            # VLM-frame objects have no YOLO confidence score.  We assign a
            # moderate placeholder (0.65) so they are not silently discarded by
            # the confidence threshold filter (0.6) while still scoring lower
            # than high-confidence structured detections.
            VLM_CONFIDENCE_PLACEHOLDER = 0.65
            extras.append({
                "camera_id": vlm_doc.get("camera_id"),
                "clip_path": vlm_doc.get("clip_path"),
                "clip_url": vlm_doc.get("clip_url"),
                "timestamp": first_ts_str,
                "start": first_ts_str,
                "end": last_ts_str,
                "duration_seconds": duration_sec,
                "object_name": ", ".join(sorted(objects_set)) if objects_set else "unknown",
                "objects": [
                    {"object_name": obj, "confidence": VLM_CONFIDENCE_PLACEHOLDER}
                    for obj in sorted(objects_set)
                ],
                "source": "vlm_fallback",
                # Use a conservative struct score – not 0.5 which would
                # inflate these above genuine low-confidence detections.
                "score_struct": 0.35,
                "score_semantic": 0.0,
            })
        logger.opt(exception=True).info("VLM-frames fallback added {} clips", len(extras))
        return extras
    except Exception:
        logger.error("VLM-frames fallback error")
        return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute_structured_query(
    parsed_filter: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Execute a structured MongoDB query against the ``detections`` collection.

    Handles timestamp normalization, count constraints, location/zone
    resolution, local-time fallback, and VLM-frames fallback automatically.
    """
    mongo_filter, count_constraint, is_zero_count, zone_ids = _build_mongo_filter(parsed_filter)

    logger.opt(exception=True).debug("MongoDB filter: {}", mongo_filter)

    # Compute effective limit (capped to avoid excessive results)
    query_limit = limit
    if parsed_filter.get("__ask_color"):
        query_limit = max(limit, min(3 * limit, 100))
    if count_constraint:
        query_limit = max(query_limit, min(3 * limit, 100))

    try:
        results = list(
            detections_col.find(mongo_filter, {"_id": 0})
            .sort("timestamp", -1)
            .limit(query_limit)
        )
        logger.debug("Initial query returned {} detection(s)", len(results))

        # Local-time fallback
        if not results and "timestamp" in mongo_filter:
            fallback = _run_local_time_fallback(mongo_filter, query_limit)
            if fallback:
                results = fallback

        # Debug diagnostic
        if settings.DEBUG and not results:
            _run_diagnostic_query(mongo_filter)

        # In-memory count-constraint enforcement
        results = filter_docs_by_count_constraint(results, parsed_filter)
        if not results and not is_zero_count:
            logger.debug("After count-constraint filtering: 0 results")

        # Confidence threshold + exact object class filter.
        # For every returned doc keep only objects that (a) meet the
        # confidence floor and (b) match the requested class exactly.
        # Docs where no object survives the filter are excluded entirely.
        if not is_zero_count:
            requested_class = parsed_filter.get("objects.object_name")
            filtered_by_conf: List[Dict[str, Any]] = []
            for doc in results:
                obj_list = doc.get("objects") or []
                matching = [
                    obj for obj in obj_list
                    if isinstance(obj, dict)
                    and float(obj.get("confidence", 0.0)) >= MIN_DETECTION_CONFIDENCE
                    and (
                        not requested_class
                        or obj.get("object_name", "").lower() == str(requested_class).lower()
                    )
                ]
                if matching:
                    doc = dict(doc)
                    doc["objects"] = matching
                    doc["matched_object_count"] = len(matching)
                    filtered_by_conf.append(doc)
            if results:  # only log if we had something to filter
                logger.info(
                    "Conf/class filter (conf>=%.2f, class=%s): %d -> %d docs",
                    MIN_DETECTION_CONFIDENCE,
                    requested_class or "any",
                    len(results),
                    len(filtered_by_conf),
                )
            results = filtered_by_conf

        # VLM-frames fallback (disabled for count-constraint queries)
        if count_constraint is None and len(results) < 5:
            logger.info("Few/no detections ({}), checking vlm_frames...", len(results))
            results.extend(_vlm_frames_fallback(mongo_filter, parsed_filter, query_limit))

        return results

    except Exception:
        logger.error("Structured query error")
        return []
