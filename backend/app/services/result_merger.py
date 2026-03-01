"""
Result merging, scoring, and ranking for the hybrid retrieval pipeline.

Combines structured detection tracks with semantic FAISS results, applying
adaptive weighting, recency boosts, semantic penalties, and MMR diversity
re-ranking.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import settings
from backend.app.services.retrieval_utils import (
    SEM_NO_STRUCTURED_THRESH,
    SEM_PENALTY_LIGHT,
    SEM_PENALTY_MED,
    SEM_PENALTY_NO_STRUCTURED,
    SEM_PENALTY_STRONG,
    SEM_RAW_THRESH_HIGH,
    SEM_RAW_THRESH_LOW,
    SEM_RAW_THRESH_MED,
    aggregate_objects,
    count_matches_constraint,
    normalize_count_constraint,
    normalize_scores,
    object_matches,
    parse_ts,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Track merging helpers
# ======================================================================

def compute_adaptive_gaps(
    structured_results: List[Dict[str, Any]],
    default_max_gap: float = 3.0,
    default_join_gap: float = 10.0,
) -> Tuple[float, float]:
    """
    Compute ``max_gap`` and ``join_gap`` from the timestamp distribution of
    structured results.  Bounded to [1, 30] seconds.
    """
    if not getattr(settings, "ENABLE_ADAPTIVE_GAPS", True) or not structured_results:
        return default_max_gap, default_join_gap
    try:
        timestamps: List[datetime] = []
        for doc in structured_results:
            ts = doc.get("timestamp")
            if ts:
                timestamps.append(parse_ts(ts))
        if len(timestamps) < 2:
            return default_max_gap, default_join_gap
        timestamps.sort()
        gaps = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        gaps = [g for g in gaps if 0 < g < 60]
        if not gaps:
            return default_max_gap, default_join_gap
        median_gap = float(sorted(gaps)[len(gaps) // 2])
        max_gap = max(1.0, min(30.0, median_gap * 2.0))
        join_gap = max(3.0, min(30.0, max_gap * 3.0))
        return max_gap, join_gap
    except Exception:
        logger.debug("Adaptive gaps computation failed", exc_info=True)
        return default_max_gap, default_join_gap


def merge_structured_tracks(
    results: List[Dict[str, Any]],
    parsed_filter: Dict[str, Any],
    max_gap: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Merge contiguous detections belonging to the same track into temporal
    segments.  Returns a flat list of merged interval dicts.
    """
    per_track: Dict[Tuple, List[datetime]] = {}
    meta: Dict[Tuple, Dict[str, Any]] = {}
    objects_by_track: Dict[Tuple, List[Dict[str, Any]]] = {}

    for doc in results:
        cam = doc.get("camera_id")
        ts = parse_ts(doc.get("timestamp", ""))
        for obj in doc.get("objects", []):
            if not object_matches(obj, parsed_filter):
                continue
            tid = obj.get("track_id", -1)
            if tid is None or (isinstance(tid, (int, float)) and tid < 0):
                tid = f"notrak_{cam}_{ts.isoformat()}_{id(obj)}"
            key = (cam, tid)
            per_track.setdefault(key, []).append(ts)
            objects_by_track.setdefault(key, []).append(obj)
            if key not in meta:
                meta[key] = {
                    "camera_id": cam,
                    "track_id": tid,
                    "object_name": obj.get("object_name"),
                    "color": obj.get("color"),
                }

    merged: List[Dict[str, Any]] = []
    for key, times in per_track.items():
        times.sort()
        if not times:
            continue
        track_objects = objects_by_track.get(key, [])
        agg_objects = aggregate_objects(track_objects)

        seg_start_idx = 0
        start = prev = times[0]
        for i, t in enumerate(times[1:], start=1):
            gap = (t - prev).total_seconds()
            if gap <= max_gap:
                prev = t
            else:
                m = dict(meta[key])
                m["start"] = start.isoformat()
                m["end"] = prev.isoformat()
                m["duration_seconds"] = max(0, int((prev - start).total_seconds()))
                m["objects"] = agg_objects
                m["matched_timestamps"] = [t_.isoformat() for t_ in times[seg_start_idx:i]]
                merged.append(m)
                start = prev = t
                seg_start_idx = i

        # Close last segment
        m = dict(meta[key])
        m["start"] = start.isoformat()
        m["end"] = prev.isoformat()
        m["duration_seconds"] = max(0, int((prev - start).total_seconds()))
        m["objects"] = agg_objects
        m["matched_timestamps"] = [t_.isoformat() for t_ in times[seg_start_idx:]]
        merged.append(m)

    merged.sort(key=lambda x: x.get("start", ""), reverse=True)
    return merged


def coalesce_tracks(
    merged: List[Dict[str, Any]],
    join_gap: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Coalesce segments across different ``track_id`` values when they are
    temporally adjacent.  Groups by ``(camera_id, object_name, color)``.
    """
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for seg in merged:
        key = (seg.get("camera_id"), seg.get("object_name"), seg.get("color"))
        groups.setdefault(key, []).append(seg)

    output: List[Dict[str, Any]] = []
    for (cam, obj, col), segs in groups.items():
        segs_sorted = sorted(segs, key=lambda s: s.get("start", ""))
        if not segs_sorted:
            continue

        cur_start = parse_ts(segs_sorted[0]["start"])
        cur_end = parse_ts(segs_sorted[0]["end"])
        all_objects = list(segs_sorted[0].get("objects", []))
        all_timestamps = list(segs_sorted[0].get("matched_timestamps", []))

        for seg in segs_sorted[1:]:
            s = parse_ts(seg["start"])
            e = parse_ts(seg["end"])
            gap = (s - cur_end).total_seconds()
            if gap <= join_gap:
                if e > cur_end:
                    cur_end = e
                all_objects.extend(seg.get("objects", []))
                all_timestamps.extend(seg.get("matched_timestamps", []))
            else:
                output.append({
                    "camera_id": cam,
                    "object_name": obj,
                    "color": col,
                    "start": cur_start.isoformat(),
                    "end": cur_end.isoformat(),
                    "duration_seconds": max(0, int((cur_end - cur_start).total_seconds())),
                    "objects": aggregate_objects(all_objects),
                    "matched_timestamps": all_timestamps,
                })
                cur_start, cur_end = s, e
                all_objects = list(seg.get("objects", []))
                all_timestamps = list(seg.get("matched_timestamps", []))

        output.append({
            "camera_id": cam,
            "object_name": obj,
            "color": col,
            "start": cur_start.isoformat(),
            "end": cur_end.isoformat(),
            "duration_seconds": max(0, int((cur_end - cur_start).total_seconds())),
            "objects": aggregate_objects(all_objects),
            "matched_timestamps": all_timestamps,
        })

    output.sort(key=lambda x: x.get("start", ""), reverse=True)
    return output


def build_count_constrained_segments(
    docs: List[Dict[str, Any]],
    parsed_filter: Dict[str, Any],
    join_gap: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Build temporal segments directly from per-detection docs that satisfy
    count constraints (e.g. ``>=2 persons``, ``==0 persons``).
    """
    from backend.app.services.retrieval_utils import (
        count_matching_objects_in_doc,
        filter_docs_by_count_constraint,
    )

    constraint = normalize_count_constraint(parsed_filter.get("count_constraint"))
    if not constraint:
        return []

    filtered = filter_docs_by_count_constraint(docs, parsed_filter)
    by_cam: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for d in filtered:
        by_cam[d.get("camera_id")].append(d)

    segments: List[Dict[str, Any]] = []
    for cam, cam_docs in by_cam.items():
        cam_docs_sorted = sorted(cam_docs, key=lambda x: x.get("timestamp", ""))

        seg_start: Optional[datetime] = None
        seg_end: Optional[datetime] = None
        seg_peak = 0
        seg_objects: List[Dict[str, Any]] = []
        seg_timestamps: List[str] = []

        def _emit() -> None:
            nonlocal seg_start, seg_end, seg_peak, seg_objects, seg_timestamps
            if seg_start is None or seg_end is None:
                return
            segments.append({
                "camera_id": cam,
                "object_name": parsed_filter.get("objects.object_name"),
                "color": parsed_filter.get("objects.color"),
                "start": seg_start.isoformat(),
                "end": seg_end.isoformat(),
                "duration_seconds": max(0, int((seg_end - seg_start).total_seconds())),
                "objects": aggregate_objects(seg_objects),
                "match_count_peak": int(seg_peak),
                "count_constraint": constraint,
                "matched_timestamps": list(seg_timestamps),
            })

        for d in cam_docs_sorted:
            ts = parse_ts(d.get("timestamp", ""))
            cnt = int(d.get("matched_object_count", count_matching_objects_in_doc(d, parsed_filter)))
            matched_objs = [
                o for o in (d.get("objects") or [])
                if isinstance(o, dict) and object_matches(o, parsed_filter)
            ]
            if seg_start is None:
                seg_start, seg_end = ts, ts
                seg_peak = cnt
                seg_objects = matched_objs.copy()
                seg_timestamps = [ts.isoformat()]
                continue
            gap = (ts - seg_end).total_seconds() if seg_end else 0
            if gap <= join_gap:
                seg_end = ts
                seg_peak = max(seg_peak, cnt)
                seg_objects.extend(matched_objs)
                seg_timestamps.append(ts.isoformat())
            else:
                _emit()
                seg_start, seg_end = ts, ts
                seg_peak = cnt
                seg_objects = matched_objs.copy()
                seg_timestamps = [ts.isoformat()]
        _emit()

    segments.sort(key=lambda x: x.get("start", ""), reverse=True)
    return segments


# ======================================================================
# Alpha / weighting
# ======================================================================

def compute_adaptive_alpha(
    parsed_filter: Dict[str, Any],
    intent: str,
    has_count_constraint: bool,
    has_action: bool,
    n_structured: int,
    n_semantic: int,
) -> float:
    """
    Compute *alpha* (structured weight) adaptively from query and result
    characteristics.  Falls back to a fixed intent-based alpha when disabled.
    """
    if not getattr(settings, "ENABLE_ADAPTIVE_ALPHA", True):
        return _fixed_alpha(intent, has_count_constraint, has_action)
    try:
        if has_count_constraint:
            return 1.0
        if has_action:
            return 0.1

        filter_count = sum(
            1 for k in ("camera_id", "objects.object_name", "objects.color", "zone", "location_hint")
            if parsed_filter.get(k)
        )
        if parsed_filter.get("timestamp"):
            filter_count += 1
        complexity = min(1.0, filter_count / 4.0)

        base = {"count": 0.9, "track": 0.5, "find": 0.7}.get(intent, 0.7)
        struct_ratio = (
            n_structured / max(1, n_structured + n_semantic)
            if (n_structured or n_semantic) else 0.5
        )
        alpha = base * 0.6 + 0.2 * complexity + 0.2 * struct_ratio
        return max(0.1, min(0.95, alpha))
    except Exception:
        logger.debug("Adaptive alpha failed, using fixed", exc_info=True)
        return _fixed_alpha(intent, has_count_constraint, has_action)


def _fixed_alpha(intent: str, has_count_constraint: bool, has_action: bool) -> float:
    if has_count_constraint:
        return 1.0
    if has_action:
        return 0.1
    return {"count": 0.9, "track": 0.5}.get(intent, 0.7)


# ======================================================================
# Semantic time-range overlap
# ======================================================================

def _semantic_time_range(sem: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime]]:
    frames = sem.get("frames") or []
    if not frames:
        return None, None
    try:
        times = [parse_ts(f["frame_ts"]) for f in frames if f.get("frame_ts")]
        return (min(times), max(times)) if times else (None, None)
    except Exception:
        return None, None


def _ranges_overlap(
    a_start: datetime, a_end: datetime,
    b_start_str: str, b_end_str: str,
) -> bool:
    try:
        b_start = parse_ts(b_start_str)
        b_end = parse_ts(b_end_str)
        return a_start <= b_end and b_start <= a_end
    except Exception:
        return False


# ======================================================================
# MMR diversity re-ranking
# ======================================================================

def apply_mmr_diversity(
    results: List[Dict[str, Any]],
    lambda_: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Re-rank with Maximal Marginal Relevance: balance relevance and diversity.
    Penalizes results that are similar in ``(camera_id, time_bucket)`` to
    spread final results across cameras and time windows.
    """
    if len(results) <= 2:
        return results
    try:
        lambda_ = max(0.0, min(1.0, lambda_))
        selected: List[Dict[str, Any]] = []
        remaining = list(results)

        def _time_bucket(r: Dict[str, Any]) -> int:
            ts_str = r.get("start") or r.get("end") or ""
            try:
                return int(parse_ts(ts_str).timestamp() / 3600) if ts_str else 0
            except Exception:
                return 0

        while remaining:
            best_idx, best_mmr = -1, -1.0
            for i, r in enumerate(remaining):
                rel = r.get("score", 0.0)
                key_r = (r.get("camera_id"), _time_bucket(r))
                max_sim = 0.0
                for s in selected:
                    key_s = (s.get("camera_id"), _time_bucket(s))
                    max_sim = max(max_sim, 1.0 if key_r == key_s else 0.0)
                mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(remaining.pop(best_idx))
        return selected
    except Exception:
        logger.debug("MMR diversity failed, returning original order", exc_info=True)
        return results


# ======================================================================
# Main merge entry point
# ======================================================================

def merge_results(
    structured: List[Dict[str, Any]],
    semantic: List[Dict[str, Any]],
    parsed_filter: Dict[str, Any],
    intent: str,
    default_max_gap: float = 3.0,
    default_join_gap: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Intelligently merge structured detection results with semantic FAISS
    results into a single ranked list.
    """
    count_constraint = normalize_count_constraint(parsed_filter.get("count_constraint"))
    has_count_constraint = count_constraint is not None
    has_action = parsed_filter.get("action") is not None

    # --- Build combined tracks ------------------------------------------
    if has_count_constraint:
        join_gap = default_join_gap
        combined_tracks = build_count_constrained_segments(structured, parsed_filter, join_gap)
        logger.debug("Count-constrained segments: %d", len(combined_tracks))
    else:
        max_gap, join_gap = compute_adaptive_gaps(structured, default_max_gap, default_join_gap)
        try:
            merged_tracks = merge_structured_tracks(structured, parsed_filter, max_gap)
            logger.debug("Merged structured tracks: %d", len(merged_tracks))
            combined_tracks = coalesce_tracks(merged_tracks, join_gap)
            logger.debug("Coalesced tracks: %d", len(combined_tracks))
        except Exception:
            logger.error("Track merging failed", exc_info=True)
            combined_tracks = []

    # --- Scoring --------------------------------------------------------
    alpha = compute_adaptive_alpha(
        parsed_filter, intent, has_count_constraint, has_action,
        n_structured=len(combined_tracks), n_semantic=len(semantic),
    )

    results_map: Dict[str, Dict[str, Any]] = {}

    # Structured tracks
    struct_scores = [t.get("duration_seconds", 0) for t in combined_tracks]
    struct_norm = normalize_scores(struct_scores)
    for track, norm_score in zip(combined_tracks, struct_norm):
        key = f"{track.get('camera_id')}_{track.get('start')}_{track.get('end')}"
        results_map[key] = {
            **track,
            "score_struct": norm_score,
            "score_semantic": 0.0,
            "source": "structured",
        }

    # Semantic results (disabled for strict count constraints)
    if not has_count_constraint:
        enable_temporal = getattr(settings, "ENABLE_TEMPORAL_OVERLAP", True)
        for sem in semantic:
            clip_url = sem.get("clip_url")
            matched = False
            # Direct clip_url match
            for _key, result in results_map.items():
                if result.get("clip_url") == clip_url:
                    result["score_semantic"] = sem.get("score_norm", 0.0)
                    result["source"] = "hybrid"
                    result["frames"] = sem.get("frames", [])
                    matched = True
                    break
            # Temporal overlap match
            if not matched and enable_temporal:
                sem_start, sem_end = _semantic_time_range(sem)
                if sem_start is not None and sem_end is not None:
                    for _key, result in results_map.items():
                        r_start, r_end = result.get("start"), result.get("end")
                        if r_start and r_end and _ranges_overlap(sem_start, sem_end, r_start, r_end):
                            result["score_semantic"] = max(
                                result.get("score_semantic", 0.0),
                                sem.get("score_norm", 0.0),
                            )
                            result["source"] = "hybrid"
                            if not result.get("frames"):
                                result["frames"] = sem.get("frames", [])
                            matched = True
                            break
            if not matched:
                key = f"sem_{sem.get('camera_id')}_{sem.get('clip_path', '')}"
                results_map[key] = {
                    "camera_id": sem.get("camera_id"),
                    "clip_url": sem.get("clip_url"),
                    "clip_path": sem.get("clip_path"),
                    "score_struct": 0.0,
                    "score_semantic": sem.get("score_norm", 0.0),
                    "score_raw_semantic": sem.get("score_raw", 0.0),
                    "source": "semantic",
                    "frames": sem.get("frames", []),
                }

    # --- Final scoring with recency boost & semantic penalties ----------
    now = datetime.now()
    halflife_hours = getattr(settings, "RECENCY_HALFLIFE_HOURS", 24.0) or 24.0
    enable_recency = getattr(settings, "ENABLE_RECENCY_BOOST", True)

    results: List[Dict[str, Any]] = []
    for result in results_map.values():
        final_score = alpha * result["score_struct"] + (1 - alpha) * result["score_semantic"]

        # Semantic-only penalty
        if result.get("source") == "semantic":
            raw_sem = result.get("score_raw_semantic", 0.0)
            if raw_sem < SEM_RAW_THRESH_LOW:
                final_score *= SEM_PENALTY_STRONG
            elif raw_sem < SEM_RAW_THRESH_MED:
                final_score *= SEM_PENALTY_MED
            elif raw_sem < SEM_RAW_THRESH_HIGH:
                final_score *= SEM_PENALTY_LIGHT
            if len(combined_tracks) == 0 and raw_sem < SEM_NO_STRUCTURED_THRESH:
                final_score *= SEM_PENALTY_NO_STRUCTURED

        # Recency boost
        if enable_recency and halflife_hours > 0:
            try:
                ts_str = result.get("start") or result.get("end") or result.get("timestamp")
                if ts_str:
                    age_hours = (now - parse_ts(ts_str)).total_seconds() / 3600.0
                    recency_boost = math.exp(-age_hours * (0.693 / halflife_hours))
                    final_score *= 1.0 + 0.5 * recency_boost
            except Exception:
                pass

        result["score"] = final_score
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    # MMR diversity
    if getattr(settings, "ENABLE_RESULT_DIVERSITY", True):
        mmr_lambda = getattr(settings, "MMR_DIVERSITY_LAMBDA", 0.3)
        results = apply_mmr_diversity(results, lambda_=mmr_lambda)

    return results
