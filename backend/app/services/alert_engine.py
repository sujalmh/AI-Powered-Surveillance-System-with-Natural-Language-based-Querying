from __future__ import annotations

import logging
import time
import zoneinfo
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.app.db.mongo import (
    alerts as alerts_col,
    alert_logs as alert_logs_col,
    detections as detections_col,
    zones as zones_col,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Zone Capacity Cache
# ---------------------------------------------------------------------

_ZONE_CAP_CACHE: Dict[Tuple[int, str], Tuple[Optional[int], float]] = {}
_ZONE_CAP_TTL = 60.0


def _get_zone_capacity(camera_id: int, zone_id: str) -> Optional[int]:
    now = time.time()
    key = (camera_id, zone_id)
    entry = _ZONE_CAP_CACHE.get(key)

    if entry and (now - entry[1]) < _ZONE_CAP_TTL:
        return entry[0]

    try:
        zone_doc = zones_col.find_one(
            {"camera_id": camera_id, "zone_id": zone_id}, {"capacity": 1}
        )
        cap = int(zone_doc["capacity"]) if zone_doc and zone_doc.get("capacity") else None
        _ZONE_CAP_CACHE[key] = (cap, now)
        return cap
    except Exception as e:
        _log.warning(
            "Failed to get zone capacity for camera_id=%s zone_id=%s: %s",
            camera_id,
            zone_id,
            e,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------
# Rule Cache + Runtime State
# ---------------------------------------------------------------------

@dataclass
class RuleCache:
    rules: List[Dict[str, Any]]
    last_refresh: float
    ttl_sec: float = 5.0


_RULE_CACHE = RuleCache(rules=[], last_refresh=0.0)
_LAST_FIRED: Dict[Tuple[str, int], float] = {}
_PER_CAMERA: Dict[int, Dict[str, Any]] = defaultdict(dict)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_active_rules() -> List[Dict[str, Any]]:
    now = time.time()
    if now - _RULE_CACHE.last_refresh > _RULE_CACHE.ttl_sec:
        _RULE_CACHE.rules = list(alerts_col.find({"enabled": True}))
        _RULE_CACHE.last_refresh = now
    return _RULE_CACHE.rules


# ---------------------------------------------------------------------
# Time Window Utilities (shared with on-demand rule evaluation)
# ---------------------------------------------------------------------


def parse_time_window(rule_time: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Build (start_iso, end_iso) from rule time spec (last_minutes, last_hours, or from/to)."""
    if not rule_time:
        return None, None
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    if "last_minutes" in rule_time:
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=int(rule_time["last_minutes"]))
        start_iso, end_iso = start.isoformat(), end.isoformat()
    elif "last_hours" in rule_time:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=int(rule_time["last_hours"]))
        start_iso, end_iso = start.isoformat(), end.isoformat()
    else:
        if "from" in rule_time:
            try:
                start_iso = datetime.fromisoformat(rule_time["from"]).isoformat()
            except Exception:
                start_iso = rule_time["from"]
        if "to" in rule_time:
            try:
                end_iso = datetime.fromisoformat(rule_time["to"]).isoformat()
            except Exception:
                end_iso = rule_time["to"]
    return start_iso, end_iso


def build_detection_query_from_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Build MongoDB query for detections from an alert rule spec."""
    q: Dict[str, Any] = {}
    cameras = rule.get("cameras")
    if cameras:
        q["camera_id"] = {"$in": cameras}

    start_iso, end_iso = parse_time_window(rule.get("time"))
    if start_iso or end_iso:
        q["timestamp"] = {}
        if start_iso:
            q["timestamp"]["$gte"] = start_iso
        if end_iso:
            q["timestamp"]["$lte"] = end_iso

    tod = rule.get("time_of_day")
    if tod and isinstance(tod, dict) and "start" in tod and "end" in tod:
        tz = tod.get("tz", "UTC")
        try:
            zoneinfo.ZoneInfo(tz)
        except Exception:
            tz = "UTC"
        dt_expr = {
            "$dateToString": {
                "format": "%H:%M",
                "date": {"$toDate": "$timestamp"},
                "timezone": tz,
            }
        }

        def parse_time_to_minutes(t_str: str) -> int:
            try:
                parts = str(t_str).split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return 0

        start_m = parse_time_to_minutes(tod["start"])
        end_m = parse_time_to_minutes(tod["end"])
        start_fmt = f"{start_m // 60:02d}:{start_m % 60:02d}"
        end_fmt = f"{end_m // 60:02d}:{end_m % 60:02d}"
        if start_m <= end_m:
            range_cond = {"$and": [{"$gte": ["$$time", start_fmt]}, {"$lte": ["$$time", end_fmt]}]}
        else:
            range_cond = {"$or": [{"$gte": ["$$time", start_fmt]}, {"$lte": ["$$time", end_fmt]}]}
        tod_cond = {"$let": {"vars": {"time": dt_expr}, "in": range_cond}}
        if "$expr" not in q:
            q["$expr"] = tod_cond
        else:
            q["$expr"] = {"$and": [q["$expr"], tod_cond]}

    objects = rule.get("objects")
    if objects and isinstance(objects, list) and len(objects) > 0:
        name = objects[0].get("name")
        if name:
            q["objects.object_name"] = name
    color = rule.get("color")
    if color:
        q["objects.color"] = color
    return q


def matches_count_condition(count: int, cond: Optional[Dict[str, Any]]) -> bool:
    """Return True if count satisfies the condition (e.g. {">=": 1})."""
    if not cond:
        return count >= 1
    for op, val in cond.items():
        try:
            v = float(val)
        except Exception:
            continue
        if op == "==" and not (count == v):
            return False
        elif op == ">=" and not (count >= v):
            return False
        elif op == "<=" and not (count <= v):
            return False
        elif op == ">" and not (count > v):
            return False
        elif op == "<" and not (count < v):
            return False
    return True


def evaluate_rule(rule_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    On-demand evaluation of an alert rule against recent detections.
    Returns an alert_log dict if triggered, otherwise None.
    """
    if not rule_doc.get("enabled", True):
        return None
    rule = rule_doc.get("rule", {}) or {}
    query = build_detection_query_from_rule(rule)
    if "timestamp" not in query:
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=30)
        query["timestamp"] = {"$gte": start.isoformat(), "$lte": end.isoformat()}

    docs = list(detections_col.find(query).sort("timestamp", -1).limit(500))
    want_name = query.get("objects.object_name")
    want_color = query.get("objects.color")
    matched_count = 0
    for d in docs:
        for obj in d.get("objects", []):
            if want_name and obj.get("object_name") != want_name:
                continue
            if want_color and obj.get("color") != want_color:
                continue
            matched_count += 1

    if matches_count_condition(matched_count, rule.get("count")):
        return {
            "alert_id": rule_doc["_id"],
            "triggered_at": _now_utc_iso(),
            "camera_id": docs[0]["camera_id"] if docs else None,
            "event_id": None,
            "detection_ids": [str(d["_id"]) for d in docs[:10]],
            "snapshot": None,
            "clip": None,
            "message": f"Rule '{rule_doc.get('name')}' matched count={matched_count}",
        }
    return None


def _parse_local_time_window(rule_time: Optional[Dict[str, Any]]) -> Optional[Tuple[dtime, dtime, str]]:
    if not rule_time:
        return None
    try:
        s_h, s_m = map(int, rule_time["start"].split(":"))
        e_h, e_m = map(int, rule_time["end"].split(":"))
        return (dtime(hour=s_h, minute=s_m), dtime(hour=e_h, minute=e_m), rule_time.get("tz", "UTC"))
    except Exception:
        return None


def _in_local_window(win: Tuple[dtime, dtime, str], now_dt: datetime) -> bool:
    start_t, end_t, _ = win
    now_t = now_dt.time()

    if start_t <= end_t:
        return start_t <= now_t <= end_t
    return now_t >= start_t or now_t <= end_t


# ---------------------------------------------------------------------
# Object Helpers
# ---------------------------------------------------------------------

def _person_boxes(objects: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
    boxes = []
    for o in objects:
        if str(o.get("object_name", "")).lower() == "person":
            bb = o.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                try:
                    x1, y1, x2, y2 = map(int, bb)
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
                except Exception:
                    pass
    return boxes


def _motion_energy(prev_gray: Optional[np.ndarray], frame_bgr: Optional[np.ndarray]):
    if frame_bgr is None:
        return prev_gray, 0.0

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

    if prev_gray is None or prev_gray.shape != gray.shape:
        return gray, 0.0

    diff = cv2.absdiff(gray, prev_gray)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    energy = float(th.mean() / 255.0)

    return gray, energy


def _fight_heuristic(person_boxes: List[Tuple[int, int, int, int]], energy_hist: Deque[float]) -> bool:
    if len(person_boxes) < 2:
        return False

    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in person_boxes]

    close = any(
        ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 < 24
        for i, (cx1, cy1) in enumerate(centers)
        for cx2, cy2 in centers[i + 1:]
    )

    if not close or not energy_hist:
        return False

    avg_energy = sum(list(energy_hist)[-10:]) / max(1, len(energy_hist))
    return avg_energy >= 0.03


def _cooldown_ok(rule_id: str, camera_id: int, cooldown_sec: float) -> bool:
    key = (rule_id, camera_id)
    now = time.time()
    last = _LAST_FIRED.get(key, 0.0)

    if now - last >= cooldown_sec:
        _LAST_FIRED[key] = now
        return True
    return False


def _insert_alert_log(rule_doc: Dict[str, Any], camera_id: int, payload: Dict[str, Any]) -> None:
    alert_logs_col.insert_one(
        {
            "alert_id": rule_doc["_id"],
            "triggered_at": _now_utc_iso(),
            "camera_id": camera_id,
            "snapshot": payload.get("snapshot"),
            "message": payload.get("message", ""),
            "payload": payload,
            "severity": rule_doc.get("severity", "info"),
        }
    )


# ---------------------------------------------------------------------
# Main Evaluation Engine
# ---------------------------------------------------------------------

def evaluate_realtime(
    camera_id: int,
    timestamp_iso: str,
    objects: List[Dict[str, Any]],
    frame_bgr: Optional[np.ndarray] = None,
    snapshot_url: Optional[str] = None,
    zone_counts: Optional[Dict[str, int]] = None,
) -> None:
    rules = _load_active_rules()
    if not rules:
        return

    st = _PER_CAMERA[camera_id]
    energy_hist = st.setdefault("energy_hist", deque(maxlen=60))
    has_fight_rule = any(rd.get("rule", {}).get("behavior") == "fight" for rd in rules)
    if has_fight_rule:
        prev_gray = st.get("prev_gray")
        cur_gray, energy = _motion_energy(prev_gray, frame_bgr)
        st["prev_gray"] = cur_gray
        energy_hist.append(energy)

    presence = defaultdict(int)
    for o in objects:
        presence[str(o.get("object_name", "")).lower()] += 1

    last_presence = st.get("presence_last", {})
    st["presence_last"] = dict(presence)

    try:
        now_dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
    except Exception:
        now_dt = datetime.now(timezone.utc)

    for rd in rules:
        rid = str(rd["_id"])
        rule = rd.get("rule", {})
        cooldown = float(rd.get("cooldown_sec", 60))
        event_type = str(rule.get("event", "")).lower()

        want_name = None
        objs = rule.get("objects") or []
        if objs:
            want_name = objs[0].get("name")

        area = rule.get("area") or {}
        zone_id = area.get("zone_id")
        
        # ---------------- Time of Day check ----------------
        tod = rule.get("time_of_day")
        if tod:
            win = _parse_local_time_window(tod)
            if not win:
                _log.warning("Skipping rule %s due to malformed time_of_day: %r", rid, tod)
                continue
            
            # Apply the requested timezone to the UTC `now_dt`
            try:
                tz_name = win[2]
                local_dt = now_dt.astimezone(zoneinfo.ZoneInfo(tz_name))
            except Exception as e:
                _log.warning("Timezone conversion failed for rule %s (tz=%s): %s", rid, win[2], e)
                continue

            if not _in_local_window(win, local_dt):
                continue

        # ---------------- Crowd Density ----------------
        if event_type == "crowd_density" and want_name:
            cnt = presence.get(want_name.lower(), 0)
            min_count = float(rule.get("min_count", 5))

            if cnt >= min_count and _cooldown_ok(rid, camera_id, cooldown):
                _insert_alert_log(
                    rd,
                    camera_id,
                    {
                        "message": f"High crowd density: {cnt} {want_name}",
                        "snapshot": snapshot_url,
                        "count": cnt,
                    },
                )
            continue

        # ---------------- Loitering ----------------
        if event_type == "loitering" and want_name:
            max_duration = float(rule.get("duration_sec", 60))
            triggered = False

            for o in objects:
                if str(o.get("object_name", "")).lower() == want_name.lower():
                    th = o.get("track_history")
                    if isinstance(th, dict):
                        try:
                            fs = datetime.fromisoformat(th["first_seen"].replace("Z", "+00:00"))
                            ls = datetime.fromisoformat(th["last_seen"].replace("Z", "+00:00"))
                            if (ls - fs).total_seconds() >= max_duration:
                                triggered = True
                        except Exception:
                            pass

            if triggered and _cooldown_ok(rid, camera_id, cooldown):
                _insert_alert_log(
                    rd,
                    camera_id,
                    {
                        "message": "Suspicious loitering detected",
                        "snapshot": snapshot_url,
                    },
                )
            continue

        # ---------------- Fight ----------------
        if rule.get("behavior") == "fight":
            if _fight_heuristic(_person_boxes(objects), energy_hist) and _cooldown_ok(
                rid, camera_id, cooldown
            ):
                _insert_alert_log(
                    rd,
                    camera_id,
                    {
                        "message": "Possible fight detected",
                        "snapshot": snapshot_url,
                    },
                )
            continue
