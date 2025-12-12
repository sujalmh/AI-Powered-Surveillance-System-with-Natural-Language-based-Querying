from __future__ import annotations

import time
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.app.db.mongo import alerts as alerts_col, alert_logs as alert_logs_col


@dataclass
class RuleCache:
  rules: List[Dict[str, Any]]
  last_refresh: float
  ttl_sec: float = 5.0


# Global state (process-local)
_RULE_CACHE = RuleCache(rules=[], last_refresh=0.0)
# Cooldown memory: (rule_id, camera_id) -> last_fired_epoch_sec
_LAST_FIRED: Dict[Tuple[str, int], float] = {}
# Per-camera state (prev gray frame, motion energy history, presence memory etc.)
_PER_CAMERA: Dict[int, Dict[str, Any]] = defaultdict(dict)


def _now_utc_iso() -> str:
  return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _load_active_rules() -> List[Dict[str, Any]]:
  """Cache enabled rules for a few seconds to avoid hot-looping Mongo."""
  now = time.time()
  if now - _RULE_CACHE.last_refresh > _RULE_CACHE.ttl_sec:
    _RULE_CACHE.rules = list(alerts_col.find({"enabled": True}))
    _RULE_CACHE.last_refresh = now
  return _RULE_CACHE.rules


def _parse_local_time_window(rule_time: Optional[Dict[str, Any]]) -> Optional[Tuple[dtime, dtime, str]]:
  """
  Supports {"start":"22:00","end":"06:00","tz":"Asia/Kolkata"} or None.
  If provided, start/end interpreted as local wall-clock for the given TZ name.
  """
  if not rule_time:
    return None
  start = rule_time.get("start")
  end = rule_time.get("end")
  tz = rule_time.get("tz", "UTC")
  if not start or not end:
    return None
  try:
    s_h, s_m = [int(x) for x in start.split(":")]
    e_h, e_m = [int(x) for x in end.split(":")]
    return (dtime(hour=s_h, minute=s_m), dtime(hour=e_h, minute=e_m), tz)
  except Exception:
    return None


def _in_local_window(win: Tuple[dtime, dtime, str], now_dt: datetime) -> bool:
  """Check if now_dt (aware or naive UTC) falls within local window (may cross midnight)."""
  start_t, end_t, tz_name = win
  # naive approach: use now_dt's time only; for cross-midnight, treat (start > end) as wrap
  now_local_t = now_dt.time()
  if start_t <= end_t:
    return start_t <= now_local_t <= end_t
  # window crosses midnight
  return now_local_t >= start_t or now_local_t <= end_t


def _class_count(objects: List[Dict[str, Any]], class_name: str) -> int:
  c = 0
  for o in objects:
    if str(o.get("object_name", "")).lower() == class_name.lower():
      c += 1
  return c


def _person_boxes(objects: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
  boxes: List[Tuple[int, int, int, int]] = []
  for o in objects:
    if str(o.get("object_name", "")).lower() == "person":
      bb = o.get("bbox")
      if isinstance(bb, (list, tuple)) and len(bb) == 4:
        try:
          x1, y1, x2, y2 = [int(v) for v in bb]
          if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
        except Exception:
          pass
  return boxes


def _motion_energy(prev_gray: Optional[np.ndarray], frame_bgr: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
  """
  Compute normalized motion energy (0..1) between prev_gray and current gray (downsampled).
  Returns (cur_gray, energy).
  """
  if frame_bgr is None:
    return prev_gray, 0.0
  gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
  gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
  if prev_gray is None or prev_gray.shape != gray.shape:
    return gray, 0.0
  diff = cv2.absdiff(gray, prev_gray)
  _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
  energy = float(th.mean() / 255.0)  # fraction of pixels over threshold
  return gray, energy


def _fight_heuristic(person_boxes: List[Tuple[int, int, int, int]], energy_hist: Deque[float]) -> bool:
  """
  Simple fight heuristic: at least 2 persons, close proximity and high recent motion energy.
  """
  if len(person_boxes) < 2:
    return False
  # closeness: any pair centers within 15% of min(frame_dim) (assume 160x90 gray for threshold basis)
  centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for (x1, y1, x2, y2) in person_boxes]
  close = False
  for i in range(len(centers)):
    for j in range(i + 1, len(centers)):
      dx = centers[i][0] - centers[j][0]
      dy = centers[i][1] - centers[j][1]
      dist = (dx * dx + dy * dy) ** 0.5
      if dist < 24.0:  # ~15% of 160
        close = True
        break
    if close:
      break
  if not close:
    return False
  # motion energy sustained
  if not energy_hist:
    return False
  recent = list(energy_hist)[-10:]  # last ~10 entries
  avg_energy = sum(recent) / max(1, len(recent))
  return avg_energy >= 0.03  # ~3% of pixels moving on average


def _cooldown_ok(rule_id: str, camera_id: int, cooldown_sec: float) -> bool:
  k = (rule_id, camera_id)
  last = _LAST_FIRED.get(k, 0.0)
  now = time.time()
  if now - last >= cooldown_sec:
    _LAST_FIRED[k] = now
    return True
  return False


def _insert_alert_log(rule_doc: Dict[str, Any], camera_id: int, payload: Dict[str, Any]) -> None:
  alert_logs_col.insert_one(
    {
      "alert_id": rule_doc["_id"],
      "triggered_at": _now_utc_iso(),
      "camera_id": int(camera_id),
      "event_id": None,
      "detection_ids": payload.get("detection_ids"),
      "snapshot": payload.get("snapshot"),
      "clip": payload.get("clip"),
      "message": payload.get("message", ""),
      "payload": payload,
      "severity": rule_doc.get("severity", "info"),
    }
  )


def evaluate_realtime(
  camera_id: int,
  timestamp_iso: str,
  objects: List[Dict[str, Any]],
  frame_bgr: Optional[np.ndarray] = None,
  snapshot_url: Optional[str] = None,
) -> None:
  """
  Evaluate all active alert rules against the latest frame objects (and optional frame).
  This is intended to be called from the detection loop after persistence.
  """
  # Prepare per-camera state
  st = _PER_CAMERA[camera_id]
  prev_gray: Optional[np.ndarray] = st.get("prev_gray")
  energy_hist = st.get("energy_hist")
  if energy_hist is None:
    energy_hist = deque(maxlen=60)  # ~30s if evaluating at ~2 Hz
    st["energy_hist"] = energy_hist
  # type: ignore[assignment]
  energy_hist = energy_hist  # for type checkers

  cur_gray, energy = _motion_energy(prev_gray, frame_bgr)
  st["prev_gray"] = cur_gray
  energy_hist.append(energy)

  # Presence map for class
  presence: Dict[str, int] = defaultdict(int)
  for o in objects:
    cname = str(o.get("object_name", "")).lower()
    presence[cname] += 1
  last_presence: Dict[str, int] = st.get("presence_last", {})
  st["presence_last"] = dict(presence)

  # Time aware object
  try:
    now_dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
  except Exception:
    now_dt = datetime.utcnow()

  rules = _load_active_rules()
  for rd in rules:
    rid = str(rd.get("_id"))
    rule = rd.get("rule", {}) or {}
    enabled = bool(rd.get("enabled", True))
    if not enabled:
      continue

    # Camera filter
    cams = rule.get("cameras")
    if isinstance(cams, list) and len(cams) > 0 and camera_id not in cams:
      continue

    cooldown = float(rd.get("cooldown_sec", 60.0))

    # 1) Count-based (e.g., "more than 3 people in frame")
    # rule.objects[0].name and rule.count {">=": N}
    # Fall back to >=1 if no count specified
    trig_count = False
    want_name: Optional[str] = None
    try:
      objs = rule.get("objects") or []
      if isinstance(objs, list) and len(objs) > 0:
        want_name = str(objs[0].get("name")) if objs[0].get("name") else None
    except Exception:
      want_name = None

    if want_name:
      cnt = presence.get(want_name.lower(), 0)
      cond = rule.get("count")
      trig = True
      if cond:
        for op, val in cond.items():
          try:
            v = float(val)
          except Exception:
            continue
          if op == "==":
            trig &= cnt == v
          elif op == ">=":
            trig &= cnt >= v
          elif op == "<=":
            trig &= cnt <= v
          elif op == ">":
            trig &= cnt > v
          elif op == "<":
            trig &= cnt < v
      else:
        trig = cnt >= 1
      if trig and _cooldown_ok(rid, camera_id, cooldown):
        _insert_alert_log(
          rd,
          camera_id,
          {
            "message": f"Rule '{rd.get('name')}' matched count: {want_name}={cnt}",
            "snapshot": snapshot_url,
            "class": want_name,
            "count": cnt,
          },
        )
        continue  # next rule

    # 2) Class enter during time (e.g., "car enters during night hours")
    # Expect rule["event"] == "class_enter_during_time", rule.objects[0].name is the class
    if str(rule.get("event", "")).lower() == "class_enter_during_time" and want_name:
      win = _parse_local_time_window(rule.get("time_of_day"))
      ok_time = True if win is None else _in_local_window(win, now_dt)
      if ok_time:
        prev_cnt = int(last_presence.get(want_name.lower(), 0)) if last_presence else 0
        curr_cnt = int(presence.get(want_name.lower(), 0))
        entered = prev_cnt == 0 and curr_cnt > 0
        if entered and _cooldown_ok(rid, camera_id, cooldown):
          _insert_alert_log(
            rd,
            camera_id,
            {
              "message": f"{want_name} entered during time window",
              "snapshot": snapshot_url,
              "class": want_name,
              "prev_count": prev_cnt,
              "curr_count": curr_cnt,
            },
          )
          continue

    # 3) Fight detection (heuristic)
    if str(rule.get("behavior", "")).lower() == "fight":
      pboxes = _person_boxes(objects)
      if _fight_heuristic(pboxes, energy_hist) and _cooldown_ok(rid, camera_id, cooldown):
        _insert_alert_log(
          rd,
          camera_id,
          {
            "message": "Possible fight detected (heuristic)",
            "snapshot": snapshot_url,
            "persons": len(pboxes),
            "avg_motion": float(sum(energy_hist) / max(1, len(energy_hist))),
          },
        )


# Optional background polling evaluator (if you want to trigger from DB-only without frame input)
def evaluate_recent_detections_poll(camera_id: Optional[int] = None, last_minutes: int = 1) -> int:
  """
  Polling fallback: can be scheduled to evaluate rules based on recent detections only.
  Returns number of alerts emitted.
  """
  # This can reuse the existing /alerts/evaluate style; left simple for now.
  return 0
