"""
Handlers for informational (non-visual) queries: alerts, cameras, system status.
Extracted from UnifiedRetrieval to keep the orchestrator focused on detection retrieval.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.app.db.mongo import (
    alert_logs as alert_logs_col,
    alerts as alerts_col,
    cameras as cameras_col,
)
from backend.app.services.detection_runner import runner

logger = logging.getLogger(__name__)


def execute_alert_query(
    parsed_filter: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Query ``alert_logs`` with optional time and camera filters.

    Returns a list of enriched alert dicts (with rule name / severity from the
    parent ``alerts`` collection when available).
    """
    q: Dict[str, Any] = {}

    # Time filter
    ts_filter = parsed_filter.get("timestamp")
    if isinstance(ts_filter, dict):
        triggered_ts: Dict[str, Any] = {}
        if ts_filter.get("$gte"):
            triggered_ts["$gte"] = ts_filter["$gte"]
        if ts_filter.get("$lte"):
            triggered_ts["$lte"] = ts_filter["$lte"]
        if triggered_ts:
            q["triggered_at"] = triggered_ts

    # Camera filter
    cam = parsed_filter.get("camera_id")
    if isinstance(cam, int):
        q["camera_id"] = cam

    logger.info("Alert query filter: %s", q)

    docs = list(
        alert_logs_col.find(q)
        .sort("triggered_at", -1)
        .limit(int(limit))
    )

    # Enrich with alert rule names
    alert_ids = [d.get("alert_id") for d in docs if d.get("alert_id") is not None]
    alert_map: Dict[str, Dict[str, Any]] = {}
    if alert_ids:
        try:
            alert_docs = list(alerts_col.find({"_id": {"$in": alert_ids}}))
            for a in alert_docs:
                alert_map[str(a.get("_id"))] = a
        except Exception:
            logger.error("Failed to enrich alerts", exc_info=True)

    results: List[Dict[str, Any]] = []
    for d in docs:
        aid_str = str(d.get("alert_id")) if d.get("alert_id") is not None else None
        rule = alert_map.get(aid_str, {}) if aid_str else {}
        results.append({
            "type": "alert_log",
            "alert_log_id": str(d.get("_id")),
            "alert_id": aid_str,
            "alert_name": rule.get("name"),
            "camera_id": d.get("camera_id"),
            "triggered_at": d.get("triggered_at"),
            "severity": d.get("severity", rule.get("severity", "info")),
            "message": d.get("message", ""),
            "snapshot": d.get("snapshot"),
            "clip": d.get("clip"),
        })
    return results


def execute_camera_query(
    parsed_filter: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Query the ``cameras`` collection and annotate each result with its
    live-running status from :pyobj:`DetectionRunner`.
    """
    cam_filter: Dict[str, Any] = {}
    cam = parsed_filter.get("camera_id")
    if isinstance(cam, int):
        cam_filter["camera_id"] = cam

    logger.info("Camera query filter: %s", cam_filter)

    cameras = list(cameras_col.find(cam_filter, {"_id": 0}).limit(int(limit)))

    try:
        running_set = runner.running_cameras()
    except Exception:
        running_set: set[int] = set()

    results: List[Dict[str, Any]] = []
    for c in cameras:
        raw_id = c.get("camera_id")
        cid: Optional[int] = None
        if raw_id is not None:
            try:
                cid = int(raw_id)
            except (TypeError, ValueError):
                pass
        result = dict(c)
        result["type"] = "camera_status"
        result["running"] = cid in running_set if cid is not None else False
        results.append(result)
    return results
