from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from bson import ObjectId
import asyncio
import json

from backend.app.db.mongo import alerts as alerts_col, alert_logs as alert_logs_col, detections as detections_col

router = APIRouter()


# ========== Models ==========

class AlertRuleSpec(BaseModel):
    time: Optional[Dict[str, Any]] = None  # {"from": ISO, "to": ISO, "last_minutes": int, "last_hours": int}
    # Optional daily time-of-day window, supports crossing midnight:
    # {"start":"22:00","end":"06:00","tz":"Asia/Kolkata"}
    time_of_day: Optional[Dict[str, Any]] = None
    # Event and behavior keywords for richer rule types
    event: Optional[str] = None         # e.g., "class_enter_during_time"
    behavior: Optional[str] = None      # e.g., "fight"
    cameras: Optional[List[int]] = None
    objects: Optional[List[Dict[str, Any]]] = None  # e.g., [{"name":"person","attributes":{"bag_confidence":{">=":0.7}}}]
    color: Optional[str] = None  # standardized color name
    count: Optional[Dict[str, Any]] = None  # {">=": 1}
    area: Optional[Dict[str, Any]] = None  # reserved for future polygon support


class CreateAlertRequest(BaseModel):
    name: str
    nl: Optional[str] = None
    rule: AlertRuleSpec
    enabled: bool = True
    actions: Optional[List[str]] = Field(default_factory=lambda: ["store_clip", "snapshot", "push_ws"])
    severity: Optional[str] = Field(default="info")
    cooldown_sec: Optional[float] = Field(default=60.0)


class AlertResponse(BaseModel):
    id: str
    name: str
    nl: Optional[str]
    rule: Dict[str, Any]
    enabled: bool
    actions: List[str]
    severity: Optional[str] = "info"
    cooldown_sec: Optional[float] = 60.0
    created_at: str
    updated_at: str


class AlertLogResponse(BaseModel):
    id: str
    alert_id: str
    triggered_at: str
    camera_id: Optional[int] = None
    event_id: Optional[str] = None
    detection_ids: Optional[List[str]] = None
    snapshot: Optional[str] = None
    clip: Optional[str] = None
    message: str


# ========== Helpers ==========

def oid_str(x: Any) -> str:
    if isinstance(x, ObjectId):
        return str(x)
    return str(x)

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def parse_time_window(rule_time: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    if not rule_time:
        return None, None
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    if "last_minutes" in rule_time:
        end = datetime.utcnow()
        start = end - timedelta(minutes=int(rule_time["last_minutes"]))
        start_iso, end_iso = start.isoformat(), end.isoformat()
    elif "last_hours" in rule_time:
        end = datetime.utcnow()
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
    q: Dict[str, Any] = {}
    # cameras -> camera_id
    cameras = rule.get("cameras")
    if cameras:
        q["camera_id"] = {"$in": cameras}

    # time window
    start_iso, end_iso = parse_time_window(rule.get("time"))
    if start_iso or end_iso:
        q["timestamp"] = {}
        if start_iso:
            q["timestamp"]["$gte"] = start_iso
        if end_iso:
            q["timestamp"]["$lte"] = end_iso

    # object / color
    objects = rule.get("objects")
    if objects and isinstance(objects, list) and len(objects) > 0:
        # for baseline, pick the first object's name
        name = objects[0].get("name")
        if name:
            q["objects.object_name"] = name

    color = rule.get("color")
    if color:
        q["objects.color"] = color

    return q

def matches_count_condition(count: int, cond: Optional[Dict[str, Any]]) -> bool:
    if not cond:
        # no condition -> at least 1
        return count >= 1
    for op, val in cond.items():
        try:
            v = float(val)
        except Exception:
            continue
        if op == "==":
            if not (count == v):
                return False
        elif op == ">=":
            if not (count >= v):
                return False
        elif op == "<=":
            if not (count <= v):
                return False
        elif op == ">":
            if not (count > v):
                return False
        elif op == "<":
            if not (count < v):
                return False
    return True

def evaluate_rule(rule_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Execute a simple evaluation of an alert rule against detections.
    Returns an alert_log dict if triggered, otherwise None.
    """
    if not rule_doc.get("enabled", True):
        return None

    rule = rule_doc.get("rule", {}) or {}
    query = build_detection_query_from_rule(rule)
    # Limit scope: last 30 minutes default if no explicit time
    if "timestamp" not in query:
        end = datetime.utcnow()
        start = end - timedelta(minutes=30)
        query["timestamp"] = {"$gte": start.isoformat(), "$lte": end.isoformat()}

    # Fetch recent detections
    docs = list(detections_col.find(query).sort("timestamp", -1).limit(500))

    # Count matched objects
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
        # Triggered
        log = {
            "alert_id": rule_doc["_id"],
            "triggered_at": now_iso(),
            "camera_id": docs[0]["camera_id"] if docs else None,
            "event_id": None,
            "detection_ids": [str(d["_id"]) for d in docs[:10]],  # sample first N
            "snapshot": None,
            "clip": None,
            "message": f"Rule '{rule_doc.get('name')}' matched count={matched_count}",
        }
        return log
    return None


# ========== Routes ==========

@router.get("/", response_model=List[AlertResponse])
def list_alerts() -> List[AlertResponse]:
    try:
        out: List[AlertResponse] = []
        for a in alerts_col.find({}).sort("created_at", -1):
            out.append(
                AlertResponse(
                    id=oid_str(a.get("_id")),
                    name=a.get("name", ""),
                    nl=a.get("nl"),
                    rule=a.get("rule", {}),
                    enabled=bool(a.get("enabled", True)),
                    actions=a.get("actions", []),
                    severity=a.get("severity", "info"),
                    cooldown_sec=float(a.get("cooldown_sec", 60.0)),
                    created_at=a.get("created_at", ""),
                    updated_at=a.get("updated_at", ""),
                )
            )
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list alerts: {e}") from e


@router.post("/", response_model=AlertResponse)
def create_alert(req: CreateAlertRequest) -> AlertResponse:
    try:
        now = now_iso()
        doc = {
            "name": req.name,
            "nl": req.nl,
            "rule": req.rule.model_dump(),
            "enabled": req.enabled,
            "actions": req.actions or [],
            "severity": (req.severity or "info"),
            "cooldown_sec": float(req.cooldown_sec or 60.0),
            "created_at": now,
            "updated_at": now,
        }
        res = alerts_col.insert_one(doc)
        doc["_id"] = res.inserted_id
        return AlertResponse(
            id=oid_str(doc["_id"]),
            name=doc["name"],
            nl=doc.get("nl"),
            rule=doc["rule"],
            enabled=doc["enabled"],
            actions=doc.get("actions", []),
            severity=doc.get("severity", "info"),
            cooldown_sec=float(doc.get("cooldown_sec", 60.0)),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {e}") from e


@router.delete("/{alert_id}", response_model=Dict[str, Any])
def delete_alert(alert_id: str) -> Dict[str, Any]:
    try:
        _id = ObjectId(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert_id")

    try:
        res = alerts_col.delete_one({"_id": _id})
        return {"ok": res.deleted_count == 1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete alert: {e}") from e


@router.post("/{alert_id}/evaluate", response_model=Dict[str, Any])
def evaluate_one(alert_id: str) -> Dict[str, Any]:
    try:
        _id = ObjectId(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert_id")

    alert_doc = alerts_col.find_one({"_id": _id})
    if not alert_doc:
        raise HTTPException(status_code=404, detail="Alert not found")

    try:
        log = evaluate_rule(alert_doc)
        if log:
            res = alert_logs_col.insert_one(log)
            return {"triggered": True, "log_id": oid_str(res.inserted_id), "message": log["message"]}
        return {"triggered": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}") from e


@router.post("/evaluate", response_model=Dict[str, Any])
def evaluate_all(limit: int = Query(10, ge=1, le=100)) -> Dict[str, Any]:
    """
    Evaluate up to 'limit' enabled alerts. This is a polling-based baseline evaluator.
    """
    try:
        triggered: List[Dict[str, Any]] = []
        for a in alerts_col.find({"enabled": True}).sort("updated_at", -1).limit(limit):
            log = evaluate_rule(a)
            if log:
                res = alert_logs_col.insert_one(log)
                triggered.append({"alert_id": oid_str(a["_id"]), "log_id": oid_str(res.inserted_id), "message": log["message"]})
        return {"ok": True, "triggered": triggered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}") from e


@router.get("/logs", response_model=List[AlertLogResponse])
def list_logs(limit: int = Query(50, ge=1, le=500)) -> List[AlertLogResponse]:
    try:
        out: List[AlertLogResponse] = []
        for l in alert_logs_col.find({}).sort("triggered_at", -1).limit(limit):
            out.append(
                AlertLogResponse(
                    id=oid_str(l.get("_id")),
                    alert_id=oid_str(l.get("alert_id")),
                    triggered_at=l.get("triggered_at", ""),
                    camera_id=l.get("camera_id"),
                    event_id=l.get("event_id"),
                    detection_ids=[str(x) for x in l.get("detection_ids", [])] if l.get("detection_ids") else None,
                    snapshot=l.get("snapshot"),
                    clip=l.get("clip"),
                    message=l.get("message", ""),
                )
            )
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list alert logs: {e}") from e


@router.get("/stream")
async def stream_alerts(
    camera_id: Optional[int] = None,
    interval_sec: float = Query(1.0, ge=0.1, le=5.0),
    last_ts: Optional[str] = None,
):
    """
    Server-Sent Events (SSE) stream of alert logs.
    Polls MongoDB every interval_sec and emits new alerts as they arrive.
    """
    async def event_gen():
        # initialize cursor watermark
        watermark = last_ts or datetime.utcnow().isoformat()
        while True:
            try:
                q: Dict[str, Any] = {"triggered_at": {"$gt": watermark}}
                if camera_id is not None:
                    q["camera_id"] = int(camera_id)
                # fetch in ascending time to advance watermark safely
                docs = list(alert_logs_col.find(q).sort("triggered_at", 1).limit(200))
                for l in docs:
                    payload = {
                        "id": oid_str(l.get("_id")),
                        "alert_id": oid_str(l.get("alert_id")),
                        "triggered_at": l.get("triggered_at", ""),
                        "camera_id": l.get("camera_id"),
                        "event_id": l.get("event_id"),
                        "detection_ids": [str(x) for x in l.get("detection_ids", [])] if l.get("detection_ids") else None,
                        "snapshot": l.get("snapshot"),
                        "clip": l.get("clip"),
                        "message": l.get("message", ""),
                        "severity": l.get("severity", "info"),
                        "payload": l.get("payload", {}),
                    }
                    # advance watermark
                    wm = l.get("triggered_at")
                    if isinstance(wm, str):
                        watermark = wm
                    yield f"data: {json.dumps(payload)}\n\n"
            except Exception as _:
                # On error, wait and continue to avoid tearing down the stream
                pass
            await asyncio.sleep(interval_sec)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
