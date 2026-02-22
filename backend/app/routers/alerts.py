from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from bson import ObjectId
import asyncio
import json

from backend.app.core.async_utils import run_sync
from backend.app.db.mongo import alerts as alerts_col, alert_logs as alert_logs_col
from backend.app.services.alert_engine import _now_utc_iso, evaluate_rule

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
    area: Optional[Dict[str, Any]] = None  # {"zone_id": "entrance"} for zone-based count
    occupancy_pct: Optional[Dict[str, Any]] = None  # {">=": 80} for occupancy threshold when zone has capacity


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


# ========== Routes ==========

@router.get("/", response_model=List[AlertResponse])
async def list_alerts() -> List[AlertResponse]:
    def _block():
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
    return await run_sync(_block)


@router.post("/", response_model=AlertResponse)
async def create_alert(req: CreateAlertRequest) -> AlertResponse:
    def _block():
        try:
            now = _now_utc_iso()
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
    return await run_sync(_block)


@router.delete("/{alert_id}", response_model=Dict[str, Any])
async def delete_alert(alert_id: str) -> Dict[str, Any]:
    def _block():
        try:
            _id = ObjectId(alert_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid alert_id")
        try:
            res = alerts_col.delete_one({"_id": _id})
            return {"ok": res.deleted_count == 1}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete alert: {e}") from e
    return await run_sync(_block)


@router.post("/{alert_id}/evaluate", response_model=Dict[str, Any])
async def evaluate_one(alert_id: str) -> Dict[str, Any]:
    def _block():
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
    return await run_sync(_block)


@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_all(limit: int = Query(10, ge=1, le=100)) -> Dict[str, Any]:
    """Evaluate up to 'limit' enabled alerts (polling-based)."""

    def _block():
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
    return await run_sync(_block)


@router.get("/logs", response_model=List[AlertLogResponse])
async def list_logs(limit: int = Query(50, ge=1, le=500)) -> List[AlertLogResponse]:
    def _block():
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
    return await run_sync(_block)


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
        # initialize cursor watermark - use UTC-aware timestamp to match triggered_at format
        watermark = last_ts or _now_utc_iso()
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
