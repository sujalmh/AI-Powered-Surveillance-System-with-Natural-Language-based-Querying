from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from backend.app.core.async_utils import run_sync
from backend.app.db.mongo import cameras as cameras_col, zones as zones_col
from backend.app.services.detection_runner import runner
from backend.app.services.video_probe import can_open_source


router = APIRouter()


class RegisterCameraRequest(BaseModel):
    camera_id: int = Field(..., ge=0)
    source: Union[int, str]
    location: Optional[str] = "Default Location"


class StartCameraRequest(BaseModel):
    source: Optional[Union[int, str]] = None
    location: Optional[str] = None
    show_window: bool = False
    check: bool = True  # preflight probe with OpenCV before starting


class ZoneBbox(BaseModel):
    """Normalized [0,1] coordinates. Requires x_min < x_max and y_min < y_max (no degenerate/inverted boxes)."""
    x_min: float = Field(..., ge=0, le=1)
    y_min: float = Field(..., ge=0, le=1)
    x_max: float = Field(..., ge=0, le=1)
    y_max: float = Field(..., ge=0, le=1)

    @model_validator(mode="after")
    def check_bbox_ranges(self) -> "ZoneBbox":
        if self.x_min >= self.x_max:
            raise ValueError(
                f"Invalid bbox: x_min ({self.x_min}) must be strictly less than x_max ({self.x_max})"
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"Invalid bbox: y_min ({self.y_min}) must be strictly less than y_max ({self.y_max})"
            )
        return self


class CreateZoneRequest(BaseModel):
    zone_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    bbox: ZoneBbox
    capacity: Optional[int] = Field(None, ge=0)
    area_sqm: Optional[float] = Field(None, ge=0)


class UpdateZoneRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    bbox: Optional[ZoneBbox] = None
    capacity: Optional[int] = Field(None, ge=0)
    area_sqm: Optional[float] = Field(None, ge=0)


@router.get("/", response_model=List[Dict[str, Any]])
async def list_cameras() -> List[Dict[str, Any]]:
    def _block():
        try:
            docs = list(cameras_col.find({}, {"_id": 0}))
            for d in docs:
                cid = int(d.get("camera_id"))
                d["running"] = runner.is_running(cid)
            return docs
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list cameras: {e}") from e
    return await run_sync(_block)


@router.get("/{camera_id}/status", response_model=Dict[str, Any])
async def camera_status(camera_id: int) -> Dict[str, Any]:
    def _block():
        try:
            doc = cameras_col.find_one({"camera_id": camera_id}, {"_id": 0}) or {}
            doc["camera_id"] = camera_id
            doc["running"] = runner.is_running(camera_id)
            return doc
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get camera status: {e}") from e
    return await run_sync(_block)


@router.post("/register", response_model=Dict[str, Any])
async def register_camera(req: RegisterCameraRequest) -> Dict[str, Any]:
    def _block():
        try:
            cameras_col.update_one(
                {"camera_id": req.camera_id},
                {
                    "$set": {
                        "camera_id": req.camera_id,
                        "source": str(req.source),
                        "location": req.location or "Default Location",
                        "status": "inactive",
                    }
                },
                upsert=True,
            )
            return {"ok": True, "camera_id": req.camera_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to register camera: {e}") from e
    return await run_sync(_block)


@router.post("/{camera_id}/start", response_model=Dict[str, Any])
async def start_camera(camera_id: int, req: StartCameraRequest) -> Dict[str, Any]:
    def _block():
        try:
            db_cam = cameras_col.find_one({"camera_id": camera_id}) or {}
            source: Union[int, str, None] = req.source if req.source is not None else db_cam.get("source")
            if source is None:
                raise HTTPException(status_code=400, detail="Missing source. Provide in request or register the camera first.")
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            location = req.location or db_cam.get("location", "Default Location")
            if req.check:
                ok, msg = can_open_source(source, timeout_seconds=8.0)
                if not ok:
                    try:
                        cameras_col.update_one(
                            {"camera_id": camera_id},
                            {"$set": {"status": "error", "last_error": msg, "source": str(source), "location": location}},
                            upsert=True,
                        )
                    except Exception:
                        pass
                    raise HTTPException(status_code=400, detail=f"Cannot connect to source: {msg}")
            started = runner.start_camera(
                camera_id=camera_id,
                source=source,
                location=location,
                show_window=req.show_window,
            )
            if not started:
                return {"ok": False, "message": "Camera already running"}
            return {"ok": True, "camera_id": camera_id, "running": True}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start camera: {e}") from e
    return await run_sync(_block)


@router.post("/{camera_id}/stop", response_model=Dict[str, Any])
async def stop_camera(camera_id: int) -> Dict[str, Any]:
    def _block():
        try:
            stopped = runner.stop_camera(camera_id)
            if not stopped:
                return {"ok": False, "message": "Camera is not running"}
            return {"ok": True, "camera_id": camera_id, "running": False}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to stop camera: {e}") from e
    return await run_sync(_block)


@router.delete("/{camera_id}", response_model=Dict[str, Any])
async def delete_camera(camera_id: int) -> Dict[str, Any]:
    """Delete a camera from the database. If running, it will be stopped first."""

    def _block():
        try:
            if runner.is_running(camera_id):
                runner.stop_camera(camera_id)
            result = cameras_col.delete_one({"camera_id": camera_id})
            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
            return {"ok": True, "camera_id": camera_id, "message": "Camera deleted successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete camera: {e}") from e
    return await run_sync(_block)


# ---------- Zones (ROIs) for crowd management ----------

@router.get("/{camera_id}/zones", response_model=List[Dict[str, Any]])
async def list_zones(camera_id: int) -> List[Dict[str, Any]]:
    """List all zones (ROIs) for a camera."""

    def _block():
        try:
            if cameras_col.find_one({"camera_id": camera_id}) is None:
                raise HTTPException(status_code=404, detail="Camera not found")
            return list(zones_col.find({"camera_id": camera_id}, {"_id": 0}))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list zones: {e}") from e
    return await run_sync(_block)


@router.post("/{camera_id}/zones", response_model=Dict[str, Any])
async def create_zone(camera_id: int, req: CreateZoneRequest) -> Dict[str, Any]:
    """Create a zone (ROI) for a camera. Bbox in normalized [0,1] coordinates."""

    def _block():
        try:
            if cameras_col.find_one({"camera_id": camera_id}) is None:
                raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
            doc = {
                "camera_id": camera_id,
                "zone_id": req.zone_id.strip(),
                "name": req.name.strip(),
                "bbox": req.bbox.model_dump(),
            }
            if req.capacity is not None:
                doc["capacity"] = req.capacity
            if req.area_sqm is not None:
                doc["area_sqm"] = req.area_sqm
            zones_col.update_one(
                {"camera_id": camera_id, "zone_id": doc["zone_id"]},
                {"$set": doc},
                upsert=True,
            )
            out = dict(doc)
            out.pop("_id", None)
            return out
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create zone: {e}") from e
    return await run_sync(_block)


@router.put("/{camera_id}/zones/{zone_id}", response_model=Dict[str, Any])
def update_zone(camera_id: int, zone_id: str, req: UpdateZoneRequest) -> Dict[str, Any]:
    """Update a zone. Only provided fields are updated."""
    try:
        doc = zones_col.find_one({"camera_id": camera_id, "zone_id": zone_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found for camera {camera_id}")
        updates = {}
        if req.name is not None:
            updates["name"] = req.name.strip()
        if req.bbox is not None:
            updates["bbox"] = req.bbox.model_dump()
        if req.capacity is not None:
            updates["capacity"] = req.capacity
        if req.area_sqm is not None:
            updates["area_sqm"] = req.area_sqm
        if not updates:
            out = dict(doc)
            out.pop("_id", None)
            return out
        zones_col.update_one(
            {"camera_id": camera_id, "zone_id": zone_id},
            {"$set": updates},
        )
        updated = zones_col.find_one({"camera_id": camera_id, "zone_id": zone_id}, {"_id": 0})
        return updated or {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update zone: {e}") from e


@router.delete("/{camera_id}/zones/{zone_id}", response_model=Dict[str, Any])
async def delete_zone(camera_id: int, zone_id: str) -> Dict[str, Any]:
    """Delete a zone."""

    def _block():
        try:
            result = zones_col.delete_one({"camera_id": camera_id, "zone_id": zone_id})
            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found for camera {camera_id}")
            return {"ok": True, "camera_id": camera_id, "zone_id": zone_id}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete zone: {e}") from e
    return await run_sync(_block)


@router.get("/probe", response_model=Dict[str, Any])
async def probe_source(
    source: str = Query(..., description="RTSP/HTTP URL or integer index (as string)"),
    timeout: float = Query(3.0, ge=0.5, le=30.0),
) -> Dict[str, Any]:
    """Check whether a camera source can be opened (OpenCV). Pass index as string or RTSP/HTTP URL."""

    def _block():
        parsed: Union[int, str] = int(source) if source.isdigit() else source
        ok, msg = can_open_source(parsed, timeout_seconds=timeout)
        return {"ok": ok, "message": msg}
    return await run_sync(_block)


@router.get("/{camera_id}/occupancy", response_model=Dict[str, Any])
async def get_occupancy(camera_id: int) -> Dict[str, Any]:
    """Return current occupancy: zones (with capacity), person_count and zone_counts."""

    def _block():
        try:
            from backend.app.db.mongo import detections as detections_col
            zone_docs = list(zones_col.find({"camera_id": camera_id}, {"_id": 0}))
            latest = detections_col.find_one(
                {"camera_id": camera_id},
                {"_id": 0, "timestamp": 1, "person_count": 1, "zone_counts": 1},
                sort=[("timestamp", -1)],
            )
            person_count = int(latest.get("person_count", 0)) if latest else 0
            zone_counts = dict(latest.get("zone_counts") or {}) if latest else {}
            zones_out = []
            for z in zone_docs:
                zid = z.get("zone_id")
                name = z.get("name", zid)
                cap = z.get("capacity")
                cnt = int(zone_counts.get(str(zid), 0))
                occ_pct = (100.0 * cnt / int(cap)) if (cap is not None and int(cap) > 0) else None
                zones_out.append({
                    "zone_id": zid,
                    "name": name,
                    "count": cnt,
                    "capacity": cap,
                    "occupancy_pct": round(occ_pct, 1) if occ_pct is not None else None,
                })
            return {
                "camera_id": camera_id,
                "person_count": person_count,
                "zone_counts": zone_counts,
                "zones": zones_out,
                "latest_timestamp": latest.get("timestamp") if latest else None,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get occupancy: {e}") from e
    return await run_sync(_block)
