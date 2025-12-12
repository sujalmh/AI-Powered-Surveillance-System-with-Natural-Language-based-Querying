from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.db.mongo import cameras as cameras_col
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


@router.get("/", response_model=List[Dict[str, Any]])
def list_cameras() -> List[Dict[str, Any]]:
    try:
        docs = list(
            cameras_col.find(
                {},
                {"_id": 0},
            )
        )
        # Inject runtime running status
        for d in docs:
            cid = int(d.get("camera_id"))
            d["running"] = runner.is_running(cid)
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list cameras: {e}") from e


@router.get("/{camera_id}/status", response_model=Dict[str, Any])
def camera_status(camera_id: int) -> Dict[str, Any]:
    try:
        doc = cameras_col.find_one({"camera_id": camera_id}, {"_id": 0}) or {}
        doc["camera_id"] = camera_id
        doc["running"] = runner.is_running(camera_id)
        return doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get camera status: {e}") from e


@router.post("/register", response_model=Dict[str, Any])
def register_camera(req: RegisterCameraRequest) -> Dict[str, Any]:
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


@router.post("/{camera_id}/start", response_model=Dict[str, Any])
def start_camera(camera_id: int, req: StartCameraRequest) -> Dict[str, Any]:
    # Determine source/location: prefer request overrides, otherwise from DB
    try:
        db_cam = cameras_col.find_one({"camera_id": camera_id}) or {}
        source: Union[int, str, None] = req.source if req.source is not None else db_cam.get("source")
        if source is None:
            raise HTTPException(status_code=400, detail="Missing source. Provide in request or register the camera first.")
        # cast numeric string to int when possible
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        location = req.location or db_cam.get("location", "Default Location")

        # Preflight probe to ensure source is reachable (unless disabled)
        if req.check:
            ok, msg = can_open_source(source, timeout_seconds=3.0)
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


@router.post("/{camera_id}/stop", response_model=Dict[str, Any])
def stop_camera(camera_id: int) -> Dict[str, Any]:
    try:
        stopped = runner.stop_camera(camera_id)
        if not stopped:
            return {"ok": False, "message": "Camera is not running"}
        return {"ok": True, "camera_id": camera_id, "running": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop camera: {e}") from e


@router.get("/probe", response_model=Dict[str, Any])
def probe_source(
    source: str = Query(..., description="RTSP/HTTP URL or integer index (as string)"),
    timeout: float = Query(3.0, ge=0.5, le=30.0),
) -> Dict[str, Any]:
    """
    Quickly check whether a camera source can be opened and a frame read using OpenCV.
    Pass camera index as a string (e.g., '0') or a full RTSP/HTTP URL.
    """
    # Cast numeric string to int index for local webcams
    parsed: Union[int, str]
    if source.isdigit():
        parsed = int(source)
    else:
        parsed = source
    ok, msg = can_open_source(parsed, timeout_seconds=timeout)
    return {"ok": ok, "message": msg}
