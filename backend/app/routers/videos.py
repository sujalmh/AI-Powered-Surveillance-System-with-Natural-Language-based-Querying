from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from pathlib import Path
import shutil

from backend.app.config import settings
from backend.app.services.frame_store import mjpeg_generator
from backend.app.services.sem_indexer import index_clip
from backend.app.db.mongo import vlm_frames, cameras as cameras_col

# Optional YOLO fallback for enriching objects in testing dashboard
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from ultralytics import YOLO as _API_YOLO  # type: ignore
except Exception:  # pragma: no cover
    _API_YOLO = None  # type: ignore

router = APIRouter()


class MediaItem(BaseModel):
    path: str
    size_bytes: int
    modified_at: str  # ISO timestamp
    url: str  # static serving URL


def _walk_media(root_dir: str, base_url: str, prefix: Optional[str] = None, limit: int = 200) -> List[MediaItem]:
    result: List[MediaItem] = []
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        return []

    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root_dir).replace("\\", "/")
            if prefix and not rel.startswith(prefix):
                continue
            try:
                stat = os.stat(fpath)
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
            except Exception:
                size = 0
                mtime = datetime.utcnow().isoformat()
            url = f"{base_url}/{rel}"
            result.append(MediaItem(path=rel, size_bytes=size, modified_at=mtime, url=url))
            count += 1
            if count >= limit:
                return result
    return result


@router.get("/recordings", response_model=List[MediaItem])
def list_recordings(prefix: Optional[str] = Query(None, description="relative path prefix filter"), limit: int = Query(200, ge=1, le=1000)) -> List[MediaItem]:
    try:
        return _walk_media(str(settings.RECORDINGS_DIR), "/media/recordings", prefix=prefix, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list recordings: {e}") from e


@router.get("/clips", response_model=List[MediaItem])
def list_clips(prefix: Optional[str] = Query(None, description="relative path prefix filter"), limit: int = Query(200, ge=1, le=1000)) -> List[MediaItem]:
    try:
        return _walk_media(str(settings.CLIPS_DIR), "/media/clips", prefix=prefix, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list clips: {e}") from e


@router.get("/snapshots", response_model=List[MediaItem])
def list_snapshots(prefix: Optional[str] = Query(None, description="relative path prefix filter"), limit: int = Query(200, ge=1, le=1000)) -> List[MediaItem]:
    try:
        return _walk_media(str(settings.SNAPSHOTS_DIR), "/media/snapshots", prefix=prefix, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list snapshots: {e}") from e


@router.get("/stream/{camera_id}")
def stream_mjpeg(
    camera_id: int,
    fps: float = Query(5.0, ge=0.2, le=30.0, description="Max frames per second to push"),
) -> StreamingResponse:
    """
    Serve an MJPEG multipart stream of the latest frames for the given camera.
    The detection loop updates an in-memory frame store; this endpoint streams it.
    """
    try:
        gen = mjpeg_generator(camera_id, fps=fps)
        return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream camera {camera_id}: {e}") from e


@router.get("/latest/{camera_id}")
def latest_jpeg(
    camera_id: int,
) -> Response:
    """
    Return the latest available JPEG frame as a single image for the given camera.
    Useful as a fallback when MJPEG streaming isn't supported by the client.
    """
    from backend.app.services.frame_store import get_latest  # local import to avoid circulars

    item = get_latest(camera_id)
    if not item:
        # return a 204 No Content if no frame yet
        return Response(status_code=204)

    jpg_bytes, _ = item
    headers = {
        "Cache-Control": "no-store, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return Response(content=jpg_bytes, media_type="image/jpeg", headers=headers)

@router.post("/upload")
async def upload_clip(
    file: UploadFile = File(...),
    camera_id: int = Form(99),
    every_sec: float = Form(1.0),
    with_captions: bool = Form(False),
) -> Dict[str, Any]:
    """
    Upload an MP4 file, save it under CLIPS_DIR/camera_{camera_id}/, and index it semantically.
    Returns the clip_url and indexing summary.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="missing filename")

        # Ensure destination
        dest_dir: Path = settings.CLIPS_DIR / f"camera_{int(camera_id)}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename and ensure .mp4
        name = Path(file.filename).name.replace("..", "_").replace("/", "_").replace("\\", "_")
        if not name.lower().endswith(".mp4"):
            name = f"{name}.mp4"
        dest_path = dest_dir / name

        # Save file to disk
        with dest_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        # Derive clip_url relative to CLIPS_DIR
        try:
            rel = dest_path.resolve().relative_to(settings.CLIPS_DIR.resolve())
            clip_url = f"/media/clips/{rel.as_posix()}"
        except Exception:
            clip_url = None

        # Create/update camera metadata in cameras collection
        try:
            now = datetime.utcnow().isoformat()
            camera_doc = cameras_col.find_one({"camera_id": int(camera_id)})
            
            if camera_doc:
                # Update existing camera
                cameras_col.update_one(
                    {"camera_id": int(camera_id)},
                    {
                        "$set": {
                            "last_seen": now,
                            "status": "active",
                            "source": str(dest_path.resolve()),
                            "last_error": None,
                        }
                    }
                )
            else:
                # Create new camera entry
                cameras_col.insert_one({
                    "camera_id": int(camera_id),
                    "location": f"Test Camera {camera_id}",
                    "source": str(dest_path.resolve()),
                    "status": "active",
                    "last_seen": now,
                    "last_error": None,
                })
        except Exception as cam_err:
            # Log but don't fail the upload
            print(f"Warning: Failed to update camera metadata: {cam_err}")

        # Index the clip (semantic VLM pipeline)
        idx_res = index_clip(
            camera_id=int(camera_id),
            clip_path=str(dest_path.resolve()),
            clip_url=clip_url,
            every_sec=float(every_sec),
            with_captions=bool(with_captions and settings.ENABLE_CAPTIONS),
        )

        return {
            "ok": True,
            "camera_id": int(camera_id),
            "clip_path": str(dest_path.resolve()),
            "clip_url": clip_url,
            "indexing": idx_res,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload/index failed: {e}") from e

@router.get("/frames")
def list_clip_frames(
    clip_path: str,
    limit: int = Query(500, ge=1, le=5000),
    enrich: Optional[str] = Query(None, description="optional enrichment: 'yolo' to compute objects on-the-fly"),
) -> List[Dict[str, Any]]:
    """
    Return frame-level metadata stored for a given clip_path from the semantic index (vlm_frames).
    Fields include: camera_id, clip_path, clip_url, frame_ts, frame_index, model, caption, hash.
    """
    try:
        cur = vlm_frames.find(
            {"clip_path": clip_path},
            {
                "_id": 0,
                "camera_id": 1,
                "clip_path": 1,
                "clip_url": 1,
                "frame_ts": 1,
                "frame_index": 1,
                "model": 1,
                "caption": 1,
                "object_captions": 1,
                "hash": 1,
                "embedding_dim": 1,
            },
        ).sort("frame_index", 1).limit(int(limit))
        docs = list(cur)

        # Best-effort enrichment: if object_captions are missing, compose from detections near frame_ts
        try:
            from backend.app.db.mongo import detections as _detections  # local import
        except Exception:
            _detections = None  # type: ignore

        if _detections is not None:
            for d in docs:
                if d.get("object_captions"):
                    continue
                ts = d.get("frame_ts")
                cam = d.get("camera_id")
                if not ts or cam is None:
                    continue
                try:
                    dt = datetime.fromisoformat(str(ts))
                    window = timedelta(seconds=3)
                    q = {
                        "camera_id": int(cam),
                        "timestamp": {"$gte": (dt - window).isoformat(), "$lte": (dt + window).isoformat()},
                    }
                    dd = list(_detections.find(q, {"_id": 0, "objects": 1}).limit(10))
                    out: List[str] = []
                    for di in dd:
                        for o in (di.get("objects") or []):
                            name = str(o.get("object_name") or "").strip().lower()
                            if not name:
                                continue
                            if name == "person":
                                color = str(o.get("color") or "").strip()
                                acc = o.get("person_attributes") or {}
                                parts: List[str] = ["person"]
                                if color and color.lower() != "unknown":
                                    parts.append(f"wearing {color.lower()} clothing")
                                if acc.get("hat_confidence", 0.0) > 0.5:
                                    parts.append("hat")
                                if acc.get("bag_confidence", 0.0) > 0.5:
                                    parts.append("carrying bag")
                                if acc.get("longsleeves_confidence", 0.0) > 0.5:
                                    parts.append("long sleeves")
                                if acc.get("longpants_confidence", 0.0) > 0.5:
                                    parts.append("long pants")
                                if acc.get("coat_jacket_confidence", 0.0) > 0.5:
                                    parts.append("coat/jacket")
                                out.append(", ".join(parts))
                            else:
                                out.append(name)
                            if len(out) >= 20:
                                break
                        if len(out) >= 20:
                            break
                    if out:
                        d["object_captions"] = out
                except Exception:
                    continue

        # Optional YOLO enrichment if requested and object_captions still missing
        if enrich == "yolo" and cv2 is not None and _API_YOLO is not None:
            try:
                model = _API_YOLO(settings.MODEL_PATH)
            except Exception:
                model = None
            if model is not None:
                try:
                    cap = cv2.VideoCapture(clip_path)
                    if cap and cap.isOpened():
                        for d in docs:
                            if d.get("object_captions"):
                                continue
                            idx = d.get("frame_index")
                            if idx is None:
                                continue
                            try:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                                ok, frame = cap.read()
                                if not ok or frame is None:
                                    continue
                                names: List[str] = []
                                res = model(frame)
                                for r in res:
                                    if getattr(r, "boxes", None) is None:
                                        continue
                                    cls = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else []
                                    for c in cls:
                                        try:
                                            nm = str(r.names[int(c)]) if hasattr(r, "names") else None
                                        except Exception:
                                            nm = None
                                        if nm:
                                            names.append(str(nm).lower())
                                        if len(names) >= 20:
                                            break
                                    if len(names) >= 20:
                                        break
                                if names:
                                    d["object_captions"] = names
                            except Exception:
                                continue
                    if cap:
                        try:
                            cap.release()
                        except Exception:
                            pass
                except Exception:
                    pass

        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to fetch frames: {e}") from e
