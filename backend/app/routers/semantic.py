from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.services.sem_indexer import index_clip
from backend.app.services.sem_search import search_unstructured
from backend.app.db.mongo import app_settings

router = APIRouter()

# indexing-mode helper (structured | semantic | both)
def _get_indexing_mode() -> str:
    try:
        doc = app_settings.find_one({"key": "indexing_mode"}, {"_id": 0, "value": 1})
        val = (doc or {}).get("value")
        return val if val in ("structured", "semantic", "both") else "both"
    except Exception:
        return "both"

# Typing-safe coercion helpers (avoid pylance complaints on int/str casting from Any)
def _as_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, bool):
            # bool is subclass of int; keep explicit mapping
            return int(x)
        if isinstance(x, (int,)):
            return x
        if isinstance(x, (float,)):
            return int(x)
        if isinstance(x, (str,)):
            return int(x.strip()) if x.strip() else default
        # fallback tries __int__
        return int(x)  # type: ignore
    except Exception:
        return default

def _as_str(x: Any) -> Optional[str]:
    try:
        if x is None:
            return None
        return str(x)
    except Exception:
        return None


class IndexClipRequest(BaseModel):
    camera_id: int
    clip_path: str = Field(..., description="Absolute or relative path to an mp4 clip on disk")
    clip_url: Optional[str] = Field(default=None, description="Public URL path (e.g., /media/clips/...)")
    every_sec: float = Field(default=1.0, ge=0.1, le=10.0)
    with_captions: Optional[bool] = Field(default=False, description="Generate BLIP-2 captions for frames if enabled")
    respect_mode: Optional[bool] = Field(default=True, description="Honor backend indexing-mode setting")


class IndexClipResponse(BaseModel):
    ok: bool
    indexed_frames: int = 0
    model: Optional[str] = None
    backend: Optional[str] = None
    message: Optional[str] = None


@router.post("/index-clip", response_model=IndexClipResponse)
def api_index_clip(req: IndexClipRequest) -> IndexClipResponse:
    try:
        # Honor runtime indexing mode unless overridden
        if bool(req.respect_mode):
            mode = _get_indexing_mode()
            if mode == "structured":
                return IndexClipResponse(
                    ok=True,
                    indexed_frames=0,
                    model=None,
                    backend=None,
                    message="semantic indexing skipped due to indexing_mode=structured",
                )
        result = index_clip(
            req.camera_id,
            req.clip_path,
            req.clip_url,
            every_sec=req.every_sec,
            with_captions=bool(req.with_captions),
        )
        # Cast into typed response to satisfy static typing
        return IndexClipResponse(
            ok=bool(result.get("ok", False)),
            indexed_frames=int(result.get("indexed_frames", 0) or 0),
            model=(str(result.get("model")) if result.get("model") is not None else None),
            backend=(str(result.get("backend")) if result.get("backend") is not None else None),
            message=(str(result.get("message")) if result.get("message") is not None else None),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"index-clip failed: {e}") from e


class IndexAllRequest(BaseModel):
    camera_id: Optional[int] = Field(default=None, description="Restrict to a camera; inferred from folder name if None")
    glob: Optional[str] = Field(default=None, description="Glob pattern; defaults to data/clips/camera_*/**/*.mp4")
    every_sec: float = Field(default=1.0, ge=0.1, le=10.0)
    with_captions: Optional[bool] = Field(default=False, description="Generate BLIP-2 captions if enabled")
    respect_mode: Optional[bool] = Field(default=True, description="Honor backend indexing-mode setting")


class IndexAllResponse(BaseModel):
    ok: bool
    indexed: int
    errors: List[str] = []


@router.post("/index-all", response_model=IndexAllResponse)
def api_index_all(req: IndexAllRequest) -> IndexAllResponse:
    # Honor runtime indexing mode unless overridden
    if bool(req.respect_mode):
        mode = _get_indexing_mode()
        if mode == "structured":
            return IndexAllResponse(ok=True, indexed=0, errors=[])

    pattern = req.glob or str(settings.CLIPS_DIR / "camera_*" / "**" / "*.mp4")
    paths = list(Path(".").glob(pattern)) if not Path(pattern).is_absolute() else list(Path(pattern).parent.glob(Path(pattern).name))
    total = 0
    errors: List[str] = []
    for p in paths:
        try:
            # infer camera_id from folder name camera_{id}
            cam_id = req.camera_id
            if cam_id is None:
                parts = p.parts
                for part in parts:
                    if part.startswith("camera_"):
                        try:
                            cam_id = int(part.split("_", 1)[1])
                        except Exception:
                            pass
                if cam_id is None:
                    # skip clips we cannot associate
                    continue
            clip_path = str(p.resolve())
            clip_url = None
            # try derive relative /media/clips url if under CLIPS_DIR
            try:
                rel = p.resolve().relative_to(settings.CLIPS_DIR)
                clip_url = f"/media/clips/{rel.as_posix()}"
            except Exception:
                clip_url = None
            res: Dict[str, Any] = index_clip(
                int(cam_id),
                clip_path,
                clip_url,
                every_sec=req.every_sec,
                with_captions=bool(req.with_captions),
            )
            try:
                total += int(res.get("indexed_frames") or 0)
            except Exception:
                pass
        except Exception as e:
            errors.append(f"{p}: {e}")
    return IndexAllResponse(ok=True, indexed=total, errors=errors)


class SemanticSearchResponse(BaseModel):
    mode: str
    semantic_results: List[Dict[str, Any]]


@router.get("/search", response_model=SemanticSearchResponse)
def api_semantic_search(
    q: str = Query(..., description="Text query"),
    top_k: int = Query(50, ge=1, le=200),
    camera_id: Optional[int] = None,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
):
    try:
        out = search_unstructured(q, top_k=top_k, camera_id=camera_id, from_iso=from_iso, to_iso=to_iso)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"semantic search failed: {e}") from e
