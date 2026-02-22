from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.core.async_utils import run_sync
from backend.app.core.settings_reader import get_indexing_mode
from backend.app.services.sem_indexer import index_clip
from backend.app.services.sem_search import search_unstructured

router = APIRouter()


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
async def api_index_clip(req: IndexClipRequest) -> IndexClipResponse:
    def _block():
        try:
            if bool(req.respect_mode):
                mode = get_indexing_mode()
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
            return IndexClipResponse(
                ok=bool(result.get("ok", False)),
                indexed_frames=int(result.get("indexed_frames", 0) or 0),
                model=(str(result.get("model")) if result.get("model") is not None else None),
                backend=(str(result.get("backend")) if result.get("backend") is not None else None),
                message=(str(result.get("message")) if result.get("message") is not None else None),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"index-clip failed: {e}") from e
    return await run_sync(_block)


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
async def api_index_all(req: IndexAllRequest) -> IndexAllResponse:
    def _block():
        if bool(req.respect_mode):
            mode = get_indexing_mode()
            if mode == "structured":
                return IndexAllResponse(ok=True, indexed=0, errors=[])
        pattern = req.glob or str(settings.CLIPS_DIR / "camera_*" / "**" / "*.mp4")
        paths = list(Path(".").glob(pattern)) if not Path(pattern).is_absolute() else list(Path(pattern).parent.glob(Path(pattern).name))
        total = 0
        errors: List[str] = []
        for p in paths:
            try:
                cam_id = req.camera_id
                if cam_id is None:
                    for part in p.parts:
                        if part.startswith("camera_"):
                            try:
                                cam_id = int(part.split("_", 1)[1])
                            except Exception:
                                pass
                            break
                    if cam_id is None:
                        continue
                clip_path = str(p.resolve())
                clip_url = None
                try:
                    rel = p.resolve().relative_to(settings.CLIPS_DIR)
                    clip_url = f"/media/clips/{rel.as_posix()}"
                except Exception:
                    pass
                res = index_clip(int(cam_id), clip_path, clip_url, every_sec=req.every_sec, with_captions=bool(req.with_captions))
                try:
                    total += int(res.get("indexed_frames") or 0)
                except Exception:
                    pass
            except Exception as e:
                errors.append(f"{p}: {e}")
        return IndexAllResponse(ok=True, indexed=total, errors=errors)
    return await run_sync(_block)


class SemanticSearchResponse(BaseModel):
    mode: str
    semantic_results: List[Dict[str, Any]]


@router.get("/search", response_model=SemanticSearchResponse)
async def api_semantic_search(
    q: str = Query(..., description="Text query"),
    top_k: int = Query(50, ge=1, le=200),
    camera_id: Optional[int] = None,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
):
    def _block():
        try:
            return search_unstructured(q, top_k=top_k, camera_id=camera_id, from_iso=from_iso, to_iso=to_iso)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"semantic search failed: {e}") from e
    return await run_sync(_block)
