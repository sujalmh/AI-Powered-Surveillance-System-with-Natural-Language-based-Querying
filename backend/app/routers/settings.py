from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.db.mongo import app_settings

router = APIRouter()

IndexingMode = Literal["structured", "semantic", "both"]


class IndexingModeResponse(BaseModel):
    indexing_mode: IndexingMode = Field(description="structured | semantic | both")


class IndexingModeRequest(BaseModel):
    indexing_mode: IndexingMode


def _get_mode() -> IndexingMode:
    doc = app_settings.find_one({"key": "indexing_mode"}, {"_id": 0, "value": 1})
    if doc and isinstance(doc.get("value"), str) and doc["value"] in ("structured", "semantic", "both"):
        return doc["value"]  # type: ignore[return-value]
    # default: both (run existing pipeline and VLM together)
    return "both"


def _set_mode(mode: IndexingMode) -> None:
    app_settings.update_one(
        {"key": "indexing_mode"},
        {"$set": {"key": "indexing_mode", "value": mode}},
        upsert=True,
    )


@router.get("/indexing-mode", response_model=IndexingModeResponse)
def get_indexing_mode() -> IndexingModeResponse:
    return IndexingModeResponse(indexing_mode=_get_mode())


@router.put("/indexing-mode", response_model=IndexingModeResponse)
def put_indexing_mode(req: IndexingModeRequest) -> IndexingModeResponse:
    try:
        _set_mode(req.indexing_mode)
        return IndexingModeResponse(indexing_mode=req.indexing_mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to save indexing mode: {e}")


@router.delete("/reset")
def reset_system():
    """
    Danger zone: Wipes all data (videos, files, DB, indices).
    """
    try:
        from backend.app.services.cleanup import perform_cleanup
        perform_cleanup()
        return {"status": "ok", "message": "System reset complete. All data wiped."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")
