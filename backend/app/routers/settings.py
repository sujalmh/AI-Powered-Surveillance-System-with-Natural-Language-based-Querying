from __future__ import annotations

from typing import Literal, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.db.mongo import app_settings

router = APIRouter()

IndexingMode = Literal["structured", "semantic", "both"]


class IndexingModeResponse(BaseModel):
    indexing_mode: IndexingMode = Field(description="structured | semantic | both")


class IndexingModeRequest(BaseModel):
    indexing_mode: IndexingMode


class LlmConfig(BaseModel):
    provider: str
    model: str
    api_key: str


class LlmConfigResponse(LlmConfig):
    pass


class LlmConfigRequest(LlmConfig):
    pass


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


def _get_llm_settings() -> dict:
    from backend.app.config import settings
    doc = app_settings.find_one({"key": "llm_config"}, {"_id": 0, "value": 1})
    if doc and isinstance(doc.get("value"), dict):
        return doc["value"]
    return {
        "provider": settings.LLM_PROVIDER,
        "model": settings.NL_DEFAULT_MODEL,
        "api_key": settings.OPENAI_API_KEY
    }


def _set_llm_settings(config: dict) -> None:
    # If the incoming api_key is masked (contains '...' or is '****'), don't overwrite the existing one
    incoming_key = config.get("api_key", "")
    if "..." in incoming_key or incoming_key == "****":
        existing = _get_llm_settings()
        config["api_key"] = existing.get("api_key", incoming_key)
        
    app_settings.update_one(
        {"key": "llm_config"},
        {"$set": {"key": "llm_config", "value": config}},
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


@router.get("/llm-config", response_model=LlmConfigResponse)
def get_llm_config() -> LlmConfigResponse:
    cfg = _get_llm_settings()
    key = cfg.get("api_key", "")
    if key and len(key) > 10:
        cfg["api_key"] = f"{key[:7]}...{key[-4:]}"
    elif key:
        cfg["api_key"] = "****"
    return LlmConfigResponse(**cfg)


@router.put("/llm-config", response_model=LlmConfigResponse)
def put_llm_config(req: LlmConfigRequest) -> LlmConfigResponse:
    try:
        _set_llm_settings(req.dict())
        return LlmConfigResponse(**req.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to save llm config: {e}")


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