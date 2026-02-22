from __future__ import annotations

from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.async_utils import run_sync
from backend.app.core.settings_reader import IndexingMode, get_indexing_mode, set_indexing_mode
from backend.app.db.mongo import app_settings

router = APIRouter()


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


# Constants for masking
MASK_ELLIPSIS = "..."
MASK_SHORT = "****"


def _is_masked(key: str) -> bool:
    """Check if the provided key is a masked visualization."""
    return MASK_ELLIPSIS in (key or "") or key == MASK_SHORT


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
    # If the incoming api_key is masked, don't overwrite the existing one
    incoming_key = config.get("api_key", "")
    if _is_masked(incoming_key):
        existing = _get_llm_settings()
        config["api_key"] = existing.get("api_key", incoming_key)
        
    app_settings.update_one(
        {"key": "llm_config"},
        {"$set": {"key": "llm_config", "value": config}},
        upsert=True,
    )


@router.get("/indexing-mode", response_model=IndexingModeResponse)
async def get_indexing_mode_route() -> IndexingModeResponse:
    return await run_sync(lambda: IndexingModeResponse(indexing_mode=get_indexing_mode()))


@router.put("/indexing-mode", response_model=IndexingModeResponse)
async def put_indexing_mode(req: IndexingModeRequest) -> IndexingModeResponse:
    def _block():
        try:
            set_indexing_mode(req.indexing_mode)
            return IndexingModeResponse(indexing_mode=req.indexing_mode)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to save indexing mode: {e}")
    return await run_sync(_block)


@router.get("/llm-config", response_model=LlmConfigResponse)
async def get_llm_config() -> LlmConfigResponse:
    def _block():
        cfg = _get_llm_settings()
        key = cfg.get("api_key", "")
        if key and len(key) > 10:
            cfg["api_key"] = f"{key[:7]}{MASK_ELLIPSIS}{key[-4:]}"
        elif key:
            cfg["api_key"] = MASK_SHORT
        return LlmConfigResponse(**cfg)
    return await run_sync(_block)


@router.put("/llm-config", response_model=LlmConfigResponse)
async def put_llm_config(req: LlmConfigRequest) -> LlmConfigResponse:
    def _block():
        try:
            _set_llm_settings(req.model_dump())
            return LlmConfigResponse(**req.model_dump())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to save llm config: {e}")
    return await run_sync(_block)


@router.delete("/reset")
async def reset_system():
    """Danger zone: Wipes all data (videos, files, DB, indices)."""

    def _block():
        try:
            from backend.app.services.cleanup import perform_cleanup
            perform_cleanup()
            return {"status": "ok", "message": "System reset complete. All data wiped."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reset failed: {e}")
    return await run_sync(_block)