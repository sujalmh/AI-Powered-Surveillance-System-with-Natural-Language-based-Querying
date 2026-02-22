"""
Central read/write for app-level settings stored in MongoDB (e.g. indexing mode).
Single source for indexing_mode used by settings router, semantic router, and object_detection.
"""
from typing import Literal, Tuple

from backend.app.db.mongo import app_settings

IndexingMode = Literal["structured", "semantic", "both"]
VALID_INDEXING_MODES: Tuple[str, ...] = ("structured", "semantic", "both")


def get_indexing_mode() -> IndexingMode:
    """Return current indexing mode from DB; default 'both' if unset or invalid."""
    try:
        doc = app_settings.find_one({"key": "indexing_mode"}, {"_id": 0, "value": 1})
        val = (doc or {}).get("value") if doc else None
        if val in VALID_INDEXING_MODES:
            return val  # type: ignore[return-value]
    except Exception:
        pass
    return "both"


def set_indexing_mode(mode: IndexingMode) -> None:
    """Persist indexing mode to DB."""
    app_settings.update_one(
        {"key": "indexing_mode"},
        {"$set": {"key": "indexing_mode", "value": mode}},
        upsert=True,
    )
