"""
Standardized API error response envelope.
All 4xx/5xx responses use the same shape: {"error": {"code": "...", "message": "...", "details": ...}}.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ApiErrorBody(BaseModel):
    """Standard error payload for API responses."""

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Extra context (e.g. validation errors)")


class ApiErrorEnvelope(BaseModel):
    """Top-level error response: {"error": {...}}."""

    error: ApiErrorBody


# Canonical error codes used by exception handlers
ERROR_VALIDATION = "VALIDATION_ERROR"
ERROR_NOT_FOUND = "NOT_FOUND"
ERROR_BAD_REQUEST = "BAD_REQUEST"
ERROR_SERVER = "SERVER_ERROR"


def error_response(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the standard error envelope dict for JSONResponse."""
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
        }
    }
