"""Role/zone access control and audit logging.

Deployment model
----------------
The bundled demo runs with ``ACCESS_CONTROL_ENABLED=false`` and therefore
behaves exactly as before.  In a deployment, terminate authentication at a
trusted reverse proxy and let that proxy inject *and strip client-supplied*
identity headers:

``X-Role``               admin | operator | viewer
``X-Actor``              stable operator/service identifier
``X-Allowed-Cameras``    comma-separated camera ids, or ``*`` for all
``X-Allowed-Zones``      comma-separated zone names/ids, or ``*`` for all

The FastAPI middleware in this module enforces a header role and camera/zone
scope when ``ACCESS_CONTROL_ENABLED=true``.  Every API/media access is appended
to ``data/logs/audit.jsonl`` (and optionally mirrored to Mongo).  This module is
deliberately dependency-light so tests and CLI tools can use the policy helpers
without starting the web app.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.app.config import settings


class Role(str, Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


ROLE_LEVEL: Dict[Role, int] = {
    Role.VIEWER: 1,
    Role.OPERATOR: 2,
    Role.ADMIN: 3,
}

# Paths that never require a role header (liveness/docs).  Surveillance media
# and every /api route do require one when access control is enabled.
OPEN_PATHS = {"/", "/health", "/favicon.ico", "/openapi.json", "/docs", "/redoc"}


class AccessDenied(PermissionError):
    """Raised when a principal lacks a role or camera/zone grant."""


@dataclass(frozen=True)
class Principal:
    """Authenticated caller plus optional camera/zone grants.

    ``None`` means "no explicit restriction" for an authenticated principal;
    an empty set means "no cameras/zones granted".  Admin always bypasses
    camera/zone restrictions.
    """

    role: Role = Role.VIEWER
    actor: str = "anonymous"
    allowed_cameras: Optional[frozenset[int]] = None
    allowed_zones: Optional[frozenset[str]] = None
    authenticated: bool = False

    def can(self, required: Role | str) -> bool:
        req = coerce_role(required)
        return ROLE_LEVEL[self.role] >= ROLE_LEVEL[req]

    @property
    def unrestricted(self) -> bool:
        return self.role == Role.ADMIN


def coerce_role(value: Role | str | None) -> Role:
    if isinstance(value, Role):
        return value
    text = str(value or "").strip().lower()
    try:
        return Role(text)
    except ValueError as exc:
        raise AccessDenied(f"unknown role: {value!r}") from exc


def _header_value(headers: Any, name: str) -> Optional[str]:
    """Case-insensitive header lookup that works for Starlette and dicts."""
    try:
        value = headers.get(name)
        if value is not None:
            return value
    except Exception:
        pass
    if isinstance(headers, Mapping):
        wanted = name.lower()
        for key, value in headers.items():
            if str(key).lower() == wanted:
                return value
    return None


def _parse_text_set(raw: Optional[str]) -> Optional[frozenset[str]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text == "*":
        return None
    return frozenset(part.strip().lower() for part in text.split(",") if part.strip())


def _parse_camera_set(raw: Optional[str]) -> Optional[frozenset[int]]:
    values = _parse_text_set(raw)
    if values is None:
        return None
    cameras: set[int] = set()
    for value in values:
        try:
            cameras.add(int(value))
        except (TypeError, ValueError):
            continue
    # An explicitly supplied header with no parseable ids is a deny-all grant.
    return frozenset(cameras)


def principal_from_headers(headers: Any) -> Principal:
    """Build a :class:`Principal` from trusted role/scope headers.

    A missing/empty role is an authentication failure when called by the
    middleware.  Policy helpers may construct a :class:`Principal` directly in
    tests or background jobs.
    """
    role_raw = _header_value(headers, settings.ACCESS_ROLE_HEADER)
    if role_raw is None or not str(role_raw).strip():
        raise AccessDenied("missing role header")
    role = coerce_role(role_raw)
    actor = str(
        _header_value(headers, settings.ACCESS_ACTOR_HEADER) or "anonymous"
    ).strip() or "anonymous"
    cameras = _parse_camera_set(
        _header_value(headers, settings.ACCESS_CAMERA_HEADER)
    )
    zones = _parse_text_set(_header_value(headers, settings.ACCESS_ZONE_HEADER))
    return Principal(
        role=role,
        actor=actor,
        allowed_cameras=cameras,
        allowed_zones=zones,
        authenticated=True,
    )


def is_camera_allowed(principal: Principal, camera_id: Any) -> bool:
    if principal.unrestricted or principal.allowed_cameras is None:
        return True
    if camera_id is None:
        return True
    try:
        return int(camera_id) in principal.allowed_cameras
    except (TypeError, ValueError):
        return False


def is_zone_allowed(principal: Principal, zone: Any) -> bool:
    if principal.unrestricted or principal.allowed_zones is None:
        return True
    if zone is None or not str(zone).strip():
        return True
    return str(zone).strip().lower() in principal.allowed_zones


def authorize_camera(principal: Principal, camera_id: Any) -> None:
    if not is_camera_allowed(principal, camera_id):
        raise AccessDenied(f"camera {camera_id} is outside the caller's grant")


def authorize_zone(principal: Principal, zone: Any) -> None:
    if not is_zone_allowed(principal, zone):
        raise AccessDenied(f"zone {zone!r} is outside the caller's grant")


def filter_authorized_results(
    principal: Principal,
    results: Iterable[Dict[str, Any]],
    camera_key: str = "camera_id",
    zone_key: str = "zone",
) -> List[Dict[str, Any]]:
    """Defence-in-depth filter for routers that return record lists."""
    out: List[Dict[str, Any]] = []
    for record in results:
        if not isinstance(record, dict):
            continue
        if not is_camera_allowed(principal, record.get(camera_key)):
            continue
        if not is_zone_allowed(principal, record.get(zone_key)):
            continue
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
_AUDIT_LOCK = threading.Lock()


def audit_log_path() -> Path:
    return Path(settings.LOGS_DIR) / "audit.jsonl"


def audit_log(
    event: str,
    *,
    actor: str = "system",
    role: str | Role | None = None,
    action: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None,
    outcome: str = "success",
    status_code: Optional[int] = None,
    camera_id: Any = None,
    zone: Any = None,
    details: Optional[Mapping[str, Any]] = None,
    log_path: Path | str | None = None,
) -> Dict[str, Any]:
    """Append one JSON object to the audit trail and return the record.

    The write is atomic at the line level (single append under a lock) and
    never raises: audit failure must not take down the request path.  Pass
    ``log_path`` in tests to avoid touching the deployment log.
    """
    target = Path(log_path) if log_path is not None else audit_log_path()
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
        "actor": str(actor or "unknown"),
        "role": role.value if isinstance(role, Role) else (str(role) if role else None),
        "action": action,
        "method": method,
        "path": path,
        "outcome": str(outcome),
        "status_code": int(status_code) if status_code is not None else None,
        "camera_id": camera_id,
        "zone": zone,
    }
    if details:
        record["details"] = dict(details)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _AUDIT_LOCK:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        # Logging must remain best-effort.  Callers can still return a result.
        pass

    if getattr(settings, "AUDIT_MONGO_ENABLED", False):
        try:
            from backend.app.db.mongo import db

            db["audit_logs"].insert_one(dict(record))
        except Exception:
            pass
    return record


# ---------------------------------------------------------------------------
# FastAPI glue
# ---------------------------------------------------------------------------
def _camera_from_path(path: str) -> Optional[int]:
    match = re.search(r"/(?:camera|cam)[_-]?(\d+)(?:/|$)", path, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _required_role_for(request: Request) -> Role:
    method = request.method.upper()
    if method in {"GET", "HEAD", "OPTIONS"}:
        return Role.VIEWER
    if request.url.path.startswith("/api/settings"):
        return Role.ADMIN
    return Role.OPERATOR


class AccessControlMiddleware(BaseHTTPMiddleware):
    """Enforce role/scope headers and write the audit trail.

    Inactive by default.  With ``ACCESS_CONTROL_ENABLED=true``:
    * ``OPTIONS`` and liveness/docs paths stay open (CORS preflight);
    * read routes require viewer, writes require operator, settings writes
      require admin;
    * camera/zone grants are checked from query/path parameters;
    * every allowed or denied API/media access is audited.

    This is a coarse, HTTP-level guard.  Authenticated identity MUST come from a
    trusted proxy; the app does not validate bearer tokens itself.
    """

    async def dispatch(self, request: Request, call_next):
        if not getattr(settings, "ACCESS_CONTROL_ENABLED", False):
            return await call_next(request)

        path = request.url.path
        if request.method.upper() == "OPTIONS" or path in OPEN_PATHS or path.startswith("/docs"):
            return await call_next(request)

        try:
            principal = principal_from_headers(request.headers)
        except AccessDenied as exc:
            audit_log(
                "access.denied",
                actor="anonymous",
                role=None,
                action="authenticate",
                path=path,
                method=request.method,
                outcome="denied",
                status_code=401,
                details={"reason": str(exc)},
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "authentication required", "code": "unauthorized"},
            )

        request.state.principal = principal
        required = _required_role_for(request)
        if not principal.can(required):
            audit_log(
                "access.denied",
                actor=principal.actor,
                role=principal.role,
                action="authorize_role",
                path=path,
                method=request.method,
                outcome="denied",
                status_code=403,
                details={"required_role": required.value},
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "insufficient role", "code": "forbidden"},
            )

        camera_id = request.query_params.get("camera_id")
        if camera_id is not None:
            try:
                camera_value: Any = int(camera_id)
            except ValueError:
                camera_value = camera_id
            if not is_camera_allowed(principal, camera_value):
                return self._deny_scope(
                    request, principal, "camera", camera_value
                )
        path_camera = _camera_from_path(path)
        if path_camera is not None and not is_camera_allowed(principal, path_camera):
            return self._deny_scope(request, principal, "camera", path_camera)

        zone = request.query_params.get("zone")
        if zone is not None and not is_zone_allowed(principal, zone):
            return self._deny_scope(request, principal, "zone", zone)

        started = datetime.now(timezone.utc)
        try:
            response = await call_next(request)
        except Exception:
            audit_log(
                "access.request",
                actor=principal.actor,
                role=principal.role,
                action=f"{request.method} {path}",
                path=path,
                method=request.method,
                outcome="error",
                status_code=500,
                camera_id=path_camera,
                zone=zone,
            )
            raise

        if path.startswith("/api") or path.startswith("/media"):
            elapsed_ms = (
                datetime.now(timezone.utc) - started
            ).total_seconds() * 1000.0
            audit_log(
                "access.request",
                actor=principal.actor,
                role=principal.role,
                action=f"{request.method} {path}",
                path=path,
                method=request.method,
                outcome="success" if response.status_code < 400 else "error",
                status_code=response.status_code,
                camera_id=path_camera,
                zone=zone,
                details={"duration_ms": round(elapsed_ms, 3)},
            )
        return response

    @staticmethod
    def _deny_scope(request: Request, principal: Principal, kind: str, value: Any):
        audit_log(
            "access.denied",
            actor=principal.actor,
            role=principal.role,
            action=f"authorize_{kind}",
            path=request.url.path,
            method=request.method,
            outcome="denied",
            status_code=403,
            camera_id=value if kind == "camera" else None,
            zone=value if kind == "zone" else None,
            details={"reason": f"{kind} outside grant"},
        )
        return JSONResponse(
            status_code=403,
            content={
                "detail": f"{kind} is outside the caller's grant",
                "code": "forbidden",
            },
        )


def require_role(required: Role | str):
    """FastAPI dependency factory for per-route role checks."""

    async def dependency(request: Request) -> Principal:
        principal = getattr(request.state, "principal", None)
        if principal is None:
            if not getattr(settings, "ACCESS_CONTROL_ENABLED", False):
                principal = Principal(role=Role.ADMIN, actor="open-access")
            else:
                principal = principal_from_headers(request.headers)
                request.state.principal = principal
        if not principal.can(required):
            from fastapi import HTTPException

            raise HTTPException(status_code=403, detail="insufficient role")
        return principal

    return dependency


__all__ = [
    "AccessControlMiddleware",
    "AccessDenied",
    "Principal",
    "Role",
    "audit_log",
    "audit_log_path",
    "authorize_camera",
    "authorize_zone",
    "coerce_role",
    "filter_authorized_results",
    "is_camera_allowed",
    "is_zone_allowed",
    "principal_from_headers",
    "require_role",
]
