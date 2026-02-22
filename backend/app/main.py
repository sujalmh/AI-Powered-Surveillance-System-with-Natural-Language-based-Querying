import logging
import os
import signal
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.app.config import settings
from backend.app.core.logging_config import configure_logging

configure_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_json=os.getenv("LOG_JSON", "false").lower() == "true",
)

from backend.app.db.mongo import get_db_info, client as mongo_client
from backend.app.routers.cameras import router as cameras_router
from backend.app.schemas.responses import (
    error_response,
    ERROR_VALIDATION,
    ERROR_NOT_FOUND,
    ERROR_BAD_REQUEST,
    ERROR_SERVER,
)

logger = logging.getLogger(__name__)

try:
    from backend.app.services.detection_runner import runner
except Exception:
    runner = None


# ── Shutdown machinery ──────────────────────────────────────────────
_shutdown_event = threading.Event()
_HARD_DEADLINE_SEC = 6.0


def _graceful_shutdown_work() -> None:
    """Best-effort cleanup: stop cameras, task pool, mongo."""
    if runner is not None:
        try:
            runner.stop_all(timeout=2.0)
        except Exception:
            pass

    try:
        from backend.app.services.task_queue import shutdown as tq_shutdown
        tq_shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

    try:
        mongo_client.close()
    except Exception:
        pass


def _watchdog() -> None:
    """Daemon thread that waits for the shutdown signal, then hard-kills after a deadline."""
    _shutdown_event.wait()
    logger.info("Shutdown watchdog started — hard deadline in %.0fs", _HARD_DEADLINE_SEC)
    _graceful_shutdown_work()
    # Give the main thread / uvicorn a moment to exit cleanly
    main_thread = threading.main_thread()
    main_thread.join(timeout=_HARD_DEADLINE_SEC)
    if main_thread.is_alive():
        logger.warning("Graceful shutdown timed out — forcing exit")
        for h in logging.root.handlers:
            try:
                h.flush()
            except Exception:
                pass
        os._exit(0)


_watchdog_thread = threading.Thread(target=_watchdog, name="shutdown-watchdog", daemon=True)
_watchdog_thread.start()


def _signal_handler(signum, frame):
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger.info("Received %s — initiating shutdown", sig_name)
    _shutdown_event.set()
    # Re-raise as KeyboardInterrupt so uvicorn's shutdown proceeds normally
    if signum == signal.SIGINT:
        raise KeyboardInterrupt


# Install signal handlers (only safe in the main thread)
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)


# ── Router imports ──────────────────────────────────────────────────
try:
    from backend.app.routers.detections import router as detections_router  # type: ignore
except Exception:
    logger.exception("Failed to import detections router")
    detections_router = None  # type: ignore

try:
    from backend.app.routers.chat import router as chat_router  # type: ignore
except Exception:
    logger.exception("Failed to import chat router")
    chat_router = None  # type: ignore

try:
    from backend.app.routers.alerts import router as alerts_router  # type: ignore
except Exception:
    logger.exception("Failed to import alerts router")
    alerts_router = None  # type: ignore

try:
    from backend.app.routers.videos import router as videos_router  # type: ignore
except Exception:
    logger.exception("Failed to import videos router")
    videos_router = None  # type: ignore

try:
    from backend.app.routers.semantic import router as semantic_router  # type: ignore
except Exception:
    logger.exception("Failed to import semantic router")
    semantic_router = None  # type: ignore

try:
    from backend.app.routers.settings import router as settings_router  # type: ignore
except Exception:
    logger.exception("Failed to import settings router")
    settings_router = None  # type: ignore

try:
    from backend.app.routers.dashboard import router as dashboard_router  # type: ignore
except Exception:
    logger.exception("Failed to import dashboard router")
    dashboard_router = None  # type: ignore


# ── Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — eagerly verify that chat dependencies can be imported so
    # mis-configurations surface at startup, not on the first /send request.
    try:
        from backend.app.services.nl_parser import parse_nl_with_llm  # noqa: F401
        from backend.app.services.unified_retrieval import UnifiedRetrieval  # noqa: F401
    except Exception:
        logger.warning(
            "Chat dependencies (parse_nl_with_llm / UnifiedRetrieval) failed to import; "
            "/api/chat/send will fall back to lazy import at request time",
            exc_info=True,
        )

    yield

    # Lifespan shutdown — signal the watchdog and let it handle cleanup.
    # Also do inline cleanup in case the watchdog hasn't fired yet.
    logger.info("Lifespan shutdown triggered")
    _shutdown_event.set()
    _graceful_shutdown_work()


app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, lifespan=lifespan)


# ── Global exception handlers (standardized error envelope) ─────────────
def _http_status_to_code(status_code: int) -> str:
    if status_code == 404:
        return ERROR_NOT_FOUND
    if status_code == 400:
        return ERROR_BAD_REQUEST
    if status_code == 422:
        return ERROR_VALIDATION
    return ERROR_SERVER


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=error_response(
            ERROR_VALIDATION,
            "Request validation failed",
            details={"errors": exc.errors()},
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("message", str(detail)) if isinstance(detail.get("message"), str) else str(detail)
        details = {k: v for k, v in detail.items() if k != "message"}
    else:
        message = str(detail)
        details = None
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            _http_status_to_code(exc.status_code),
            message,
            details=details or None,
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=error_response(
            ERROR_SERVER,
            "Internal server error",
            details={"detail": str(exc)} if getattr(settings, "DEBUG", False) else None,
        ),
    )


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS or [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3012",
        "http://127.0.0.1:3012",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(cameras_router, prefix="/api/cameras", tags=["cameras"])
if detections_router:
    app.include_router(detections_router, prefix="/api/detections", tags=["detections"])
if chat_router:
    app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
if alerts_router:
    app.include_router(alerts_router, prefix="/api/alerts", tags=["alerts"])
if videos_router:
    app.include_router(videos_router, prefix="/api/videos", tags=["videos"])
if semantic_router:
    app.include_router(semantic_router, prefix="/api/semantic", tags=["semantic"])
if settings_router:
    app.include_router(settings_router, prefix="/api/settings", tags=["settings"])
if dashboard_router:
    app.include_router(dashboard_router, prefix="/api/dashboard", tags=["dashboard"])

# Static media mounts (local disk). Access: /media/{type}/...
app.mount("/media/recordings", StaticFiles(directory=str(settings.RECORDINGS_DIR)), name="recordings")
app.mount("/media/clips", StaticFiles(directory=str(settings.CLIPS_DIR)), name="clips")
app.mount("/media/snapshots", StaticFiles(directory=str(settings.SNAPSHOTS_DIR)), name="snapshots")


@app.get("/health")
def health():
    info = {}
    try:
        info = get_db_info()
    except Exception as e:
        info = {"error": f"db: {e}"}
    return {"status": "ok", "version": app.version, "db": info}


@app.get("/")
def root():
    return {
        "name": settings.APP_NAME,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "routers": ["/api/cameras", "/api/detections", "/api/chat", "/api/alerts", "/api/videos", "/api/semantic"],
        "media": {
            "recordings": "/media/recordings",
            "clips": "/media/clips",
            "snapshots": "/media/snapshots",
        },
    }
