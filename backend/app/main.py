import atexit
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.db.mongo import get_db_info, client as mongo_client
from backend.app.routers.cameras import router as cameras_router

logger = logging.getLogger(__name__)

try:
    from backend.app.services.detection_runner import runner
except Exception:
    runner = None


# ── Force-exit handler ──────────────────────────────────────────────
# PyTorch, OpenCLIP, PyMongo, and other ML libraries create non-daemon
# background threads.  On Windows, when uvicorn --reload tries to shut
# down the worker process, those threads keep the process alive forever.
# This atexit handler runs AFTER the asyncio loop and lifespan shutdown
# have completed and forcibly terminates the process so it doesn't hang.
#
# NOTE: uvicorn installs its own SIGTERM handler, so no application-level
# SIGTERM registration is performed here.  Cleanup relies on atexit/_force_exit.
def _force_exit():
    """Last-resort exit: kill the process even if stray threads linger."""
    os._exit(0)


atexit.register(_force_exit)


# ── Router imports ──────────────────────────────────────────────────
try:
    from backend.app.routers.detections import router as detections_router  # type: ignore
except Exception:
    detections_router = None  # type: ignore

try:
    from backend.app.routers.chat import router as chat_router  # type: ignore
except Exception:
    logger.exception("Failed to import chat router")
    chat_router = None  # type: ignore

try:
    from backend.app.routers.alerts import router as alerts_router  # type: ignore
except Exception:
    alerts_router = None  # type: ignore

try:
    from backend.app.routers.videos import router as videos_router  # type: ignore
except Exception:
    videos_router = None  # type: ignore

try:
    from backend.app.routers.semantic import router as semantic_router  # type: ignore
except Exception:
    semantic_router = None  # type: ignore

try:
    from backend.app.routers.settings import router as settings_router  # type: ignore
except Exception:
    settings_router = None  # type: ignore

try:
    from backend.app.routers.dashboard import router as dashboard_router  # type: ignore
except Exception:
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

    # Shutdown — best-effort graceful cleanup before atexit fires
    if runner is not None:
        try:
            for cid in list(runner.list_running().keys()):
                try:
                    runner.stop_camera(cid, timeout=2)
                except Exception:
                    logger.exception("Error stopping camera %s during shutdown", cid)
        except Exception:
            logger.exception("Error listing running cameras during shutdown")
    try:
        mongo_client.close()
    except Exception:
        logger.exception("Error closing MongoDB connection during shutdown")


app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, lifespan=lifespan)

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
