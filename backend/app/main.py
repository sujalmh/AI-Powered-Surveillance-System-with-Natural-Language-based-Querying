from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.db.mongo import get_db_info
from backend.app.routers.cameras import router as cameras_router

# Optional future routers (to be added as files are created)
try:
    from backend.app.routers.detections import router as detections_router  # type: ignore
except Exception:
    detections_router = None  # type: ignore

try:
    from backend.app.routers.chat import router as chat_router  # type: ignore
except Exception:
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


app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

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
