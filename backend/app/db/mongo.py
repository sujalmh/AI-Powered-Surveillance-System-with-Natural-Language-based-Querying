from typing import Any, Dict
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.collection import Collection
from pymongo.database import Database

from backend.app.config import settings


client: MongoClient = MongoClient(settings.MONGODB_URI)
db: Database = client[settings.MONGO_DB_NAME]

# Collections
cameras: Collection = db["cameras"]
detections: Collection = db["detections"]
tracks: Collection = db["tracks"]
events: Collection = db["events"]
alerts: Collection = db["alerts"]
alert_logs: Collection = db["alert_logs"]
chat_messages: Collection = db["chat_messages"]
videos: Collection = db["videos"]  # optional manifest of recording segments
vlm_frames: Collection = db["vlm_frames"]  # semantic frame-level metadata (FAISS stores vectors)
app_settings: Collection = db["settings"]  # app-level settings (e.g., indexing mode)
anomaly_events: Collection = db["anomaly_events"]  # cached daily anomaly detection results


def init_indexes() -> None:
    # cameras
    cameras.create_index([("camera_id", ASCENDING)], unique=True)
    # app settings
    try:
        app_settings.create_index([("key", ASCENDING)], unique=True)
    except Exception:
        pass

    # detections: frequent queries by time/object/color
    detections.create_index([("camera_id", ASCENDING), ("timestamp", DESCENDING)])
    detections.create_index(
        [("objects.object_name", ASCENDING), ("objects.color", ASCENDING), ("timestamp", DESCENDING)]
    )
    try:
        detections.create_index([("person_count", ASCENDING), ("timestamp", DESCENDING)])
    except Exception:
        pass

    # tracks: per-camera persistent trackers (from BoT-SORT)
    try:
        tracks.create_index([("camera_id", ASCENDING), ("track_id", ASCENDING)], unique=True)
        tracks.create_index([("last_seen", DESCENDING)])
        tracks.create_index([("first_seen", DESCENDING)])
    except Exception:
        pass

    # events: time-based queries, summary search
    events.create_index([("camera_id", ASCENDING), ("start_ts", DESCENDING)])
    try:
        events.create_index([("summary", TEXT)])
    except Exception:
        # Text index may already exist or not supported by deployment tier
        pass

    # alerts
    alerts.create_index([("enabled", ASCENDING)])
    alerts.create_index([("rule.cameras", ASCENDING)])

    # alert logs
    alert_logs.create_index([("alert_id", ASCENDING), ("triggered_at", DESCENDING)])
    alert_logs.create_index([("camera_id", ASCENDING), ("triggered_at", DESCENDING)])

    # chat messages
    chat_messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])

    # videos (recording segments manifest)
    videos.create_index([("camera_id", ASCENDING), ("start_ts", DESCENDING)])

    # vlm_frames: semantic search metadata (FAISS holds vectors)
    try:
        vlm_frames.create_index([("clip_path", ASCENDING), ("frame_index", ASCENDING)], unique=True)
        vlm_frames.create_index([("camera_id", ASCENDING), ("frame_ts", DESCENDING)])
        vlm_frames.create_index([("hash", ASCENDING)])
        vlm_frames.create_index([("model", ASCENDING)])
    except Exception:
        pass

    # anomaly_events: daily anomaly detection cache
    try:
        anomaly_events.create_index([("date", ASCENDING)], unique=True)
    except Exception:
        pass


def get_db_info() -> Dict[str, Any]:
    return {
        "name": settings.MONGO_DB_NAME,
        "collections": sorted(db.list_collection_names()),
    }


# Initialize indexes when module is imported
try:
    init_indexes()
except Exception as e:
    # Avoid crashing the app if the DB is temporarily unavailable at import time.
    # The application can provide a /health endpoint to retry initialization on demand.
    print(f"Warning: Failed to initialize MongoDB indexes at import time: {e}")
