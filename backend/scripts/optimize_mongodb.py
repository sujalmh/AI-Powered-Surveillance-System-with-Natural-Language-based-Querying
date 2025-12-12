from __future__ import annotations

from pymongo import MongoClient, ASCENDING, DESCENDING


def optimize_mongodb_schema() -> None:
    client = MongoClient()
    db = client["ai_surveillance"]
    detections = db["detections"]

    # Add critical indexes for hybrid search and structured queries
    print("[Mongo] Creating indexes on detections ...")
    detections.create_index([("timestamp", DESCENDING), ("camera_id", ASCENDING)])
    detections.create_index([("objects.object_name", ASCENDING), ("timestamp", DESCENDING)])
    detections.create_index([("objects.color", ASCENDING), ("timestamp", DESCENDING)])
    detections.create_index([("track_id", ASCENDING), ("camera_id", ASCENDING)])
    # Accessory presence flags (if schema is extended later)
    try:
        detections.create_index([("person_attributes.bag_confidence", DESCENDING)])
        detections.create_index([("person_attributes.hat_confidence", DESCENDING)])
        detections.create_index([("person_attributes.coat_jacket_confidence", DESCENDING)])
    except Exception:
        # Ignore if attributes not present in deployment
        pass

    # Tracks collection: ensure exists and has proper indexes
    tracks = db["tracks"]
    print("[Mongo] Creating indexes on tracks ...")
    tracks.create_index([("camera_id", ASCENDING), ("track_id", ASCENDING)], unique=True)
    tracks.create_index([("last_seen", DESCENDING)])
    tracks.create_index([("first_seen", DESCENDING)])

    # Semantic frames (if present)
    vlm_frames = db["vlm_frames"]
    print("[Mongo] Creating indexes on vlm_frames ...")
    try:
        vlm_frames.create_index([("clip_path", ASCENDING), ("frame_index", ASCENDING)], unique=True)
        vlm_frames.create_index([("camera_id", ASCENDING), ("frame_ts", DESCENDING)])
        vlm_frames.create_index([("hash", ASCENDING)])
        vlm_frames.create_index([("model", ASCENDING)])
    except Exception:
        pass

    print("[Mongo] Indexes created successfully")
    print("[Mongo] Current indexes by collection:")
    for name in ("detections", "tracks", "vlm_frames"):
        try:
            print(f" - {name}")
            for idx in db[name].list_indexes():
                print(f"   {idx}")
        except Exception as e:
            print(f"   (could not list indexes for {name}: {e})")


if __name__ == "__main__":
    optimize_mongodb_schema()
