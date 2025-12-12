import shutil
import os
from pathlib import Path
from backend.app.config import settings
from backend.app.db.mongo import (
    detections,
    videos,
    alerts,
    cameras,
    tracks,
    events,
    alert_logs,
    chat_messages,
    vlm_frames
)

def perform_cleanup():
    """
    Wipes all data to start fresh:
    1. Deletes all files in media directories (recordings, clips, snapshots)
    2. Deletes FAISS indices
    3. Drops MongoDB collections
    """
    print("!!! Performing System Reset. Wiping all data... !!!")
    
    # 1. Clear directories
    dirs_to_clear = [
        settings.RECORDINGS_DIR,
        settings.CLIPS_DIR,
        settings.SNAPSHOTS_DIR,
        settings.FAISS_DIR,
        settings.FAISS_PERSON_DIR
    ]
    
    for d in dirs_to_clear:
        if d.exists():
            print(f"Cleaning directory: {d}")
            for item in d.iterdir():
                try:
                    if item.name == ".gitkeep":
                        continue
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"Failed to delete {item}: {e}")
    
    # 2. Drop DB collections
    print("Dropping MongoDB collections...")
    try:
        detections.drop()
        videos.drop()
        alerts.drop()
        cameras.drop()
        tracks.drop()
        events.drop()
        alert_logs.drop()
        chat_messages.drop()
        vlm_frames.drop()
        
        # We might want to keep app_settings to preserve the user's config
        # app_settings.drop() 
        print("MongoDB collections dropped.")
    except Exception as e:
        print(f"Error dropping collections: {e}")

    print("!!! Data wipe complete. System reset. !!!")
