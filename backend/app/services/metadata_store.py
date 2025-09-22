import os
import uuid
import shutil
import datetime
from moviepy.editor import VideoFileClip
from pymongo import MongoClient, errors
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
# mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")


def store_video_metadata(file_path, original_filename):
    dest_folder = "uploaded_videos"
    os.makedirs(dest_folder, exist_ok=True)

    video_id = str(uuid.uuid4())

    # Extract extension and rename file to a unique filename
    extension = os.path.splitext(original_filename)[1]  # e.g. ".mp4"
    unique_filename = f"{video_id}{extension}"
    dest_path = os.path.join(dest_folder, unique_filename)

    shutil.copy(file_path, dest_path)

    upload_time = datetime.datetime.utcnow()
    file_size = os.path.getsize(dest_path)

    try:
        clip = VideoFileClip(dest_path)
        duration = clip.duration
        width, height = clip.size 
        clip.reader.close()
        if clip.audio:
            clip.audio.reader.close_proc()
    except Exception:
        duration = None
        width, height = None, None

    metadata = {
        "video_id": video_id,
        "original_filename": original_filename,
        "stored_filename": unique_filename,
        "path": dest_path,
        "upload_time": upload_time,
        "file_size": file_size,
        "duration": duration,
        "resolution": {"width": width, "height": height}
    }

    try:
        client = MongoClient(mongo_uri)
        db = client["SurveillanceAI"]
        collection = db["video_metadata"]

        # Optional: check for duplicate video_id (very rare)
        if collection.find_one({"video_id": video_id}):
            return {"error": "Duplicate video_id detected. Skipping insert."}

        result = collection.insert_one(metadata)

        metadata["_id"] = str(result.inserted_id)
        metadata["upload_time"] = metadata["upload_time"].isoformat()

        return metadata

    except errors.DuplicateKeyError:
        return {"error": "Duplicate video_id found. Not inserted."}

    except Exception as e:
        return {"error": str(e)}
