import os
import uuid
from moviepy.editor import VideoFileClip
from app.services.query import nl_to_query
from flask import current_app

def send_response(query):
    video_path = os.path.join(current_app.config["UPLOAD_FOLDER"], "test.mp4")

    st, et, reply = nl_to_query(query)

    clip = VideoFileClip(video_path).subclip(st, et)
    output_path = os.path.join(current_app.config["VIDEO_FOLDER"], f"output_clip_{uuid.uuid4()}.mp4")

    clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    return {
        "start_time": st,
        "end_time": et,
        "clip_path": output_path,
        "reply": reply
    }
