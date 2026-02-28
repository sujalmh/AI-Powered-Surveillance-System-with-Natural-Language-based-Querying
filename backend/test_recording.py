import sys
import os
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.app.config import settings
from backend.object_detection import ContinuousRecorder
from backend.app.services.clip_builder import build_clip_from_snapshots

def run_test():
    cam_id = 999
    fps = 10.0
    recorder = ContinuousRecorder(camera_id=cam_id, fps=fps, chunk_duration_sec=3)
    
    print("Writing frames...")
    start_time = datetime.now(timezone.utc)
    
    for i in range(50):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (i * 5 % 255, 0, 0)
        recorder.write(frame)
        time.sleep(0.1)
    
    recorder.release()
    time.sleep(2)  # Wait for H264 conversion queue
    
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(seconds=3)
    
    print(f"Extracting clip from {start_dt.isoformat()} to {end_dt.isoformat()}...")
    try:
        clip = build_clip_from_snapshots(
            camera_id=cam_id,
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            fps=fps
        )
        print("Success! Extracted clip:")
        print(f"Path: {clip.path}")
        print(f"Duration frames: {clip.frame_count}")
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_test()
