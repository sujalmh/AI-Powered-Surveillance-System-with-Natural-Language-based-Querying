import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.app.config import settings
from backend.app.services.clip_builder import build_clip_from_snapshots


@pytest.mark.timeout(30)
def test_recording_and_clip_extraction(tmp_path):
    from backend.object_detection import ContinuousRecorder, _conversion_queue

    cam_id = 999
    fps = 5.0

    # Isolate test artifacts
    orig_snap = settings.SNAPSHOTS_DIR
    orig_rec = settings.RECORDINGS_DIR
    orig_clip = settings.CLIPS_DIR
    settings.SNAPSHOTS_DIR = tmp_path / "snapshots"
    settings.RECORDINGS_DIR = tmp_path / "recordings"
    settings.CLIPS_DIR = tmp_path / "clips"
    settings.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    settings.CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        recorder = ContinuousRecorder(camera_id=cam_id, fps=fps, chunk_duration_sec=1)

        # Write a handful of frames deterministically
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (i * 10 % 255, 0, 0)
            recorder.write(frame)

        recorder.release()
        _conversion_queue.join()

        rec_dir = settings.RECORDINGS_DIR / f"camera_{cam_id}"
        mp4_files = sorted(rec_dir.glob("*.mp4"))
        assert mp4_files, "Recorder did not emit any MP4 files"

        # Use the last recording timestamps for clip window
        end_dt = datetime.fromtimestamp(mp4_files[-1].stat().st_mtime, tz=timezone.utc)
        start_dt = end_dt - timedelta(seconds=2)

        clip = build_clip_from_snapshots(
            camera_id=cam_id,
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            fps=fps,
        )

        assert clip is not None
        assert Path(clip.path).exists(), "Clip file missing on disk"
        assert clip.frame_count > 0, "Clip has zero frames"
    finally:
        settings.SNAPSHOTS_DIR = orig_snap
        settings.RECORDINGS_DIR = orig_rec
        settings.CLIPS_DIR = orig_clip
