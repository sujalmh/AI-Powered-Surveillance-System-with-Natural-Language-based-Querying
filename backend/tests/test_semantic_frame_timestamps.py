"""Frame timestamps must never be null, or time-windowed semantic search drops them."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.sem_indexer import sample_frames_from_clip  # noqa: E402


def _write_clip(path: Path, frames: int = 10, fps: float = 5.0) -> Path:
    h, w = 64, 64
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        img = np.full((h, w, 3), i * 20 % 255, dtype=np.uint8)
        writer.write(img)
    writer.release()
    return path


def test_untimed_filename_falls_back_to_mtime(tmp_path):
    clip = _write_clip(tmp_path / "upload.mp4")
    bundles = list(sample_frames_from_clip(str(clip), every_sec=1.0))
    assert bundles, "expected sampled frames"
    assert all(b.frame_ts for b in bundles), "frame_ts must not be None"
    first = datetime.fromisoformat(bundles[0].frame_ts)
    # frame_ts is naive UTC, matching the rest of the pipeline
    expected = datetime.fromtimestamp(clip.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)
    assert abs((first - expected).total_seconds()) < 5


def test_timestamped_filename_is_used(tmp_path):
    clip = _write_clip(tmp_path / "2026-03-01T10-00-00-000000__to__2026-03-01T10-00-04-000000.mp4")
    bundles = list(sample_frames_from_clip(str(clip), every_sec=1.0))
    assert bundles
    assert bundles[0].frame_ts.startswith("2026-03-01T10:00:00")


def test_explicit_start_iso_wins(tmp_path):
    clip = _write_clip(tmp_path / "2026-03-01T10-00-00-000000__to__2026-03-01T10-00-04-000000.mp4")
    bundles = list(
        sample_frames_from_clip(str(clip), every_sec=1.0, start_iso="2026-05-05T08:30:00+00:00")
    )
    assert bundles
    assert bundles[0].frame_ts.startswith("2026-05-05T08:30:00")
