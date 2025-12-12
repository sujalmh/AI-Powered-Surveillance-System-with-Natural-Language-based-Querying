from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from backend.app.config import settings


@dataclass
class BuiltClip:
    camera_id: int
    path: str
    url: str
    frame_count: int
    fps: float
    width: int
    height: int


_TS_PATTERNS = [
    # Example: 2025-10-23T20-56-25-468181.jpg (from code that replaces : and . with -)
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})T(?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})(?:-(?P<us>\d+))?"),
    # Fallback: plain ISO-like without changes: 2025-10-23T20:56:25(.468181)?
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})T(?P<H>\d{2}):(?P<M>\d{2}):(?P<S>\d{2})(?:\.(?P<us>\d+))?"),
]


def _parse_ts_from_filename(name: str) -> Optional[datetime]:
    stem = Path(name).stem
    for pat in _TS_PATTERNS:
        m = pat.search(stem)
        if m:
            try:
                us = int(m.group("us")) if m.groupdict().get("us") else 0
                return datetime(
                    int(m.group("y")),
                    int(m.group("m")),
                    int(m.group("d")),
                    int(m.group("H")),
                    int(m.group("M")),
                    int(m.group("S")),
                    us,
                )
            except Exception:
                continue
    return None


def _collect_snapshots(camera_id: int, start_dt: datetime, end_dt: datetime) -> List[Tuple[datetime, Path]]:
    cam_dir = settings.SNAPSHOTS_DIR / f"camera_{camera_id}"
    if not cam_dir.exists():
        return []
    files: List[Tuple[datetime, Path]] = []
    for p in cam_dir.glob("*.jpg"):
        ts = _parse_ts_from_filename(p.name)
        if ts is None:
            # fall back to file mtime if parsing fails
            try:
                ts = datetime.fromtimestamp(p.stat().st_mtime)
            except Exception:
                continue
        if start_dt <= ts <= end_dt:
            files.append((ts, p))
    files.sort(key=lambda x: x[0])
    return files


def build_clip_from_snapshots(
    camera_id: int,
    start_iso: str,
    end_iso: str,
    fps: float = 5.0,
) -> BuiltClip:
    """
    Build an MP4 clip by stitching snapshots between [start_iso, end_iso] for a camera.
    Returns BuiltClip with filesystem path and public URL.
    """
    try:
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = datetime.fromisoformat(end_iso)
    except Exception as e:
        raise ValueError(f"Invalid start/end ISO timestamps: {e}")

    if end_dt < start_dt:
        raise ValueError("end must be >= start")

    items = _collect_snapshots(camera_id, start_dt, end_dt)
    if not items:
        raise FileNotFoundError("No snapshots found in the requested interval")

    # Read first frame to get frame size
    first_img = cv2.imread(str(items[0][1]))
    if first_img is None:
        raise RuntimeError(f"Failed to read first snapshot: {items[0][1]}")
    height, width = first_img.shape[:2]

    # Prepare output path
    out_dir: Path = settings.CLIPS_DIR / f"camera_{camera_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_start = start_iso.replace(":", "-").replace(".", "-")
    safe_end = end_iso.replace(":", "-").replace(".", "-")
    base_name = f"{safe_start}__to__{safe_end}"
    writer = None
    out_path = None

    # Initialize writer with a set of browser-friendly codecs, preferring H.264 if available
    _fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if _fourcc_fn is None:
        raise RuntimeError("OpenCV build missing VideoWriter_fourcc")

    # Try a sequence of codecs and containers for best browser compatibility
    # Note: H.264 requires an FFmpeg-enabled OpenCV build with libx264 support.
    candidates = [
        ("mp4", "avc1"),  # H.264
        ("mp4", "H264"),  # H.264 alias
        ("mp4", "X264"),  # H.264 alias
        ("mp4", "mp4v"),  # MPEG-4 Part 2 (may not play in some browsers)
        ("avi", "MJPG"),  # MJPEG fallback (may not play inline; download should work)
    ]

    target_size = (width, height)
    for ext, codec in candidates:
        # H.264 typically requires even dimensions
        if ext == "mp4" and codec in ("avc1", "H264", "X264"):
            w = width - (width % 2)
            h = height - (height % 2)
            target_size = (max(2, w), max(2, h))
        else:
            target_size = (width, height)

        fourcc = _fourcc_fn(*codec)
        test_path = out_dir / f"{base_name}.{ext}"
        wtr = cv2.VideoWriter(str(test_path), fourcc, max(0.5, float(fps)), target_size)
        if wtr is not None and wtr.isOpened():
            writer = wtr
            out_path = test_path
            width, height = target_size  # update if adjusted for H.264
            break

    if writer is None or out_path is None:
        raise RuntimeError("Failed to open VideoWriter with any supported codec")

    frame_count = 0
    try:
        for ts, img_path in items:
            img = cv2.imread(str(img_path))
            if img is None:
                # skip unreadable frames
                continue
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(img)
            frame_count += 1
    finally:
        writer.release()

    if frame_count == 0:
        # Cleanup empty file
        try:
            out_path.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass
        raise RuntimeError("No valid frames written to clip")

    # Public URL (served by static /media/clips)
    rel = out_path.relative_to(settings.CLIPS_DIR).as_posix()
    url = f"/media/clips/{rel}"
    return BuiltClip(
        camera_id=camera_id,
        path=str(out_path),
        url=url,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )
