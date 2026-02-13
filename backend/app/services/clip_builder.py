from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

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


def _parse_whitelist(stamps: List[str]) -> Set[datetime]:
    out = set()
    for s in stamps:
        try:
            # Replace Z with +00:00 to handle UTC explicitly
            safe = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(safe)
            if dt.tzinfo is not None:
                # Convert to UTC and strip timezone info
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            out.add(dt)
        except Exception:
            pass
    return out


def _collect_snapshots(
    camera_id: int,
    start_dt: datetime,
    end_dt: datetime,
    allowed_timestamps: Optional[List[str]] = None,
    whitelist_dts: Optional[Set[datetime]] = None,
    whitelist_tolerance_sec: float = 5.0,
) -> List[Tuple[datetime, Path]]:
    cam_dir = settings.SNAPSHOTS_DIR / f"camera_{camera_id}"
    if not cam_dir.exists():
        return []
    files: List[Tuple[datetime, Path]] = []
    
    # Parse allowed timestamps into a set of datetime objects for efficient lookup
    whitelist_dts: Optional[Set[datetime]] = None
    if allowed_timestamps:
        whitelist_dts = _parse_whitelist(allowed_timestamps)

    for p in cam_dir.glob("*.jpg"):
        ts = _parse_ts_from_filename(p.name)
        if ts is None:
            # fall back to file mtime if parsing fails
            try:
                ts = datetime.fromtimestamp(p.stat().st_mtime)
            except Exception:
                continue
        if start_dt <= ts <= end_dt:
            # If whitelist is provided, only include snapshots near a whitelisted timestamp
            if whitelist_dts is not None:
                if not any(abs((ts - wt).total_seconds()) <= whitelist_tolerance_sec for wt in whitelist_dts):
                    continue
            files.append((ts, p))
    files.sort(key=lambda x: x[0])
    return files


def _parse_whitelist(stamps: List[str]) -> Set[datetime]:
    out = set()
    for s in stamps:
        try:
            # Handle possible variations in ISO format
            # Replace Z with +00:00 to handle UTC explicitly
            safe = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(safe)
            if dt.tzinfo is not None:
                # Convert to UTC and strip timezone info to match naive file timestamps (which are UTC)
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            out.add(dt)
        except Exception:
            pass
    return out


def build_clip_from_snapshots(
    camera_id: int,
    start_iso: str,
    end_iso: str,
    fps: float = 5.0,
    allowed_timestamps: Optional[List[str]] = None,
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

    items = _collect_snapshots(
        camera_id,
        start_dt,
        end_dt,
        whitelist_dts=_parse_whitelist(allowed_timestamps) if allowed_timestamps else None,
    )
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

    # Initialize writer with browser-compatible codec
    _fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if _fourcc_fn is None:
        raise RuntimeError("OpenCV build missing VideoWriter_fourcc")

    # Use MP4V codec which has universal browser support
    # MP4V (MPEG-4 Part 2) works in all major browsers and doesn't require external libraries
    candidates = [
        ("mp4", "mp4v"),  # MPEG-4 Part 2 - universal browser support
        ("mp4", "XVID"),  # Xvid fallback
        ("avi", "MJPG"),  # Last resort
    ]

    # Ensure even dimensions for codec compatibility
    width = width - (width % 2) if width % 2 != 0 else width
    height = height - (height % 2) if height % 2 != 0 else height

    target_size = (width, height)
    for ext, codec in candidates:
        # Only adjust dimensions if needed (MJPEG doesn't require even dimensions)
        target_size = (width, height)

        fourcc = _fourcc_fn(*codec)
        test_path = out_dir / f"{base_name}.{ext}"
        wtr = cv2.VideoWriter(str(test_path), fourcc, max(0.5, float(fps)), target_size)
        if wtr is not None and wtr.isOpened():
            writer = wtr
            out_path = test_path
            width, height = target_size
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

    # ==========================================
    # CRITICAL: Convert to H.264 using FFmpeg
    # ==========================================
    # OpenCV's codecs don't work reliably in browsers.
    # Use imageio-ffmpeg (bundles FFmpeg) to convert to H.264.
    final_path = out_path
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        import subprocess
        
        # Get bundled ffmpeg executable
        ffmpeg_exe = get_ffmpeg_exe()
        
        # Create H.264 output path
        h264_path = out_path.with_suffix('.mp4') if out_path.suffix != '.mp4' else out_path.parent / f"{out_path.stem}_h264.mp4"
        
        # FFmpeg command for browser-compatible H.264
        cmd = [
            ffmpeg_exe,
            "-y",  # overwrite
            "-i", str(out_path),  # input
            "-c:v", "libx264",  # H.264 codec
            "-preset", "fast",  # encoding speed
            "-crf", "23",  # quality
            "-pix_fmt", "yuv420p",  # browser compatibility
            "-movflags", "+faststart",  # web optimization
            "-loglevel", "error",  # suppress verbose output
            str(h264_path)
        ]
        
        # Run FFmpeg conversion
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        
        if result.returncode == 0 and h264_path.exists():
            # Success! Replace original with H.264 version
            if h264_path != out_path:
                try:
                    out_path.unlink()  # Delete original
                except Exception:
                    pass
                # Rename H.264 to original name
                final_out_path = out_path.parent / f"{base_name}.mp4"
                h264_path.rename(final_out_path)
                final_path = final_out_path
            else:
                final_path = h264_path
            print(f"✅ Converted to H.264 using FFmpeg: {final_path.name}")
        else:
            err_msg = result.stderr.decode()[:200] if result.stderr else "Unknown error"
            print(f"⚠️ FFmpeg conversion failed: {err_msg}")
            final_path = out_path
            
    except ImportError:
        print("⚠️ imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")
        final_path = out_path
    except Exception as e:
        print(f"⚠️ FFmpeg conversion error: {e}. Using original video.")
        final_path = out_path

    # Public URL (served by static /media/clips)
    rel = final_path.relative_to(settings.CLIPS_DIR).as_posix()
    url = f"/media/clips/{rel}"
    return BuiltClip(
        camera_id=camera_id,
        path=str(final_path),
        url=url,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )
