from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple
from loguru import logger

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
    # Also handles legacy filenames with +00-00 or Z suffix after microseconds
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})T(?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})(?:-(?P<us>\d+))?"),
    # Fallback: plain ISO-like without changes: 2025-10-23T20:56:25(.468181)?
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})T(?P<H>\d{2}):(?P<M>\d{2}):(?P<S>\d{2})(?:\.(?P<us>\d+))?"),
]


def _probe_video_duration(path: Path) -> float:
    """Return the video file's internal duration in seconds, or 0 on failure."""
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0 and frame_count > 0:
            return frame_count / fps
    except Exception:
        pass
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
    return 0.0


def _wall_to_video_offset(
    wall_offset_sec: float,
    wall_duration_sec: float,
    video_duration_sec: float,
) -> float:
    """
    Convert a wall-clock offset (seconds from chunk start) to a video-file
    offset, accounting for FPS mismatch between recording header and actual
    frame delivery rate.

    If the VideoWriter was opened at 25fps but frames arrived at ~20fps,
    the video's internal timeline is shorter than wall-clock time.  Without
    scaling, FFmpeg would seek ~22% past the correct position.
    """
    if wall_duration_sec > 1.0 and video_duration_sec > 1.0:
        scaling = video_duration_sec / wall_duration_sec
        # Sanity: only apply scaling if ratio is within reasonable bounds
        # (0.3x – 3.0x).  Outside that range, one of the durations is
        # probably wrong (e.g. corrupt file).
        if 0.3 <= scaling <= 3.0:
            return max(0.0, wall_offset_sec * scaling)
    return max(0.0, wall_offset_sec)


def _map_wall_range_to_video_range(
    items: List[Tuple[datetime, datetime, Path]],
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[float, float]:
    """
    Map a wall-clock time range across multiple video files to per-file
    video-timeline offsets and durations, accounting for per-file FPS mismatch.
    
    Returns (offset_sec, duration_sec) for FFmpeg -ss and -t flags.
    Falls back to wall times if video durations cannot be determined.
    """
    if not items:
        return 0.0, (end_dt - start_dt).total_seconds()
    
    overall_start = items[0][0]
    overall_end = items[-1][1]
    total_wall_dur = (overall_end - overall_start).total_seconds()
    
    # Compute per-file video durations
    file_video_durs = [_probe_video_duration(p) for _, _, p in items]
    total_video_dur = sum(file_video_durs)
    
    # If we can't determine video durations, fall back to wall times
    if total_video_dur <= 0 or total_wall_dur <= 1.0:
        wall_offset = max(0.0, (start_dt - overall_start).total_seconds())
        wall_duration = (end_dt - start_dt).total_seconds()
        return wall_offset, wall_duration
    
    # Piecewise mapping: walk items and accumulate video time for the requested range
    offset_sec = 0.0
    duration_sec = 0.0
    found_start = False
    
    for i, (chunk_start, chunk_end, p) in enumerate(items):
        wall_chunk_dur = (chunk_end - chunk_start).total_seconds()
        video_chunk_dur = file_video_durs[i]
        
        # Skip segments before start_dt
        if chunk_end <= start_dt:
            # Accumulate offset in case start is in a later chunk
            if video_chunk_dur > 0:
                ratio = video_chunk_dur / wall_chunk_dur if wall_chunk_dur > 0 else 1.0
                offset_sec += video_chunk_dur
            continue
        
        # Handle segments that overlap with [start_dt, end_dt]
        seg_start = max(chunk_start, start_dt)
        seg_end = min(chunk_end, end_dt)
        
        if seg_start < seg_end:
            wall_seg_offset = (seg_start - chunk_start).total_seconds()
            wall_seg_dur = (seg_end - seg_start).total_seconds()
            
            if not found_start:
                # First segment: compute absolute offset in video timeline
                if video_chunk_dur > 0 and wall_chunk_dur > 0:
                    ratio = video_chunk_dur / wall_chunk_dur
                    offset_sec += wall_seg_offset * ratio
                else:
                    offset_sec += wall_seg_offset
                found_start = True
            
            # Accumulate duration for this segment
            if video_chunk_dur > 0 and wall_chunk_dur > 0:
                ratio = video_chunk_dur / wall_chunk_dur
                duration_sec += wall_seg_dur * ratio
            else:
                duration_sec += wall_seg_dur
        
        # Stop if we've covered end_dt
        if seg_end >= end_dt:
            break
    
    return max(0.0, offset_sec), max(0.0, duration_sec)


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
    
    parsed_whitelist = whitelist_dts
    if parsed_whitelist is None and allowed_timestamps:
        parsed_whitelist = _parse_whitelist(allowed_timestamps)

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
            if parsed_whitelist is not None:
                if not any(abs((ts - wt).total_seconds()) <= whitelist_tolerance_sec for wt in parsed_whitelist):
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


def _has_moov_atom(path: Path) -> bool:
    """
    Quick binary check for an MP4 moov atom.

    ISO base media files store data in boxes (atoms) with a 4-byte size
    followed by a 4-byte type.  We scan for a box whose type is ``moov``.
    This avoids opening the file with cv2/FFmpeg which would print noisy
    "moov atom not found" errors to stderr that we cannot capture.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(min(path.stat().st_size, 64 * 1024))  # first 64KB
        # Also check at end-of-file (moov may be at the tail in non-faststart files)
        if path.stat().st_size > 64 * 1024:
            with open(path, "rb") as f:
                f.seek(max(0, path.stat().st_size - 64 * 1024))
                data += f.read()
        return b"moov" in data
    except Exception:
        return False


def _is_mp4_readable(path: Path) -> bool:
    """
    Check that an MP4 file is valid and usable for clip extraction.

    Filters out:
    - Files smaller than 1KB (incomplete)
    - Files modified in the last 2 seconds (still being written)
    - Files missing the moov atom (truncated/corrupt recordings)
    - Files that fail to open with cv2.VideoCapture
    """
    try:
        # Skip files that are too small or too recent
        stat = path.stat()
        if stat.st_size < 1024:  # Less than 1KB
            return False
        age_sec = (datetime.now().timestamp() - stat.st_mtime)
        if age_sec < 2.0:  # Modified less than 2 seconds ago
            logger.debug("Skipping very recent file (age=%.1fs): %s", age_sec, path.name)
            return False

        # Fast binary check for moov atom — avoids noisy FFmpeg stderr
        # when cv2.VideoCapture tries to open a corrupt file.
        if not _has_moov_atom(path):
            logger.debug("Skipping recording without moov atom: %s", path.name)
            return False

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        # Try to read at least one frame to confirm usability
        ok = cap.grab()
        cap.release()
        return ok
    except Exception:
        return False


def _collect_recordings(
    camera_id: int,
    start_dt: datetime,
    end_dt: datetime,
) -> List[Tuple[datetime, datetime, Path]]:
    cam_dir = settings.RECORDINGS_DIR / f"camera_{camera_id}"
    if not cam_dir.exists():
        return []
    
    files: List[Tuple[datetime, Path]] = []
    
    for p in cam_dir.glob("*.mp4"):
        # Skip temporary files from FFmpeg background conversion
        if p.stem.endswith("_h264") or p.stem.endswith("._converting"):
            # Clean up stale converting files (<1KB = failed conversion)
            if p.stem.endswith("._converting"):
                try:
                    if p.stat().st_size < 1024:
                        p.unlink(missing_ok=True)
                        logger.debug("Cleaned up failed conversion file: %s", p.name)
                except Exception:
                    pass
            continue
        ts = _parse_ts_from_filename(p.name)
        if ts is None:
            try:
                ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)
            except Exception:
                continue
        files.append((ts, p))
        
    files.sort(key=lambda x: x[0])
    
    overlapping = []
    for i in range(len(files)):
        chunk_start, p = files[i]
        
        if i + 1 < len(files):
            chunk_end = files[i+1][0]
        else:
            try:
                chunk_end = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)
            except Exception:
                chunk_end = datetime.now(timezone.utc).replace(tzinfo=None)
                
        if chunk_end < chunk_start:
            chunk_end = chunk_start + timedelta(seconds=300)

        if start_dt <= chunk_end and end_dt >= chunk_start:
            if _is_mp4_readable(p):
                overlapping.append((chunk_start, chunk_end, p))
            else:
                logger.debug("Skipping unreadable/incomplete recording: {}", p.name)
            
    return overlapping


def _build_clip_from_snapshots_fallback(
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
        if start_dt.tzinfo is not None:
            start_dt = start_dt.astimezone(timezone.utc).replace(tzinfo=None)
        if end_dt.tzinfo is not None:
            end_dt = end_dt.astimezone(timezone.utc).replace(tzinfo=None)
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
        
        # Create H.264 output path (use ._converting suffix to avoid being picked up by _collect_recordings)
        h264_path = out_path.parent / f"{out_path.stem}._converting.mp4"
        
        # FFmpeg command for browser-compatible H.264
        cmd = [
            ffmpeg_exe,
            "-y",  # overwrite
            "-nostdin",  # don't wait for user input on errors
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
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.error("FFmpeg conversion failed: {}", e)
            final_path = out_path  # Fall back to original file
            result = None
            
        if result is not None and result.returncode == 0 and h264_path.exists():
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
            logger.info("Converted snapshot clip to H.264 via FFmpeg: {}", final_path.name)
        elif result is not None:
            err_msg = result.stderr.decode()[:200] if result.stderr else "Unknown error"
            logger.warning("FFmpeg snapshot-fallback conversion failed: {}", err_msg)
            final_path = out_path
            
    except ImportError:
        logger.warning("imageio-ffmpeg not installed — snapshot fallback clip uses mp4v codec (may not play in browsers). Install: pip install imageio-ffmpeg")
        final_path = out_path
    except Exception as e:
        logger.warning("FFmpeg snapshot-fallback conversion error: {}. Using original video.", e)
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


def build_clip_from_snapshots(
    camera_id: int,
    start_iso: str,
    end_iso: str,
    fps: float = 5.0,
    allowed_timestamps: Optional[List[str]] = None,
) -> BuiltClip:
    """
    Build an MP4 clip by extracting from continuous recordings between [start_iso, end_iso].
    Returns BuiltClip with filesystem path and public URL.
    Note: kept the name `build_clip_from_snapshots` for backwards compatibility with callers.
    """
    try:
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        if start_dt.tzinfo is not None:
            start_dt = start_dt.astimezone(timezone.utc).replace(tzinfo=None)
        if end_dt.tzinfo is not None:
            end_dt = end_dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception as e:
        raise ValueError(f"Invalid start/end ISO timestamps: {e}")

    if end_dt < start_dt:
        raise ValueError("end must be >= start")

    items = _collect_recordings(camera_id, start_dt, end_dt)
    if not items:
        # Fallback to old behavior if no recordings exist
        return _build_clip_from_snapshots_fallback(camera_id, start_iso, end_iso, fps, allowed_timestamps)
        
    out_dir: Path = settings.CLIPS_DIR / f"camera_{camera_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_start = start_iso.replace(":", "-").replace(".", "-")
    safe_end = end_iso.replace(":", "-").replace(".", "-")
    base_name = f"{safe_start}__to__{safe_end}"
    out_path = out_dir / f"{base_name}.mp4"
    
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_exe = get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"
        
    import subprocess
    
    if len(items) == 1:
        # Single file — apply FPS scaling to correct for mismatch between
        # the VideoWriter's header FPS and actual frame delivery rate.
        chunk_start, chunk_end, p = items[0]
        wall_offset = max(0.0, (start_dt - chunk_start).total_seconds())
        wall_duration = (end_dt - start_dt).total_seconds()
        wall_chunk_dur = (chunk_end - chunk_start).total_seconds()
        vid_dur = _probe_video_duration(p)

        offset_sec = _wall_to_video_offset(wall_offset, wall_chunk_dur, vid_dur)
        duration_sec = _wall_to_video_offset(wall_duration, wall_chunk_dur, vid_dur) if vid_dur > 0 else wall_duration

        if vid_dur > 0 and wall_chunk_dur > 1.0:
            logger.debug(
                "Clip FPS scaling: wall_chunk=%.1fs vid=%.1fs ratio=%.3f  "
                "wall_offset=%.2fs→vid_offset=%.2fs wall_dur=%.2fs→vid_dur=%.2fs",
                wall_chunk_dur, vid_dur, vid_dur / wall_chunk_dur,
                wall_offset, offset_sec, wall_duration, duration_sec,
            )

        # -ss AFTER -i for frame-accurate seeking (important for mp4v recordings
        # whose keyframes may be sparse).
        cmd = [
            ffmpeg_exe, "-y",
            "-nostdin",  # Don't wait for user input on errors
            "-i", str(p),
            "-ss", str(offset_sec),
            "-t", str(duration_sec),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-loglevel", "error", str(out_path)
        ]
        try:
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.error("build_clip_from_snapshots: ffmpeg trim failed (cmd={}): {}", cmd, exc)
            out_path.unlink(missing_ok=True)
            return _build_clip_from_snapshots_fallback(camera_id, start_iso, end_iso, fps, allowed_timestamps)
        if cp.returncode != 0:
            err = cp.stderr.decode(errors="ignore") if cp.stderr else ""
            logger.error("build_clip_from_snapshots: ffmpeg trim failed (cmd={}): {}", cmd, err[:500])
            out_path.unlink(missing_ok=True)
            return _build_clip_from_snapshots_fallback(camera_id, start_iso, end_iso, fps, allowed_timestamps)
    else:
        # Merge via concat demuxer — apply piecewise FPS scaling per file
        concat_file = out_dir / f"{base_name}_concat.txt"
        with concat_file.open("w") as f:
            for _, _, p in items:
                # ffmpeg requires full paths properly formatted for concat
                f.write(f"file '{p.resolve().as_posix()}'\n")
                
        # Use piecewise mapping to account for per-file FPS differences
        offset_sec, duration_sec = _map_wall_range_to_video_range(items, start_dt, end_dt)

        overall_start = items[0][0]
        overall_end = items[-1][1]
        wall_chunk_dur = (overall_end - overall_start).total_seconds()
        total_vid_dur = sum(_probe_video_duration(p) for _, _, p in items)

        if total_vid_dur > 0 and wall_chunk_dur > 1.0:
            logger.debug(
                "Clip concat piecewise mapping: wall_total=%.1fs vid_total=%.1fs overall_ratio=%.3f offset=%.2fs dur=%.2fs",
                wall_chunk_dur, total_vid_dur, total_vid_dur / wall_chunk_dur, offset_sec, duration_sec,
            )

        # -ss AFTER -i for frame-accurate seeking
        cmd = [
            ffmpeg_exe, "-y",
            "-nostdin",  # Don't wait for user input on errors
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-ss", str(offset_sec),
            "-t", str(duration_sec),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-loglevel", "error", str(out_path)
        ]
        try:
            try:
                cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
                logger.error("build_clip_from_snapshots: ffmpeg concat failed (cmd={}): {}", cmd, exc)
                out_path.unlink(missing_ok=True)
                return _build_clip_from_snapshots_fallback(camera_id, start_iso, end_iso, fps, allowed_timestamps)
            if cp.returncode != 0:
                err = cp.stderr.decode(errors="ignore") if cp.stderr else ""
                logger.error("build_clip_from_snapshots: ffmpeg concat failed (cmd={}): {}", cmd, err[:500])
                out_path.unlink(missing_ok=True)
                return _build_clip_from_snapshots_fallback(camera_id, start_iso, end_iso, fps, allowed_timestamps)
        finally:
            try:
                concat_file.unlink(missing_ok=True)
            except Exception:
                pass
            
    if not out_path.exists():
        # Fallback to snaps if ffmpeg fails
        return _build_clip_from_snapshots_fallback(camera_id, start_iso, end_iso, fps, allowed_timestamps)
        
    rel = out_path.relative_to(settings.CLIPS_DIR).as_posix()
    url = f"/media/clips/{rel}"
    
    # Try to grab frame count/width/height (approx)
    try:
        cap = cv2.VideoCapture(str(out_path))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        cap.release()
    except Exception:
        fc, w, h = 0, 640, 480
    
    return BuiltClip(
        camera_id=camera_id,
        path=str(out_path),
        url=url,
        frame_count=fc,
        fps=fps,
        width=w,
        height=h,
    )
