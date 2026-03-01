import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Iterator

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

# Stores latest JPEG bytes per camera_id
# camera_id -> (jpg_bytes, updated_epoch_seconds)
_frame_store: Dict[int, Tuple[bytes, float]] = {}
_lock = threading.Lock()

# Eviction: drop entries not updated for this many seconds (default 30 min). Set FRAME_STORE_TTL_SEC env to override.
FRAME_STORE_TTL_SEC: float = float(os.getenv("FRAME_STORE_TTL_SEC", str(30 * 60)))
# Cap number of cameras to avoid unbounded memory (0 = no cap). Set FRAME_STORE_MAX_CAMERAS env to override.
FRAME_STORE_MAX_CAMERAS: int = int(os.getenv("FRAME_STORE_MAX_CAMERAS", "0"))


def _evict_stale_and_cap() -> None:
    """Caller must hold _lock. Remove stale entries and optionally cap size by oldest."""
    now = time.time()
    stale = [cid for cid, (_, ts) in _frame_store.items() if (now - ts) > FRAME_STORE_TTL_SEC]
    for cid in stale:
        del _frame_store[cid]

    if FRAME_STORE_MAX_CAMERAS <= 0 or len(_frame_store) <= FRAME_STORE_MAX_CAMERAS:
        return
    # Sort by timestamp ascending and drop oldest until under cap
    by_ts: List[Tuple[int, float]] = [(cid, ts) for cid, (_, ts) in _frame_store.items()]
    by_ts.sort(key=lambda x: x[1])
    for cid, _ in by_ts[: len(_frame_store) - FRAME_STORE_MAX_CAMERAS]:
        _frame_store.pop(cid, None)


def update_frame(camera_id: int, jpg_bytes: bytes) -> None:
    now = time.time()
    with _lock:
        _frame_store[camera_id] = (jpg_bytes, now)
        _evict_stale_and_cap()


def get_latest(camera_id: int) -> Optional[Tuple[bytes, float]]:
    with _lock:
        entry = _frame_store.get(camera_id)
        if entry is None:
            return None
        jpg, ts = entry
        if (time.time() - ts) > FRAME_STORE_TTL_SEC:
            del _frame_store[camera_id]
            return None
        return entry


def mjpeg_generator(camera_id: int, fps: float = 5.0, stall_timeout: float = 10.0) -> Iterator[bytes]:
    """
    Yields an MJPEG multipart stream for the given camera_id.
    - fps: how often to push a frame (upper bound; if frames arrive slower, we push latest)
    - stall_timeout: if no new frame has arrived for this many seconds, yield a placeholder
      and eventually terminate the generator to free server resources.
    """
    boundary = b"--frame\r\n"
    content_type = b"Content-Type: image/jpeg\r\n"

    interval = 1.0 / max(0.1, fps)

    # 1x1 black JPEG placeholder for stale streams
    _PLACEHOLDER_JPG: Optional[bytes] = None

    last_frame_epoch: Optional[float] = None
    stall_start: Optional[float] = None
    # Terminate after 3x stall_timeout of total inactivity
    hard_timeout = stall_timeout * 3.0
    placeholder_interval = 1.0  # send placeholder at most once per second

    while True:
        start = time.time()
        item = get_latest(camera_id)
        if item is not None:
            jpg, ts = item
            # Only yield if this is a genuinely new/recent frame
            if last_frame_epoch is None or ts != last_frame_epoch:
                last_frame_epoch = ts
                stall_start = None  # reset stall timer
                header = boundary + content_type + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                yield header + jpg + b"\r\n"
            elif stall_start is None:
                # Same frame as before — start stall timer
                stall_start = time.time()
        else:
            if stall_start is None:
                stall_start = time.time()

        # Handle stall: no new frame for a while
        if stall_start is not None:
            stalled_for = time.time() - stall_start
            if stalled_for >= hard_timeout:
                # Terminate generator — camera likely stopped
                return
            if stalled_for >= stall_timeout:
                # Yield a tiny placeholder to keep connection alive and signal stale
                if _PLACEHOLDER_JPG is None:
                    try:
                        import numpy as np
                        blank = np.zeros((2, 2, 3), dtype=np.uint8)
                        ok, buf = cv2.imencode(".jpg", blank)
                        _PLACEHOLDER_JPG = buf.tobytes() if ok else b""
                    except Exception:
                        _PLACEHOLDER_JPG = b""
                if _PLACEHOLDER_JPG:
                    header = boundary + content_type + f"Content-Length: {len(_PLACEHOLDER_JPG)}\r\n\r\n".encode("ascii")
                    yield header + _PLACEHOLDER_JPG + b"\r\n"
                    time.sleep(placeholder_interval)
                    continue

        elapsed = time.time() - start
        sleep_for = max(0.0, interval - elapsed)
        time.sleep(sleep_for)
