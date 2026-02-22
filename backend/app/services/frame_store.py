import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Iterator

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
    - stall_timeout: if no frame has been updated for this many seconds, we yield blank frames periodically
    """
    boundary = b"--frame\r\n"
    content_type = b"Content-Type: image/jpeg\r\n"

    interval = 1.0 / max(0.1, fps)

    while True:
        start = time.time()
        item = get_latest(camera_id)
        if item is not None:
            jpg, ts = item
            # Build a proper multipart with Content-Length to improve client compatibility
            header = boundary + content_type + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
            yield header + jpg + b"\r\n"
        else:
            # No frame yet; keep connection alive by sleeping to next interval
            pass

        elapsed = time.time() - start
        sleep_for = max(0.0, interval - elapsed)
        time.sleep(sleep_for)
