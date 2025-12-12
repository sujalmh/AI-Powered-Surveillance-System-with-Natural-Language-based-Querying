import threading
import time
from typing import Dict, Optional, Tuple, Iterator

# Stores latest JPEG bytes per camera_id
# camera_id -> (jpg_bytes, updated_epoch_seconds)
_frame_store: Dict[int, Tuple[bytes, float]] = {}
_lock = threading.Lock()


def update_frame(camera_id: int, jpg_bytes: bytes) -> None:
    now = time.time()
    with _lock:
        _frame_store[camera_id] = (jpg_bytes, now)


def get_latest(camera_id: int) -> Optional[Tuple[bytes, float]]:
    with _lock:
        return _frame_store.get(camera_id)


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
