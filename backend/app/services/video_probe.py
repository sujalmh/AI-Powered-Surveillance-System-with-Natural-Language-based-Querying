from __future__ import annotations

import os
import time
from typing import Union, Tuple

import cv2


def can_open_source(source: Union[int, str], timeout_seconds: float = 10.0) -> Tuple[bool, str]:
    """
    Lightweight preflight check to validate if OpenCV can open and read from the given source.

    - For network streams (http/rtsp), retries opening up to 3 times.
    - Tries to grab a frame within timeout_seconds.
    - Always releases the capture.

    Returns:
        (ok, message)
    """
    is_network = isinstance(source, str) and (
        source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://")
    )

    # Help OpenCV/FFMPEG handle network streams more reliably
    if is_network:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;10000000")

    cap = None
    max_open_attempts = 3 if is_network else 1

    try:
        # Retry opening for network streams (DroidCam, IP cameras, etc.)
        for attempt in range(max_open_attempts):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                break
            cap.release()
            cap = None
            if attempt < max_open_attempts - 1:
                time.sleep(1.0)

        if cap is None or not cap.isOpened():
            return False, f"cv2.VideoCapture could not open source: {source}"

        # Try to read at least one frame within timeout window
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            ok, _ = cap.read()
            if ok:
                return True, "Opened and read frame successfully"
            # Small sleep to avoid hot loop; some streams need a moment to warm up
            time.sleep(0.1)

        return False, f"Opened source but failed to read a frame within {timeout_seconds:.1f}s"
    except Exception as e:
        return False, f"Exception during probe: {e}"
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
