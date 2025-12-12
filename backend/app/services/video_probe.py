from __future__ import annotations

import time
from typing import Union, Tuple

import cv2


def can_open_source(source: Union[int, str], timeout_seconds: float = 3.0) -> Tuple[bool, str]:
    """
    Lightweight preflight check to validate if OpenCV can open and read from the given source.

    - Attempts to open the capture.
    - Tries to grab a frame within timeout_seconds.
    - Always releases the capture.

    Returns:
        (ok, message)
    """
    cap = None
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return False, f"cv2.VideoCapture could not open source: {source}"

        # Try to read at least one frame within timeout window
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            ok, _ = cap.read()
            if ok:
                return True, "Opened and read frame successfully"
            # Small sleep to avoid hot loop; some streams need a moment to warm up
            time.sleep(0.05)

        return False, f"Opened source but failed to read a frame within {timeout_seconds:.1f}s"
    except Exception as e:
        return False, f"Exception during probe: {e}"
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
