from __future__ import annotations

import os
import time
import threading
from typing import Union, Tuple

import cv2


def can_open_source(source: Union[int, str], timeout_seconds: float = 10.0) -> Tuple[bool, str]:
    """
    Lightweight preflight check to validate if OpenCV can open and read from the given source.

    - For network streams (http/rtsp), retries opening up to 3 times.
    - Tries to grab a frame within timeout_seconds.
    - Always releases the capture.
    - Uses threading to prevent cv2.VideoCapture from hanging indefinitely.

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
    open_timeout = min(5.0, timeout_seconds * 0.5)  # Max 5 seconds to open, or half of total timeout

    try:
        # Retry opening for network streams (DroidCam, IP cameras, etc.)
        for attempt in range(max_open_attempts):
            cap_result = [None]
            cap_error = [None]
            
            def _open_capture():
                try:
                    # Use DirectShow for local webcams on Windows
                    if isinstance(source, int):
                        c = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                    else:
                        c = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    cap_result[0] = c
                except Exception as e:
                    cap_error[0] = e
            
            # Open capture in a thread with timeout
            thread = threading.Thread(target=_open_capture, daemon=True)
            thread.start()
            thread.join(timeout=open_timeout)
            
            if thread.is_alive():
                # Thread is still running, capture is hanging
                if cap_result[0] is not None:
                    try:
                        cap_result[0].release()
                    except Exception:
                        pass
                if attempt < max_open_attempts - 1:
                    time.sleep(0.5)
                    continue
                return False, f"Timeout opening camera source {source} (took >{open_timeout:.1f}s)"
            
            if cap_error[0] is not None:
                return False, f"Error opening source: {cap_error[0]}"
            
            cap = cap_result[0]
            if cap is not None and cap.isOpened():
                break
            
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = None
            
            if attempt < max_open_attempts - 1:
                time.sleep(1.0)

        if cap is None or not cap.isOpened():
            return False, f"cv2.VideoCapture could not open source: {source}"

        # Try to read at least one frame within remaining timeout window
        frame_read_timeout = timeout_seconds - open_timeout
        deadline = time.time() + frame_read_timeout
        while time.time() < deadline:
            ok, _ = cap.read()
            if ok:
                return True, "Opened and read frame successfully"
            # Small sleep to avoid hot loop; some streams need a moment to warm up
            time.sleep(0.1)

        return False, f"Opened source but failed to read a frame within {frame_read_timeout:.1f}s"
    except Exception as e:
        return False, f"Exception during probe: {e}"
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
