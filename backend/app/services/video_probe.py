from __future__ import annotations

import os
import time
import threading
from typing import Union, Tuple

import cv2


def can_open_source(source: Union[int, str], timeout_seconds: float = 10.0) -> Tuple[bool, str]:
    """
    Lightweight preflight check to validate if OpenCV can open and read from the given source.

    - Retries opening up to 3 times for network streams, 2 for local cameras.
    - Tries to grab a frame within timeout_seconds.
    - Always releases the capture.
    - Uses threading to prevent cv2.VideoCapture from hanging indefinitely.

    Returns:
        (ok, message)
    """
    is_local = isinstance(source, int)
    is_network = isinstance(source, str) and (
        source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://")
    )

    if is_network:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;10000000")

    cap = None
    # Local cameras (DirectShow) can be slow to init on Windows; give 2 attempts.
    max_open_attempts = 3 if is_network else 2
    # Local cameras need most of the budget for the open call itself;
    # network streams split more evenly between open and first-read.
    open_timeout = (
        min(timeout_seconds * 0.85, 10.0) if is_local
        else min(timeout_seconds * 0.5, 5.0)
    )

    try:
        start_time = time.time()
        last_error_msg = ""

        for attempt in range(max_open_attempts):
            cap_result: list = [None]
            cap_error: list = [None]

            def _open_capture(result_list=cap_result, error_list=cap_error):
                try:
                    if isinstance(source, int):
                        c = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                    else:
                        c = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    result_list[0] = c
                except Exception as e:
                    error_list[0] = e

            thread = threading.Thread(target=_open_capture, daemon=True)
            thread.start()
            thread.join(timeout=open_timeout)

            if thread.is_alive():
                if cap_result[0] is not None:
                    try:
                        cap_result[0].release()
                    except Exception:
                        pass
                last_error_msg = (
                    f"Timeout opening camera source {source} on attempt "
                    f"{attempt + 1}/{max_open_attempts} (>{open_timeout:.1f}s)"
                )
                if attempt < max_open_attempts - 1:
                    time.sleep(0.5)
                    continue
                return False, last_error_msg

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
            last_error_msg = f"cv2.VideoCapture opened but isOpened()=False for source {source} (attempt {attempt + 1})"

            if attempt < max_open_attempts - 1:
                time.sleep(0.5)

        if cap is None or not cap.isOpened():
            return False, last_error_msg or f"cv2.VideoCapture could not open source: {source}"

        elapsed = time.time() - start_time
        remaining = max(1.0, timeout_seconds - elapsed)
        deadline = time.time() + remaining
        while time.time() < deadline:
            ok, _ = cap.read()
            if ok:
                return True, "Opened and read frame successfully"
            time.sleep(0.1)

        return False, f"Opened source but failed to read a frame within {remaining:.1f}s"
    except Exception as e:
        return False, f"Exception during probe: {e}"
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
