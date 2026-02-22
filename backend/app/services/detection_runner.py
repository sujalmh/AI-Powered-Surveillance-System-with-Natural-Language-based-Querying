import logging
import threading
from typing import Dict, Optional, Set, Union

from backend.app.db.mongo import cameras as cameras_col

logger = logging.getLogger(__name__)


class DetectionRunner:
    """
    Manages live detection threads for cameras.
    """

    def __init__(self) -> None:
        self._workers: Dict[int, tuple[threading.Thread, threading.Event]] = {}
        self._lock = threading.Lock()

    def is_running(self, camera_id: int) -> bool:
        with self._lock:
            return camera_id in self._workers and self._workers[camera_id][0].is_alive()

    def running_cameras(self) -> Set[int]:
        """Return set of camera_ids that are currently running (one lock, no per-camera calls)."""
        with self._lock:
            return {cid for cid, (th, _) in self._workers.items() if th.is_alive()}

    def start_camera(
        self,
        camera_id: int,
        source: Union[int, str],
        location: str = "Default Location",
        show_window: bool = False,
    ) -> bool:
        """
        Start detection on a camera. Returns True if started, False if already running.
        """
        with self._lock:
            if camera_id in self._workers and self._workers[camera_id][0].is_alive():
                return False

            stop_event = threading.Event()

            def _target():
                try:
                    # Lazy import to avoid loading heavy ML deps at API startup
                    from backend.object_detection import process_live_stream  # type: ignore

                    process_live_stream(
                        camera_id=camera_id,
                        source=source,
                        location=location,
                        show_window=show_window,
                        stop_event=stop_event,
                    )
                finally:
                    # Mark camera inactive if process exits
                    try:
                        cameras_col.update_one(
                            {"camera_id": camera_id},
                            {"$set": {"status": "inactive"}},
                            upsert=True,
                        )
                    except Exception:
                        pass
                    # Remove worker entry
                    with self._lock:
                        self._workers.pop(camera_id, None)

            t = threading.Thread(target=_target, name=f"detector_cam_{camera_id}", daemon=True)
            self._workers[camera_id] = (t, stop_event)

            # Mark camera record and start thread
            try:
                cameras_col.update_one(
                    {"camera_id": camera_id},
                    {"$set": {"source": str(source), "location": location, "status": "starting"}},
                    upsert=True,
                )
            except Exception:
                pass

            t.start()
            return True

    def stop_camera(self, camera_id: int, timeout: float = 5.0) -> bool:
        """
        Stop detection on a camera. Returns True if stop signal was sent, False if not running.
        """
        with self._lock:
            pair = self._workers.get(camera_id)
            if not pair:
                return False
            t, stop_event = pair

        stop_event.set()
        t.join(timeout=timeout)

        with self._lock:
            # Clean if thread finished
            if not t.is_alive():
                self._workers.pop(camera_id, None)

        try:
            cameras_col.update_one({"camera_id": camera_id}, {"$set": {"status": "inactive"}}, upsert=True)
        except Exception:
            pass

        return True

    def stop_all(self, timeout: float = 3.0) -> None:
        """Signal every camera to stop in parallel, then join all threads."""
        with self._lock:
            pairs = list(self._workers.items())

        if not pairs:
            return

        # Signal all stop events at once so cameras shut down concurrently
        for cid, (t, stop_event) in pairs:
            stop_event.set()

        # Wait for each thread with a per-thread budget
        for cid, (t, _) in pairs:
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("Camera %s thread did not exit within %.1fs", cid, timeout)

        # Clean up finished entries
        with self._lock:
            for cid, (t, _) in list(self._workers.items()):
                if not t.is_alive():
                    self._workers.pop(cid, None)

        # Best-effort DB update
        for cid, _ in pairs:
            try:
                cameras_col.update_one({"camera_id": cid}, {"$set": {"status": "inactive"}}, upsert=True)
            except Exception:
                pass

    def list_running(self) -> Dict[int, str]:
        """
        Return a map of camera_id -> thread_status ("alive" | "stopped")
        """
        with self._lock:
            return {cid: ("alive" if th.is_alive() else "stopped") for cid, (th, _) in self._workers.items()}


runner = DetectionRunner()
