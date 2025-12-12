import json
import os
import sys
import requests
from pathlib import Path

def main():
    # Pick a single clip to (re)index with captions enabled.
    # Adjust path if you want a different clip.
    clip_path = Path("data/clips/camera_0/2025-10-23T20-58-27-447915__to__2025-10-23T20-58-48-276134.mp4").resolve()
    if not clip_path.exists():
        print("Clip not found:", clip_path)
        sys.exit(1)

    # Derive a clip_url that maps to the static mount /media/clips
    try:
        # If CLIPS_DIR is data/clips, then relative to that:
        rel = clip_path.relative_to(Path("data").resolve() / "clips")
        clip_url = f"/media/clips/{rel.as_posix()}"
    except Exception:
        clip_url = None

    payload = {
        "camera_id": 0,
        "clip_path": str(clip_path),
        "clip_url": clip_url,
        "every_sec": 1.0,
        "with_captions": True,
    }
    url = "http://127.0.0.1:8000/api/semantic/index-clip"
    print("POST", url)
    print("payload=", json.dumps(payload, indent=2))
    try:
        r = requests.post(url, json=payload, timeout=1200)
        print("status:", r.status_code)
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text)
    except Exception as e:
        print("request error:", repr(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
