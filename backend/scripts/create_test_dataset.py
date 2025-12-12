from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _unsanitize_iso_from_name(s: str) -> Optional[datetime]:
    """
    Expect: YYYY-MM-DDTHH-MM-SS(-us optional)
    Converts to datetime if possible, else None.
    """
    try:
        parts = s.split("T")
        if len(parts) != 2:
            return None
        date_part, time_part = parts
        tparts = time_part.split("-")
        if len(tparts) < 3:
            return None
        H, M, S = tparts[0], tparts[1], tparts[2]
        us = tparts[3] if len(tparts) > 3 else "0"
        dt = datetime.strptime(f"{date_part}T{H}:{M}:{S}", "%Y-%m-%dT%H:%M:%S")
        try:
            u = int(us)
            dt = dt.replace(microsecond=u)
        except Exception:
            pass
        return dt
    except Exception:
        return None


def _parse_start_end_from_clip_path(clip_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to parse start/end ISO timestamps from our file naming convention:
    {safe_start}__to__{safe_end}.mp4  where safe_start/end replaced ':' and '.' with '-'
    Example: 2025-10-23T21-08-09-698850__to__2025-10-23T21-09-22-527513.mp4
    """
    try:
        name = Path(clip_path).stem
        if "__to__" not in name:
            return None, None
        safe_start, safe_end = name.split("__to__", 1)
        sdt = _unsanitize_iso_from_name(safe_start)
        edt = _unsanitize_iso_from_name(safe_end)
        return (sdt.isoformat() if sdt else None, edt.isoformat() if edt else None)
    except Exception:
        return None, None


def build_default_queries() -> List[Dict[str, Any]]:
    """
    Provide a starter set of queries for manual labeling.
    """
    return [
        {"query": "person wearing red jacket", "expected_ids": []},
        {"query": "person with bag", "expected_ids": []},
        {"query": "person wearing black pants", "expected_ids": []},
        {"query": "count people in the last minute", "expected_ids": []},
    ]


def derive_camera_id_from_path(p: Path) -> Optional[int]:
    """
    Try to infer camera_id from 'camera_{id}' in the path.
    """
    for part in p.parts:
        if part.startswith("camera_"):
            try:
                return int(part.split("_", 1)[1])
            except Exception:
                return None
    return None


def derive_clip_url(p: Path, clips_root: Path) -> Optional[str]:
    """
    Try to map a clip under data/clips to a /media/clips/... URL.
    """
    try:
        rel = p.resolve().relative_to(clips_root.resolve())
        return f"/media/clips/{rel.as_posix()}"
    except Exception:
        return None


def collect_clips(max_items: int, clips_root: Path) -> List[Path]:
    """
    Scan data/clips/camera_*/**/*.mp4 and return up to max_items.
    """
    paths: List[Path] = []
    if not clips_root.exists():
        return paths
    for mp4 in sorted(clips_root.glob("camera_*/**/*.mp4")):
        paths.append(mp4)
        if len(paths) >= max_items:
            break
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a starter test dataset manifest from existing clips.")
    ap.add_argument("--max", type=int, default=10, help="Max number of clips to include (default: 10)")
    ap.add_argument("--out", type=str, default="tests/data/dataset.json", help="Output JSON path")
    ap.add_argument("--clips_root", type=str, default="data/clips", help="Root folder for clips")
    args = ap.parse_args()

    clips_root = Path(args.clips_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clips = collect_clips(args.max, clips_root=clips_root)

    items: List[Dict[str, Any]] = []
    for p in clips:
        start_iso, end_iso = _parse_start_end_from_clip_path(str(p))
        cam_id = derive_camera_id_from_path(p) or 0
        clip_url = derive_clip_url(p, clips_root=clips_root)

        items.append(
            {
                "camera_id": cam_id,
                "clip_path": str(p.resolve()),
                "clip_url": clip_url,
                "start_iso": start_iso,
                "end_iso": end_iso,
                "notes": "Fill expected_ids with detection document IDs or (camera_id, track_id) tuples for ground truth.",
                "ground_truth": build_default_queries(),
            }
        )

    dataset = {
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "count": len(items),
        "items": items,
        "instructions": (
            "For each item, edit ground_truth[*].expected_ids with the list of expected detection identifiers "
            "(e.g., Mongo _id strings) or a tuple-like string 'cam{camera_id}:track{track_id}' to refer to tracks. "
            "This manifest can be used by integration tests to evaluate recall and query accuracy."
        ),
    }

    out_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote dataset manifest with {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()
