"""Privacy helpers: retention pruning, face blur, and gender-off filtering.

Design notes
------------
* Retention is **never** performed automatically at import/startup.  The
  functions below are the tested policy primitive; a cron/systemd timer (or an
  explicit admin command) must call :func:`run_retention_policy` with
  ``dry_run=False`` and ``RETENTION_ENABLED=true``.
* Face blur uses OpenCV's bundled Haar cascade only -- no model download and no
  GPU requirement.  It is a coarse detector, not a guarantee of anonymisation;
  the switch is a deployment option.
* ``INCLUDE_GENDER_ATTRIBUTES=false`` removes gender terms from generated
  captions and from person-search result text.  It does not retrain or rewrite
  an already-built FAISS index.
"""
from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.app.config import settings


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------
def _media_dirs(media_dirs: Optional[Mapping[str, os.PathLike | str]] = None):
    if media_dirs is not None:
        return media_dirs
    return {
        "clips": settings.CLIPS_DIR,
        "snapshots": settings.SNAPSHOTS_DIR,
        "recordings": settings.RECORDINGS_DIR,
    }


def _iter_media_files(root: Path) -> Iterable[Path]:
    """Recursively yield regular files without following symlinks."""
    try:
        entries = list(os.scandir(root))
    except (FileNotFoundError, NotADirectoryError):
        return
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                yield from _iter_media_files(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path)
        except OSError:
            continue


def prune_old_media(
    media_dirs: Optional[Mapping[str, os.PathLike | str]] = None,
    retention_hours: Optional[float] = None,
    *,
    dry_run: bool = True,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Find (and optionally delete) media files older than the retention window.

    Returns a JSON-serialisable report.  Dry-run is the default; callers must
    opt in explicitly before any file is removed.  Only files are removed;
    directory structure is left intact.
    """
    hours = float(settings.RETENTION_HOURS if retention_hours is None else retention_hours)
    if hours <= 0:
        raise ValueError("retention_hours must be > 0")
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    cutoff = reference - timedelta(hours=hours)

    by_category: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, str]] = []
    candidates: List[Dict[str, Any]] = []
    scanned = 0

    for category, directory in _media_dirs(media_dirs).items():
        root = Path(directory)
        category_report = {
            "directory": str(root),
            "scanned_files": 0,
            "candidate_files": 0,
            "deleted_files": 0,
            "bytes_reclaimable": 0,
            "bytes_reclaimed": 0,
        }
        by_category[category] = category_report
        if not root.exists():
            continue
        for path in _iter_media_files(root):
            scanned += 1
            category_report["scanned_files"] += 1
            try:
                stat = path.stat()
            except OSError as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if mtime >= cutoff:
                continue
            candidate = {
                "path": str(path),
                "category": category,
                "size_bytes": int(stat.st_size),
                "mtime": mtime.isoformat(),
                "age_hours": round((reference - mtime).total_seconds() / 3600.0, 3),
                "deleted": False,
            }
            category_report["candidate_files"] += 1
            category_report["bytes_reclaimable"] += int(stat.st_size)
            if not dry_run:
                try:
                    path.unlink()
                    candidate["deleted"] = True
                    category_report["deleted_files"] += 1
                    category_report["bytes_reclaimed"] += int(stat.st_size)
                except OSError as exc:
                    errors.append({"path": str(path), "error": str(exc)})
            candidates.append(candidate)

    return {
        "dry_run": bool(dry_run),
        "retention_hours": hours,
        "cutoff": cutoff.isoformat(),
        "reference_time": reference.isoformat(),
        "scanned_files": scanned,
        "candidate_files": len(candidates),
        "deleted_files": sum(1 for c in candidates if c["deleted"]),
        "bytes_reclaimable": sum(c["size_bytes"] for c in candidates),
        "bytes_reclaimed": sum(
            c["size_bytes"] for c in candidates if c["deleted"]
        ),
        "by_category": by_category,
        "candidates": candidates,
        "errors": errors,
    }


def run_retention_policy(
    *,
    dry_run: bool = True,
    force: bool = False,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Run the configured retention policy.

    Refuses to act unless ``RETENTION_ENABLED=true`` (or ``force=True`` is
    passed by an explicit admin tool/test).  This function is not called from
    application startup.
    """
    if not getattr(settings, "RETENTION_ENABLED", False) and not force:
        return {
            "skipped": True,
            "reason": "RETENTION_ENABLED=false; no files were inspected or deleted",
            "dry_run": bool(dry_run),
            "retention_hours": float(settings.RETENTION_HOURS),
        }
    report = prune_old_media(
        retention_hours=float(settings.RETENTION_HOURS),
        dry_run=dry_run,
        now=now,
    )
    report["skipped"] = False
    return report


# ---------------------------------------------------------------------------
# Face blur (OpenCV Haar cascade, bundled with opencv)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        raise RuntimeError(f"OpenCV Haar cascade not found: {cascade_path}")
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"failed to load OpenCV Haar cascade: {cascade_path}")
    return cascade


def detect_faces(
    image: np.ndarray,
    *,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (20, 20),
) -> List[Tuple[int, int, int, int]]:
    if image is None or getattr(image, "size", 0) == 0:
        return []
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("image must be HxW, HxWx3, or HxWx4")
    gray = cv2.equalizeHist(gray)
    faces = get_face_cascade().detectMultiScale(
        gray,
        scaleFactor=float(scale_factor),
        minNeighbors=int(min_neighbors),
        minSize=tuple(int(v) for v in min_size),
    )
    return [tuple(int(v) for v in face) for face in faces]


def blur_faces_with_count(
    image: np.ndarray,
    *,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    margin: float = 0.2,
) -> Tuple[np.ndarray, int]:
    """Blur detected faces, returning ``(image, number_of_faces)``.

    With no face present the original array is returned unchanged.  The
    function works for BGR, BGRA, and grayscale images.
    """
    if image is None:
        raise ValueError("image is None")
    array = np.asarray(image)
    faces = detect_faces(
        array, scale_factor=scale_factor, min_neighbors=min_neighbors
    )
    if not faces:
        return array, 0

    out = array.copy()
    height, width = out.shape[:2]
    for x, y, w, h in faces:
        mx = int(round(w * max(0.0, margin)))
        my = int(round(h * max(0.0, margin)))
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(width, x + w + mx)
        y2 = min(height, y + h + my)
        if x2 <= x1 or y2 <= y1:
            continue
        region = out[y1:y2, x1:x2]
        if region.size == 0:
            continue
        shortest = max(1, min(region.shape[:2]))
        ksize = max(3, int(shortest * 0.25))
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 101)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), 0)
    return out, len(faces)


def blur_faces(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Face-blur convenience wrapper (returns only the image)."""
    blurred, _ = blur_faces_with_count(image, **kwargs)
    return blurred


def face_blur_enabled() -> bool:
    return bool(getattr(settings, "FACE_BLUR_ENABLED", False))


def blur_faces_in_file(
    input_path: Path | str,
    output_path: Optional[Path | str] = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Blur faces in a still image; write atomically when replacing in place.

    Refuses to touch the file unless ``FACE_BLUR_ENABLED=true`` or the caller
    explicitly passes ``force=True``.  Returns a skipped report otherwise.
    """
    source = Path(input_path)
    target = Path(output_path) if output_path is not None else source
    if not force and not face_blur_enabled():
        return {
            "source": str(source),
            "output": str(target),
            "faces_blurred": 0,
            "skipped": True,
            "reason": "FACE_BLUR_ENABLED=false; pass force=True to override",
        }
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {source}")
    blurred, count = blur_faces_with_count(image)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target == source:
        temp = target.with_name(target.name + ".blur.tmp" + target.suffix)
        if not cv2.imwrite(str(temp), blurred):
            raise RuntimeError(f"could not write blurred image: {temp}")
        shutil.move(str(temp), str(target))
    else:
        if not cv2.imwrite(str(target), blurred):
            raise RuntimeError(f"could not write blurred image: {target}")
    return {
        "source": str(source),
        "output": str(target),
        "faces_blurred": count,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Gender-off filtering
# ---------------------------------------------------------------------------
_GENDER_WORDS = {
    "male",
    "female",
    "man",
    "woman",
    "men",
    "women",
    "boy",
    "girl",
    "masculine",
    "feminine",
    "gender",
}
_GENDER_RE = re.compile(
    r"\b(" + "|".join(sorted(_GENDER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def gender_attributes_enabled() -> bool:
    return bool(getattr(settings, "INCLUDE_GENDER_ATTRIBUTES", True))


def strip_gender_terms(text: str) -> str:
    """Remove gender tokens from a comma-separated attribute caption.

    ``"person, wearing red clothing, female"`` -> ``"person, wearing red
    clothing"``.  Punctuation/whitespace left behind by the removal is
    normalised.
    """
    if not text:
        return text
    parts: List[str] = []
    for raw in str(text).split(","):
        cleaned = _GENDER_RE.sub("", raw)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t-;:")
        if cleaned:
            parts.append(cleaned)
    return ", ".join(parts)


def strip_gender_from_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy a detection/person result and strip gender from text fields.

    Nested result containers (``objects``, ``person_hits``, ``frames``,
    ``detection``) are handled recursively so person-search output cannot leak
    gender text through a child record.
    """
    out = dict(record)
    for key in ("attribute_text", "caption", "semantic_query"):
        if isinstance(out.get(key), str):
            out[key] = strip_gender_terms(out[key])
    if isinstance(out.get("object_captions"), list):
        out["object_captions"] = [
            strip_gender_terms(item) if isinstance(item, str) else item
            for item in out["object_captions"]
        ]
    for key in ("objects", "person_hits", "frames", "detection"):
        value = out.get(key)
        if isinstance(value, list):
            out[key] = [
                strip_gender_from_record(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        elif isinstance(value, Mapping):
            out[key] = strip_gender_from_record(value)
    return out


def strip_gender_from_results(
    results: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if gender_attributes_enabled():
        return [dict(r) for r in results]
    return [strip_gender_from_record(r) for r in results]


__all__ = [
    "blur_faces",
    "blur_faces_in_file",
    "blur_faces_with_count",
    "detect_faces",
    "face_blur_enabled",
    "gender_attributes_enabled",
    "get_face_cascade",
    "prune_old_media",
    "run_retention_policy",
    "strip_gender_from_record",
    "strip_gender_from_results",
    "strip_gender_terms",
]
