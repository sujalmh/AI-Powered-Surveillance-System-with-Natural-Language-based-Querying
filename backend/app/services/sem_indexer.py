from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import imagehash

# Optional YOLO fallback for object captions when structured detections are unavailable (e.g., uploaded clips without timestamps)
try:
    from ultralytics import YOLO as _UltralyticsYOLO  # type: ignore
except Exception:  # pragma: no cover
    _UltralyticsYOLO = None  # type: ignore

_YOLO_FALLBACK = None  # type: ignore

def _get_yolo_fallback():
    global _YOLO_FALLBACK
    if _YOLO_FALLBACK is None and _UltralyticsYOLO is not None:
        try:
            _YOLO_FALLBACK = _UltralyticsYOLO(settings.MODEL_PATH)
        except Exception:
            _YOLO_FALLBACK = None
    return _YOLO_FALLBACK

def _object_captions_from_yolo(img: np.ndarray, limit: int = 20) -> List[str]:
    """
    Lightweight, best-effort object captions using YOLO on the frame image.
    Provides class names; does not compute colors (kept simple for testing).
    """
    out: List[str] = []
    try:
        model = _get_yolo_fallback()
        if model is None:
            return out
        res = model(img)
        for r in res:
            if r.boxes is None:
                continue
            cls = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else []
            for c in cls:
                try:
                    name = str(r.names[int(c)]) if hasattr(r, "names") else None
                except Exception:
                    name = None
                if name:
                    out.append(name.lower())
                if len(out) >= limit:
                    return out
    except Exception:
        pass
    return out

from backend.app.config import settings
from backend.app.services.sem_embedder import get_embedder
from backend.app.services.sem_store import get_faiss_store
from backend.app.services.sem_captioner import get_captioner
from backend.app.db.mongo import vlm_frames, detections


@dataclass
class FrameBundle:
    img: np.ndarray
    frame_index: int
    frame_ts: Optional[str]  # ISO timestamp if derivable, else None


def _parse_start_end_from_clip_path(clip_path: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Attempt to parse start/end ISO timestamps from our file naming convention:
    {safe_start}__to__{safe_end}.mp4  where safe_start/end replaced ":" and "." with "-"
    Example: 2025-10-23T21-08-09-698850__to__2025-10-23T21-09-22-527513.mp4
    """
    try:
        name = Path(clip_path).stem
        if "__to__" not in name:
            return None, None
        safe_start, safe_end = name.split("__to__", 1)

        def _unsanitize(s: str) -> Optional[datetime]:
            # Expect: YYYY-MM-DDTHH-MM-SS(-us)?
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
                fmt = "%Y-%m-%dT%H:%M:%S"
                dt = datetime.strptime(f"{date_part}T{H}:{M}:{S}", fmt)
                # microseconds if present and numeric
                try:
                    u = int(us)
                    dt = dt.replace(microsecond=u)
                except Exception:
                    pass
                return dt
            except Exception:
                return None

        return _unsanitize(safe_start), _unsanitize(safe_end)
    except Exception:
        return None, None


def _frame_time_from_index(frame_idx: int, fps: float, start_dt: Optional[datetime]) -> Optional[str]:
    try:
        if start_dt is None or fps <= 0:
            return None
        sec = frame_idx / max(fps, 1e-6)
        ts = start_dt + timedelta(seconds=sec)
        return ts.isoformat()
    except Exception:
        return None


def _compose_caption_from_detections(camera_id: int, frame_ts: Optional[str], top_k: int = 3) -> str:
    """
    Build a richer, multi-part caption from structured detections near the frame time:
    - Summarize persons and clothing colors
    - List top environment objects with counts
    """
    try:
        if not frame_ts:
            return ""
        # Query a small time window around frame_ts for this camera
        ts = datetime.fromisoformat(frame_ts)
        window = timedelta(seconds=3)
        q = {
            "camera_id": int(camera_id),
            "timestamp": {"$gte": (ts - window).isoformat(), "$lte": (ts + window).isoformat()},
        }
        docs = list(detections.find(q, {"_id": 0, "objects": 1}).limit(5))
        if not docs:
            return ""

        # Aggregate across matched docs
        person_colors: Dict[str, int] = {}
        persons = 0
        obj_counts: Dict[str, int] = {}
        for d in docs:
            for o in (d.get("objects") or []):
                name = str(o.get("object_name") or "").strip().lower()
                if not name:
                    continue
                if name == "person":
                    persons += 1
                    col = str(o.get("color") or "").strip()
                    if col and col.lower() != "unknown":
                        person_colors[col] = person_colors.get(col, 0) + 1
                else:
                    obj_counts[name] = obj_counts.get(name, 0) + 1

        parts: List[str] = []

        # Person summary with clothing colors
        if persons > 0:
            color_list = sorted(person_colors.items(), key=lambda x: x[1], reverse=True)[: top_k]
            if color_list:
                colors_str = ", ".join([f"{c} ({n})" for c, n in color_list])
                parts.append(f"{persons} person(s), clothing colors: {colors_str}")
            else:
                parts.append(f"{persons} person(s) present")

        # Environment objects
        env_top = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[: top_k]
        if env_top:
            env_str = ", ".join([f"{n}× {obj}" for obj, n in env_top])
            parts.append(f"environment: {env_str}")

        return ". ".join(parts)
    except Exception:
        return ""


def _object_captions_from_detections(camera_id: int, frame_ts: Optional[str], limit: int = 20) -> List[str]:
    """
    Build a list of per-object captions near the frame time for richer testing output.
    Example items:
      - "person wearing red clothing, hat"
      - "car"
      - "person wearing black clothing, carrying bag"
    """
    out: List[str] = []
    try:
        if not frame_ts:
            return out
        ts = datetime.fromisoformat(frame_ts)
        window = timedelta(seconds=3)
        q = {
            "camera_id": int(camera_id),
            "timestamp": {"$gte": (ts - window).isoformat(), "$lte": (ts + window).isoformat()},
        }
        docs = list(detections.find(q, {"_id": 0, "objects": 1}).limit(10))
        for d in docs:
            for o in (d.get("objects") or []):
                name = str(o.get("object_name") or "").strip().lower()
                if not name:
                    continue
                if name == "person":
                    color = str(o.get("color") or "").strip()
                    acc = o.get("person_attributes") or {}
                    parts: List[str] = ["person"]
                    if color and color.lower() != "unknown":
                        parts.append(f"wearing {color.lower()} clothing")
                    # accessories
                    if acc.get("hat_confidence", 0.0) > 0.5:
                        parts.append("hat")
                    if acc.get("bag_confidence", 0.0) > 0.5:
                        parts.append("carrying bag")
                    if acc.get("longsleeves_confidence", 0.0) > 0.5:
                        parts.append("long sleeves")
                    if acc.get("longpants_confidence", 0.0) > 0.5:
                        parts.append("long pants")
                    if acc.get("coat_jacket_confidence", 0.0) > 0.5:
                        parts.append("coat/jacket")
                    out.append(", ".join(parts))
                else:
                    out.append(name)
                if len(out) >= limit:
                    return out
    except Exception:
        pass
    return out

def sample_frames_from_clip(clip_path: str, every_sec: float = 1.0) -> Iterator[FrameBundle]:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_step = max(1, int(round(fps * every_sec))) if fps > 0 else 1
    start_dt, _ = _parse_start_end_from_clip_path(clip_path)

    try:
        idx = 0
        out_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_step == 0:
                ts_iso = _frame_time_from_index(idx, fps, start_dt)
                yield FrameBundle(img=frame, frame_index=idx, frame_ts=ts_iso)
                out_idx += 1
            idx += 1
    finally:
        cap.release()


def _phash_hex(img: np.ndarray) -> str:
    try:
        # Convert to RGB PIL for imagehash
        if img.ndim == 2:
            x = np.stack([img, img, img], axis=-1)
        else:
            x = img
            if x.shape[-1] == 3 and x.flags["C_CONTIGUOUS"]:
                x = x[..., ::-1]  # BGR->RGB heuristic
        pil = Image.fromarray(x)
        return str(imagehash.phash(pil))
    except Exception:
        return ""


def index_clip(
    camera_id: int,
    clip_path: str,
    clip_url: Optional[str] = None,
    every_sec: float = 1.0,
    with_captions: bool = False,
) -> Dict[str, object]:
    """
    Index a clip by sampling frames every `every_sec`, deduplicating via phash,
    embedding with CLIP, and inserting vectors into FAISS + metadata into Mongo.
    """
    if not settings.ENABLE_SEMANTIC:
        return {"ok": False, "message": "semantic indexing disabled via config"}

    embedder = get_embedder()
    store = get_faiss_store(dim=512)

    # Optional captioner
    captioner = None
    if with_captions and settings.ENABLE_CAPTIONS:
        try:
            captioner = get_captioner()
        except Exception:
            captioner = None

    # Dedup across clip using phash set and also skip if already exists in DB (clip_path+frame_index unique)
    seen_hashes: set[str] = set()
    images: List[np.ndarray] = []
    metas: List[Dict[str, object]] = []

    now_iso = datetime.now().isoformat()
    added = 0

    for bundle in sample_frames_from_clip(clip_path, every_sec=every_sec):
        # DB-level uniqueness guard
        existing = vlm_frames.find_one(
            {"clip_path": clip_path, "frame_index": int(bundle.frame_index)},
            {"_id": 1, "caption": 1},
        )
        if existing:
            # Backfill caption without adding a new vector if missing
            if captioner is not None and not existing.get("caption"):
                try:
                    cap_txts = captioner.caption_images_batched([bundle.img])
                    cap_txt = cap_txts[0] if cap_txts else None
                except Exception:
                    cap_txt = None
                try:
                    vlm_frames.update_one(
                        {"_id": existing["_id"]},
                        {"$set": {"caption": cap_txt, "updated_at": now_iso}},
                    )
                except Exception:
                    pass
            continue
        h = _phash_hex(bundle.img)
        if h and h in seen_hashes:
            continue
        seen_hashes.add(h)

        images.append(bundle.img)
        metas.append(
            {
                "camera_id": int(camera_id),
                "clip_path": clip_path,
                "clip_url": clip_url,
                "frame_ts": bundle.frame_ts,
                "frame_index": int(bundle.frame_index),
                "model": settings.SEMANTIC_MODEL,
                "embedding_dim": 512,
                "hash": h,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

        # To bound memory, embed+flush in micro-batches ~ batch_size
        if len(images) >= embedder.batch_size:
            embs = embedder.image_embed(images)
            metas_to_add = metas
            if captioner is not None:
                try:
                    caps = captioner.caption_images_batched(images)
                except Exception:
                    caps = []
                metas_to_add = []
                for j, m in enumerate(metas):
                    m2 = dict(m)
                    base_cap = caps[j] if j < len(caps) else None
                    rich_cap = _compose_caption_from_detections(camera_id=int(m.get("camera_id", -1)), frame_ts=str(m.get("frame_ts") or ""))
                    if base_cap and rich_cap:
                        m2["caption"] = f"{base_cap}. {rich_cap}"
                    elif rich_cap:
                        m2["caption"] = rich_cap
                    else:
                        m2["caption"] = base_cap
                    # Save per-object captions for testing (richer like realtime camera indexing)
                    caps_list = _object_captions_from_detections(
                        camera_id=int(m.get("camera_id", -1)),
                        frame_ts=str(m.get("frame_ts") or "")
                    )
                    if not caps_list:
                        # Fallback to YOLO on the sampled frame when structured detections are not available
                        caps_list = _object_captions_from_yolo(images[j])
                    m2["object_captions"] = caps_list
                    metas_to_add.append(m2)
            else:
                # No captioner: still compose rich captions from structured detections and save per-object captions
                metas_to_add = []
                for j, m in enumerate(metas):
                    m2 = dict(m)
                    rich_cap = _compose_caption_from_detections(camera_id=int(m.get("camera_id", -1)), frame_ts=str(m.get("frame_ts") or ""))
                    m2["caption"] = rich_cap or None
                    caps_list = _object_captions_from_detections(
                        camera_id=int(m.get("camera_id", -1)),
                        frame_ts=str(m.get("frame_ts") or "")
                    )
                    if not caps_list:
                        caps_list = _object_captions_from_yolo(images[j])
                    m2["object_captions"] = caps_list
                    metas_to_add.append(m2)
            store.vector_add(embs, metas_to_add, save=False)
            images.clear()
            metas.clear()
            added += embs.shape[0]

    # Flush remainder
    if images:
        embs = embedder.image_embed(images)
        metas_to_add = metas
        if captioner is not None:
            try:
                caps = captioner.caption_images_batched(images)
            except Exception:
                caps = []
            metas_to_add = []
            for j, m in enumerate(metas):
                m2 = dict(m)
                base_cap = caps[j] if j < len(caps) else None
                rich_cap = _compose_caption_from_detections(camera_id=int(m.get("camera_id", -1)), frame_ts=str(m.get("frame_ts") or ""))
                if base_cap and rich_cap:
                    m2["caption"] = f"{base_cap}. {rich_cap}"
                elif rich_cap:
                    m2["caption"] = rich_cap
                else:
                    m2["caption"] = base_cap
                # Save per-object captions for testing (richer like realtime camera indexing)
                caps_list = _object_captions_from_detections(
                    camera_id=int(m.get("camera_id", -1)),
                    frame_ts=str(m.get("frame_ts") or "")
                )
                if not caps_list:
                    caps_list = _object_captions_from_yolo(images[j])
                m2["object_captions"] = caps_list
                metas_to_add.append(m2)
        else:
            # No captioner: compose rich captions from structured detections and save per-object captions
            metas_to_add = []
            for j, m in enumerate(metas):
                m2 = dict(m)
                rich_cap = _compose_caption_from_detections(camera_id=int(m.get("camera_id", -1)), frame_ts=str(m.get("frame_ts") or ""))
                m2["caption"] = rich_cap or None
                caps_list = _object_captions_from_detections(
                    camera_id=int(m.get("camera_id", -1)),
                    frame_ts=str(m.get("frame_ts") or "")
                )
                if not caps_list:
                    caps_list = _object_captions_from_yolo(images[j])
                m2["object_captions"] = caps_list
                metas_to_add.append(m2)
        store.vector_add(embs, metas_to_add, save=False)
        added += embs.shape[0]

    # Persist index and metadata mapping
    if added > 0:
        store.save()

    return {"ok": True, "indexed_frames": added, "model": settings.SEMANTIC_MODEL, "backend": settings.VECTOR_BACKEND}
