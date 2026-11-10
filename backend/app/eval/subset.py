"""Released-subset loaders, track proxies, and motion-energy features.

Everything here reads only the artefacts shipped in ``data/``.  There is no
MongoDB dependency and no synthetic embedding.  The module is deliberately
small so the benchmark builder and the two evaluators share exactly the same
clip/time/track definitions.

Released scope (verified at import time by the builder, not assumed here):
  * scene index: 242 frames -> 121 clips, camera_0 only
  * person index: 1,778 descriptors, camera_0 only, track_id = -1 for all
  * 2026-03-01 17:01--18:56
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
SCENE_META = DATA_DIR / "index" / "faiss" / "meta.json"
SCENE_INDEX = DATA_DIR / "index" / "faiss" / "index.bin"
PERSON_META = DATA_DIR / "index" / "faiss_person" / "meta.json"
PERSON_INDEX = DATA_DIR / "index" / "faiss_person" / "index.bin"
SNAPSHOT_DIR = DATA_DIR / "snapshots" / "camera_0"
CLIP_DIR = DATA_DIR / "clips" / "camera_0"

# The released person descriptors are [0.6*v_unit || 0.4*t_unit] / norm.
# These sub-vector norms let us recover the two unit branches without
# re-running either encoder.
PERSON_VISUAL_DIM = 768
PERSON_TEXT_DIM = 384
PERSON_VISUAL_SUBNORM = 0.8320493
PERSON_TEXT_SUBNORM = 0.5546990

_COUNT_RE = re.compile(r"(\d+)\s+person\(s\)")
_COLOR_RE = re.compile(
    r"\b(Olive|Red|Brown|Black|Blue|White|Green|Gray|Grey|Yellow|Purple|Navy|Orange|Pink)\b",
    re.IGNORECASE,
)
_CLIP_RE = re.compile(r"^(?P<start>.+)__to__(?P<end>.+)\.mp4$")


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def parse_clip_ts(text: str) -> _dt.datetime:
    """Parse ``2026-03-01T17-01-00-052839`` (clip filename form)."""
    date, clock = text.split("T")
    parts = clock.split("-")
    if len(parts) < 3:
        raise ValueError(f"bad clip timestamp: {text!r}")
    micro = parts[3] if len(parts) > 3 else "0"
    return _dt.datetime.fromisoformat(f"{date}T{parts[0]}:{parts[1]}:{parts[2]}.{micro}")


def parse_det_ts(text: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(text)


def parse_clip_stem(stem: str) -> Tuple[_dt.datetime, _dt.datetime]:
    m = _CLIP_RE.match(stem + ".mp4")
    if not m:
        raise ValueError(f"bad clip name: {stem!r}")
    return parse_clip_ts(m.group("start")), parse_clip_ts(m.group("end"))


def fmt_hms(ts: _dt.datetime, seconds: bool = False) -> str:
    return ts.strftime("%H:%M:%S" if seconds else "%H:%M")


def five_min_bucket(ts: _dt.datetime) -> str:
    minute = (ts.minute // 5) * 5
    return ts.strftime("%Y-%m-%dT%H:") + f"{minute:02d}"


# ---------------------------------------------------------------------------
# Scene metadata
# ---------------------------------------------------------------------------

def _first_zone(caption: str) -> Optional[str]:
    head = (caption or "").split(".")[0].strip().lower()
    return head or None


def _count_from_caption(caption: str) -> Optional[int]:
    m = _COUNT_RE.search(caption or "")
    return int(m.group(1)) if m else None


def _colors_from_caption(caption: str) -> List[str]:
    return [c.capitalize() for c in _COLOR_RE.findall(caption or "")]


def _colors_from_objects(object_captions: Sequence[str]) -> List[str]:
    out: List[str] = []
    for cap in object_captions or []:
        m = re.search(r"wearing\s+(.+?)\s+clothing", cap, re.IGNORECASE)
        if m:
            out.extend(c.capitalize() for c in _COLOR_RE.findall(m.group(1)))
    return out


def load_scene_frames() -> List[Dict[str, Any]]:
    raw = json.loads(SCENE_META.read_text(encoding="utf-8"))
    frames: List[Dict[str, Any]] = []
    for item in raw:
        path = Path(str(item["clip_path"]))
        stem = path.stem
        start, end = parse_clip_stem(stem)
        colors = _colors_from_caption(item.get("caption", ""))
        colors += _colors_from_objects(item.get("object_captions", []))
        frames.append(
            {
                "faiss_id": int(item["faiss_id"]),
                "camera_id": int(item.get("camera_id", 0)),
                "clip_id": stem,
                "clip_name": path.name,
                "clip_path": str(path),
                "clip_start": start,
                "clip_end": end,
                "frame_ts": parse_det_ts(item["frame_ts"]),
                "caption": item.get("caption", ""),
                "object_captions": list(item.get("object_captions") or []),
                "count": _count_from_caption(item.get("caption", "")),
                "zone": _first_zone(item.get("caption", "")),
                "colors": sorted(set(colors)),
                "event": _first_zone(item.get("caption", ""))
                if _first_zone(item.get("caption", "")) in {"fighting", "smoke"}
                else None,
            }
        )
    frames.sort(key=lambda x: (x["frame_ts"], x["faiss_id"]))
    return frames


def build_clip_table(frames: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    clips: Dict[str, Dict[str, Any]] = {}
    for fr in frames:
        cid = fr["clip_id"]
        clip = clips.setdefault(
            cid,
            {
                "clip_id": cid,
                "camera_id": fr["camera_id"],
                "clip_path": fr["clip_path"],
                "clip_name": fr["clip_name"],
                "start": fr["clip_start"],
                "end": fr["clip_end"],
                "frames": [],
            },
        )
        clip["frames"].append(fr)
    for clip in clips.values():
        clip["frames"].sort(key=lambda x: x["frame_ts"])
        counts = [f["count"] for f in clip["frames"] if f["count"] is not None]
        clip["counts"] = counts
        clip["modal_count"] = (
            max(set(counts), key=lambda c: (counts.count(c), c)) if counts else None
        )
        clip["zones"] = sorted({f["zone"] for f in clip["frames"] if f["zone"]})
        clip["colors"] = sorted({c for f in clip["frames"] for c in f["colors"]})
        clip["events"] = sorted({f["event"] for f in clip["frames"] if f["event"]})
    return dict(sorted(clips.items(), key=lambda kv: kv[1]["start"]))


# ---------------------------------------------------------------------------
# Person metadata / vectors
# ---------------------------------------------------------------------------

def load_person_meta(sort_by_time: bool = True) -> List[Dict[str, Any]]:
    """Load person metadata.

    ``row_index`` is the FAISS row / sidecar position and is the only unique
    identifier (``detection_id`` and ``faiss_id`` both contain duplicates in
    the release).  Rows must stay in sidecar order when they are paired with
    ``load_index_vectors`` output; pass ``sort_by_time=False`` for that use.
    The default sorted order is kept for query construction, where the order
    only affects deterministic sampling.
    """
    raw = json.loads(PERSON_META.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for row_index, item in enumerate(raw):
        out.append(
            {
                "row_index": row_index,
                "faiss_id": int(item["faiss_id"]),
                "detection_id": str(item["detection_id"]),
                "camera_id": int(item.get("camera_id", 0)),
                "timestamp": parse_det_ts(item["timestamp"]),
                "timestamp_str": item["timestamp"],
                "color": str(item.get("color") or ""),
                "attribute_text": str(item.get("attribute_text") or ""),
                "object_index": int(item.get("object_index", 0)),
            }
        )
    if sort_by_time:
        out.sort(key=lambda x: (x["timestamp"], x["faiss_id"]))
    return out


def load_index_vectors(path: Path, dim: int) -> np.ndarray:
    import faiss

    index = faiss.read_index(str(path))
    if index.d != dim:
        raise ValueError(f"{path} has dim {index.d}, expected {dim}")
    return index.reconstruct_n(0, index.ntotal).astype(np.float32)


def split_person_vectors(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Recover unit visual/text branches from the stored weighted concat."""
    if vectors.ndim != 2 or vectors.shape[1] != 1152:
        raise ValueError("expected an (N, 1152) person matrix")
    v = vectors[:, :PERSON_VISUAL_DIM]
    t = vectors[:, PERSON_VISUAL_DIM:]
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    v_unit = np.divide(v, np.maximum(vn, 1e-9)).astype(np.float32)
    t_unit = np.divide(t, np.maximum(tn, 1e-9)).astype(np.float32)
    return v_unit, t_unit


def assign_detections_to_clips(
    detections: Sequence[Dict[str, Any]], clips: Dict[str, Dict[str, Any]]
) -> List[Optional[str]]:
    ordered = sorted(clips.items(), key=lambda kv: kv[1]["start"])
    out: List[Optional[str]] = []
    for det in detections:
        ts = det["timestamp"]
        hit = None
        for cid, clip in ordered:
            if clip["start"] <= ts <= clip["end"]:
                hit = cid
                break
        out.append(hit)
    return out


# ---------------------------------------------------------------------------
# Deterministic single-camera track proxies
# ---------------------------------------------------------------------------

def build_tracks(
    detections: Sequence[Dict[str, Any]],
    visual_unit: np.ndarray,
    max_gap: float = 3.0,
    threshold: float = 0.95,
    max_duration: float = 60.0,
    max_size: int = 20,
) -> Tuple[List[int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy nearest-previous visual matching with a temporal gate.

    A detection joins the most similar active track whose last detection is
    at most ``max_gap`` seconds older, whose first detection is at most
    ``max_duration`` seconds older, and whose cosine is at least
    ``threshold``.  Same-timestamp detections are never linked.  The duration
    cap prevents chained matches from producing hour-long pseudo-tracks.
    This is a *proxy* for identity: the release has no track ids (all -1), so
    the resulting tracks are pseudo-labels, not ground-truth identities.
    """
    n = len(detections)
    if visual_unit.shape[0] != n:
        raise ValueError("visual_unit rows must match detections")
    order = sorted(range(n), key=lambda i: (detections[i]["timestamp"], detections[i]["faiss_id"]))
    track_of = [-1] * n
    tracks: List[Dict[str, Any]] = []
    active: List[Tuple[int, int]] = []  # (track_id, last_detection_index)
    edges: List[Dict[str, Any]] = []

    for i in order:
        ts_i = detections[i]["timestamp"]
        best_tid = -1
        best_j = -1
        best_sim = -2.0
        for tid, j in active:
            dt = (ts_i - detections[j]["timestamp"]).total_seconds()
            if dt <= 0 or dt > max_gap:
                continue
            if (ts_i - tracks[tid]["start"]).total_seconds() > max_duration:
                continue
            if tracks[tid]["size"] >= max_size:
                continue
            sim = float(visual_unit[i] @ visual_unit[j])
            if sim > best_sim or (sim == best_sim and tid < best_tid):
                best_sim, best_tid, best_j = sim, tid, j
        if best_tid >= 0 and best_sim >= threshold:
            track_of[i] = best_tid
            edges.append(
                {
                    "query_index": i,
                    "gallery_index": best_j,
                    "track_id": best_tid,
                    "same": True,
                    "dt": (ts_i - detections[best_j]["timestamp"]).total_seconds(),
                    "cosine": best_sim,
                }
            )
            for k, (tid, j) in enumerate(active):
                if tid == best_tid:
                    active[k] = (tid, i)
                    break
        else:
            tid = len(tracks)
            track_of[i] = tid
            tracks.append(
                {"track_id": tid, "indices": [], "start": ts_i, "end": ts_i, "size": 0}
            )
            active.append((tid, i))
        tracks[track_of[i]]["indices"].append(i)
        tracks[track_of[i]]["size"] += 1
        tracks[track_of[i]]["start"] = min(tracks[track_of[i]]["start"], ts_i)
        tracks[track_of[i]]["end"] = max(tracks[track_of[i]]["end"], ts_i)
        active = [
            (tid, j)
            for tid, j in active
            if (ts_i - detections[j]["timestamp"]).total_seconds() <= max_gap
        ]

    for tr in tracks:
        tr["size"] = len(tr["indices"])
        tr["track_id_str"] = f"t{tr['track_id']:05d}"
    return track_of, tracks, edges


def track_quality(
    detections: Sequence[Dict[str, Any]],
    visual_unit: np.ndarray,
    track_of: Sequence[int],
    max_gap: float = 3.0,
) -> Dict[str, float]:
    """Self-supervised quality proxy used to pick the track threshold.

    ``separation`` is mean same-track link cosine minus mean best cosine to a
    different track within the same temporal gate.  ``coverage`` is the
    fraction of detections in tracks with at least two members.  The product
    rewards discriminative links without collapsing every detection into a
    singleton.  No human identity labels are used.
    """
    n = len(detections)
    same: List[float] = []
    for i in range(n):
        for j in range(n):
            if j == i or track_of[i] != track_of[j]:
                continue
            dt = abs((detections[i]["timestamp"] - detections[j]["timestamp"]).total_seconds())
            if 0 < dt <= max_gap:
                same.append(float(visual_unit[i] @ visual_unit[j]))
    diff: List[float] = []
    for i in range(n):
        best = None
        for j in range(n):
            if j == i or track_of[j] == track_of[i]:
                continue
            dt = abs((detections[i]["timestamp"] - detections[j]["timestamp"]).total_seconds())
            if 0 < dt <= max_gap:
                sim = float(visual_unit[i] @ visual_unit[j])
                best = sim if best is None else max(best, sim)
        if best is not None:
            diff.append(best)
    counts: Dict[int, int] = {}
    for t in track_of:
        counts[t] = counts.get(t, 0) + 1
    coverage = sum(1 for t in track_of if counts[t] >= 2) / float(max(1, n))
    within = float(np.mean(same)) if same else 0.0
    between = float(np.mean(diff)) if diff else 0.0
    return {
        "separation": within - between,
        "within": within,
        "between": between,
        "coverage": coverage,
        "objective": (within - between) * coverage,
        "n_tracks": float(len(counts)),
        "n_multi": float(sum(1 for v in counts.values() if v >= 2)),
    }


def tune_track_threshold(
    detections: Sequence[Dict[str, Any]],
    visual_unit: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
    max_gap: float = 3.0,
) -> Tuple[float, List[Dict[str, float]]]:
    thresholds = list(thresholds or [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    rows: List[Dict[str, float]] = []
    for thr in thresholds:
        track_of, tracks, _ = build_tracks(detections, visual_unit, max_gap, thr)
        q = track_quality(detections, visual_unit, track_of, max_gap)
        rows.append({"threshold": float(thr), **q})
    best_obj = max(r["objective"] for r in rows)
    # The calibration objective is flat around the optimum.  Pick the
    # *strictest* threshold within 5% of the best objective (a one-standard-
    # error-style rule) rather than the loosest threshold at the edge of the
    # plateau.  This reduces false merges without materially changing the
    # calibration score.
    stable = [r for r in rows if r["objective"] >= best_obj * 0.95]
    best_thr = max(r["threshold"] for r in stable)
    return best_thr, rows


# ---------------------------------------------------------------------------
# Motion-energy proxy for action-intensity queries
# ---------------------------------------------------------------------------

def compute_motion_energy(
    clips: Dict[str, Dict[str, Any]], max_side: int = 96
) -> Dict[str, Dict[str, float]]:
    """Mean absolute grayscale difference between consecutive snapshots.

    Snapshots are read from the released ``data/snapshots/camera_0`` folder.
    The value is an action-intensity proxy, not a semantic action label.
    """
    import cv2

    snaps = sorted(SNAPSHOT_DIR.glob("*.jpg"))
    snap_ts = []
    for p in snaps:
        stem = p.stem  # 2026-03-01T17-01-00-052839
        date, clock = stem.split("T")
        parts = clock.split("-")
        snap_ts.append(
            (
                _dt.datetime.fromisoformat(
                    f"{date}T{parts[0]}:{parts[1]}:{parts[2]}.{parts[3]}"
                ),
                p,
            )
        )
    snap_ts.sort(key=lambda x: x[0])
    out: Dict[str, Dict[str, float]] = {}
    for cid, clip in clips.items():
        files = [p for ts, p in snap_ts if clip["start"] <= ts <= clip["end"]]
        if len(files) < 2:
            out[cid] = {"mean": 0.0, "median": 0.0, "n_snapshots": float(len(files))}
            continue
        imgs: List[np.ndarray] = []
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (max_side, max_side), interpolation=cv2.INTER_AREA)
            imgs.append(img.astype(np.float32) / 255.0)
        diffs = [float(np.mean(np.abs(imgs[i + 1] - imgs[i]))) for i in range(len(imgs) - 1)]
        out[cid] = {
            "mean": float(np.mean(diffs)) if diffs else 0.0,
            "median": float(np.median(diffs)) if diffs else 0.0,
            "n_snapshots": float(len(imgs)),
        }
    return out


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def overlap_seconds(
    a_start: _dt.datetime, a_end: _dt.datetime, b_start: _dt.datetime, b_end: _dt.datetime
) -> float:
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0.0, (hi - lo).total_seconds())


def clip_sort_key(clip: Dict[str, Any]) -> _dt.datetime:
    return clip["start"]


def modality_norm(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else x


def json_default(obj: Any) -> Any:
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")
