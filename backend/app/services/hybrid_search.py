from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.config import settings
from backend.app.services.attribute_encoder import get_attribute_encoder
from backend.app.services.fusion import MultimodalFusion
from backend.app.services.nl_parser import parse_nl_with_llm
from backend.app.services.person_store import get_person_store
from backend.app.services.retrieval_utils import parse_ts


@dataclass
class ParsedQuery:
    embedding_text: str
    camera_id: Optional[int]
    time_filter: Optional[Dict[str, Any]]
    colors: List[str]


def _derive_embedding_text(parsed_filter: Dict[str, Any], raw_query: str) -> str:
    # Prefer explicit color + person for better recall
    color = parsed_filter.get("objects.color")
    obj = parsed_filter.get("objects.object_name") or "person"
    if color:
        return f"{obj} wearing {str(color).lower()} clothing"
    # Fallback to raw query (shortened)
    rq = str(parsed_filter.get("__raw") or raw_query).strip()
    return rq if len(rq) <= 128 else rq[:128]


def _normalize_colors(parsed_filter: Dict[str, Any]) -> List[str]:
    c = parsed_filter.get("objects.color")
    if not c:
        return []
    return [str(c)]


def parse_for_hybrid(query: str) -> ParsedQuery:
    pf = parse_nl_with_llm(query)
    emb_txt = _derive_embedding_text(pf, query)
    cam = pf.get("camera_id")
    time_filter = pf.get("timestamp")
    colors = _normalize_colors(pf)
    return ParsedQuery(embedding_text=emb_txt, camera_id=cam, time_filter=time_filter, colors=colors)


def _norm_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-9:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _clip_bounds(clip_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Recover the clip interval from ``...__to__...`` filenames, if present."""
    if not clip_path:
        return None, None
    name = Path(str(clip_path)).stem
    if "__to__" not in name:
        return None, None
    left, right = name.split("__to__", 1)

    def _decode(token: str) -> Optional[str]:
        # Filenames use dashes in the time component: 2026-03-01T17-01-00-052839
        try:
            date, time = token.split("T", 1)
            hh, mm, ss, frac = time.split("-")
            return f"{date}T{hh}:{mm}:{ss}.{frac}"
        except Exception:
            return None

    return _decode(left), _decode(right)


class HybridSearchEngine:
    """
    Person-level hybrid search.

    Two entry points:

    * :meth:`search_meta` -- offline/metadata path.  Uses only the person FAISS
      index + ``meta.json`` and maps each person hit's timestamp to the nearest
      scene clip in ``faiss/meta.json``.  No MongoDB, LLM, or OpenCLIP needed
      (MiniLM encodes the text query).  This is the path used by
      ``UnifiedRetrieval`` and offline evaluation.
    * :meth:`search` -- legacy path that additionally parses the query with the
      LLM and fetches detection documents from Mongo when available.

    Person gallery vectors are 1152-d (SigLIP 768 + MiniLM 384).  A text-only
    query uses ``visual=0`` and the same weighted concat/L2 normalisation as the
    gallery, matching Algorithm 2 in the paper.
    """

    def __init__(
        self,
        person_top_k: int = 50,
        *,
        person_store: Any = None,
        attr_encoder: Any = None,
        fusion: Optional[MultimodalFusion] = None,
        scene_meta_path: Optional[Path | str] = None,
        person_meta_path: Optional[Path | str] = None,
        enable_mongo: Optional[bool] = None,
    ) -> None:
        self.person_store = (
            person_store if person_store is not None else get_person_store(dim=1152)
        )
        self.attr_encoder = (
            attr_encoder if attr_encoder is not None else get_attribute_encoder()
        )
        self.fusion = fusion or MultimodalFusion()
        self.top_k = int(person_top_k)
        self.scene_meta_path = Path(
            scene_meta_path or settings.FAISS_DIR / "meta.json"
        )
        self.person_meta_path = Path(
            person_meta_path or settings.FAISS_PERSON_DIR / "meta.json"
        )
        self._enable_mongo = enable_mongo
        self._mongo_checked = enable_mongo is False
        self._mongo_available: bool = bool(enable_mongo) if enable_mongo else False
        self._scene_cache: Optional[Dict[int, Tuple[List[datetime], List[Dict[str, Any]]]]] = None
        self._scene_all: Optional[Tuple[List[datetime], List[Dict[str, Any]]]] = None

    # ------------------------------------------------------------------
    # Availability / lazy metadata
    # ------------------------------------------------------------------
    @staticmethod
    def person_index_available(
        person_dir: Optional[Path | str] = None,
        person_meta_path: Optional[Path | str] = None,
    ) -> bool:
        """Cheap check used to decide whether person routing is possible."""
        directory = Path(person_dir or settings.FAISS_PERSON_DIR)
        index_path = directory / "index.bin"
        meta_path = Path(person_meta_path or directory / "meta.json")
        try:
            if not index_path.exists() or index_path.stat().st_size <= 0:
                return False
            if not meta_path.exists() or meta_path.stat().st_size <= 2:
                return False
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return isinstance(meta, list) and len(meta) > 0
        except Exception:
            return False

    def _load_scene_meta(self) -> None:
        if self._scene_cache is not None:
            return
        by_camera: Dict[int, List[Tuple[datetime, Dict[str, Any]]]] = {}
        all_rows: List[Tuple[datetime, Dict[str, Any]]] = []
        try:
            rows = json.loads(self.scene_meta_path.read_text(encoding="utf-8"))
        except Exception:
            rows = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict) or not row.get("clip_path"):
                    continue
                ts = row.get("frame_ts")
                if not ts:
                    continue
                try:
                    dt = parse_ts(str(ts))
                except Exception:
                    continue
                try:
                    camera_id = int(row.get("camera_id", -1))
                except Exception:
                    camera_id = -1
                by_camera.setdefault(camera_id, []).append((dt, row))
                all_rows.append((dt, row))
        cache: Dict[int, Tuple[List[datetime], List[Dict[str, Any]]]] = {}
        for camera_id, items in by_camera.items():
            items.sort(key=lambda item: item[0])
            cache[camera_id] = ([item[0] for item in items], [item[1] for item in items])
        all_rows.sort(key=lambda item: item[0])
        self._scene_cache = cache
        self._scene_all = (
            ([item[0] for item in all_rows], [item[1] for item in all_rows])
            if all_rows
            else None
        )

    def _nearest_scene(
        self, timestamp: Any, camera_id: Any
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        self._load_scene_meta()
        if not timestamp:
            return None, None
        try:
            dt = parse_ts(str(timestamp))
        except Exception:
            return None, None
        rows: Optional[Tuple[List[datetime], List[Dict[str, Any]]]] = None
        cache = self._scene_cache or {}
        try:
            camera_key: Optional[int] = int(camera_id)
        except Exception:
            camera_key = None
        if camera_key is not None and camera_key >= 0:
            rows = cache.get(camera_key)
            # Never map a known camera to another camera's clip when the scene
            # index has no vectors for it.
            if rows is None and cache:
                return None, None
        if not rows:
            rows = self._scene_all
        if not rows or not rows[0]:
            return None, None
        times, metas = rows
        idx = bisect_left(times, dt)
        candidates = []
        if idx < len(times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best_idx = min(candidates, key=lambda i: abs((times[i] - dt).total_seconds()))
        gap = abs((times[best_idx] - dt).total_seconds())
        return metas[best_idx], gap

    # ------------------------------------------------------------------
    # Encoding / filtering
    # ------------------------------------------------------------------
    def _encode_query(self, text: str) -> np.ndarray:
        t = self.attr_encoder.encode_text(text)  # 384-dim normalized
        v = np.zeros((768,), dtype=np.float32)   # No visual query available
        f = self.fusion.fuse(v, t)               # 1152-dim
        return f.astype(np.float32)

    def _build_filter_pred(self, parsed: ParsedQuery):
        colors = {str(c).lower() for c in parsed.colors if c}

        def _pred(meta: Dict[str, Any]) -> bool:
            if parsed.camera_id is not None:
                try:
                    if int(meta.get("camera_id", -1)) != int(parsed.camera_id):
                        return False
                except Exception:
                    return False
            if parsed.time_filter:
                try:
                    ts = meta.get("timestamp")
                    if ts and (parsed.time_filter.get("$gte") or parsed.time_filter.get("$lte")):
                        dt = parse_ts(str(ts))
                        if parsed.time_filter.get("$gte") and dt < parse_ts(parsed.time_filter["$gte"]):
                            return False
                        if parsed.time_filter.get("$lte") and dt > parse_ts(parsed.time_filter["$lte"]):
                            return False
                except Exception:
                    pass
            if colors:
                color = str(meta.get("color") or "").strip().lower()
                if color and color not in colors:
                    return False
            return True

        return _pred

    def _search_hits(
        self, parsed: ParsedQuery, top_k: int, fetch_multiplier: int = 3
    ) -> List[Dict[str, Any]]:
        qv = self._encode_query(parsed.embedding_text)
        return self.person_store.vector_search(
            qv,
            top_k=max(1, int(top_k)) * max(1, int(fetch_multiplier)),
            filter_pred=self._build_filter_pred(parsed),
        )

    # ------------------------------------------------------------------
    # Mongo enrichment (optional, cached probe)
    # ------------------------------------------------------------------
    def _mongo_is_available(self) -> bool:
        if self._enable_mongo is False:
            return False
        if self._mongo_checked:
            return self._mongo_available
        self._mongo_checked = True
        try:
            from backend.app.db.mongo import client

            client.admin.command("ping")
            self._mongo_available = True
        except Exception:
            self._mongo_available = False
        return self._mongo_available

    @staticmethod
    def _detection_summary(det_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not det_id:
            return None
        try:
            from bson import ObjectId  # type: ignore
            from backend.app.db.mongo import detections as detections_col

            doc = detections_col.find_one({"_id": ObjectId(det_id)})
            if not doc:
                return None
            return {
                "detection_id": str(det_id),
                "timestamp": doc.get("timestamp"),
                "camera_id": doc.get("camera_id"),
                "person_count": doc.get("person_count"),
                "objects": doc.get("objects", []),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Offline metadata search
    # ------------------------------------------------------------------
    def search_meta(
        self,
        query: str,
        top_k: int = 10,
        parsed_filter: Optional[Dict[str, Any]] = None,
        *,
        enrich_mongo: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Search the person index and map hits to scene clips, without Mongo.

        Returns clip-level dicts with ``clip_path``, ``clip_url``, ``timestamp``,
        ``score``/``score_raw``/``score_norm``, ``source='person'``, and the
        contributing person hits.  If the scene metadata is unavailable, the
        result set is empty so callers can fall back to the scene index.
        """
        parsed_filter = parsed_filter or {}
        text = str(
            query
            or parsed_filter.get("__semantic_query")
            or parsed_filter.get("__embedding_text")
            or parsed_filter.get("__raw")
            or ""
        ).strip()
        if not text:
            return []
        camera_id = parsed_filter.get("camera_id")
        if not isinstance(camera_id, int):
            camera_id = None
        parsed = ParsedQuery(
            embedding_text=text,
            camera_id=camera_id,
            time_filter=parsed_filter.get("timestamp")
            if isinstance(parsed_filter.get("timestamp"), dict)
            else None,
            colors=_normalize_colors(parsed_filter),
        )
        hits = self._search_hits(parsed, top_k=max(1, int(top_k)))
        if not hits:
            return []

        use_mongo = self._mongo_is_available() if enrich_mongo is None else bool(enrich_mongo)
        by_clip: Dict[str, Dict[str, Any]] = {}
        for hit in hits:
            meta = hit.get("meta") or {}
            scene, gap = self._nearest_scene(meta.get("timestamp"), meta.get("camera_id"))
            if scene is None:
                continue
            clip_path = scene.get("clip_path")
            if not clip_path:
                continue
            score = float(hit.get("score") or 0.0)
            entry = by_clip.get(clip_path)
            person_hit = {
                "faiss_id": meta.get("faiss_id"),
                "detection_id": meta.get("detection_id"),
                "object_index": meta.get("object_index"),
                "camera_id": meta.get("camera_id"),
                "track_id": meta.get("track_id"),
                "timestamp": meta.get("timestamp"),
                "color": meta.get("color"),
                "attribute_text": meta.get("attribute_text"),
                "score": score,
                "scene_gap_seconds": gap,
            }
            if entry is None:
                clip_start, clip_end = _clip_bounds(clip_path)
                entry = {
                    "camera_id": scene.get("camera_id", meta.get("camera_id")),
                    "clip_path": clip_path,
                    "clip_url": scene.get("clip_url"),
                    "timestamp": meta.get("timestamp"),
                    "start": clip_start or meta.get("timestamp"),
                    "end": clip_end or meta.get("timestamp"),
                    "score": score,
                    "score_raw": score,
                    "source": "person",
                    "retrieval_branch": "person",
                    "person_hits": [person_hit],
                    "objects": [
                        {
                            "object_name": "person",
                            "color": meta.get("color"),
                            "attribute_text": meta.get("attribute_text"),
                        }
                    ],
                    "frames": [
                        {
                            "frame_ts": meta.get("timestamp"),
                            "frame_index": scene.get("frame_index"),
                            "score": score,
                            "caption": meta.get("attribute_text"),
                        }
                    ],
                    "scene_frame_ts": scene.get("frame_ts"),
                    "scene_gap_seconds": gap,
                }
                by_clip[clip_path] = entry
            else:
                entry["person_hits"].append(person_hit)
                entry.setdefault("objects", []).append(
                    {
                        "object_name": "person",
                        "color": meta.get("color"),
                        "attribute_text": meta.get("attribute_text"),
                    }
                )
                if score > entry["score_raw"]:
                    entry["score_raw"] = score
                    entry["timestamp"] = meta.get("timestamp")
                    entry["frames"][0] = {
                        "frame_ts": meta.get("timestamp"),
                        "frame_index": scene.get("frame_index"),
                        "score": score,
                        "caption": meta.get("attribute_text"),
                    }

        results = list(by_clip.values())
        for entry in results:
            entry["score"] = float(entry["score_raw"])
            entry["frames"] = sorted(
                entry.get("frames", []),
                key=lambda f: float(f.get("score") or 0.0),
                reverse=True,
            )[:5]
        norms = _norm_scores([float(r["score_raw"]) for r in results])
        for entry, norm in zip(results, norms):
            entry["score_norm"] = norm
            if use_mongo:
                det = self._detection_summary(
                    (entry.get("person_hits") or [{}])[0].get("detection_id")
                )
                if det:
                    entry["detection"] = det
                    if det.get("objects"):
                        entry["objects"] = det["objects"]
        results.sort(key=lambda r: float(r.get("score_raw") or 0.0), reverse=True)
        if not getattr(settings, "INCLUDE_GENDER_ATTRIBUTES", True):
            from backend.app.core.privacy import strip_gender_from_results

            results = strip_gender_from_results(results)
        return results

    # ------------------------------------------------------------------
    # Legacy Mongo-backed search
    # ------------------------------------------------------------------
    def _apply_mongo_filters(self, cand: List[Dict[str, Any]], parsed: ParsedQuery, top_k: int) -> List[Dict[str, Any]]:
        from bson import ObjectId  # type: ignore
        from backend.app.db.mongo import detections as detections_col

        out: List[Dict[str, Any]] = []
        for c in cand:
            meta = c.get("meta", {})
            det_id = meta.get("detection_id")
            if not det_id:
                continue

            # Fetch the detection document by ObjectId
            try:
                doc = detections_col.find_one({"_id": ObjectId(det_id)})
            except Exception:
                doc = None
            if doc is None:
                continue

            # Color filter at object level
            if parsed.colors:
                try:
                    obj_idx = int(meta.get("object_index", -1))
                    if 0 <= obj_idx < len(doc.get("objects", [])):
                        ob = doc["objects"][obj_idx]
                        col = (ob.get("color") or "").strip()
                        if col and col not in parsed.colors:
                            continue
                except Exception:
                    pass

            doc2 = dict(doc)
            doc2["_id"] = str(doc2.get("_id")) if doc2.get("_id") is not None else None
            doc2["_similarity"] = float(c.get("score") or 0.0)
            doc2["_person_meta"] = meta
            doc2["source"] = "person"
            out.append(doc2)

        out.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)
        return out[:top_k]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        parsed = parse_for_hybrid(query)
        qv = self._encode_query(parsed.embedding_text)
        preds = self.person_store.vector_search(qv, top_k=max(1, int(top_k)) * 3, filter_pred=self._build_filter_pred(parsed))
        results = self._apply_mongo_filters(preds, parsed, top_k=max(1, int(top_k)))
        return results
