from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymongo.collection import Collection

from backend.app.db.mongo import detections as detections_col
from backend.app.services.person_store import get_person_store
from backend.app.services.attribute_encoder import get_attribute_encoder
from backend.app.services.fusion import MultimodalFusion
from backend.app.services.nl_parser import parse_nl_with_llm


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


class HybridSearchEngine:
    """
    Person-level hybrid search:
    - Build fused query vector: visual=0 (no image), text = MiniLM(embedding_text)
    - Search person FAISS index for top_k*3 candidates
    - Apply Mongo filters (camera/time/color) by fetching detection docs
    - Re-score by similarity and return top_k documents with similarity
    """

    def __init__(self, person_top_k: int = 50) -> None:
        self.person_store = get_person_store(dim=1152)
        self.attr_encoder = get_attribute_encoder()
        self.fusion = MultimodalFusion()
        self.top_k = int(person_top_k)

    def _encode_query(self, text: str) -> np.ndarray:
        t = self.attr_encoder.encode_text(text)  # 384-dim normalized
        v = np.zeros((768,), dtype=np.float32)   # No visual query available
        f = self.fusion.fuse(v, t)               # 1152-dim
        return f.astype(np.float32)

    def _build_filter_pred(self, parsed: ParsedQuery):
        def _pred(meta: Dict[str, Any]) -> bool:
            if parsed.camera_id is not None and int(meta.get("camera_id", -1)) != int(parsed.camera_id):
                return False
            if parsed.time_filter:
                try:
                    ts = meta.get("timestamp")
                    if ts and (parsed.time_filter.get("$gte") or parsed.time_filter.get("$lte")):
                        from datetime import datetime
                        dt = datetime.fromisoformat(ts)
                        if parsed.time_filter.get("$gte") and dt < datetime.fromisoformat(parsed.time_filter["$gte"]):
                            return False
                        if parsed.time_filter.get("$lte") and dt > datetime.fromisoformat(parsed.time_filter["$lte"]):
                            return False
                except Exception:
                    pass
            return True
        return _pred

    def _apply_mongo_filters(self, cand: List[Dict[str, Any]], parsed: ParsedQuery, top_k: int) -> List[Dict[str, Any]]:
        from bson import ObjectId  # type: ignore

        # Batch-fetch all detection docs in a single $in query (avoids N+1)
        det_id_by_idx: Dict[int, str] = {}
        for i, c in enumerate(cand):
            det_id = (c.get("meta") or {}).get("detection_id")
            if det_id:
                det_id_by_idx[i] = det_id

        if not det_id_by_idx:
            return []

        unique_ids = list(set(det_id_by_idx.values()))
        try:
            oid_list = [ObjectId(d) for d in unique_ids]
            docs_cursor = detections_col.find({"_id": {"$in": oid_list}})
            docs_map: Dict[str, Dict[str, Any]] = {str(d["_id"]): d for d in docs_cursor}
        except Exception:
            docs_map = {}

        # Normalize colors for case-insensitive comparison
        parsed_colors_lower = [c.lower() for c in parsed.colors]

        out: List[Dict[str, Any]] = []
        for i, c in enumerate(cand):
            det_id = det_id_by_idx.get(i)
            if not det_id:
                continue
            doc = docs_map.get(det_id)
            if doc is None:
                continue
            meta = c.get("meta", {})

            # Color filter at object level (case-insensitive)
            # Check against the top-3 colors array, with fallback to legacy single color
            if parsed_colors_lower:
                try:
                    obj_idx = int(meta.get("object_index", -1))
                    if not (0 <= obj_idx < len(doc.get("objects", []))):
                        # Invalid/missing object index — treat as non-match when color filter is active
                        continue
                    ob = doc["objects"][obj_idx]
                    # Gather all color names: top-3 array + upper/lower body + legacy single
                    all_colors = set()
                    for col in (ob.get("colors") or []):
                        all_colors.add(str(col).strip().lower())
                    for col in (ob.get("upper_body_colors") or []):
                        all_colors.add(str(col).strip().lower())
                    for col in (ob.get("lower_body_colors") or []):
                        all_colors.add(str(col).strip().lower())
                    legacy = (ob.get("color") or "").strip().lower()
                    if legacy:
                        all_colors.add(legacy)
                    all_colors.discard("")
                    all_colors.discard("unknown")
                    if not all_colors or not all_colors.intersection(parsed_colors_lower):
                        # No color metadata at all, or no intersection — reject
                        continue
                except Exception as e:
                    # On unexpected errors, be conservative and reject
                    from loguru import logger
                    logger.debug("Failed color metadata parsing for doc _id={}: {}", doc.get("_id"), e, exc_info=True)
                    continue

            doc2 = dict(doc)
            doc2["_similarity"] = float(c.get("score") or 0.0)
            doc2["_person_meta"] = meta
            out.append(doc2)

        out.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)
        return out[:top_k]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        parsed = parse_for_hybrid(query)
        qv = self._encode_query(parsed.embedding_text)
        preds = self.person_store.vector_search(qv, top_k=max(1, int(top_k)) * 3, filter_pred=self._build_filter_pred(parsed))
        results = self._apply_mongo_filters(preds, parsed, top_k=max(1, int(top_k)))
        return results
