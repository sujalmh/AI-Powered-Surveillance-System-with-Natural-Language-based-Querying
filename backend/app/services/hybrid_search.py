from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymongo.collection import Collection

from backend.app.db.mongo import detections as detections_col, db as _db
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
        out: List[Dict[str, Any]] = []
        for c in cand:
            meta = c.get("meta", {})
            det_id = meta.get("detection_id")
            try:
                doc = _db["detections"].find_one({"_id": {"$eq": _db.codec_options.uuid_representation and None or None, "$oid": det_id}})  # not used; fallback below
            except Exception:
                doc = None
            if doc is None and det_id:
                # Try standard ObjectId conversion
                try:
                    from bson import ObjectId  # type: ignore
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
            doc2["_similarity"] = float(c.get("score") or 0.0)
            doc2["_person_meta"] = meta
            out.append(doc2)

        # Sort by similarity desc
        out.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)
        return out[:top_k]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        parsed = parse_for_hybrid(query)
        qv = self._encode_query(parsed.embedding_text)
        preds = self.person_store.vector_search(qv, top_k=max(1, int(top_k)) * 3, filter_pred=self._build_filter_pred(parsed))
        results = self._apply_mongo_filters(preds, parsed, top_k=max(1, int(top_k)))
        return results
