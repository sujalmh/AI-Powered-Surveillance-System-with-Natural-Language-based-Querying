from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.config import settings
from backend.app.db.mongo import db, vlm_frames

# FAISS import (faiss-cpu)
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore


class FaissStore:
    """
    FAISS + Mongo metadata store.
    - FAISS holds L2-normalized vectors (IndexFlatIP with inner-product == cosine similarity).
    - Mongo holds frame-level metadata (camera_id, clip_path, frame_ts, frame_index, model, hash).
    - A local meta map (faiss_id -> mongo_id & quick fields) is persisted to meta.json for fast lookup.
    """

    def __init__(self, dim: int = 512) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not available. Ensure faiss-cpu is installed.")
        self.dim = dim
        self.dir: Path = settings.FAISS_DIR
        self.index_path: Path = self.dir / "index.bin"
        self.meta_path: Path = self.dir / "meta.json"
        self._index = None  # type: ignore
        self._meta: List[Dict[str, Any]] = []
        self._load_or_init()

    def _load_or_init(self) -> None:
        # Load meta
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                self._meta = []
        else:
            self._meta = []

        # Load index
        if self.index_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                # Basic sanity: dimension must match
                if self._index.d != self.dim:
                    # Rebuild new
                    self._index = faiss.IndexFlatIP(self.dim)
                    self._meta = []
            except Exception:
                self._index = faiss.IndexFlatIP(self.dim)
                self._meta = []
        else:
            self._index = faiss.IndexFlatIP(self.dim)

        # Set number of threads (optional tuning)
        try:
            faiss.omp_set_num_threads(os.cpu_count() or 4)
        except Exception:
            pass

    # Persist atomically
    def _atomic_write(self, path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def save(self) -> None:
        # Persist index
        try:
            faiss.write_index(self._index, str(self.index_path))
        except Exception as e:
            raise RuntimeError(f"Failed to save FAISS index: {e}")
        # Persist meta
        try:
            data = json.dumps(self._meta, ensure_ascii=False).encode("utf-8")
            self._atomic_write(self.meta_path, data)
        except Exception as e:
            raise RuntimeError(f"Failed to save FAISS meta: {e}")

    def count(self) -> int:
        try:
            return self._index.ntotal  # type: ignore
        except Exception:
            return 0

    def vector_add(self, embeddings: np.ndarray, metas: List[Dict[str, Any]], save: bool = True) -> List[int]:
        """
        Add vectors (NxD float32, L2-normalized) and metadata in the same order.
        Returns list of FAISS ids assigned.
        """
        if embeddings.size == 0 or len(metas) == 0:
            return []
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        # Ensure contiguous C-order
        embeddings = np.ascontiguousarray(embeddings)
        start_id = self.count()
        # Add to index
        self._index.add(embeddings)  # type: ignore

        # Upsert Mongo docs and record faiss meta map
        faiss_ids: List[int] = []
        for i, m in enumerate(metas):
            # Upsert by clip_path + frame_index uniqueness (enforced by index)
            filter_q = {"clip_path": m["clip_path"], "frame_index": int(m["frame_index"])}
            update_doc = {
                "$setOnInsert": {"created_at": m.get("created_at")},
                "$set": {
                    "camera_id": int(m["camera_id"]),
                    "clip_url": m.get("clip_url"),
                    "frame_ts": m.get("frame_ts"),
                    "model": m.get("model"),
                    "embedding_dim": int(m.get("embedding_dim", embeddings.shape[1])),
                    "hash": m.get("hash"),
                    "caption": m.get("caption"),
                    "object_captions": m.get("object_captions"),
                    "updated_at": m.get("updated_at"),
                },
            }
            res = vlm_frames.update_one(filter_q, update_doc, upsert=True)
            if res.upserted_id is not None:
                mongo_id = res.upserted_id
            else:
                # Fetch _id of existing doc
                doc = vlm_frames.find_one(filter_q, {"_id": 1})
                mongo_id = doc["_id"] if doc else None

            row_id = start_id + i
            faiss_ids.append(row_id)
            self._meta.append({
                "faiss_id": row_id,
                "mongo_id": str(mongo_id) if mongo_id is not None else None,
                "camera_id": int(m["camera_id"]),
                "clip_path": m["clip_path"],
                "clip_url": m.get("clip_url"),
                "frame_ts": m.get("frame_ts"),
                "frame_index": int(m["frame_index"]),
                "model": m.get("model"),
                "caption": m.get("caption"),
                "object_captions": m.get("object_captions"),
            })

        if save:
            self.save()
        return faiss_ids

    def vector_search(self, query_vec: np.ndarray, top_k: int = 50, filter_pred: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        query_vec: shape (D,) or (1,D) float32 normalized
        filter_pred: optional callable(meta_dict)->bool to filter meta after search (e.g., camera_id filter)
        returns: [{score: float, meta: {...}}]
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        query_vec = np.ascontiguousarray(query_vec)

        top_k = max(1, int(top_k))
        D, I = self._index.search(query_vec, top_k)  # type: ignore
        scores = D[0].tolist() if len(D) > 0 else []
        ids = I[0].tolist() if len(I) > 0 else []

        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(self._meta):
                continue
            meta = self._meta[idx]
            if filter_pred and not filter_pred(meta):
                continue
            out.append({"score": float(score), "meta": meta})
        return out


# Global singleton
_store: Optional[FaissStore] = None


def get_faiss_store(dim: int = 512) -> FaissStore:
    global _store
    if _store is None:
        _store = FaissStore(dim=dim)
    return _store
