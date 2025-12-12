from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.config import settings

# FAISS import (faiss-cpu)
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore


class PersonFaissStore:
    """
    Person-level FAISS + Mongo-independent meta store.
    - FAISS holds L2-normalized vectors (IndexFlatIP with inner-product == cosine similarity).
    - Meta is persisted to meta.json alongside the index for fast lookup.
    - Meta fields include detection reference and per-object index within detection doc.
    Directory: settings.FAISS_PERSON_DIR
    """

    def __init__(self, dim: int = 1152) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not available. Ensure faiss-cpu is installed.")
        self.dim = int(dim)
        self.dir: Path = settings.FAISS_PERSON_DIR
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
                if getattr(self._index, "d", None) != self.dim:
                    # Rebuild new if dim mismatch
                    self._index = faiss.IndexFlatIP(self.dim)
                    self._meta = []
            except Exception:
                self._index = faiss.IndexFlatIP(self.dim)
                self._meta = []
        else:
            self._index = faiss.IndexFlatIP(self.dim)

        # Tune threads
        try:
            faiss.omp_set_num_threads(os.cpu_count() or 4)
        except Exception:
            pass

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
        Add person vectors (NxD float32, L2-normalized) and metadata (same order).
        Meta fields expected:
          - detection_id: str (Mongo detections _id) or None
          - object_index: int (index within detection.objects)
          - camera_id: int
          - track_id: int
          - timestamp: str (ISO)
          - color: Optional[str]
          - attribute_text: Optional[str]
        Returns FAISS ids.
        """
        if embeddings.size == 0 or not metas:
            return []
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        embeddings = np.ascontiguousarray(embeddings)
        start_id = self.count()
        self._index.add(embeddings)  # type: ignore

        ids: List[int] = []
        for i, m in enumerate(metas):
            row_id = start_id + i
            ids.append(row_id)
            # Store minimal, query-relevant meta
            self._meta.append(
                {
                    "faiss_id": row_id,
                    "detection_id": m.get("detection_id"),
                    "object_index": int(m.get("object_index", -1)),
                    "camera_id": int(m.get("camera_id", -1)),
                    "track_id": int(m.get("track_id", -1)),
                    "timestamp": m.get("timestamp"),
                    "color": m.get("color"),
                    "attribute_text": m.get("attribute_text"),
                }
            )

        if save:
            self.save()
        return ids

    def vector_search(self, query_vec: np.ndarray, top_k: int = 50, filter_pred: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        query_vec: (D,) or (1,D) float32 normalized.
        filter_pred: optional callable(meta)->bool to filter results based on meta.
        Returns list of {score: float, meta: {...}}.
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
_STORE: Optional[PersonFaissStore] = None


def get_person_store(dim: int = 1152) -> PersonFaissStore:
    global _STORE
    if _STORE is None:
        _STORE = PersonFaissStore(dim=dim)
    return _STORE
