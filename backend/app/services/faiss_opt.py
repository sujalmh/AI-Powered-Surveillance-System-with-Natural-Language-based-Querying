from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# FAISS import (faiss-cpu)
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore


def create_optimized_index(embedding_dim: int, nlist: int = 100, use_gpu: bool = False) -> "faiss.Index":  # type: ignore[name-defined]
    """
    Create an IVF-Flat index optimized for cosine-similarity search via inner product.
    """
    if faiss is None:
        raise RuntimeError("faiss is not available. Ensure faiss-cpu is installed.")
    quantizer = faiss.IndexFlatIP(embedding_dim)  # inner product (cosine if vectors are L2-normalized)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
    # Optionally move to GPU
    if use_gpu:
        try:
            # Only proceed if GPU APIs are available (faiss-gpu); faiss-cpu won't have these symbols.
            if hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
                res = faiss.StandardGpuResources()  # type: ignore[attr-defined]
                index = faiss.index_cpu_to_gpu(res, 0, index)  # type: ignore[attr-defined]
        except Exception:
            # Fallback to CPU if GPU not available
            pass
    return index


def train_ivf(index_ivf: "faiss.IndexIVFFlat", sample_vectors: np.ndarray) -> None:  # type: ignore[name-defined]
    """
    Train IVF index with a sample of vectors (NxD float32). Vectors must be L2-normalized.
    """
    if sample_vectors.dtype != np.float32:
        sample_vectors = sample_vectors.astype(np.float32)
    sample_vectors = np.ascontiguousarray(sample_vectors)
    index_ivf.train(sample_vectors)  # type: ignore[attr-defined]


def migrate_flat_to_ivf(flat_index: "faiss.IndexFlatIP", nlist: int = 100, use_gpu: bool = False) -> "faiss.IndexIVFFlat":  # type: ignore[name-defined]
    """
    Migrate a Flat IP index to IVF-Flat. Reconstructs all vectors from the flat index if supported.
    """
    if faiss is None:
        raise RuntimeError("faiss is not available. Ensure faiss-cpu is installed.")
    d = int(getattr(flat_index, "d"))
    ivf = create_optimized_index(d, nlist=nlist, use_gpu=use_gpu)
    # Gather all vectors from flat index
    ntotal = int(getattr(flat_index, "ntotal"))
    # Try reconstruct per id (works on Flat in modern faiss)
    vecs = []
    for i in range(ntotal):
        try:
            v = faiss.vector_to_array(flat_index.reconstruct(i))  # type: ignore[attr-defined]
            vecs.append(np.asarray(v, dtype=np.float32))
        except Exception:
            # Fallback: not supported; break
            vecs = []
            break
    if not vecs and ntotal > 0:
        # Fallback approach: sample centroids by searching random queries; not ideal.
        # In typical FlatIP, reconstruct is supported; if not, user should re-build from source vectors.
        raise RuntimeError("Failed to reconstruct vectors from Flat index. Rebuild IVF from source embeddings.")
    arr = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, d), dtype=np.float32)
    if arr.size > 0:
        # Train and add
        # Use a subset to train
        train_sample = arr if arr.shape[0] <= 10000 else arr[np.random.choice(arr.shape[0], 10000, replace=False)]
        train_ivf(ivf, train_sample)  # type: ignore[arg-type]
        ivf.add(arr)  # type: ignore[attr-defined]
    return ivf  # type: ignore[return-value]
