from __future__ import annotations

import numpy as np


class MultimodalFusion:
    """Weighted concatenation of visual (SigLIP 768) and text (MiniLM 384) embeddings.
    Returns a 1152-dim L2-normalized float32 vector.
    """

    def __init__(self, visual_weight: float = 0.6, text_weight: float = 0.4) -> None:
        self.visual_weight = float(visual_weight)
        self.text_weight = float(text_weight)

    def fuse(self, visual_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        """Fuse visual and text embeddings via weighted concatenation.

        Args:
            visual_emb: shape (768,) float32
            text_emb: shape (384,) float32

        Returns:
            np.ndarray: shape (1152,) float32 L2-normalized
        """
        v = np.asarray(visual_emb, dtype=np.float32).reshape(-1)
        t = np.asarray(text_emb, dtype=np.float32).reshape(-1)

        # Defensive shape correction
        if v.shape[0] != 768:
            v2 = np.zeros((768,), dtype=np.float32)
            n = min(768, v.shape[0])
            if n > 0:
                v2[:n] = v[:n]
            v = v2
        if t.shape[0] != 384:
            t2 = np.zeros((384,), dtype=np.float32)
            n = min(384, t.shape[0])
            if n > 0:
                t2[:n] = t[:n]
            t = t2

        v *= self.visual_weight
        t *= self.text_weight
        fused = np.concatenate([v, t]).astype(np.float32)

        # L2 normalize
        norm = float(np.linalg.norm(fused)) + 1e-6
        fused = (fused / norm).astype(np.float32)
        return fused
