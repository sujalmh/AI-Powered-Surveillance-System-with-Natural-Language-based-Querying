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

        When the visual embedding is all-zeros (text-only query), the visual
        weight is redistributed to the text portion so that the resulting
        vector lives entirely in the text sub-space.  This prevents the 768
        zero dimensions from diluting cosine similarity against stored
        vectors that have real visual features.

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

        # Detect text-only query (visual is all zeros) and redistribute
        # weight entirely to text to avoid diluting the signal.
        visual_is_zero = float(np.linalg.norm(v)) < 1e-8
        if visual_is_zero:
            v_weight = 0.0
            t_weight = 1.0
        else:
            v_weight = self.visual_weight
            t_weight = self.text_weight

        v *= v_weight
        t *= t_weight
        fused = np.concatenate([v, t]).astype(np.float32)

        # L2 normalize
        norm = float(np.linalg.norm(fused)) + 1e-6
        fused = (fused / norm).astype(np.float32)
        return fused
