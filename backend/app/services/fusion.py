from __future__ import annotations

import math

import numpy as np


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def fuse_gated(
    visual_emb: np.ndarray,
    text_emb: np.ndarray,
    visual_quality: float,
    text_quality: float,
    lam: float = 4.0,
    base_visual_weight: float = 0.6,
    base_text_weight: float = 0.4,
) -> np.ndarray:
    """Quality-gated heterogeneous concat fusion.

    The gate ``gw = sigmoid(lam * (qv - qt))`` (clipped to [0.05, 0.95])
    scales each branch around the static (0.6, 0.4) operating point:
    ``wv = 1.2 * gw``, ``wt = 0.8 * (1 - gw)``.  When qualities tie,
    ``gw = 0.5`` and the weights reduce exactly to the static (0.6, 0.4)
    point.  Output is a 1152-d L2-normalized float32 vector, identical in
    format to :meth:`MultimodalFusion.fuse`.
    """
    v = np.asarray(visual_emb, dtype=np.float32).reshape(-1)
    t = np.asarray(text_emb, dtype=np.float32).reshape(-1)
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
    qv = max(0.0, min(1.0, float(visual_quality)))
    qt = max(0.0, min(1.0, float(text_quality)))
    gw = _sigmoid(float(lam) * (qv - qt))
    gw = max(0.05, min(0.95, gw))
    # Tie (gw=0.5) reproduces the static weights exactly; the gate
    # interpolates each branch between 0.1x and ~1.6x of its static mass.
    wv = (base_visual_weight * 2.0) * gw
    wt = (base_text_weight * 2.0) * (1.0 - gw)
    v = (v * wv).astype(np.float32)
    t = (t * wt).astype(np.float32)
    fused = np.concatenate([v, t]).astype(np.float32)
    norm = float(np.linalg.norm(fused)) + 1e-6
    return (fused / norm).astype(np.float32)


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

    def fuse_gated(
        self,
        visual_emb: np.ndarray,
        text_emb: np.ndarray,
        visual_quality: float,
        text_quality: float,
        lam: float = 4.0,
    ) -> np.ndarray:
        """Per-item quality-gated fusion; ties reduce to this instance's weights."""
        return fuse_gated(
            visual_emb, text_emb, visual_quality, text_quality, lam=lam,
            base_visual_weight=self.visual_weight,
            base_text_weight=self.text_weight,
        )
