from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from backend.app.config import settings


class VisualEmbedder:
    """
    SigLIP image encoder for person ROI crops.

    - Model: google/siglip-base-patch16-224 (configurable via SIGLIP_MODEL)
    - Returns L2-normalized float32 embeddings of shape (N, D)
    """

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or getattr(settings, "SIGLIP_MODEL", "google/siglip-base-patch16-224")
        device_str = settings.EMBED_DEVICE if torch.cuda.is_available() and settings.EMBED_DEVICE == "cuda" else "cpu"
        self.device = torch.device(device_str)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    @torch.no_grad()
    def encode(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Args:
            images: list of ROI arrays in HxWxC (BGR or RGB)

        Returns:
            np.ndarray: (N, D) float32 L2-normalized
        """
        if not images:
            return np.zeros((0, 768), dtype=np.float32)

        def to_pil_rgb(x: np.ndarray) -> Image.Image:
            if x.ndim == 2:
                x = np.stack([x, x, x], axis=-1)
            if x.shape[-1] == 3 and x.flags["C_CONTIGUOUS"]:
                x = x[..., ::-1]  # BGR->RGB heuristic
            return Image.fromarray(x)

        pil_imgs = [to_pil_rgb(arr) for arr in images]
        inputs = self.processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats.float().cpu().numpy().astype(np.float32)


# Singleton accessor (lazy)
_VISUAL: Optional[VisualEmbedder] = None


def get_visual_embedder() -> VisualEmbedder:
    global _VISUAL
    if _VISUAL is None:
        _VISUAL = VisualEmbedder()
    return _VISUAL
