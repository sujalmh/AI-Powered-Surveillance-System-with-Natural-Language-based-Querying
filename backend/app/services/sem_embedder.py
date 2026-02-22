from __future__ import annotations

import functools
from typing import List, Optional, Tuple

import numpy as np
import torch
import open_clip
from PIL import Image

from backend.app.config import settings


def _parse_openclip_model(model_str: str) -> Tuple[str, Optional[str]]:
    """
    Parse SEMANTIC_MODEL like "ViT-B-32/laion2b_s34b_b79k" into (model_name, pretrained_tag).
    If no slash present, returns (model_str, None) and will fall back to open_clip default.
    """
    if "/" in model_str:
        name, pretrained = model_str.split("/", 1)
        return name.strip(), pretrained.strip()
    return model_str.strip(), None


class _Embedder:
    """
    Singleton CLIP embedder wrapper (OpenCLIP) with GPU support and batching.
    Provides L2-normalized embeddings for cosine similarity via inner-product in FAISS.
    """

    def __init__(self) -> None:
        self.device = torch.device(settings.EMBED_DEVICE if torch.cuda.is_available() and settings.EMBED_DEVICE == "cuda" else "cpu")
        model_name, pretrained = _parse_openclip_model(settings.SEMANTIC_MODEL or "ViT-B-32/laion2b_s34b_b79k")

        if pretrained:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.batch_size = int(settings.EMBED_BATCH_SIZE or 32)

    @torch.no_grad()
    def image_embed(self, images: List[np.ndarray]) -> np.ndarray:
        """
        images: list of numpy arrays in HxWxC (BGR or RGB). We'll convert to RGB PIL.
        returns: float32 np array (N, D) L2-normalized
        """
        if not images:
            return np.zeros((0, 512), dtype=np.float32)

        def to_pil_rgb(x: np.ndarray) -> Image.Image:
            if x.ndim == 2:
                # grayscale -> RGB
                x = np.stack([x, x, x], axis=-1)
            # if BGR (common from cv2), swap to RGB by heuristic: assume cv2 produced BGR if last dim is 3
            if x.shape[-1] == 3 and x.flags['C_CONTIGUOUS']:
                # we cannot always know if it's BGR; typical pipeline feeds cv2 frames (BGR)
                # allow callers to pass RGB too; converting BGR->RGB twice is harmless visually here
                x = x[..., ::-1]
            return Image.fromarray(x)

        outs: List[torch.Tensor] = []
        idx = 0
        while idx < len(images):
            batch = images[idx: idx + self.batch_size]
            pil_batch = [self.preprocess(to_pil_rgb(arr)).unsqueeze(0) for arr in batch]
            if not pil_batch:
                break
            x = torch.cat(pil_batch, dim=0).to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                feats = self.model.encode_image(x)
            # L2 normalize
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            outs.append(feats.float().cpu())
            idx += self.batch_size
        emb = torch.cat(outs, dim=0).numpy().astype(np.float32)
        return emb

    @torch.no_grad()
    def text_embed(self, texts: List[str]) -> np.ndarray:
        """
        returns: float32 np array (N, D) L2-normalized
        """
        if not texts:
            return np.zeros((0, 512), dtype=np.float32)

        outs: List[torch.Tensor] = []
        idx = 0
        while idx < len(texts):
            batch = texts[idx: idx + self.batch_size]
            tokens = self.tokenizer(batch).to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            outs.append(feats.float().cpu())
            idx += self.batch_size
        emb = torch.cat(outs, dim=0).numpy().astype(np.float32)
        return emb


# Singleton accessor
@functools.lru_cache(maxsize=1)
def get_embedder() -> _Embedder:
    return _Embedder()
