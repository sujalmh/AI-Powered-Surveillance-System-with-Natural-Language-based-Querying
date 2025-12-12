from __future__ import annotations

import functools
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from backend.app.config import settings
import os
from backend.app.services.sem_embedder import get_embedder

# Lazy import transformers so backend can start even if captions are disabled
try:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor, pipeline  # type: ignore
except Exception:  # pragma: no cover
    Blip2ForConditionalGeneration = None  # type: ignore
    Blip2Processor = None  # type: ignore
    pipeline = None  # type: ignore


def _to_pil_rgb(x: np.ndarray) -> Image.Image:
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] == 3 and x.flags["C_CONTIGUOUS"]:
        # Heuristic for cv2 BGR -> RGB; harmless if already RGB
        x = x[..., ::-1]
    return Image.fromarray(x)


class _Captioner:
    """
    BLIP-2 captioner for per-frame captions.
    Loads only if ENABLE_CAPTIONS=true. Uses CPU by default unless EMBED_DEVICE=cuda and torch has CUDA.
    """

    def __init__(self) -> None:
        if not settings.ENABLE_SEMANTIC or not settings.ENABLE_CAPTIONS:
            raise RuntimeError("Captions are disabled by config.")

        if Blip2ForConditionalGeneration is None or Blip2Processor is None:
            raise RuntimeError("transformers not installed. Please install transformers/accelerate/sentencepiece.")

        self.device = torch.device(
            "cuda" if (settings.EMBED_DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        model_id = settings.CAPTION_MODEL or "Salesforce/blip2-flan-t5-base"
        # On CPU, avoid fp16
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Fast path: allow forcing lightweight CLIP label guesses without HF downloads
        if str(settings.CAPTION_MODEL).lower() == "clip_labels":
            self.backend = "clip_labels"
            self.labels = [
                "person", "backpack", "bag", "crowd", "vehicle", "car", "bike", "motorcycle",
                "helmet", "police", "security", "running", "fighting", "smoke", "fire",
                "knife", "gun", "doorway", "entrance", "staircase", "corridor"
            ]
            prompts = [f"a photo of {lbl}" for lbl in self.labels]
            try:
                self._label_text_emb = get_embedder().text_embed(prompts)  # (L, D) normalized
            except Exception:
                self._label_text_emb = None
            # Set reasonable defaults and skip HF model loading
            self.max_new_tokens = 0
            self.batch_size = int(getattr(settings, "EMBED_BATCH_SIZE", 16) or 16)
            return

        # Hugging Face auth / cache
        hf_token = settings.HF_TOKEN or os.getenv("HF_TOKEN") or None
        cache_dir = settings.HF_HOME or os.getenv("HF_HOME") or None

        self.processor = None
        self.model = None
        self.pipe = None  # fallback pipeline

        # Try BLIP-2 first
        try:
            self.processor = Blip2Processor.from_pretrained(
                model_id,
                token=hf_token,
                cache_dir=cache_dir,
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                token=hf_token,
                cache_dir=cache_dir,
            ).to(self.device)
            self.model.eval()
            self.backend = "blip2"
        except Exception:
            # Fallback to a public captioner with image-to-text pipeline to avoid blocking on BLIP-2
            # Choices: "Salesforce/blip-image-captioning-base" (BLIP) or "nlpconnect/vit-gpt2-image-captioning"
            fallback_model = os.getenv("CAPTION_FALLBACK_MODEL", "nlpconnect/vit-gpt2-image-captioning")
            # Force PyTorch backend to avoid TF/Keras dependency issues
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            if pipeline is not None:
                try:
                    self.pipe = pipeline(
                        task="image-to-text",
                        model=fallback_model,
                        device=0 if self.device.type == "cuda" else -1,
                        token=hf_token,
                        framework="pt",
                    )
                    self.backend = "pipeline"
                except Exception:
                    self.pipe = None
            # Final fallback: CLIP label guesses (lightweight, no HF dependency)
            if self.pipe is None:
                self.backend = "clip_labels"
                # A compact surveillance-oriented label set
                self.labels = [
                    "person", "backpack", "bag", "crowd", "vehicle", "car", "bike", "motorcycle",
                    "helmet", "police", "security", "running", "fighting", "smoke", "fire",
                    "knife", "gun", "doorway", "entrance", "staircase", "corridor"
                ]
                prompts = [f"a photo of {lbl}" for lbl in self.labels]
                try:
                    self._label_text_emb = get_embedder().text_embed(prompts)  # (L, D) normalized
                except Exception:
                    # As an absolute last resort if embedder init fails
                    self._label_text_emb = None

        self.max_new_tokens = int(getattr(settings, "CAPTION_MAX_NEW_TOKENS", 50) or 50)
        # Reuse embedding batch size as a rough default
        self.batch_size = int(getattr(settings, "EMBED_BATCH_SIZE", 16) or 16)

    @torch.no_grad()
    def caption_images_batched(self, images: List[np.ndarray]) -> List[str]:
        """
        Generate captions for a batch of images using BLIP-2.
        For BLIP-2 FLAN-T5 checkpoints, provide a text prompt to steer captioning.
        """
        if not images:
            return []
        out: List[str] = []
        i = 0
        # A concise, generic caption prompt that works well with FLAN-T5-based BLIP-2
        prompt = "Question: describe the image in a short sentence. Answer:"
        while i < len(images):
            chunk = images[i : i + self.batch_size]
            pil_list = [_to_pil_rgb(arr) for arr in chunk]

            if self.backend == "blip2" and self.processor is not None and self.model is not None:
                # Pass a prompt per image to avoid broadcast issues on some processors
                texts = [prompt] * len(pil_list)
                inputs = self.processor(images=pil_list, text=texts, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
                captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            elif self.backend == "clip_labels" and getattr(self, "_label_text_emb", None) is not None:
                # Zero-shot label guessing with CLIP embeddings (fast, always available)
                try:
                    img_emb = get_embedder().image_embed([np.array(p) for p in pil_list])  # (N, D), normalized
                    # similarity (N, L)
                    sims = img_emb @ self._label_text_emb.T  # type: ignore[attr-defined]
                    top_idx = sims.argmax(axis=1).tolist()
                    captions = [self.labels[i] if 0 <= i < len(self.labels) else "" for i in top_idx]  # type: ignore[attr-defined]
                except Exception:
                    captions = ["" for _ in pil_list]
            elif self.pipe is not None:
                # Pipeline returns list of list[{'generated_text': str}]
                results = self.pipe(pil_list, max_new_tokens=self.max_new_tokens)
                captions = []
                for r in results:
                    if isinstance(r, list) and r and isinstance(r[0], dict):
                        captions.append(str(r[0].get("generated_text", "")).strip())
                    else:
                        captions.append("")
            else:
                captions = [""] * len(pil_list)

            # Normalize/strip and guard against empty strings
            captions = [(c.strip() if isinstance(c, str) else "") for c in captions]
            out.extend(captions)
            i += self.batch_size
        return out


@functools.lru_cache(maxsize=1)
def get_captioner() -> _Captioner:
    """
    Returns a singleton captioner instance. Will raise if captions are disabled or transformers not installed.
    """
    return _Captioner()
