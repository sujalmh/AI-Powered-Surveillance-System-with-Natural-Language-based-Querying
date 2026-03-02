from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import os

# Avoid importing TensorFlow/Keras via Transformers when not needed
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# Defer SentenceTransformer import to runtime to prevent heavy optional deps during test collection
SentenceTransformer = None  # type: ignore
try:
    from sentence_transformers import SentenceTransformer as _ST  # type: ignore
    SentenceTransformer = _ST
except Exception:
    SentenceTransformer = None  # type: ignore


def _join_color_names(colors: List[str], limit: int = 2) -> str:
    """Join up to `limit` unique color names with 'and', lowercased."""
    seen: List[str] = []
    for c in colors:
        cl = str(c).strip().lower()
        if cl and cl != "unknown" and cl not in seen:
            seen.append(cl)
        if len(seen) >= limit:
            break
    return " and ".join(seen) if seen else ""


class AttributeEncoder:
    """
    Convert structured attributes (OpenVINO + multi-color data) into natural language and encode to text embeddings.
    Default model: all-MiniLM-L12-v2 (384-dim). Returns L2-normalized float32 vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L12-v2") -> None:
        global SentenceTransformer
        if SentenceTransformer is None:
            # Ensure TF backend is disabled before importing transformers submodules
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            from sentence_transformers import SentenceTransformer as _ST  # type: ignore
            SentenceTransformer = _ST
        self.model = SentenceTransformer(model_name)  # type: ignore

    def attributes_to_text(
        self,
        openvino_attrs: Dict[str, float],
        color_name: Optional[str],
        color_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build concise textual description for a person based on attributes and color.

        When ``color_info`` is provided (dict with upper_body_colors / lower_body_colors),
        produces richer text like:
            "person wearing red and white top, blue and black pants, carrying bag"
        Falls back to the single ``color_name`` for backward compatibility.
        """
        parts: List[str] = ["person"]

        # --- Rich upper/lower body color description ---
        upper_desc = ""
        lower_desc = ""
        if color_info and isinstance(color_info, dict):
            upper_colors = color_info.get("upper_body_colors") or []
            lower_colors = color_info.get("lower_body_colors") or []
            upper_desc = _join_color_names(upper_colors)
            lower_desc = _join_color_names(lower_colors)

        if upper_desc and lower_desc:
            parts.append(f"wearing {upper_desc} top")
            parts.append(f"{lower_desc} pants")
        elif upper_desc:
            parts.append(f"wearing {upper_desc} top")
        elif lower_desc:
            parts.append(f"wearing {lower_desc} pants")
        elif color_name and str(color_name).lower() not in ("", "unknown"):
            # Fallback to single overall color (backward compat with old data)
            parts.append(f"wearing {color_name.lower()} clothing")

        # Accessories and clothing (threshold at 0.5)
        if openvino_attrs:
            if openvino_attrs.get("bag_confidence", 0.0) > 0.5:
                parts.append("carrying bag")
            if openvino_attrs.get("hat_confidence", 0.0) > 0.5:
                parts.append("wearing hat")
            if openvino_attrs.get("longsleeves_confidence", 0.0) > 0.5:
                parts.append("long sleeves")
            if openvino_attrs.get("longpants_confidence", 0.0) > 0.5:
                parts.append("long pants")
            if openvino_attrs.get("coat_jacket_confidence", 0.0) > 0.5:
                parts.append("coat or jacket")

            # Gender (soft)
            male = openvino_attrs.get("male_confidence", 0.0)
            if male >= 0.65:
                parts.append("male")
            elif male <= 0.35:
                parts.append("female")

        return ", ".join(parts)

    def object_to_caption(self, obj: dict) -> str:
        """
        Convert a stored detection object dict into a human-readable caption string.
        Single source of truth — replaces inline attribute-to-text logic scattered
        across sem_indexer.py and videos.py.

        For non-person objects returns the object name (with colors if available).
        For persons, builds a rich description using upper/lower body colors + OpenVINO attributes.

        Example output: "person wearing red and white top, blue pants, carrying bag, long sleeves"
        """
        name = str(obj.get("object_name", "")).strip().lower()
        if name != "person":
            # Include primary color for non-person objects
            color = str(obj.get("color") or "").strip().lower()
            if color and color != "unknown":
                return f"{color} {name}" if name else "unknown"
            return name or "unknown"
        color = str(obj.get("color") or "").strip()
        color = color if color.lower() not in ("", "unknown") else None
        attrs = obj.get("person_attributes") or {}
        # Build color_info from stored object fields for rich description
        color_info: Dict[str, Any] = {}
        if obj.get("upper_body_colors"):
            color_info["upper_body_colors"] = obj["upper_body_colors"]
        if obj.get("lower_body_colors"):
            color_info["lower_body_colors"] = obj["lower_body_colors"]
        if obj.get("colors"):
            color_info["colors"] = obj["colors"]
        return self.attributes_to_text(attrs, color, color_info if color_info else None)

    def encode_text(self, text: str) -> np.ndarray:
        emb = self.model.encode([text])
        v = emb[0].astype(np.float32)
        # L2 normalize
        n = np.linalg.norm(v) + 1e-6
        return (v / n).astype(np.float32)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Batch encode texts; returns (N, D) float32 L2-normalized."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        emb = self.model.encode(texts)
        v = np.asarray(emb, dtype=np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
        return (v / n).astype(np.float32)


# Singleton accessor
_ENCODER: Optional[AttributeEncoder] = None


def get_attribute_encoder() -> AttributeEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = AttributeEncoder()
    return _ENCODER
