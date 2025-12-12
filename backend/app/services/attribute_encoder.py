from __future__ import annotations

from typing import Dict, List, Optional

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


class AttributeEncoder:
    """
    Convert structured attributes (OpenVINO + CIEDE2000 colors) into natural language and encode to text embeddings.
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

    def attributes_to_text(self, openvino_attrs: Dict[str, float], color_name: Optional[str]) -> str:
        """
        Build concise textual description for a person based on attributes and color.
        Example: "person wearing navy blue top, black pants, carrying bag"
        """
        parts: List[str] = ["person"]

        # Color (single color field available in current pipeline)
        if color_name and str(color_name).lower() not in ("", "unknown"):
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

    def encode_text(self, text: str) -> np.ndarray:
        emb = self.model.encode([text])
        v = emb[0].astype(np.float32)
        # L2 normalize
        n = np.linalg.norm(v) + 1e-6
        return (v / n).astype(np.float32)


# Singleton accessor
_ENCODER: Optional[AttributeEncoder] = None


def get_attribute_encoder() -> AttributeEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = AttributeEncoder()
    return _ENCODER
