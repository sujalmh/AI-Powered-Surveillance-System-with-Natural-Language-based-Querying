from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root on sys.path so "backend" package resolves when running this file directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure Transformers avoids importing TensorFlow/Keras backend
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# Import modules that previously triggered tf_keras issues
import backend.app.services.attribute_encoder as attribute_encoder  # noqa: F401
import backend.app.services.visual_embedder as visual_embedder      # noqa: F401

print("imports OK")
