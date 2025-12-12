import sys
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.config import settings
from backend.app.services.sem_captioner import get_captioner


def load_image(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im)


def main():
    # Choose a snapshot as a test image (adjust if needed)
    # Fallbacks across a few known snapshot names
    candidates = list(Path("data/snapshots/camera_0").glob("*.jpg"))
    if not candidates:
        print("No snapshots found under data/snapshots/camera_0")
        sys.exit(1)

    img_path = candidates[-1]
    print("Testing captioner on:", img_path)

    try:
        captioner = get_captioner()
        arr = load_image(img_path)
        caps = captioner.caption_images_batched([arr])
        print("Captions:", caps)
    except Exception as e:
        import traceback
        print("Captioner error:", repr(e))
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
