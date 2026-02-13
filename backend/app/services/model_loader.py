import os
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Official OpenVINO storage URL for the model (2023.0 release)
MODEL_BASE_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"
MODEL_NAME = "person-attributes-recognition-crossroad-0238"
MODEL_SUBPATH = f"{MODEL_NAME}/FP32"

def ensure_model_downloaded(model_xml_path: str) -> None:
    """
    Ensure the OpenVINO model XML and BIN files exist locally.
    If not, download them from the official Open Model Zoo repository.

    Args:
        model_xml_path: Local path where the .xml file is expected to be.
                        We infer the .bin path by replacing extension.
    """
    xml_path = Path(model_xml_path)
    bin_path = xml_path.with_suffix(".bin")

    # If both files exist, no action needed
    if xml_path.exists() and bin_path.exists():
        return

    # Create parent directory if needed
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"OpenVINO model not found at {xml_path}. Downloading...")
    print(f"⬇️ OpenVINO model missing. Downloading from Open Model Zoo...")

    try:
        # Download XML
        xml_url = f"{MODEL_BASE_URL}/{MODEL_SUBPATH}/{xml_path.name}"
        _download_file(xml_url, xml_path)

        # Download BIN
        bin_url = f"{MODEL_BASE_URL}/{MODEL_SUBPATH}/{bin_path.name}"
        _download_file(bin_url, bin_path)

        print(f"✅ Download complete: {xml_path.name}")
        logger.info(f"Successfully downloaded OpenVINO model to {xml_path.parent}")

    except Exception as e:
        logger.error(f"Failed to download OpenVINO model: {e}")
        print(f"❌ Failed to download model: {e}")
        # Clean up partial downloads
        if xml_path.exists(): xml_path.unlink()
        if bin_path.exists(): bin_path.unlink()
        raise

def _download_file(url: str, dest_path: Path) -> None:
    """Helper to download a file with progress indication."""
    print(f"   Getting {dest_path.name}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192): 
            if chunk:
                f.write(chunk)
