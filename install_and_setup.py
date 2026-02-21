import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    logger.info("Installing requirements from backend/requirements.txt...")
    req_file = os.path.join("backend", "requirements.txt")
    if not os.path.exists(req_file):
        logger.error(f"Requirements file not found at {req_file}")
        return

    # Try reading with different encodings to handle potential issues
    content = ""
    try:
        with open(req_file, "r", encoding="utf-16") as f:
            content = f.read()
    except Exception:
        try:
            with open(req_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read requirements file: {e}")
            return

    # Write to a temp file with utf-8 to be sure pip can read it if needed, 
    # but pip install -r usually works with files. 
    # However, since we had issues reading it, let's just use pip install -r directly 
    # and hope pip handles the encoding, or we manually install the packages.
    # Actually, simpler to just run pip install -r backend/requirements.txt
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        logger.info("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        # If utf-16 failed, maybe try converting it?
        # But let's assume pip handles it or the previous error was just my display tool.

def download_openvino_model():
    logger.info("Checking OpenVINO model...")
    # Add project root to sys.path to allow imports
    sys.path.append(os.getcwd())
    
    try:
        from backend.app.services.model_loader import ensure_model_downloaded, MODEL_NAME
        # The constant OPENVINO_MODEL_XML is defined in object_detection.py, let's hardcode or import it
        # In object_detection.py: 
        # "intel/person-attributes-recognition-crossroad-0238/FP32/person-attributes-recognition-crossroad-0238.xml"
        
        xml_path = f"intel/{MODEL_NAME}/FP32/{MODEL_NAME}.xml"
        ensure_model_downloaded(xml_path)
        logger.info("OpenVINO model check complete.")
    except Exception as e:
        logger.error(f"Failed to download OpenVINO model: {e}")

def download_yolo_model():
    logger.info("Checking YOLO model...")
    try:
        from ultralytics import YOLO
        model_name = "yolo11m-seg.pt"
        if not os.path.exists(model_name):
            logger.info(f"Downloading {model_name}...")
        model = YOLO(model_name)
        logger.info(f"YOLO model {model_name} loaded/downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download YOLO model: {e}")

def download_botsort_model():
    logger.info("Checking BoTSORT ReID model...")
    try:
        import boxmot
        import torch
        BoTSORT = getattr(boxmot, "BoTSORT", getattr(boxmot, "BotSort", None))
        if BoTSORT is None:
            logger.warning("BoTSORT not found in boxmot module.")
            return

        reid_weights = "osnet_x0_25_msmt17.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing BoTSORT to trigger download of {reid_weights} if needed...")
        # This triggers download if weights are missing
        tracker = BoTSORT(
            reid_weights=reid_weights,
            device=device,
            half=(device == "cuda"),
        )
        logger.info("BoTSORT model check complete.")
    except Exception as e:
        logger.error(f"Failed to download BoTSORT model: {e}")

if __name__ == "__main__":
    # install_requirements()
    download_yolo_model()
    download_openvino_model()
    download_botsort_model()
    logger.info("Setup script finished.")
