import cv2
from loguru import logger
import numpy as np
import datetime
import time
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import MiniBatchKMeans
Core = None  # lazy import in ensure_openvino_loaded
import math
import os
import threading
from typing import Union, Optional, Dict, List, Tuple, Any
from backend.app.config import settings
import torch
try:
    import boxmot  # type: ignore
    # Try BoTSORT (all caps) first, then falling back to BotSort (PascalCase)
    BoTSORT = getattr(boxmot, "BoTSORT", getattr(boxmot, "BotSort", None))
except Exception:  # pragma: no cover
    BoTSORT = None  # type: ignore
from backend.app.services.frame_store import update_frame
from backend.app.services.clip_builder import build_clip_from_snapshots
from backend.app.services.sem_indexer import index_clip
from backend.app.core.settings_reader import get_indexing_mode
from pymongo import UpdateOne
from backend.app.db.mongo import tracks, cameras as cameras_col, zones as zones_col, detections as detections_col
from backend.app.services.visual_embedder import get_visual_embedder
from backend.app.services.attribute_encoder import get_attribute_encoder
from backend.app.services.person_store import get_person_store
from backend.app.services.fusion import MultimodalFusion
from backend.app.services.alert_engine import evaluate_realtime
from backend.app.services.task_queue import submit as task_submit


def _get_zones_for_camera(zones_collection: Any, camera_id: int) -> List[Dict[str, Any]]:
    """Return list of zone docs for camera, with short TTL cache."""
    global _ZONES_CACHE
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    entry = _ZONES_CACHE.get(camera_id)
    if entry is not None and (now - entry[1]) < _ZONES_CACHE_TTL_SEC:
        return entry[0]
    try:
        zone_list = list(zones_collection.find({"camera_id": camera_id}))
        _ZONES_CACHE[camera_id] = (zone_list, now)
        return zone_list
    except Exception:
        return []


def _compute_zone_counts(
    objects: List[Dict[str, Any]],
    zones_list: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> Dict[str, int]:
    """
    Count persons per zone. Zone bbox is normalized [0,1].
    Person bbox in objects is pixel [x1,y1,x2,y2]. Use centroid and normalize.
    """
    zone_counts: Dict[str, int] = {}
    if not frame_width or not frame_height:
        return zone_counts
    for z in zones_list:
        zid = z.get("zone_id")
        if not zid:
            continue
        zone_counts[str(zid)] = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if str(obj.get("object_name", "")).strip().lower() != "person":
            continue
        bbox = obj.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        nx = cx / float(frame_width)
        ny = cy / float(frame_height)
        for z in zones_list:
            zid = z.get("zone_id")
            if not zid:
                continue
            b = z.get("bbox") or {}
            x_min = float(b.get("x_min", 0))
            y_min = float(b.get("y_min", 0))
            x_max = float(b.get("x_max", 1))
            y_max = float(b.get("y_max", 1))
            if x_min <= nx <= x_max and y_min <= ny <= y_max:
                zone_counts[str(zid)] = zone_counts.get(str(zid), 0) + 1
    return zone_counts


def to_numpy(x):
    if x is None:
        return None
    try:
        if hasattr(x, "cpu"):
            return x.cpu().numpy()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)
    except Exception:
        return np.asarray(x)

# =========================================
# CONFIGURATION (centralized via backend.app.config)
# =========================================
MODEL_PATH = settings.MODEL_PATH
OPENVINO_MODEL_XML = settings.OPENVINO_MODEL_XML
CONF_THRESHOLD = 0.5
NUM_CLUSTERS = 5
MIN_PERSON_WIDTH_PIXELS = 60  # Minimum width of a person to trigger attribute recognition

# Zones cache for crowd/ROI counts: camera_id -> (list of zone docs, cache_time)
_ZONES_CACHE: Dict[int, Tuple[List[Dict[str, Any]], float]] = {}
_ZONES_CACHE_TTL_SEC = 60.0

# =========================================
# LOAD MODELS
# =========================================
logger.info("Loading YOLO26l-seg model...")
model = YOLO(MODEL_PATH)
logger.info("YOLO model loaded successfully.")

# Defer attributes model loading until needed to avoid import-time crashes
compiled_model = None  # OpenVINO
input_layer = None     # OpenVINO
output_layer = None    # OpenVINO
onnx_session = None    # ONNX Runtime

def ensure_attributes_model_loaded() -> bool:
    """
    Lazily loads the attributes model (OpenVINO or ONNX) the first time it's needed.
    Returns True if the model is available, False otherwise.
    """
    global compiled_model, input_layer, output_layer, onnx_session
    
    # Check if already loaded
    if str(OPENVINO_MODEL_XML).lower().endswith(".onnx"):
        if onnx_session is not None:
            return True
    else:
        if compiled_model is not None and input_layer is not None and output_layer is not None:
            return True

    try:
        from backend.app.services.model_loader import ensure_model_downloaded
        # Ensure model files exist before loading
        ensure_model_downloaded(OPENVINO_MODEL_XML)
        
        if str(OPENVINO_MODEL_XML).lower().endswith(".onnx"):
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
            try:
                # Try with CUDA first, fall back to CPU if it fails
                onnx_session = ort.InferenceSession(OPENVINO_MODEL_XML, providers=providers)
            except Exception:
                onnx_session = ort.InferenceSession(OPENVINO_MODEL_XML, providers=['CPUExecutionProvider'])
            logger.info("ONNX attributes model loaded successfully (lazy).")
            return True
        else:
            # Import here to avoid failing at module import time if OpenVINO is misconfigured
            from openvino.runtime import Core as OVCore  # type: ignore
            ie = OVCore()
            cm = ie.compile_model(model=OPENVINO_MODEL_XML, device_name="CPU")
            compiled_model = cm
            input_layer = compiled_model.input(0)
            output_layer = compiled_model.output(0)
            logger.info("OpenVINO attributes model loaded successfully (lazy).")
            return True
    except Exception as e:
        logger.warning("Attributes model disabled (lazy loading failed): {}", e)
        return False


# =================================================================
# ADVANCED COLOR UTILITIES (CIEDE2000 Implementation)
# =================================================================
CSS3_COLORS = {
    'Red': '#FF0000', 'Green': '#008000', 'Blue': '#0000FF', 'Yellow': '#FFFF00',
    'Black': '#000000', 'White': '#FFFFFF', 'Purple': '#800080', 'Orange': '#FFA500',
    'Pink': '#FFC0CB', 'Brown': '#A52A2A', 'Gray': '#808080', 'Cyan': '#00FFFF',
    'Magenta': '#FF00FF', 'Lime': '#00FF00', 'Navy': '#000080', 'Olive': '#808000',
    'Teal': '#008080', 'Violet': '#EE82EE', 'Maroon': '#800000', 'Silver': '#C0C0C0',
    'Gold': '#FFD700', 'Coral': '#FF7F50', 'Turquoise': '#40E0D0', 'Salmon': '#FA8072',
    'Indigo': '#4B0082'
}

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

CSS3_RGB_COLORS = {name: hex_to_rgb(code) for name, code in CSS3_COLORS.items()}

def rgb_to_lab(rgb):
    """Convert RGB to LAB using OpenCV."""
    rgb_arr = np.array([[rgb]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)[0][0]
    return tuple(map(float, lab))

CSS3_LAB_COLORS = {name: rgb_to_lab(rgb) for name, rgb in CSS3_RGB_COLORS.items()}

def delta_e_ciede2000(lab1, lab2, k_L=1, k_C=1, k_H=1):
    """Compute CIEDE2000 color difference between two LAB colors."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    bar_C = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(bar_C**7 / (bar_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)

    h1p = math.degrees(math.atan2(b1, a1p))
    if h1p < 0: h1p += 360

    h2p = math.degrees(math.atan2(b2, a2p))
    if h2p < 0: h2p += 360

    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p

    diff_h = h2p - h1p
    if C1p * C2p == 0:
        delta_hp = 0
    elif abs(diff_h) <= 180:
        delta_hp = diff_h
    elif diff_h > 180:
        delta_hp = diff_h - 360
    else:
        delta_hp = diff_h + 360
    
    delta_Hp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(delta_hp) / 2)

    bar_Lp = (L1 + L2) / 2
    bar_Cp = (C1p + C2p) / 2

    if abs(diff_h) <= 180:
        bar_hp = (h1p + h2p) / 2
    else:
        if (h1p + h2p) < 360:
            bar_hp = (h1p + h2p + 360) / 2
        else:
            bar_hp = (h1p + h2p - 360) / 2

    T = 1 - 0.17 * math.cos(math.radians(bar_hp - 30)) + \
        0.24 * math.cos(math.radians(2 * bar_hp)) + \
        0.32 * math.cos(math.radians(3 * bar_hp + 6)) - \
        0.20 * math.cos(math.radians(4 * bar_hp - 63))

    delta_theta = 30 * math.exp(-(((bar_hp - 275) / 25)**2))
    R_C = 2 * math.sqrt(bar_Cp**7 / (bar_Cp**7 + 25**7))
    S_L = 1 + (0.015 * (bar_Lp - 50)**2) / math.sqrt(20 + (bar_Lp - 50)**2)
    S_C = 1 + 0.045 * bar_Cp
    S_H = 1 + 0.015 * bar_Cp * T
    R_T = -math.sin(math.radians(2 * delta_theta)) * R_C

    dE = math.sqrt(
        (delta_Lp / (k_L * S_L))**2 +
        (delta_Cp / (k_C * S_C))**2 +
        (delta_Hp / (k_H * S_H))**2 +
        R_T * (delta_Cp / (k_C * S_C)) * (delta_Hp / (k_H * S_H))
    )
    return dE

def closest_color(rgb_color):
    """Find closest named color to given RGB color using CIEDE2000."""
    if rgb_color == (0, 0, 0):
        return "Unknown"
    
    lab_color = rgb_to_lab(rgb_color)
    return min(CSS3_LAB_COLORS, key=lambda name: delta_e_ciede2000(lab_color, CSS3_LAB_COLORS[name]))


def _extract_masked_lab_pixels(roi_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract LAB pixels from only the segmented (masked) region of an ROI.
    Applies a hard binary threshold (>0.5) on the mask to exclude soft boundary pixels.
    Returns Nx3 LAB array or None if insufficient pixels.
    """
    if roi_bgr is None or roi_bgr.size == 0 or mask is None or mask.size == 0:
        return None
    # Hard binary threshold — exclude soft boundary pixels that leak background
    if mask.dtype in (np.float32, np.float64):
        mask_bin = (mask > 0.5).astype(np.uint8)
    else:
        mask_bin = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
    # Ensure mask dimensions match ROI
    if mask_bin.shape[:2] != roi_bgr.shape[:2]:
        mask_bin = cv2.resize(mask_bin, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    lab_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab_roi[mask_bin > 0]
    if pixels.shape[0] < 10:
        return None
    return pixels


def _kmeans_dominant_colors(lab_pixels: np.ndarray, k: int = 3) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    Run MiniBatchKMeans on LAB pixels and return top-k named colors sorted by cluster size.
    Each entry is (color_name, (R, G, B)).
    """
    # Clamp k to available pixel count
    n_pixels = lab_pixels.shape[0]
    actual_k = min(k, max(1, n_pixels // 10))
    actual_k = max(1, actual_k)

    km = MiniBatchKMeans(n_clusters=actual_k, n_init=1, max_iter=20, batch_size=min(512, n_pixels), random_state=42)
    km.fit(lab_pixels.astype(np.float32))

    # Sort clusters by size (largest first)
    counts = np.bincount(km.labels_, minlength=actual_k)
    sorted_indices = np.argsort(-counts)

    results: List[Tuple[str, Tuple[int, int, int]]] = []
    for idx in sorted_indices[:k]:
        center_lab = km.cluster_centers_[idx]
        lab_arr = np.array([[[int(center_lab[0]), int(center_lab[1]), int(center_lab[2])]]], dtype=np.uint8)
        rgb = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)[0][0]
        rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        name = closest_color(rgb_tuple)
        results.append((name, rgb_tuple))
    return results


def get_top_colors(roi_bgr: np.ndarray, mask: np.ndarray, k: int = 3) -> List[str]:
    """
    Extract top-k dominant named colors from the segmented (masked) region only.
    Background pixels within the bounding box are excluded via the segmentation mask.
    Returns list of color names sorted by dominance, e.g. ["Red", "White", "Black"].
    Falls back to single histogram color if k-means fails.
    """
    try:
        pixels = _extract_masked_lab_pixels(roi_bgr, mask)
        if pixels is None:
            # Fallback: mean color from whatever pixels exist
            if roi_bgr is not None and roi_bgr.size > 0:
                mean_bgr = cv2.mean(roi_bgr)[:3]
                rgb_fb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
                return [closest_color(rgb_fb)]
            return ["Unknown"]
        results = _kmeans_dominant_colors(pixels, k=k)
        if not results:
            return ["Unknown"]
        # Deduplicate names while preserving order
        seen = set()
        names: List[str] = []
        for name, _ in results:
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names if names else ["Unknown"]
    except Exception as e:
        logger.warning("get_top_colors failed: {}", e)
        return ["Unknown"]


def get_person_body_colors(roi_bgr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Split a person ROI into upper body (~top 40%) and lower body (~bottom 60%)
    and extract top-3 dominant colors from ONLY the segmented pixels in each region.
    Background pixels are fully excluded via the segmentation mask.

    Returns:
        {
            "color": "Red",                          # primary overall (backward compat)
            "colors": ["Red", "White", "Blue"],      # top-3 overall
            "upper_body_colors": ["Red", "White"],   # top-3 upper body
            "lower_body_colors": ["Blue", "Black"],  # top-3 lower body
        }
    """
    h = roi_bgr.shape[0]
    split_y = int(h * 0.40)  # head+torso vs legs

    # Overall colors from full segmented region
    overall = get_top_colors(roi_bgr, mask, k=3)

    # Upper body (top 40%)
    upper_roi = roi_bgr[:split_y, :]
    upper_mask = mask[:split_y, :] if mask.shape[0] >= split_y else mask
    upper_colors = get_top_colors(upper_roi, upper_mask, k=3)

    # Lower body (bottom 60%)
    lower_roi = roi_bgr[split_y:, :]
    lower_mask = mask[split_y:, :] if mask.shape[0] > split_y else mask
    lower_colors = get_top_colors(lower_roi, lower_mask, k=3)

    return {
        "color": overall[0] if overall else "Unknown",
        "colors": overall,
        "upper_body_colors": upper_colors,
        "lower_body_colors": lower_colors,
    }


def get_dominant_color(roi, mask, k=NUM_CLUSTERS):
    """Legacy wrapper — returns primary (R,G,B) via new k-means pipeline for backward compatibility."""
    try:
        pixels = _extract_masked_lab_pixels(roi, mask)
        if pixels is None:
            return (0, 0, 0)
        results = _kmeans_dominant_colors(pixels, k=k)
        if results:
            return results[0][1]  # (R, G, B)
        return (0, 0, 0)
    except Exception:
        return (0, 0, 0)

# =========================================
# PERSON ATTRIBUTES PREPROCESSING & INFERENCE
# =========================================
def preprocess_for_attributes(roi, target_size=(80, 160)):
    """
    Resizes and pads an image to a target size while maintaining aspect ratio (letterboxing).
    This prevents feature distortion.
    """
    target_w, target_h = target_size
    h, w = roi.shape[:2]

    # Determine the scale factor and new dimensions
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black canvas of the target size
    padded = np.full((target_h, target_w, 3), 0, dtype=np.uint8)

    # Paste the resized image into the center of the canvas
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    return padded

def get_person_attributes(person_roi, person_mask):
    """
    IMPROVED: Infers attributes from a masked, aspect-ratio-preserved ROI.
    Supports both OpenVINO and ONNX backends.
    Returns confidence scores instead of booleans for more nuanced data.
    """
    if person_roi is None or person_roi.size == 0:
        return {}
    if not ensure_attributes_model_loaded():
        return {}
    
    try:
        # 1. Apply mask to ROI to remove background noise
        mask_u8 = (person_mask * 255).astype(np.uint8) if person_mask.dtype != np.uint8 else person_mask
        mask_3ch = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        masked_roi = cv2.bitwise_and(person_roi, mask_3ch)

        # 2. Preprocess with letterboxing to preserve aspect ratio
        # For modern PAR models (like PA100k ONNX), standard size is typically 256x128.
        # But we'll keep 160x80 to remain compatible with Intel model unless changed.
        target_size = (80, 160) if not onnx_session else (128, 256) # (w, h)
        preprocessed_img = preprocess_for_attributes(masked_roi, target_size=target_size)

        # 3. Inference
        img = preprocessed_img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)

        if onnx_session is not None:
            # ONNX Inference (usually normalizes by 255 and maybe mean/std, but keep generic here)
            img = img / 255.0
            input_name = onnx_session.get_inputs()[0].name
            preds = onnx_session.run(None, {input_name: img})[0]
        else:
            # OpenVINO Inference (Intel model takes raw 0-255 inputs)
            preds = compiled_model([img])[output_layer]

        attrs = preds.flatten()

        # 4. Convert model logits to probabilities using sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Detect if outputs need sigmoid (some ONNX models already output 0-1)
        needs_sigmoid = True
        try:
            if attrs.size > 0 and np.all((attrs >= 0.0) & (attrs <= 1.0)):
                needs_sigmoid = False
        except Exception:
            needs_sigmoid = True

        # 5. Return confidence scores for richer data
        # Note: Index mapping assumes Intel model format for both for now. 
        # A true ONNX PAR replacement would need an index mapping dict.
        return {
            "male_confidence": round(float(sigmoid(attrs[0]) if needs_sigmoid else attrs[0]), 2),
            "bag_confidence": round(float(sigmoid(attrs[1]) if needs_sigmoid else attrs[1]), 2),
            "hat_confidence": round(float(sigmoid(attrs[2]) if needs_sigmoid else attrs[2]), 2),
            "longhair_confidence": round(float(sigmoid(attrs[3]) if needs_sigmoid else attrs[3]), 2),
            "longpants_confidence": round(float(sigmoid(attrs[4]) if needs_sigmoid else attrs[4]), 2),
            "longsleeves_confidence": round(float(sigmoid(attrs[5]) if needs_sigmoid else attrs[5]), 2),
            "coat_jacket_confidence": round(float(sigmoid(attrs[6]) if needs_sigmoid else attrs[6]), 2),
        }
    except Exception as e:
        logger.warning("Failed to get person attributes: {}", e)
        return {}


def get_person_attributes_batch(
    person_rois: List[np.ndarray], person_masks: List[np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Batch infer person attributes for multiple ROIs.
    Supports OpenVINO and ONNX backends.
    """
    if not person_rois or not person_masks:
        return []
    if not ensure_attributes_model_loaded():
        return [{} for _ in person_rois]
    try:
        batch_arr: List[np.ndarray] = []
        target_size = (80, 160) if not onnx_session else (128, 256)
        
        for roi, mask in zip(person_rois, person_masks):
            if roi is None or roi.size == 0:
                batch_arr.append(np.zeros((3, target_size[1], target_size[0]), dtype=np.float32))
                continue
            mask_u8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
            mask_3ch = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
            masked_roi = cv2.bitwise_and(roi, mask_3ch)
            preprocessed = preprocess_for_attributes(masked_roi, target_size=target_size)
            img = preprocessed.transpose((2, 0, 1)).astype(np.float32)
            if onnx_session is not None:
                img = img / 255.0
            batch_arr.append(img)
            
        batch = np.stack(batch_arr, axis=0).astype(np.float32)
        
        if onnx_session is not None:
            input_name = onnx_session.get_inputs()[0].name
            preds = onnx_session.run(None, {input_name: batch})[0]
        else:
            # OpenVINO model expects batch=1; infer one at a time
            pred_list = []
            for img in batch_arr:
                single = img[np.newaxis, ...]  # (1,3,H,W)
                pred_list.append(compiled_model([single])[output_layer])
            preds = np.concatenate(pred_list, axis=0)
            
        out: List[Dict[str, Any]] = []

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        needs_sigmoid = True
        try:
            if preds.size > 0 and np.all((preds >= 0.0) & (preds <= 1.0)):
                needs_sigmoid = False
        except Exception:
            needs_sigmoid = True
        for i in range(preds.shape[0]):
            attrs = preds[i].flatten()
            out.append({
                "male_confidence": round(float(sigmoid(attrs[0]) if needs_sigmoid else attrs[0]), 2),
                "bag_confidence": round(float(sigmoid(attrs[1]) if needs_sigmoid else attrs[1]), 2),
                "hat_confidence": round(float(sigmoid(attrs[2]) if needs_sigmoid else attrs[2]), 2),
                "longhair_confidence": round(float(sigmoid(attrs[3]) if needs_sigmoid else attrs[3]), 2),
                "longpants_confidence": round(float(sigmoid(attrs[4]) if needs_sigmoid else attrs[4]), 2),
                "longsleeves_confidence": round(float(sigmoid(attrs[5]) if needs_sigmoid else attrs[5]), 2),
                "coat_jacket_confidence": round(float(sigmoid(attrs[6]) if needs_sigmoid else attrs[6]), 2),
            })
        return out
    except Exception as e:
        logger.warning("Batch person attributes failed: {}", e)
        return [{} for _ in person_rois]


def _run_person_embeddings_async(
    camera_id: int,
    timestamp: str,
    det_id_str: Optional[str],
    person_rois: List[np.ndarray],
    person_attr_texts: List[str],
    person_meta: List[Dict[str, object]],
) -> None:
    """
    Run in task queue: batch SigLIP + attribute encode, fuse, add to person FAISS.
    Decouples embedding compute from the detection loop.
    """
    if not person_rois or not person_attr_texts or len(person_rois) != len(person_attr_texts):
        return
    try:
        visual_encoder = get_visual_embedder()
        attr_encoder = get_attribute_encoder()
        fusion = MultimodalFusion(
            visual_weight=getattr(settings, "FUSION_VISUAL_WEIGHT", 0.6),
            text_weight=getattr(settings, "FUSION_TEXT_WEIGHT", 0.4),
        )
        person_index = get_person_store(dim=1152)
        v_embs = visual_encoder.encode(person_rois)
        t_embs = attr_encoder.encode_texts(person_attr_texts)
        fused_vecs: List[np.ndarray] = []
        for j in range(len(person_rois)):
            v_vec = v_embs[j] if j < v_embs.shape[0] else np.zeros((768,), dtype=np.float32)
            t_vec = t_embs[j] if j < t_embs.shape[0] else attr_encoder.encode_text(person_attr_texts[j])
            fused_vecs.append(fusion.fuse(v_vec, t_vec).astype(np.float32))
        arr = np.vstack(fused_vecs).astype(np.float32)
        metas = [dict(m, detection_id=det_id_str) for m in person_meta]
        person_index.vector_add(arr, metas, save=False)
    except Exception as e:
        logger.warning("Async person embeddings failed: {}", e)


# =========================================
# AUTO INDEXING HELPERS (VLM)
# =========================================
def _auto_build_and_index(camera_id: int, start_iso: str, end_iso: str, every_sec: float = 1.0) -> None:
    """
    Build a clip from snapshots between [start_iso, end_iso] and index it semantically.
    Non-blocking: intended for background thread usage.
    """
    try:
        # Respect runtime mode: skip semantic work if structured-only
        if get_indexing_mode() == "structured":
            return
        built = build_clip_from_snapshots(camera_id=camera_id, start_iso=start_iso, end_iso=end_iso, fps=5.0)
        index_clip(
            camera_id=camera_id,
            clip_path=built.path,
            clip_url=built.url,
            every_sec=every_sec,
            with_captions=bool(settings.ENABLE_CAPTIONS),
        )
    except Exception as e:
        # Log to stdout; do not interrupt detection
        logger.warning("Auto VLM indexing failed for cam {} [{}..{}]: {}", camera_id, start_iso, end_iso, e)


def cleanup_corrupt_recordings() -> int:
    """
    Remove corrupt recordings (missing moov atom) and stale conversion files.

    Called at startup to clean up after unclean shutdowns.  mp4v recordings
    write the moov atom only on ``VideoWriter.release()``; if the process was
    killed, the file is large but unreadable.

    Returns the number of files removed.
    """
    removed = 0
    recordings_dir: Path = settings.RECORDINGS_DIR
    if not recordings_dir.exists():
        return 0
    for cam_dir in recordings_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        for p in cam_dir.glob("*.mp4"):
            try:
                # Remove stale conversion intermediaries
                if "._converting" in p.name:
                    if p.stat().st_size < 1024:
                        p.unlink(missing_ok=True)
                        removed += 1
                        logger.info("Removed stale conversion file: {}", p.name)
                    continue
                # Skip very small files (already handled elsewhere)
                if p.stat().st_size < 1024:
                    continue
                # Quick binary check for moov atom
                has_moov = False
                with open(p, "rb") as f:
                    head = f.read(min(p.stat().st_size, 64 * 1024))
                if b"moov" in head:
                    has_moov = True
                elif p.stat().st_size > 64 * 1024:
                    with open(p, "rb") as f:
                        f.seek(max(0, p.stat().st_size - 64 * 1024))
                        tail = f.read()
                    has_moov = b"moov" in tail
                if not has_moov:
                    p.unlink(missing_ok=True)
                    removed += 1
                    logger.warning(
                        "Removed corrupt recording (no moov atom, {}MB): {}",
                        p.stat().st_size / 1024 / 1024 if p.exists() else 0,
                        p.name,
                    )
            except Exception as e:
                logger.debug("cleanup_corrupt_recordings: skip {}: {}", p.name, e)
    if removed:
        logger.info("Startup cleanup removed {} corrupt/stale recording(s)", removed)
    return removed


import queue

def _ffmpeg_convert_worker(q: queue.Queue):
    import subprocess
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_exe = get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"
        
    while True:
        try:
            item = q.get()
            if item is None:
                break
            # Support both old format (str) and new format (str, actual_fps)
            if isinstance(item, tuple):
                filepath, actual_fps = item
            else:
                filepath, actual_fps = item, 0.0
            filepath = Path(filepath)
            h264_path = filepath.parent / f"{filepath.stem}._converting.mp4"
            cmd = [
                ffmpeg_exe, "-y", "-i", str(filepath),
            ]
            # If we measured actual FPS, tell FFmpeg to re-stamp frames at
            # the correct rate so video-time matches wall-clock time.
            if actual_fps >= 1.0:
                cmd += ["-r", f"{actual_fps:.2f}"]
            cmd += [
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-loglevel", "error", str(h264_path)
            ]
            success = False
            try:
                cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
                success = cp.returncode == 0
                if not success:
                    err = cp.stderr.decode(errors="ignore") if cp.stderr else ""
                    logger.warning("FFmpeg background conversion failed for {}: {}", filepath, err[:500])
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg background conversion timed out for {}", filepath)
            if success and h264_path.exists() and h264_path != filepath:
                filepath.unlink(missing_ok=True)
                h264_path.rename(filepath)
        except Exception as e:
            logger.warning("FFmpeg background conversion failed: {}", e)
        finally:
            q.task_done()

MAX_FFMPEG_QUEUE_SIZE = 32
_conversion_queue = queue.Queue(maxsize=MAX_FFMPEG_QUEUE_SIZE)
_conversion_thread = threading.Thread(target=_ffmpeg_convert_worker, args=(_conversion_queue,), daemon=True)
_conversion_thread.start()

class ContinuousRecorder:
    def __init__(self, camera_id: int, fps: float, chunk_duration_sec: int = 300):
        self.camera_id = camera_id
        self.fps = fps if fps > 0 else 30.0
        self.chunk_duration_sec = chunk_duration_sec
        self.writer = None
        self.current_chunk_start_time = 0.0
        self.current_filepath = None
        self.frame_size = None
        self._frame_count = 0  # track actual frames written per chunk
        
        self.out_dir = settings.RECORDINGS_DIR / f"camera_{camera_id}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def _open_writer(self, frame):
        h, w = frame.shape[:2]
        # Ensure even dimensions for mp4v
        w = w - (w % 2)
        h = h - (h % 2)
        self.frame_size = (w, h)
        target_size = (w, h)
        
        now = datetime.datetime.now(datetime.timezone.utc)
        # Strip tz info BEFORE replacing separators so the filename is clean
        # (consistent with snapshot filenames which use naive-UTC timestamps).
        safe_ts = now.replace(tzinfo=None).isoformat().replace(":", "-").replace(".", "-")
        filename = f"{safe_ts}.mp4"
        self.current_filepath = self.out_dir / filename
        
        _fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        fourcc = _fourcc_fn(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.current_filepath), fourcc, max(1.0, float(self.fps)), target_size)
        self.current_chunk_start_time = time.time()
        self._frame_count = 0
        
    def write(self, frame):
        now = time.time()
        if self.writer is None:
            self._open_writer(frame)
        elif now - self.current_chunk_start_time >= self.chunk_duration_sec:
            self._close_writer()
            self._open_writer(frame)
            
        if self.writer is not None and self.writer.isOpened():
            h, w = frame.shape[:2]
            target_w, target_h = self.frame_size
            if w != target_w or h != target_h:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            self.writer.write(frame)
            self._frame_count += 1
            
    def _close_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            if self.current_filepath and self.current_filepath.exists():
                # Compute actual FPS from frames written / wall-clock duration
                # so the FFmpeg re-encode can fix the video timebase.
                wall_dur = time.time() - self.current_chunk_start_time
                actual_fps = self._frame_count / wall_dur if wall_dur > 1.0 else 0.0
                try:
                    _conversion_queue.put(
                        (str(self.current_filepath), actual_fps),
                        timeout=5.0,
                    )
                except queue.Full:
                    logger.warning("FFmpeg conversion queue full; dropping file {}", self.current_filepath)

    def release(self):
        self._close_writer()

# =========================================
# Threaded frame reader — keeps capture flowing
# even while YOLO detection blocks the main loop.
# =========================================
class ThreadedFrameReader:
    """Continuously reads frames on a background thread, feeds MJPEG, and records."""

    def __init__(self, cap: "cv2.VideoCapture", camera_id: int):
        self._cap = cap
        self._camera_id = camera_id
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._stopped = False
        
        # Initialize continuous recorder
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._recorder = ContinuousRecorder(camera_id, fps, chunk_duration_sec=300)
        
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopped:
            try:
                ret, frame = self._cap.read()
            except Exception:
                break
            if self._stopped:
                break
            with self._lock:
                self._ret = ret
                self._frame = frame
                
            if ret and frame is not None:
                # Record to disk
                self._recorder.write(frame)
                
                try:
                    encode_ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if encode_ok:
                        update_frame(self._camera_id, jpg.tobytes())
                except Exception:
                    pass
            else:
                import time
                time.sleep(0.01)

    def read(self):
        """Return the latest frame (non-blocking)."""
        with self._lock:
            return self._ret, self._frame

    def stop(self) -> None:
        self._stopped = True
        # Release the capture first to unblock any pending cap.read()
        try:
            self._cap.release()
        except Exception:
            pass
        self._thread.join(timeout=2)
        if hasattr(self, '_recorder'):
            self._recorder.release()

    def release(self) -> None:
        self.stop()

    def reopen(self, source, backend=None):
        """Release old cap, open a new one, restart reader thread."""
        self.stop()  # sets _stopped, releases cap, joins thread
        if backend is not None:
            self._cap = cv2.VideoCapture(source, backend)
        else:
            self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # update recorder fps just in case
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._recorder.fps = fps if fps > 0 else 30.0
        
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._cap.isOpened()


# =========================================
# HELPER: ENRICH DETECTED OBJECT (COLOR + TRACK + META)
# =========================================
def enrich_detected_object(
    obj_name: str,
    masked_roi: np.ndarray,
    mask_roi: np.ndarray,
    x1: int, x2: int, y1: int, y2: int,
    track_id: Any,
    conf_val: Any,
    timestamp: str,
    camera_id: Any,
    obj_index: int,
    track_state: Dict[int, Any],
    track_bulk_ops: List[Any],
    person_attr_defer: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], int, int]],
    person_rois: List[np.ndarray],
    person_attr_texts: List[str],
    person_meta: List[Dict[str, Any]],
    attr_encoder: Any
) -> Dict[str, Any]:
    """Helper to unify bounding-box extraction logic in both live and file stream."""
    from pymongo import UpdateOne
    # Color extraction: persons get upper/lower body split; others get top-3
    if obj_name == "person":
        color_info = get_person_body_colors(masked_roi, mask_roi)
        color_name = color_info["color"]
    else:
        top_colors = get_top_colors(masked_roi, mask_roi, k=3)
        color_name = top_colors[0] if top_colors else "Unknown"
        color_info = {"color": color_name, "colors": top_colors}

    tid = int(track_id) if track_id is not None else -1
    cval = float(conf_val)

    obj: Dict[str, Any] = {
        "object_name": obj_name,
        "track_id": tid,
        "confidence": round(cval, 2),
        "color": color_name,
        "colors": color_info.get("colors", [color_name]),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
    }
    if obj_name == "person":
        obj["upper_body_colors"] = color_info.get("upper_body_colors", [])
        obj["lower_body_colors"] = color_info.get("lower_body_colors", [])

    # Update track history (persistent across frames)
    if tid >= 0:
        if track_state is not None:
            st = track_state.get(tid)
            if st is None:
                st = {"first_seen": timestamp, "last_seen": timestamp, "frame_count": 1}
            else:
                st["last_seen"] = timestamp
                try:
                    prev_fc = st.get("frame_count", 0)
                    prev_fc = int(prev_fc) if isinstance(prev_fc, (int, float, str)) else 0
                    st["frame_count"] = prev_fc + 1
                except Exception:
                    st["frame_count"] = 1
            track_state[tid] = st
            obj["track_history"] = st

        track_bulk_ops.append(
            UpdateOne(
                {"camera_id": int(camera_id), "track_id": tid},
                {
                    "$setOnInsert": {"first_seen": timestamp if not track_state else st["first_seen"], "camera_id": int(camera_id)},
                    "$set": {"last_seen": timestamp},
                    "$inc": {"frame_count": 1},
                },
                upsert=True,
            )
        )

    # Person attributes — quality filter + batch defer
    if obj_name == "person":
        if (x2 - x1) >= MIN_PERSON_WIDTH_PIXELS:
            person_attr_defer.append((masked_roi, mask_roi, color_info, tid, obj_index))
            obj["person_attributes"] = {}
            obj["attribute_text"] = ""
        else:
            obj["person_attributes"] = {"status": "skipped_too_small"}
            attr_text = attr_encoder.attributes_to_text({}, color_name, color_info) if attr_encoder else "person"
            obj["attribute_text"] = attr_text
            person_rois.append(masked_roi)
            person_attr_texts.append(attr_text)
            person_meta.append({
                "object_index": obj_index,
                "camera_id": int(camera_id),
                "track_id": tid,
                "timestamp": timestamp,
                "color": color_name,
                "colors": color_info.get("colors", [color_name]),
                "upper_body_colors": color_info.get("upper_body_colors", []),
                "lower_body_colors": color_info.get("lower_body_colors", []),
                "attribute_text": attr_text,
            })
    return obj

# =========================================
# MAIN DETECTION & TRACKING LOOP
# =========================================
def process_live_stream(
    camera_id: int = 1,
    source: Union[int, str] = 0,
    location: str = "Default Location",
    show_window: bool = False,
    stop_event: Optional[threading.Event] = None
):
    # Use shared app DB layer (backend.app.db.mongo)
    cameras_collection = cameras_col
    detections_collection = detections_col
    zones_collection = zones_col

    # Use DirectShow for local webcams on Windows to improve reliability
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        # For HTTP/MJPEG streams (e.g. DroidCam), use FFMPEG backend
        # and reduce buffer to 1 frame to avoid latency/stale-frame issues
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        msg = f"Cannot open camera source {source}"
        logger.error("{}", msg)
        try:
            cameras_collection.update_one(
                {"camera_id": camera_id},
                {"$set": {"status": "error", "last_error": msg, "source": str(source), "location": location}},
                upsert=True,
            )
        except Exception:
            pass
        return

    cameras_collection.update_one(
        {"camera_id": camera_id},
        {"$set": {
            "source": str(source),
            "location": location,
            "status": "active",
            "last_seen": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "").rstrip("Z")
        }},
        upsert=True
    )
    logger.info("Camera {} registered in the database.", camera_id)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps / 2) # Process 2 frames per second
    frame_count = 0

    # Wrap capture in a threaded reader so frames flow to the MJPEG stream
    # even while YOLO inference blocks the detection loop.
    reader = ThreadedFrameReader(cap, camera_id)

    # Initialize BoT-SORT tracker (if available)
    tracker = None

    # Initialize multimodal encoders and person index
    attr_encoder = get_attribute_encoder()
    person_index = get_person_store(dim=1152)
    if BoTSORT is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reid_weights = os.getenv("REID_WEIGHTS", "osnet_x1_0_msmt17.pt")
        # Detect actual camera FPS so track_buffer scales correctly
        _cam_fps = int(cap.get(cv2.CAP_PROP_FPS) or 30) if hasattr(cap, 'get') else 30
        try:
            tracker = BoTSORT(
                reid_weights=Path(reid_weights),
                device=device,
                half=(device == "cuda"),
                # --- Surveillance-tuned params ---
                # Keep a lost person alive for 60 frames (2 s at 30 fps)
                # so they can be re-identified after walking behind an object.
                track_buffer=int(os.getenv("BOTSORT_TRACK_BUFFER", "60")),
                # Stricter ReID: only match if appearance distance < 0.4
                # (default 0.25 is too loose and causes ID swaps).
                appearance_thresh=float(os.getenv("BOTSORT_APPEARANCE_THRESH", "0.4")),
                # Scale buffer relative to actual FPS
                frame_rate=_cam_fps,
            )
            logger.info(
                "BoTSORT tracker active — ReID: %s | device: %s | buffer: %s frames | appearance_thresh: %s",
                reid_weights, device,
                os.getenv("BOTSORT_TRACK_BUFFER", "60"),
                os.getenv("BOTSORT_APPEARANCE_THRESH", "0.4"),
            )
        except Exception as e:
            logger.error(
                "BoTSORT init FAILED (falling back to YOLO built-in tracker). "
                "Ensure boxmot is installed. Error: %s",
                e,
            )
            tracker = None

    # In-memory track history across frames: track_id -> {first_seen, last_seen, frame_count}
    track_state: Dict[int, Dict[str, Any]] = {}

    # Auto-indexing window (seconds) — builds short clips from snapshots and indexes them
    try:
        window_sec = float(os.getenv("AUTO_INDEX_WINDOW_SEC", "15"))
    except Exception:
        window_sec = 15.0
    window_start_iso: Optional[str] = None

    logger.info("Starting live detection on Camera {} at '{}'...", camera_id, location)

    def _should_stop() -> bool:
        return stop_event is not None and stop_event.is_set()

    read_failures = 0
    max_read_failures = 60
    reconnect_attempts = 0
    max_reconnects = 3
    last_heartbeat_time = 0.0
    last_person_save_time = 0.0

    while not _should_stop():
        ret, frame = reader.read()
        if not ret or frame is None:
            read_failures += 1
            if read_failures > max_read_failures:
                # For HTTP/MJPEG streams, try to reconnect before giving up
                if isinstance(source, str) and reconnect_attempts < max_reconnects:
                    reconnect_attempts += 1
                    logger.info("Stream dropped. Reconnecting (attempt {}/{})...", reconnect_attempts, max_reconnects)
                    import time
                    for _ in range(20):
                        if _should_stop():
                            break
                        time.sleep(0.1)
                    if reader.reopen(source, cv2.CAP_FFMPEG):
                        logger.info("Reconnected to stream on attempt {}", reconnect_attempts)
                        read_failures = 0
                        continue
                    else:
                        logger.warning("Reconnect attempt {} failed", reconnect_attempts)
                        read_failures = 0
                        continue

                msg = f"No frames from source after {max_read_failures} attempts"
                logger.error("{}", msg)
                try:
                    cameras_collection.update_one(
                        {"camera_id": camera_id},
                        {"$set": {"status": "error", "last_error": msg}},
                        upsert=True,
                    )
                except Exception:
                    pass
                break
            # brief delay before next read attempt
            import time
            time.sleep(0.03)  # 30ms — check ~30 times/sec for a new frame
            continue
        else:
            read_failures = 0
            reconnect_attempts = 0  # Reset reconnect counter on successful read

        # Heartbeat: mark camera active and update last_seen (throttled to every 5s to reduce DB load)
        import time as _time
        now_ts = _time.time()
        if now_ts - last_heartbeat_time >= 5.0:
            try:
                cameras_collection.update_one(
                    {"camera_id": camera_id},
                    {"$set": {"status": "active", "last_seen": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "").rstrip("Z"), "last_error": None}},
                    upsert=True,
                )
                last_heartbeat_time = now_ts
            except Exception:
                pass

        # NOTE: Frame store update is handled by ThreadedFrameReader continuously.
        # No need to call update_frame() here — the reader thread does it at full FPS.

        if frame_count % frame_interval == 0:
            use_half = torch.cuda.is_available()
            if tracker is None:
                results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False, half=use_half)
            else:
                results = model(frame, half=use_half)
            if _should_stop():
                break
            # Use UTC and strip timezone suffix so MongoDB string comparison matches chat retrieval (which uses UTC range with normalized timestamps)
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "").rstrip("Z")

            doc = {"camera_id": camera_id, "timestamp": timestamp, "objects": []}

            # Collect person ROIs and attribute texts for async batch encoding (task queue)
            person_rois: List[np.ndarray] = []
            person_attr_texts: List[str] = []
            person_meta: List[Dict[str, object]] = []
            track_bulk_ops: List[Any] = []  # Batch track updates for bulk_write
            # Defer batch OpenVINO: (masked_roi, mask_roi, color_info, track_id, obj_index) for each person with width >= MIN
            person_attr_defer: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], int, int]] = []

            for r in results:
                if r.masks is None or r.boxes is None:
                    continue

                masks_data = r.masks.data if r.masks is not None else None
                if masks_data is None:
                    continue
                masks = to_numpy(masks_data)

                b = r.boxes
                if b is None:
                    continue

                xyxy = to_numpy(b.xyxy)
                cls_arr = to_numpy(b.cls)
                conf_arr = to_numpy(b.conf)

                # Use BoT-SORT to get persistent track IDs, fallback to Ultralytics IDs if unavailable
                ids_arr = None
                if tracker is not None and isinstance(xyxy, np.ndarray) and xyxy.size > 0:
                    det_to_tid: Dict[int, int] = {}
                    try:
                        # tracker.update expects xyxy, conf, cls and the frame
                        tracks = tracker.update(
                            xyxy.astype(np.float32),
                            (conf_arr.astype(np.float32) if isinstance(conf_arr, np.ndarray) else np.asarray(conf_arr, dtype=np.float32)),
                            (cls_arr.astype(np.int32) if isinstance(cls_arr, np.ndarray) else np.asarray(cls_arr, dtype=np.int32)),
                            frame,
                        )
                        # tracks is expected to be Nx(>=5): [x1, y1, x2, y2, track_id, ...]
                        def _iou(a: np.ndarray, b: np.ndarray) -> float:
                            xx1 = float(max(a[0], b[0])); yy1 = float(max(a[1], b[1]))
                            xx2 = float(min(a[2], b[2])); yy2 = float(min(a[3], b[3]))
                            w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
                            inter = w * h
                            area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
                            area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
                            union = area_a + area_b - inter + 1e-6
                            return inter / union

                        track_boxes: List[Tuple[np.ndarray, int]] = []
                        for t in tracks:
                            try:
                                x1t, y1t, x2t, y2t, tid = float(t[0]), float(t[1]), float(t[2]), float(t[3]), int(t[4])
                                track_boxes.append((np.array([x1t, y1t, x2t, y2t], dtype=np.float32), tid))
                            except Exception:
                                continue

                        for di in range(xyxy.shape[0]):
                            db = xyxy[di].astype(np.float32)
                            best_iou, best_tid = 0.0, -1
                            for tb, tid in track_boxes:
                                iou = _iou(db, tb)
                                if iou > best_iou:
                                    best_iou, best_tid = iou, tid
                            if best_iou >= 0.3:
                                det_to_tid[di] = best_tid

                        ids_arr = np.array([det_to_tid.get(i, -1) for i in range(xyxy.shape[0])], dtype=np.int32)
                    except Exception:
                        ids_arr = None

                # Fallback: Ultralytics-provided IDs if present
                if ids_arr is None:
                    ids_arr = to_numpy(getattr(b, "id", None)) if getattr(b, "id", None) is not None else None

                # Defensive checks to satisfy static typing and runtime safety
                if xyxy is None or cls_arr is None or conf_arr is None:
                    continue
                if not isinstance(xyxy, np.ndarray) or xyxy.size == 0:
                    continue

                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    cls_idx = int(cls_arr[i]) if np.ndim(cls_arr) == 1 else int(cls_arr[i][0])
                    obj_name = model.names[cls_idx]
                    
                    roi = frame[y1:y2, x1:x2]
                    
                    # Handle possible mismatch between boxes and masks length
                    if isinstance(masks, np.ndarray) and masks.ndim >= 1 and i < masks.shape[0]:
                        mask_src = masks[i]
                    else:
                        mask_src = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    mask_full = cv2.resize(mask_src, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Hard binary threshold — exclude soft boundary pixels that leak background
                    if mask_full.dtype in (np.float32, np.float64):
                        mask_full = (mask_full > 0.5).astype(np.uint8) * 255
                    mask_roi = mask_full[y1:y2, x1:x2]
                    # Apply mask to ROI so only segmented pixels are visible (background zeroed)
                    masked_roi = cv2.bitwise_and(roi, roi, mask=mask_roi if mask_roi.dtype == np.uint8 else (mask_roi > 0).astype(np.uint8))
                    
                    obj = enrich_detected_object(
                        obj_name=obj_name,
                        masked_roi=masked_roi,
                        mask_roi=mask_roi,
                        x1=x1, x2=x2, y1=y1, y2=y2,
                        track_id=ids_arr[i] if ids_arr is not None else -1,
                        conf_val=conf_arr[i] if np.ndim(conf_arr) == 1 else conf_arr[i][0],
                        timestamp=timestamp,
                        camera_id=camera_id,
                        obj_index=len(doc["objects"]),
                        track_state=track_state,
                        track_bulk_ops=track_bulk_ops,
                        person_attr_defer=person_attr_defer,
                        person_rois=person_rois,
                        person_attr_texts=person_attr_texts,
                        person_meta=person_meta,
                        attr_encoder=attr_encoder
                    )
                    doc["objects"].append(obj)

            # Batch OpenVINO attribute inference for deferred persons; then add them to person_rois/person_attr_texts/person_meta
            if person_attr_defer:
                rois_attr = [t[0] for t in person_attr_defer]
                masks_attr = [t[1] for t in person_attr_defer]
                batch_attrs = get_person_attributes_batch(rois_attr, masks_attr)
                for j, (mroi, _mask, c_info, track_id, obj_idx) in enumerate(person_attr_defer):
                    if j < len(batch_attrs):
                        doc["objects"][obj_idx]["person_attributes"] = batch_attrs[j]
                    c_name = c_info.get("color", "Unknown")
                    attr_text = attr_encoder.attributes_to_text(
                        doc["objects"][obj_idx].get("person_attributes") or {}, c_name, c_info
                    )
                    doc["objects"][obj_idx]["attribute_text"] = attr_text
                    person_rois.append(mroi)
                    person_attr_texts.append(attr_text)
                    person_meta.append(
                        {
                            "object_index": obj_idx,
                            "camera_id": int(camera_id),
                            "track_id": track_id,
                            "timestamp": timestamp,
                            "color": c_name,
                            "colors": c_info.get("colors", [c_name]),
                            "upper_body_colors": c_info.get("upper_body_colors", []),
                            "lower_body_colors": c_info.get("lower_body_colors", []),
                            "attribute_text": attr_text,
                        }
                    )

            # Person embeddings are computed asynchronously in task queue (see below after insert)

            # Batch persist track updates (one bulk_write per frame)
            if track_bulk_ops:
                try:
                    tracks.bulk_write(track_bulk_ops)
                except Exception:
                    pass

            # Persist object count summaries for robust retrieval (e.g., no-person / multi-person queries)
            try:
                objs = doc.get("objects", []) if isinstance(doc.get("objects"), list) else []
                person_count = 0
                object_counts: Dict[str, int] = {}
                for _o in objs:
                    if not isinstance(_o, dict):
                        continue
                    _name = str(_o.get("object_name", "")).strip().lower()
                    if not _name:
                        continue
                    object_counts[_name] = object_counts.get(_name, 0) + 1
                    if _name == "person":
                        person_count += 1
                doc["person_count"] = int(person_count)
                doc["object_counts"] = object_counts
            except Exception:
                # Keep detection loop resilient if summary bookkeeping fails
                pass

            # Zone-level counts for crowd management (when zones exist for this camera)
            try:
                zones_list = _get_zones_for_camera(zones_collection, int(camera_id))
                if zones_list and doc.get("objects"):
                    h, w = frame.shape[0], frame.shape[1]
                    doc["zone_counts"] = _compute_zone_counts(
                        doc["objects"], zones_list, w, h
                    )
            except Exception as e:
                logger.exception(
                    "Zone-counting failed for camera_id=%s timestamp=%s: %s",
                    camera_id,
                    timestamp,
                    e,
                )

            if doc["objects"]:
                try:
                    safe_ts = timestamp.replace(":", "-").replace(".", "-")
                    cam_dir = settings.SNAPSHOTS_DIR / f"camera_{camera_id}"
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = cam_dir / f"{safe_ts}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    doc["snapshot"] = f"/media/snapshots/camera_{camera_id}/{snap_path.name}"
                except Exception as e:
                    logger.warning("Failed to save snapshot: {}", e)
                ins = detections_collection.insert_one(doc)
                det_id = ins.inserted_id if ins is not None else None

                # Evaluate real-time alert rules with current frame context (incl. zone_counts for zone-based rules)
                try:
                    evaluate_realtime(
                        camera_id=int(camera_id),
                        timestamp_iso=timestamp,
                        objects=doc.get("objects", []),
                        frame_bgr=frame,
                        snapshot_url=doc.get("snapshot"),
                        zone_counts=doc.get("zone_counts"),
                    )
                except Exception as _e:
                    # Do not interrupt detection loop on alert evaluation errors
                    pass

                # Run person embeddings in background task (SigLIP + fusion + FAISS add)
                if person_rois:
                    try:
                        task_submit(
                            _run_person_embeddings_async,
                            int(camera_id),
                            timestamp,
                            str(det_id) if det_id is not None else None,
                            [roi.copy() for roi in person_rois],
                            list(person_attr_texts),
                            [dict(m) for m in person_meta],
                        )
                    except Exception:
                        pass
                # Periodic persist of person index (background task also adds to same store)
                try:
                    import time as _time
                    now_ps = _time.time()
                    if now_ps - last_person_save_time >= 30.0:
                        person_index.save()
                        last_person_save_time = now_ps
                except Exception:
                    pass

                # Auto-index trigger: roll a short window into a clip and index semantically in background
                try:
                    # Initialize window start on first saved snapshot
                    if window_start_iso is None:
                        window_start_iso = timestamp
                    # Roll window if duration reached
                    else:
                        now_dt = datetime.datetime.fromisoformat(timestamp)
                        start_dt = datetime.datetime.fromisoformat(window_start_iso)
                        if (now_dt - start_dt).total_seconds() >= window_sec:
                            # Spawn background job via bounded task queue (backpressure, error logging)
                            task_submit(_auto_build_and_index, camera_id, window_start_iso, timestamp, 1.0)
                            # Start next window from current timestamp
                            window_start_iso = timestamp
                except Exception as _e:
                    # Ignore auto-index scheduling errors
                    pass

        frame_count += 1
        
        # Note: show_window requires GUI-enabled OpenCV build
        # Removed cv2.imshow/waitKey to prevent crashes with opencv-python-headless

    
    logger.info("Camera {} loop exited, cleaning up...", camera_id)
    try:
        cameras_collection.update_one(
            {"camera_id": camera_id},
            {"$set": {"status": "inactive"}},
        )
    except Exception:
        pass
    reader.release()
    logger.info("Camera {} processing completed.", camera_id)


# =========================================
# VIDEO FILE DETECTION (mirrors live stream pipeline for uploaded videos)
# =========================================
def process_video_file(
    camera_id: int,
    video_path: str,
    location: str = "Uploaded Video",
    target_fps: float = 2.0,
    start_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process an uploaded video file through the full detection pipeline:
      YOLO seg -> color extraction -> OpenVINO batch person attributes ->
      attribute text encoding -> MongoDB detection insert -> snapshot save ->
      SigLIP+FAISS person embedding (via async task queue).

    This mirrors process_live_stream() so that uploaded videos are as
    queryable as live cameras.  Call index_clip() AFTER this function so
    the semantic captioner can pull rich detections from MongoDB.

    Args:
        camera_id:  Camera / source ID to tag detections with.
        video_path: Absolute path to the MP4 file.
        location:   Human-readable location label.
        target_fps: How many frames per second to sample (default 2).
        start_iso:  Optional ISO timestamp for the video start time.
                    If None, inferred from the file's mtime.

    Returns:
        dict with keys: ok, frames_processed, detections_inserted.
    """
    import time as _time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("process_video_file: Cannot open video {}", video_path)
        return {"ok": False, "error": f"Cannot open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = max(1, int(round(fps / max(target_fps, 0.1))))

    # Determine start datetime for timestamp assignment
    if start_iso:
        try:
            from dateutil import parser
            dt = parser.isoparse(start_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            start_dt = dt.astimezone(datetime.timezone.utc)
        except Exception:
            start_dt = datetime.datetime.now(datetime.timezone.utc)
    else:
        try:
            mtime = os.path.getmtime(video_path)
            duration = total_frames / fps if fps > 0 and total_frames > 0 else 0
            start_dt = datetime.datetime.fromtimestamp(max(0, mtime - duration), tz=datetime.timezone.utc)
        except Exception:
            start_dt = datetime.datetime.now(datetime.timezone.utc)
    
    # Ensure start_dt explicitly drops the explicit '+00:00' for a canonical 'Z' if aware
    def to_canonical_iso(dt: datetime.datetime) -> str:
        s = dt.isoformat()
        if s.endswith("+00:00"):
            return s.replace("+00:00", "Z")
        return s
    canonical_start_iso = to_canonical_iso(start_dt)

    logger.info(
        "process_video_file: camera_id=%s path=%s fps=%.1f target_fps=%.1f interval=%d start=%s",
        camera_id, video_path, fps, target_fps, frame_interval, canonical_start_iso,
    )

    attr_encoder = None  # type: ignore
    person_index = None  # type: ignore
    frames_processed = 0
    detections_inserted = 0
    frame_idx = 0
    last_person_save_ts = _time.time()

    try:
        attr_encoder = get_attribute_encoder()
        person_index = get_person_store(dim=1152)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            frames_processed += 1

            # Timestamp from video position
            pos_sec = frame_idx / max(fps, 1e-6)
            timestamp = (start_dt + datetime.timedelta(seconds=pos_sec)).isoformat()

            use_half = torch.cuda.is_available()
            try:
                # persist=True gives consistent YOLO track IDs across frames in the file
                results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False, half=use_half)
            except Exception as e:
                logger.warning("process_video_file: YOLO track failed frame {}: {}", frame_idx, e)
                frame_idx += 1
                continue

            doc: Dict[str, Any] = {
                "camera_id": camera_id,
                "timestamp": timestamp,
                "location": location,
                "objects": [],
            }
            person_rois: List[np.ndarray] = []
            person_attr_texts: List[str] = []
            person_meta: List[Dict[str, object]] = []
            track_bulk_ops: List[Any] = []
            person_attr_defer: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], int, int]] = []

            for r in results:
                if r.masks is None or r.boxes is None:
                    continue
                masks_data = r.masks.data if r.masks is not None else None
                if masks_data is None:
                    continue
                masks = to_numpy(masks_data)
                b = r.boxes
                if b is None:
                    continue
                xyxy = to_numpy(b.xyxy)
                cls_arr = to_numpy(b.cls)
                conf_arr = to_numpy(b.conf)
                ids_arr = to_numpy(getattr(b, "id", None)) if getattr(b, "id", None) is not None else None

                if xyxy is None or cls_arr is None or conf_arr is None:
                    continue
                if not isinstance(xyxy, np.ndarray) or xyxy.size == 0:
                    continue

                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x1 >= x2 or y1 >= y2:
                        continue

                    cls_idx = int(cls_arr[i]) if np.ndim(cls_arr) == 1 else int(cls_arr[i][0])
                    obj_name = model.names[cls_idx]
                    roi = frame[y1:y2, x1:x2]

                    if isinstance(masks, np.ndarray) and masks.ndim >= 1 and i < masks.shape[0]:
                        mask_src = masks[i]
                    else:
                        mask_src = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    mask_full = cv2.resize(mask_src, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Hard binary threshold — exclude soft boundary pixels that leak background
                    if mask_full.dtype in (np.float32, np.float64):
                        mask_full = (mask_full > 0.5).astype(np.uint8) * 255
                    mask_roi = mask_full[y1:y2, x1:x2]
                    # Apply mask to ROI so only segmented pixels are visible (background zeroed)
                    masked_roi = cv2.bitwise_and(roi, roi, mask=mask_roi if mask_roi.dtype == np.uint8 else (mask_roi > 0).astype(np.uint8))

                    obj = enrich_detected_object(
                        obj_name=obj_name,
                        masked_roi=masked_roi,
                        mask_roi=mask_roi,
                        x1=x1, x2=x2, y1=y1, y2=y2,
                        track_id=ids_arr[i] if ids_arr is not None else -1,
                        conf_val=conf_arr[i] if np.ndim(conf_arr) == 1 else conf_arr[i][0],
                        timestamp=timestamp,
                        camera_id=camera_id,
                        obj_index=len(doc["objects"]),
                        track_state={},  # file mode does not carry track_state across frames in the identical way (bulk op uses atomic upsert)
                        track_bulk_ops=track_bulk_ops,
                        person_attr_defer=person_attr_defer,
                        person_rois=person_rois,
                        person_attr_texts=person_attr_texts,
                        person_meta=person_meta,
                        attr_encoder=attr_encoder
                    )
                    doc["objects"].append(obj)

            # Batch OpenVINO attribute inference for deferred persons
            if person_attr_defer:
                rois_attr = [t[0] for t in person_attr_defer]
                masks_attr = [t[1] for t in person_attr_defer]
                batch_attrs = get_person_attributes_batch(rois_attr, masks_attr)
                for j, (mroi, _mask, c_info, track_id, obj_idx) in enumerate(person_attr_defer):
                    if j < len(batch_attrs):
                        doc["objects"][obj_idx]["person_attributes"] = batch_attrs[j]
                    c_name = c_info.get("color", "Unknown")
                    attr_text = attr_encoder.attributes_to_text(
                        doc["objects"][obj_idx].get("person_attributes") or {}, c_name, c_info
                    )
                    doc["objects"][obj_idx]["attribute_text"] = attr_text
                    person_rois.append(mroi)
                    person_attr_texts.append(attr_text)
                    person_meta.append({
                        "object_index": obj_idx,
                        "camera_id": int(camera_id),
                        "track_id": track_id,
                        "timestamp": timestamp,
                        "color": c_name,
                        "colors": c_info.get("colors", [c_name]),
                        "upper_body_colors": c_info.get("upper_body_colors", []),
                        "lower_body_colors": c_info.get("lower_body_colors", []),
                        "attribute_text": attr_text,
                    })

            # Batch persist track updates
            if track_bulk_ops:
                try:
                    tracks.bulk_write(track_bulk_ops)
                except Exception:
                    pass

            # Compute object count summaries
            try:
                pcount = 0
                ocounts: Dict[str, int] = {}
                for o in doc["objects"]:
                    if isinstance(o, dict):
                        n = str(o.get("object_name", "")).lower()
                        if n:
                            ocounts[n] = ocounts.get(n, 0) + 1
                            if n == "person":
                                pcount += 1
                doc["person_count"] = pcount
                doc["object_counts"] = ocounts
            except Exception:
                pass

            # Zone-level counts — mirrors live stream pipeline for full parity
            try:
                zones_list = _get_zones_for_camera(zones_col, int(camera_id))
                if zones_list and doc.get("objects"):
                    h, w = frame.shape[0], frame.shape[1]
                    doc["zone_counts"] = _compute_zone_counts(
                        doc["objects"], zones_list, w, h
                    )
            except Exception:
                pass

            if doc["objects"]:
                # Save snapshot
                try:
                    safe_ts = timestamp.replace(":", "-").replace(".", "-")
                    cam_dir = settings.SNAPSHOTS_DIR / f"camera_{camera_id}"
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = cam_dir / f"{safe_ts}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    doc["snapshot"] = f"/media/snapshots/camera_{camera_id}/{snap_path.name}"
                except Exception as e:
                    logger.warning("process_video_file: snapshot failed: {}", e)

                # Insert detection document
                det_id: Optional[Any] = None
                try:
                    ins = detections_col.insert_one(doc)
                    det_id = ins.inserted_id if ins is not None else None
                    detections_inserted += 1
                except Exception as e:
                    logger.warning("process_video_file: detection insert failed: {}", e)

                # Async SigLIP + attribute fusion → person FAISS
                if person_rois:
                    try:
                        task_submit(
                            _run_person_embeddings_async,
                            int(camera_id),
                            timestamp,
                            str(det_id) if det_id is not None else None,
                            [roi.copy() for roi in person_rois],
                            list(person_attr_texts),
                            [dict(m) for m in person_meta],
                        )
                    except Exception:
                        pass

                # Periodically persist person index
                now_ps = _time.time()
                if now_ps - last_person_save_ts >= 30.0:
                    try:
                        person_index.save()
                    except Exception:
                        pass
                    last_person_save_ts = now_ps

            frame_idx += 1

    finally:
        cap.release()
        # Final person index flush
        try:
            if person_index is not None:
                person_index.save()
        except Exception:
            pass

    logger.info(
        "process_video_file: finished camera_id=%s frames_processed=%d detections_inserted=%d",
        camera_id, frames_processed, detections_inserted,
    )
    return {
        "ok": True,
        "frames_processed": frames_processed,
        "detections_inserted": detections_inserted,
    }


# =========================================
if __name__ == "__main__":
    process_live_stream(camera_id=1, source="sample.mp4", location="Main Entrance")

