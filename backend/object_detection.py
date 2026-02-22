import cv2
import logging
import numpy as np
import datetime
from ultralytics import YOLO
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

logger = logging.getLogger(__name__)


def _get_zones_for_camera(zones_collection: Any, camera_id: int) -> List[Dict[str, Any]]:
    """Return list of zone docs for camera, with short TTL cache."""
    global _ZONES_CACHE
    now = datetime.datetime.now().timestamp()
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
logger.info("Loading YOLOv11m-seg model...")
model = YOLO(MODEL_PATH)
logger.info("YOLO model loaded successfully.")

# Defer OpenVINO model loading until needed to avoid import-time crashes
compiled_model = None
input_layer = None
output_layer = None

def ensure_openvino_loaded() -> bool:
    """
    Lazily loads the OpenVINO attributes model the first time it's needed.
    Returns True if the model is available, False otherwise.
    """
    global compiled_model, input_layer, output_layer
    if compiled_model is not None and input_layer is not None and output_layer is not None:
        return True
    try:
        # Import here to avoid failing at module import time if OpenVINO is misconfigured
        from openvino.runtime import Core as OVCore  # type: ignore
        from backend.app.services.model_loader import ensure_model_downloaded
        
        # Ensure model files exist before loading
        ensure_model_downloaded(OPENVINO_MODEL_XML)
        
        ie = OVCore()
        cm = ie.compile_model(model=OPENVINO_MODEL_XML, device_name="CPU")
        compiled_model = cm
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        logger.info("OpenVINO attributes model loaded successfully (lazy).")
        return True
    except Exception as e:
        compiled_model = None
        input_layer = None
        output_layer = None
        logger.warning("OpenVINO attributes model disabled (lazy): %s", e)
        return False


# =================================================================
# ADVANCED COLOR UTILITIES (CIEDE2000 Implementation)
# =================================================================
CSS3_COLORS = {
    'Red': '#FF0000', 'Green': '#008000', 'Blue': '#0000FF', 'Yellow': '#FFFF00',
    'Black': '#000000', 'White': '#FFFFFF', 'Purple': '#800080', 'Orange': '#FFA500',
    'Pink': '#FFC0CB', 'Brown': '#A52A2A', 'Gray': '#808080', 'Cyan': '#00FFFF',
    'Magenta': '#FF00FF', 'Lime': '#00FF00', 'Navy': '#000080', 'Olive': '#800000',
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


def get_dominant_color(roi, mask, k=NUM_CLUSTERS):
    """Extract dominant color from masked region using histogram on quantized LAB (fast, no K-Means)."""
    if roi is None or roi.size == 0 or mask is None or mask.size == 0:
        return (0, 0, 0)
    try:
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        mask_bool = mask.astype(bool) if mask.dtype != bool else (mask > 0.5)
        masked_pixels = lab_roi[mask_bool]
        n_pixels = masked_pixels.shape[0]
        if n_pixels == 0:
            return (0, 0, 0)
        if n_pixels < 10:
            mean_lab = np.mean(masked_pixels, axis=0)
            lab_arr = np.array([[mean_lab]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)[0][0]
            return tuple(map(int, rgb_color))
        # Quantize LAB into 8x8x8 bins; L in [0,255], A/B in [0,255] with 128 neutral
        L, A, B = masked_pixels[:, 0], masked_pixels[:, 1], masked_pixels[:, 2]
        qL = np.minimum(L // 32, 7)
        qA = np.minimum(np.maximum(A.astype(np.int32) + 128, 0) // 32, 7)
        qB = np.minimum(np.maximum(B.astype(np.int32) + 128, 0) // 32, 7)
        flat_idx = (qL.astype(np.int32) * 64 + qA * 8 + qB).astype(np.int32)
        hist = np.bincount(flat_idx, minlength=512)
        dominant_bin = int(np.argmax(hist))
        # Bin center back to LAB (uint8 for OpenCV: L 0-255, A/B 0-255)
        Lc = min(255, dominant_bin // 64 * 32 + 16)
        Ac = (dominant_bin // 8 % 8) * 32 + 16
        Bc = (dominant_bin % 8) * 32 + 16
        lab_arr = np.array([[[Lc, Ac, Bc]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)[0][0]
        return tuple(map(int, rgb_color))
    except Exception as e:
        logger.warning("Histogram color extraction failed: %s", e)
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
    Returns confidence scores instead of booleans for more nuanced data.
    """
    if person_roi is None or person_roi.size == 0:
        return {}
    # Ensure attributes model is available; otherwise skip gracefully
    if not ensure_openvino_loaded() or compiled_model is None or output_layer is None:
        return {}
    
    try:
        # 1. Apply mask to ROI to remove background noise
        mask_u8 = (person_mask * 255).astype(np.uint8) if person_mask.dtype != np.uint8 else person_mask
        mask_3ch = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        masked_roi = cv2.bitwise_and(person_roi, mask_3ch)

        # 2. Preprocess with letterboxing to preserve aspect ratio
        preprocessed_img = preprocess_for_attributes(masked_roi, target_size=(80, 160))

        # 3. Standard OpenVINO inference steps
        img = preprocessed_img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        preds = compiled_model([img])[output_layer]
        attrs = preds.flatten()

        # 4. Convert model logits to probabilities using sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # 5. Return confidence scores for richer data
        return {
            "male_confidence": round(float(sigmoid(attrs[0])), 2),
            "bag_confidence": round(float(sigmoid(attrs[1])), 2),
            "hat_confidence": round(float(sigmoid(attrs[2])), 2),
            "longhair_confidence": round(float(sigmoid(attrs[3])), 2),
            "longpants_confidence": round(float(sigmoid(attrs[4])), 2),
            "longsleeves_confidence": round(float(sigmoid(attrs[5])), 2),
            "coat_jacket_confidence": round(float(sigmoid(attrs[6])), 2),
        }
    except Exception as e:
        logger.warning("Failed to get person attributes: %s", e)
        return {}


def get_person_attributes_batch(
    person_rois: List[np.ndarray], person_masks: List[np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Batch infer person attributes for multiple ROIs in one OpenVINO call.
    Returns a list of attribute dicts (same format as get_person_attributes), one per ROI.
    """
    if not person_rois or not person_masks:
        return []
    if not ensure_openvino_loaded() or compiled_model is None or output_layer is None:
        return [{} for _ in person_rois]
    try:
        batch_arr: List[np.ndarray] = []
        for roi, mask in zip(person_rois, person_masks):
            if roi is None or roi.size == 0:
                batch_arr.append(np.zeros((3, 160, 80), dtype=np.float32))
                continue
            mask_u8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
            mask_3ch = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
            masked_roi = cv2.bitwise_and(roi, mask_3ch)
            preprocessed = preprocess_for_attributes(masked_roi, target_size=(80, 160))
            img = preprocessed.transpose((2, 0, 1)).astype(np.float32)
            batch_arr.append(img)
        batch = np.stack(batch_arr, axis=0).astype(np.float32)
        preds = compiled_model([batch])[output_layer]
        out: List[Dict[str, Any]] = []
        for i in range(preds.shape[0]):
            attrs = preds[i].flatten()
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            out.append({
                "male_confidence": round(float(sigmoid(attrs[0])), 2),
                "bag_confidence": round(float(sigmoid(attrs[1])), 2),
                "hat_confidence": round(float(sigmoid(attrs[2])), 2),
                "longhair_confidence": round(float(sigmoid(attrs[3])), 2),
                "longpants_confidence": round(float(sigmoid(attrs[4])), 2),
                "longsleeves_confidence": round(float(sigmoid(attrs[5])), 2),
                "coat_jacket_confidence": round(float(sigmoid(attrs[6])), 2),
            })
        return out
    except Exception as e:
        logger.warning("Batch person attributes failed: %s", e)
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
        logger.warning("Async person embeddings failed: %s", e)


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
        logger.warning("Auto VLM indexing failed for cam %s [%s..%s]: %s", camera_id, start_iso, end_iso, e)


# =========================================
# Threaded frame reader — keeps capture flowing
# even while YOLO detection blocks the main loop.
# =========================================
class ThreadedFrameReader:
    """Continuously reads frames on a background thread and feeds the MJPEG store."""

    def __init__(self, cap: "cv2.VideoCapture", camera_id: int):
        self._cap = cap
        self._camera_id = camera_id
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._stopped = False
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
            if ret:
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
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._cap.isOpened()


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
        logger.error("%s", msg)
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
            "last_seen": datetime.datetime.now().isoformat()
        }},
        upsert=True
    )
    logger.info("Camera %s registered in the database.", camera_id)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps / 2) # Process 2 frames per second
    frame_count = 0

    # Wrap capture in a threaded reader so frames flow to the MJPEG stream
    # even while YOLO inference blocks the detection loop.
    reader = ThreadedFrameReader(cap, camera_id)

    # Initialize BoT-SORT tracker (if available)
    tracker = None

    # Initialize multimodal encoders and person index
    visual_encoder = get_visual_embedder()
    attr_encoder = get_attribute_encoder()
    fusion = MultimodalFusion(
        visual_weight=getattr(settings, "FUSION_VISUAL_WEIGHT", 0.6),
        text_weight=getattr(settings, "FUSION_TEXT_WEIGHT", 0.4),
    )
    person_index = get_person_store(dim=1152)
    if BoTSORT is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            tracker = BoTSORT(
                reid_weights=os.getenv("REID_WEIGHTS", "osnet_x0_25_msmt17.pt"),
                device=device,
                half=(device == "cuda"),
            )
        except Exception as e:
            logger.warning("BoTSORT disabled: %s", e)
            tracker = None

    # In-memory track history across frames: track_id -> {first_seen, last_seen, frame_count}
    track_state: Dict[int, Dict[str, Any]] = {}

    # Auto-indexing window (seconds) — builds short clips from snapshots and indexes them
    try:
        window_sec = float(os.getenv("AUTO_INDEX_WINDOW_SEC", "15"))
    except Exception:
        window_sec = 15.0
    window_start_iso: Optional[str] = None

    logger.info("Starting live detection on Camera %s at '%s'...", camera_id, location)

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
                    logger.info("Stream dropped. Reconnecting (attempt %s/%s)...", reconnect_attempts, max_reconnects)
                    import time
                    for _ in range(20):
                        if _should_stop():
                            break
                        time.sleep(0.1)
                    if reader.reopen(source, cv2.CAP_FFMPEG):
                        logger.info("Reconnected to stream on attempt %s", reconnect_attempts)
                        read_failures = 0
                        continue
                    else:
                        logger.warning("Reconnect attempt %s failed", reconnect_attempts)
                        read_failures = 0
                        continue

                msg = f"No frames from source after {max_read_failures} attempts"
                logger.error("%s", msg)
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
                    {"$set": {"status": "active", "last_seen": datetime.datetime.now().isoformat(), "last_error": None}},
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
            timestamp = datetime.datetime.now().isoformat()

            doc = {"camera_id": camera_id, "timestamp": timestamp, "objects": []}

            # Collect person ROIs and attribute texts for async batch encoding (task queue)
            person_rois: List[np.ndarray] = []
            person_attr_texts: List[str] = []
            person_meta: List[Dict[str, object]] = []
            track_bulk_ops: List[Any] = []  # Batch track updates for bulk_write
            # Defer batch OpenVINO: (roi, mask_roi, color_name, track_id, obj_index) for each person with width >= MIN
            person_attr_defer: List[Tuple[np.ndarray, np.ndarray, str, int, int]] = []

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
                    mask_roi = mask_full[y1:y2, x1:x2]
                    
                    rgb_color = get_dominant_color(roi, mask_roi)
                    color_name = closest_color(rgb_color)

                    track_id = int(ids_arr[i]) if ids_arr is not None else -1
                    conf_val = float(conf_arr[i]) if np.ndim(conf_arr) == 1 else float(conf_arr[i][0])
                    
                    obj = {
                        "object_name": obj_name,
                        "track_id": track_id,
                        "confidence": round(conf_val, 2),
                        "color": color_name,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }

                    # Update track history (persistent across frames)
                    if track_id is not None and int(track_id) >= 0:
                        tid = int(track_id)
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
                        # Queue per-track update for batch bulk_write
                        track_bulk_ops.append(
                            UpdateOne(
                                {"camera_id": int(camera_id), "track_id": tid},
                                {
                                    "$setOnInsert": {"first_seen": st["first_seen"], "camera_id": int(camera_id)},
                                    "$set": {"last_seen": st["last_seen"]},
                                    "$inc": {"frame_count": 1},
                                },
                                upsert=True,
                            )
                        )

                    # --- IMPROVEMENT: Quality filter + batch OpenVINO ---
                    # Only run attribute inference if the person is wide enough; batch inference after loop
                    if obj_name == "person":
                        person_width = x2 - x1
                        obj_index = len(doc["objects"])
                        if person_width >= MIN_PERSON_WIDTH_PIXELS:
                            person_attr_defer.append((roi, mask_roi, color_name, int(track_id), obj_index))
                            obj["person_attributes"] = {}  # Filled after batch inference
                            obj["attribute_text"] = ""  # Filled after batch
                            # person_rois / person_attr_texts / person_meta added after batch below
                        else:
                            obj["person_attributes"] = {"status": "skipped_too_small"}
                            attr_text = attr_encoder.attributes_to_text({}, color_name)
                            obj["attribute_text"] = attr_text
                            person_rois.append(roi)
                            person_attr_texts.append(attr_text)
                            person_meta.append(
                                {
                                    "object_index": obj_index,
                                    "camera_id": int(camera_id),
                                    "track_id": int(track_id),
                                    "timestamp": timestamp,
                                    "color": color_name,
                                    "attribute_text": attr_text,
                                }
                            )

                    doc["objects"].append(obj)

            # Batch OpenVINO attribute inference for deferred persons; then add them to person_rois/person_attr_texts/person_meta
            if person_attr_defer:
                rois_attr = [t[0] for t in person_attr_defer]
                masks_attr = [t[1] for t in person_attr_defer]
                batch_attrs = get_person_attributes_batch(rois_attr, masks_attr)
                for j, (roi, _mask, color_name, track_id, obj_idx) in enumerate(person_attr_defer):
                    if j < len(batch_attrs):
                        doc["objects"][obj_idx]["person_attributes"] = batch_attrs[j]
                    attr_text = attr_encoder.attributes_to_text(
                        doc["objects"][obj_idx].get("person_attributes") or {}, color_name
                    )
                    doc["objects"][obj_idx]["attribute_text"] = attr_text
                    person_rois.append(roi)
                    person_attr_texts.append(attr_text)
                    person_meta.append(
                        {
                            "object_index": obj_idx,
                            "camera_id": int(camera_id),
                            "track_id": track_id,
                            "timestamp": timestamp,
                            "color": color_name,
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
                    logger.warning("Failed to save snapshot: %s", e)
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

    
    logger.info("Camera %s loop exited, cleaning up...", camera_id)
    try:
        cameras_collection.update_one(
            {"camera_id": camera_id},
            {"$set": {"status": "inactive"}},
        )
    except Exception:
        pass
    reader.release()
    logger.info("Camera %s processing completed.", camera_id)

# =========================================
if __name__ == "__main__":
    process_live_stream(camera_id=1, source="sample.mp4", location="Main Entrance")
