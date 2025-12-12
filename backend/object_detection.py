import cv2
import numpy as np
import datetime
from ultralytics import YOLO
from sklearn.cluster import KMeans
from pymongo import MongoClient
Core = None  # lazy import in ensure_openvino_loaded
import math
import os
import threading
from typing import Union, Optional, Dict, List, Tuple, Any
from backend.app.config import settings
import torch
try:
    import boxmot  # type: ignore
    BoTSORT = getattr(boxmot, "BoTSORT", None)  # BoT-SORT tracker
except Exception:  # pragma: no cover
    BoTSORT = None  # type: ignore
from backend.app.services.frame_store import update_frame
from backend.app.services.clip_builder import build_clip_from_snapshots
from backend.app.services.sem_indexer import index_clip
from backend.app.db.mongo import app_settings, tracks
from backend.app.services.visual_embedder import get_visual_embedder
from backend.app.services.attribute_encoder import get_attribute_encoder
from backend.app.services.person_store import get_person_store
from backend.app.services.fusion import MultimodalFusion
from backend.app.services.alert_engine import evaluate_realtime

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
# CONFIGURATION
# =========================================
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11m-seg.pt")   # YOLOv11 segmentation model for detection
OPENVINO_MODEL_XML = os.getenv(
    "OPENVINO_MODEL_XML",
    "intel/person-attributes-recognition-crossroad-0238/FP32/person-attributes-recognition-crossroad-0238.xml"
)
CONF_THRESHOLD = 0.5
NUM_CLUSTERS = 5
MIN_PERSON_WIDTH_PIXELS = 60 # NEW: Minimum width of a person to trigger attribute recognition

# MongoDB connection (use env; never hardcode secrets)
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(mongo_uri)
db = client[os.getenv("MONGO_DB_NAME", "ai_surveillance")]

# =========================================
# LOAD MODELS
# =========================================
print("🔄 Loading YOLOv11m-seg model...")
model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded successfully.")

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
        ie = OVCore()
        cm = ie.compile_model(model=OPENVINO_MODEL_XML, device_name="CPU")
        compiled_model = cm
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        print("✅ OpenVINO attributes model loaded successfully (lazy).")
        return True
    except Exception as e:
        compiled_model = None
        input_layer = None
        output_layer = None
        print(f"⚠️ OpenVINO attributes model disabled (lazy): {e}")
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
    """Extract dominant color from masked region using K-Means in LAB space."""
    if roi is None or roi.size == 0 or mask is None or mask.size == 0:
        return (0, 0, 0)
    
    try:
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        mask_bool = (mask > 0.5)
        
        masked_pixels = lab_roi[mask_bool]
        
        if masked_pixels.shape[0] < k:
            if masked_pixels.shape[0] == 0: return (0, 0, 0)
            # Not enough pixels for clustering, find the mean color instead
            mean_lab = np.mean(masked_pixels, axis=0)
            lab_arr = np.array([[mean_lab]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)[0][0]
            return tuple(map(int, rgb_color))
        
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        kmeans.fit(masked_pixels)
        
        dominant_lab = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        lab_arr = np.array([[dominant_lab]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)[0][0]
        
        return tuple(map(int, rgb_color))
    except Exception as e:
        print(f"⚠️ Warning: K-Means or color conversion failed: {e}")
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
        print(f"⚠️ Warning: Failed to get person attributes: {e}")
        return {}


# =========================================
# AUTO INDEXING HELPERS (VLM)
# =========================================
def _get_indexing_mode() -> str:
    try:
        doc = app_settings.find_one({"key": "indexing_mode"}, {"_id": 0, "value": 1})
        val = (doc or {}).get("value")
        return val if val in ("structured", "semantic", "both") else "both"
    except Exception:
        return "both"


def _auto_build_and_index(camera_id: int, start_iso: str, end_iso: str, every_sec: float = 1.0) -> None:
    """
    Build a clip from snapshots between [start_iso, end_iso] and index it semantically.
    Non-blocking: intended for background thread usage.
    """
    try:
        # Respect runtime mode: skip semantic work if structured-only
        if _get_indexing_mode() == "structured":
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
        print(f"⚠️ Auto VLM indexing failed for cam {camera_id} [{start_iso}..{end_iso}]: {e}")


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
    global db
    cameras_collection = db["cameras"]
    detections_collection = db["detections"]

    # Use DirectShow for local webcams on Windows to improve reliability
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        msg = f"Cannot open camera source {source}"
        print(f"❌ {msg}")
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
    print(f"✅ Camera {camera_id} registered in the database.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps / 2) # Process 2 frames per second
    frame_count = 0

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
            print(f"⚠️ BoTSORT disabled: {e}")
            tracker = None

    # In-memory track history across frames: track_id -> {first_seen, last_seen, frame_count}
    track_state: Dict[int, Dict[str, Any]] = {}

    # Auto-indexing window (seconds) — builds short clips from snapshots and indexes them
    try:
        window_sec = float(os.getenv("AUTO_INDEX_WINDOW_SEC", "15"))
    except Exception:
        window_sec = 15.0
    window_start_iso: Optional[str] = None

    print(f"📹 Starting live detection on Camera {camera_id} at '{location}'...")
    # Allow some tolerance for webcams/RTSP streams that may take a moment to provide frames
    read_failures = 0
    max_read_failures = 60  # ~2 seconds at 30fps (adjust as needed)

    while True:
        if stop_event is not None and stop_event.is_set():
            print("🛑 Stop signal received.")
            break
        ret, frame = cap.read()
        if not ret:
            read_failures += 1
            if read_failures > max_read_failures:
                msg = f"No frames from source after {max_read_failures} attempts"
                print(f"🔴 {msg}")
                try:
                    cameras_collection.update_one(
                        {"camera_id": camera_id},
                        {"$set": {"status": "error", "last_error": msg}},
                        upsert=True,
                    )
                except Exception:
                    pass
                break
            # brief delay before next read attempt; helps some drivers/RTSP warm-up
            cv2.waitKey(1)
            continue
        else:
            read_failures = 0

        # Heartbeat: mark camera active and update last_seen on successful frame
        try:
            cameras_collection.update_one(
                {"camera_id": camera_id},
                {"$set": {"status": "active", "last_seen": datetime.datetime.now().isoformat(), "last_error": None}},
                upsert=True,
            )
        except Exception:
            pass

        # Update live preview frame (MJPEG)
        try:
            # Optionally resize for bandwidth; keep original for now
            encode_ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if encode_ok:
                update_frame(camera_id, jpg.tobytes())
        except Exception as _e:
            # Do not interrupt pipeline on preview failure
            pass

        if frame_count % frame_interval == 0:
            # Run YOLO with or without external tracker
            if tracker is None:
                results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)
            else:
                results = model(frame)  # list of Results with .boxes and .masks
            timestamp = datetime.datetime.now().isoformat()

            doc = {"camera_id": camera_id, "timestamp": timestamp, "objects": []}

            # Accumulate person-level embeddings to add after Mongo insert (to attach detection_id/object_index)
            fused_vecs: List[np.ndarray] = []
            fused_meta: List[Dict[str, object]] = []

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
                        # Persist per-track summary in Mongo (upsert)
                        try:
                            tracks.update_one(
                                {"camera_id": int(camera_id), "track_id": tid},
                                {
                                    "$setOnInsert": {"first_seen": st["first_seen"], "camera_id": int(camera_id)},
                                    "$set": {"last_seen": st["last_seen"]},
                                    "$inc": {"frame_count": 1},
                                },
                                upsert=True,
                            )
                        except Exception:
                            pass

                    # --- IMPROVEMENT: Quality filter based on OpenVINO docs ---
                    # Only run attribute inference if the person is wide enough
                    attributes: Dict[str, Any] = {}
                    if obj_name == "person":
                        person_width = x2 - x1
                        if person_width >= MIN_PERSON_WIDTH_PIXELS:
                            attributes = get_person_attributes(roi, mask_roi)
                            obj["person_attributes"] = attributes
                        else:
                            # Optionally, note that attributes were skipped
                            obj["person_attributes"] = {"status": "skipped_too_small"}

                        # Build person-level fused embedding (SigLIP visual + attribute text)
                        try:
                            # Visual embedding from masked ROI
                            # Ensure mask applies: keep ROI; mask already applied for attributes, but reuse raw roi for encoder
                            v_emb = visual_encoder.encode([roi])
                            if v_emb.shape[0] == 1:
                                v_vec = v_emb[0]
                            else:
                                v_vec = np.zeros((768,), dtype=np.float32)

                            # Attribute text + text embedding
                            attr_text = attr_encoder.attributes_to_text(attributes or {}, color_name)
                            t_vec = attr_encoder.encode_text(attr_text)

                            # Fuse to 1152-dim vector
                            f_vec = fusion.fuse(v_vec, t_vec)
                            obj["attribute_text"] = attr_text  # optional for traceability

                            # Defer add to person FAISS until we have detection_id
                            fused_vecs.append(f_vec.astype(np.float32))
                            fused_meta.append(
                                {
                                    "object_index": len(doc["objects"]),  # current index before append
                                    "camera_id": int(camera_id),
                                    "track_id": int(track_id),
                                    "timestamp": timestamp,
                                    "color": color_name,
                                    "attribute_text": attr_text,
                                }
                            )
                        except Exception as _ex:
                            # Do not block detection on embedding errors
                            pass


                    doc["objects"].append(obj)

            if doc["objects"]:
                try:
                    safe_ts = timestamp.replace(":", "-").replace(".", "-")
                    cam_dir = settings.SNAPSHOTS_DIR / f"camera_{camera_id}"
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = cam_dir / f"{safe_ts}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    doc["snapshot"] = f"/media/snapshots/camera_{camera_id}/{snap_path.name}"
                except Exception as e:
                    print(f"⚠️ Warning: Failed to save snapshot: {e}")
                ins = detections_collection.insert_one(doc)
                det_id = ins.inserted_id if ins is not None else None

                # Evaluate real-time alert rules with current frame context
                try:
                    evaluate_realtime(
                        camera_id=int(camera_id),
                        timestamp_iso=timestamp,
                        objects=doc.get("objects", []),
                        frame_bgr=frame,
                        snapshot_url=doc.get("snapshot"),
                    )
                except Exception as _e:
                    # Do not interrupt detection loop on alert evaluation errors
                    pass

                # Flush person-level vectors to FAISS (with detection_id/object_index linkage)
                try:
                    if fused_vecs:
                        arr = np.vstack(fused_vecs).astype(np.float32)
                        metas = []
                        for m in fused_meta:
                            mi = dict(m)
                            mi["detection_id"] = str(det_id) if det_id is not None else None
                            metas.append(mi)
                        person_index.vector_add(arr, metas, save=True)
                except Exception:
                    # Ignore person-index failures; structured path must not be interrupted
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
                            # Spawn background job so detection loop is not blocked
                            threading.Thread(
                                target=_auto_build_and_index,
                                args=(camera_id, window_start_iso, timestamp, 1.0),
                                daemon=True,
                            ).start()
                            # Start next window from current timestamp
                            window_start_iso = timestamp
                except Exception as _e:
                    # Ignore auto-index scheduling errors
                    pass

        frame_count += 1
        
        if show_window:
            cv2.imshow(f"Camera {camera_id} - {location}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cameras_collection.update_one(
        {"camera_id": camera_id},
        {"$set": {"status": "inactive"}}
    )
    
    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    print(f"✅ Camera {camera_id} processing completed.")

# =========================================
if __name__ == "__main__":
    process_live_stream(camera_id=1, source="sample.mp4", location="Main Entrance")
