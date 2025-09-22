#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import argparse
import uuid
from datetime import datetime
from scipy.spatial import distance as dist

import ailia

# Import helper functions from YOLOv9 code
sys.path.append('./app/detection/util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import plot_results, write_predictions  # noqa
from nms_utils import batched_nms
from webcamera_utils import get_capture, get_writer  # noqa

from logging import getLogger
logger = getLogger(__name__)

# MongoDB imports
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")

# ==============================
# YOLOv9 Model Parameters
# ==============================
WEIGHT_YOLOV9E_PATH = 'yolov9e.onnx'
MODEL_YOLOV9E_PATH  = 'yolov9e.onnx.prototxt'
WEIGHT_YOLOV9C_PATH = 'yolov9c.onnx'
MODEL_YOLOV9C_PATH  = 'yolov9c.onnx.prototxt'
REMOTE_YOLOV9_PATH = 'https://storage.googleapis.com/ailia-models/yolov9/'

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

THRESHOLD = 0.25
IOU = 0.7
DETECTION_SIZE = 640

# ==============================
# Person Attributes Model Parameters
# ==============================
DEFAULT_ATTR_MODEL = "0234"
REMOTE_ATTR_PATH = "https://storage.googleapis.com/ailia-models/person-attributes-recognition-crossroad/"
WEIGHT_ATTR_PATH = "person-attributes-recognition-crossroad-{}.onnx".format(DEFAULT_ATTR_MODEL)
MODEL_ATTR_PATH  = "person-attributes-recognition-crossroad-{}.onnx.prototxt".format(DEFAULT_ATTR_MODEL)

# We expect 8 attribute scores; here we use these keys
ATTR_LABELS = ['is_male', 'has_bag', 'has_backpack', 'has_hat',
               'has_longsleeves', 'has_longpants', 'has_longhair', 'has_coat_jacket']

# ==============================
# Argument Parser
# ==============================
parser = get_base_parser('YOLOv9 Person Detection + Attributes + Tracking + MongoDB', 'input.jpg', 'output.png')
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for YOLOv9.'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='The detection IOU threshold for YOLOv9.'
)
parser.add_argument(
    '-ds', '--detection_size',
    default=DETECTION_SIZE, type=int,
    help='The detection input size for YOLOv9.'
)
parser.add_argument(
    '-m', '--model_type', default='v9e',
    choices=('v9e', 'v9c'),
    help='YOLOv9 model type to use.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='Use onnxruntime instead of ailia.Net.'
)
parser.add_argument(
    '-v', '--video', default=None,
    help='Path to video file or camera index (default: None for image mode).'
)
parser.add_argument(
    '--attr_model', default=DEFAULT_ATTR_MODEL, choices=['0230', '0234'],
    help='Person attributes model version (default: 0234).'
)
parser.add_argument(
    '--camera_id', default="cam0",
    help='Camera ID', type=str
)
parser.add_argument(
    '--env_id', default=0, type=int,
    help='Environment id for ailia.'
)
parser.add_argument(
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output predictions to txt or json file.'
)
parser.add_argument(
    '--savepath',
    default=None,
    help='Path to save the processed video or image results (default: None).'
)
args = update_parser(parser)

# Update attribute model paths if a different version is selected
WEIGHT_ATTR_PATH = "person-attributes-recognition-crossroad-{}.onnx".format(args.attr_model)
MODEL_ATTR_PATH  = "person-attributes-recognition-crossroad-{}.onnx.prototxt".format(args.attr_model)

# ==============================
# Simple Centroid Tracker for Persons
# ==============================
class CentroidTracker:
    def __init__(self, maxDisappeared=20, maxDistance=50):
        self.nextObjectID = 0
        self.objects = {}  # objectID -> (bbox, centroid)
        self.disappeared = {}  # objectID -> disappearance count
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, bbox):
        cX = int(bbox[0] + bbox[2] / 2)
        cY = int(bbox[1] + bbox[3] / 2)
        self.objects[self.nextObjectID] = (bbox, (cX, cY))
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w/2)
            cY = int(y + h/2)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for bbox in rects:
                self.register(bbox)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = np.array([self.objects[objectID][1] for objectID in objectIDs])
            D = dist.cdist(objectCentroids, inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = (rects[col], (inputCentroids[col][0], inputCentroids[col][1]))
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(rects[col])
        return self.objects

def match_tracker(detection_box, tracker_objects, max_distance=30):
    cX = detection_box[0] + detection_box[2] / 2
    cY = detection_box[1] + detection_box[3] / 2
    best_id = None
    best_dist = max_distance
    for tid, (bbox, centroid) in tracker_objects.items():
        d = np.linalg.norm(np.array([cX, cY]) - np.array(centroid))
        if d < best_dist:
            best_dist = d
            best_id = tid
    return best_id

# ==============================
# YOLOv9 Helper Functions (from provided code)
# ==============================
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def scale_boxes(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])
    return boxes

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_h, im_w, _ = img.shape
    size = args.detection_size
    r = min(size / im_h, size / im_w)
    oh, ow = int(round(im_h * r)), int(round(im_w * r))
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
    dh, dw = size - oh, size - ow
    stride = 32
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    img = normalize_image(img, normalize_type='255')
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def post_processing(preds, img, orig_shape):
    conf_thres = args.threshold
    iou_thres = args.iou
    xc = np.max(preds[:, 4:], axis=1) > conf_thres
    none_out = np.zeros((0, 6))
    x = preds[0].T[xc[0]]
    if not x.shape[0]:
        return none_out
    box, cls = np.split(x, [4], axis=1)
    box = xywh2xyxy(box)
    j = np.argmax(cls, axis=1)
    conf = cls[np.arange(len(cls)), j]
    x = np.concatenate((box, conf.reshape(-1,1), j.reshape(-1,1)), axis=1)
    n = x.shape[0]
    if not n:
        return none_out
    max_nms = 30000
    x = x[np.argsort(-x[:,4])[:max_nms]]
    c = x[:, 5]
    boxes, scores = x[:, :4], x[:, 4]
    i = batched_nms(boxes, scores, c, iou_thres)
    max_det = 300
    i = i[:max_det]
    preds = x[i]
    preds[:, :4] = np.round(scale_boxes(img.shape[2:], preds[:, :4], orig_shape))
    return preds

def predict(net, img):
    orig_shape = img.shape
    inp = preprocess(img)
    if not args.onnx:
        output = net.predict([inp])
    else:
        output = net.run([x.name for x in net.get_outputs()], {net.get_inputs()[0].name: inp})
    preds = output[0]
    preds = post_processing(preds, inp, orig_shape)
    return preds

def convert_to_detector_object(preds, im_w, im_h):
    det_objs = []
    for i in range(len(preds)):
        (x1, y1, x2, y2) = preds[i, :4]
        score = float(preds[i, 4])
        cls = int(preds[i, 5])
        obj = ailia.DetectorObject(
            category=COCO_CATEGORY[cls],
            prob=score,
            x=x1 / im_w,
            y=y1 / im_h,
            w=(x2 - x1) / im_w,
            h=(y2 - y1) / im_h,
        )
        det_objs.append(obj)
    return det_objs

# ==============================
# Person Attributes Functions
# ==============================
def crop_and_resize(img, x, y, w, h):
    if w * 2 < h:
        nw = h // 2
        x = x + (w - nw) // 2
        w = nw
    else:
        nh = w * 2
        y = y + (h - nh) // 2
        h = nh
    ih, iw, _ = img.shape
    if x < 0:
        w = w + x
        x = 0
    if y < 0:
        h = h + y
        y = 0
    if x + w > iw:
        w = iw - x
    if y + h > ih:
        h = ih - y
    cropped = img[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (80, 160))
    return resized, x, y, w, h

def draw_attributes(img, x, y, attributes, labels):
    n = min(len(attributes), len(labels))
    attr_text = ", ".join([f"{labels[i]}:{attributes[i]:.2f}" for i in range(n)])
    cv2.putText(img, attr_text, (x, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

# ==============================
# Recognition Functions (Video Mode with Tracking & MongoDB)
# ==============================

import os
def recognize_from_video(yolo_net, attr_net):
    cap = get_capture(args.video if args.video is not None else 0)
    assert cap.isOpened(), "Cannot open video source"
    writer = None
    if args.savepath and args.savepath != "output.png":
        f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        save_dir = os.path.dirname(args.savepath)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        writer = get_writer(args.savepath, f_h, f_w)
    im_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize MongoDB connection
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["SurveillanceAI"]
    collection = db["detected_objects"]

    # Generate a unique video id
    processed_video_path = args.savepath
    video_id = processed_video_path.split('/')[-1].split('.')[0]
    
    
    frame_count = 0
    tracker = CentroidTracker(maxDisappeared=20, maxDistance=50)
    
    print("Starting video processing. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        preds = predict(yolo_net, frame)
        det_objs = convert_to_detector_object(preds, im_w, im_h)
        res_img = frame.copy()

        # Lists to accumulate detection info for MongoDB
        frame_detections = []

        # Separate person and non-person detections
        person_detections = []  # list of tuples: (detector_object, bbox)
        for obj in det_objs:
            x_abs = int(obj.x * im_w)
            y_abs = int(obj.y * im_h)
            w_abs = int(obj.w * im_w)
            h_abs = int(obj.h * im_h)
            if obj.category == "person":
                person_detections.append((obj, (x_abs, y_abs, w_abs, h_abs)))
            else:
                # Draw non-person detection
                cv2.rectangle(res_img, (x_abs, y_abs), (x_abs+w_abs, y_abs+h_abs), (0,255,0), 2)
                label = f"{obj.category} {obj.prob:.2f}"
                cv2.putText(res_img, label, (x_abs, y_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                detection = {
                    "object_id": str(uuid.uuid4()),
                    "class": obj.category,
                    "bounding_box": {
                        "x_min": x_abs,
                        "y_min": y_abs,
                        "x_max": x_abs + w_abs,
                        "y_max": y_abs + h_abs
                    },
                    "confidence": obj.prob,
                    "attributes": {}  # No extra attributes for non-person objects
                }
                frame_detections.append(detection)
        
        # Update tracker with person bounding boxes
        person_boxes = [box for (_, box) in person_detections]
        tracker_objects = tracker.update(person_boxes)
        
        # Process each person detection
        for (obj, box) in person_detections:
            x_abs, y_abs, w_abs, h_abs = box
            tracking_id = match_tracker(box, tracker_objects, max_distance=50)
            # Run attribute recognition for person
            cropped, cx, cy, cw, ch = crop_and_resize(frame, x_abs, y_abs, w_abs, h_abs)
            input_data = cropped.transpose(2, 0, 1).astype(np.float32)
            result = attr_net.run(input_data)
            # Extract attribute scores; here we assume 8 values are returned
            attributes_scores = result[0][0][:min(8, len(result[0][0]))]
            draw_attributes(res_img, x_abs, y_abs, attributes_scores, ATTR_LABELS)
            # Draw bounding box and tracking id on frame
            cv2.rectangle(res_img, (x_abs, y_abs), (x_abs+w_abs, y_abs+h_abs), (255,0,0), 2)
            label = f"person {obj.prob:.2f} ID:{tracking_id if tracking_id is not None else 'N/A'}"
            cv2.putText(res_img, label, (x_abs, y_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
            # Convert attribute scores to booleans (using 0.5 threshold) and add color placeholders
            person_attributes = {}
            for i, key in enumerate(ATTR_LABELS[:len(attributes_scores)]):
                person_attributes[key] = True if attributes_scores[i] > 0.5 else False

            # Placeholders for colors (could be enhanced with an actual color recognition module)
            person_attributes["top_color"] = None
            person_attributes["bottom_color"] = None
            person_attributes["headwear_color"] = None

            detection = {
                "object_id": str(uuid.uuid4()),
                "class": "person",
                "bounding_box": {
                    "x_min": x_abs,
                    "y_min": y_abs,
                    "x_max": x_abs + w_abs,
                    "y_max": y_abs + h_abs
                },
                "confidence": obj.prob,
                "attributes": person_attributes,
                "tracking_id": tracking_id
            }
            frame_detections.append(detection)
        
        # Build frame document
        frame_doc = {
            "video_id": video_id,
            "camera_id": args.camera_id,
            "frame_timestamp": datetime.utcnow(),
            "frame_number": frame_count,
            "detections": frame_detections
        }
        # Insert document into MongoDB
        collection.insert_one(frame_doc)
        
        cv2.imshow("Detection & Tracking", res_img)
        if writer is not None:
            writer.write(res_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Processing finished.")

def recognize_from_image(yolo_net, attr_net):
    # For image mode, we only display results (MongoDB logging is omitted here)
    for image_path in args.input:
        logger.info("Processing " + image_path)
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to load image " + image_path)
            continue
        preds = predict(yolo_net, img)
        det_objs = convert_to_detector_object(preds, img.shape[1], img.shape[0])
        res_img = img.copy()
        for obj in det_objs:
            x_abs = int(obj.x * img.shape[1])
            y_abs = int(obj.y * img.shape[0])
            w_abs = int(obj.w * img.shape[1])
            h_abs = int(obj.h * img.shape[0])
            if obj.category == "person":
                cropped, cx, cy, cw, ch = crop_and_resize(img, x_abs, y_abs, w_abs, h_abs)
                input_data = cropped.transpose(2, 0, 1).astype(np.float32)
                result = attr_net.run(input_data)
                attributes_scores = result[0][0][:min(8, len(result[0][0]))]
                draw_attributes(res_img, x_abs, y_abs, attributes_scores, ATTR_LABELS)
            cv2.rectangle(res_img, (x_abs, y_abs), (x_abs+w_abs, y_abs+h_abs), (0,255,0), 2)
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        cv2.imwrite(savepath, res_img)
        logger.info("Saved result at " + savepath)
    print("Image processing finished.")

def main():
    dic_model = {
        'v9e': (WEIGHT_YOLOV9E_PATH, MODEL_YOLOV9E_PATH),
        'v9c': (WEIGHT_YOLOV9C_PATH, MODEL_YOLOV9C_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = dic_model[args.model_type]
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_YOLOV9_PATH)
    check_and_download_models(WEIGHT_ATTR_PATH, MODEL_ATTR_PATH, REMOTE_ATTR_PATH)
    if not args.onnx:
        yolo_net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        yolo_net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
    attr_net = ailia.Net(MODEL_ATTR_PATH, WEIGHT_ATTR_PATH, env_id=args.env_id)
    if args.video is not None:
        recognize_from_video(yolo_net, attr_net)
    else:
        recognize_from_image(yolo_net, attr_net)

if __name__ == '__main__':
    main()
