import sys
import time
from collections import OrderedDict

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('./app/detection/util')
from model_utils import check_and_download_models
from detector_utils import plot_results, load_image, write_predictions
from webcamera_utils import get_capture, get_writer, calc_adjust_fsize

# logger
from logging import getLogger
logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clothing-detection/'

DATASETS_MODEL_PATH = OrderedDict([
    ('modanet', ['yolov3-modanet.opt.onnx', 'yolov3-modanet.opt.onnx.prototxt']),
    ('df2', ['yolov3-df2.opt.onnx', 'yolov3-df2.opt.onnx.prototxt'])
])

DATASETS_CATEGORY = {
    'modanet': [
        "bag", "belt", "boots", "footwear", "outer", "dress", "sunglasses",
        "pants", "top", "shorts", "skirt", "headwear", "scarf/tie"
    ],
    'df2': [
        "short sleeve top", "long sleeve top", "short sleeve outwear",
        "long sleeve outwear", "vest", "sling", "shorts", "trousers", "skirt",
        "short sleeve dress", "long sleeve dress", "vest dress", "sling dress"
    ]
}
THRESHOLD = 0.39
IOU = 0.4
DETECTION_WIDTH = 416
SAVE_IMAGE_PATH = 'output_with_bboxes.png'

# Default dataset
DATASET = 'modanet'
weight_path, model_path = DATASETS_MODEL_PATH[DATASET]
category = DATASETS_CATEGORY[DATASET]

# ======================
# Secondary Functions
# ======================

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def preprocess(img, resize):
    image = Image.fromarray(img)
    boxed_image = letterbox_image(image, (resize, resize))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data = np.transpose(image_data, [0, 3, 1, 2])
    return image_data

def post_processing(img_shape, all_boxes, all_scores, indices):
    indices = indices.astype(int)

    bboxes = []
    for idx_ in indices[0]:
        cls_ind = idx_[1]
        score = all_scores[tuple(idx_)]

        idx_1 = (idx_[0], idx_[2])
        box = all_boxes[idx_1]
        y, x, y2, x2 = box
        w = (x2 - x) / img_shape[1]
        h = (y2 - y) / img_shape[0]
        x /= img_shape[1]
        y /= img_shape[0]

        r = ailia.DetectorObject(
            category=cls_ind, prob=score,
            x=x, y=y, w=w, h=h,
        )
        bboxes.append(r)

    return bboxes

def draw_bounding_boxes(image, detect_objects, categories):
    """Draw bounding boxes and labels on the image."""
    img_height, img_width = image.shape[:2]
    result_img = image.copy()
    
    for obj in detect_objects:
        x1 = int(obj.x * img_width)
        y1 = int(obj.y * img_height)
        x2 = int((obj.x + obj.w) * img_width)
        y2 = int((obj.y + obj.h) * img_height)
        
        # Draw rectangle
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{categories[obj.category]} ({obj.prob:.2f})"
        cv2.putText(result_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    return result_img

# ======================
# Main functions
# ======================

def detect_objects(img, detector):
    img_shape = img.shape[:2]

    # initial preprocesses
    img_processed = preprocess(img, resize=DETECTION_WIDTH)

    # feedforward
    all_boxes, all_scores, indices = detector.predict({
        'input_1': img_processed,
        'image_shape': np.array([img_shape], np.float32),
        'layer.score_threshold': np.array([THRESHOLD], np.float32),
        'iou_threshold': np.array([IOU], np.float32),
    })

    # post processes
    detect_object = post_processing(img_shape, all_boxes, all_scores, indices)

    return detect_object

def recognize_from_image(image_frame, detector):
    # prepare input data
    x = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)

    # inference
    logger.info('Start inference...')
    detect_objects_list = detect_objects(x, detector)

    # Draw bounding boxes and save the image
    result_img = draw_bounding_boxes(image_frame, detect_objects_list, category)
    cv2.imwrite(SAVE_IMAGE_PATH, result_img)
    logger.info(f'Image with bounding boxes saved at: {SAVE_IMAGE_PATH}')

    # extract clothing items and their bounding coordinates
    results = []
    for obj in detect_objects_list:
        clothing_item = category[obj.category]
        # Calculate bounding coordinates in pixel values
        img_height, img_width = image_frame.shape[:2]
        x1 = int(obj.x * img_width)
        y1 = int(obj.y * img_height)
        x2 = int((obj.x + obj.w) * img_width)
        y2 = int((obj.y + obj.h) * img_height)
        results.append({
            "item": clothing_item,
            "bounding_box": [x1, y1, x2, y2]  # [top-left-x, top-left-y, bottom-right-x, bottom-right-y]
        })

    return results

def recognize_from_video(video, detector):
    capture = get_capture(video)
    writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_object = detect_objects(x, detector)
        res_img = plot_results(detect_object, frame, category)
        cv2.imshow('frame', res_img)
        frame_shown = True

        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

def detect_cloths(image_frame=None, video=None):
    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    detector = ailia.Net(model_path, weight_path)
    id_image_shape = detector.find_blob_index_by_name("image_shape")
    detector.set_input_shape((1, 3, DETECTION_WIDTH, DETECTION_WIDTH))
    detector.set_input_blob_shape((1, 2), id_image_shape)

    if video is not None:
        # video mode
        recognize_from_video(video, detector)
    elif image_frame is not None:
        # image mode
        results = recognize_from_image(image_frame, detector)
        logger.info('Script finished successfully.')
        return results

    logger.info('No input provided.')
