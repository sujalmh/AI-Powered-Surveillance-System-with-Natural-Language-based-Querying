from flask import Flask, Response, jsonify, Blueprint, request
import cv2
import threading
from app.detection.detect import start_analysis
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid
from app.ai.gemini_vid_analysis import start_gemini_analysis
load_dotenv()
mongo_uri = os.getenv('MONGO_URI')

# available_cameras = {}
# camera_locks = {}

mongo_client = MongoClient(mongo_uri)
db = mongo_client["SurveillanceAI"]
camera_collection = db["camera_feeds"]
reports_collection = db["reports"]

camera_bp = Blueprint('cam', __name__)

# def detect_cameras(max_cams=1):
#     for cam_id in range(max_cams):
#         cap = cv2.VideoCapture(cam_id)
#         if cap.isOpened():
#             available_cameras[f"cam{cam_id}"] = cap
#             camera_locks[f"cam{cam_id}"] = threading.Lock()
#         else:
#             cap.release()

# detect_cameras()

# @camera_bp.route('/cameras', methods=['GET'])
# def get_cameras():
#     return jsonify(list(available_cameras.keys()))

# @camera_bp.route('/stream/<camera_id>', methods=['GET'])
# def stream_camera(camera_id):
#     if camera_id not in available_cameras:
#         return jsonify({'error': 'Camera not found'}), 404

#     def generate_frames(cam_id):
#         cap = available_cameras[cam_id]
#         lock = camera_locks[cam_id]
#         while True:
#             with lock:
#                 success, frame = cap.read()
#             if not success:
#                 break
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     return Response(generate_frames(camera_id),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

CAMERA_ROOT = "cameras"

@camera_bp.route('/add/camera', methods=['POST'])
def add_camera():
    location = request.form.get('location')
    file = request.files.get('file')
    camera_name = request.form.get('name')
    resolution = request.form.get('resolution')
    fps = request.form.get('fps')
    status = request.form.get('status') == 'true'
    audio = request.form.get('audio') == 'true'

    if not location or not file:
        return jsonify({'error': 'Missing location or video_file'}), 400

    # Ensure folder exists
    location_path = os.path.join(CAMERA_ROOT, secure_filename(location))
    os.makedirs(location_path, exist_ok=True)

    # Save the file
    # Generate unique camera ID
    camera_id = str(uuid.uuid4())
    filename = secure_filename(camera_id + ".mp4")
    print(f"{filename=}")
    save_path = os.path.join(location_path, filename)
    file.save(save_path)

    # Test if video can be opened
    cap = cv2.VideoCapture(save_path)
    if not cap.isOpened():
        return jsonify({'error': 'Video file saved but cannot be opened'}), 500
    cap.release()

    # Store metadata in MongoDB
    camera_doc = {
        "camera_id": camera_id,
        "location": location,
        "camera_name": camera_name,
        "filename": filename,
        "resolution": resolution,
        "fps": fps,
        "status": status,
        "audio": audio,
        "storage": os.path.getsize(save_path),
        "path": save_path,
        "added_on": datetime.utcnow(),
        "active": False,
        "alerts": False
    }

    camera_collection.insert_one(camera_doc)

    return jsonify({
        'message': f'Camera added at location: {location}',
        'camera_id': camera_id,
        'path': save_path
    }), 200

@camera_bp.route('/list/cameras', methods=['GET'])
def list_cameras():
    cameras = list(camera_collection.find({}, {'_id': 0}))  # exclude MongoDB's internal _id
    for camera in cameras:
        if 'storage' in camera:
            camera['storage'] = round(camera['storage'] / (1024 ** 3), 4)
    result = []
    print(f"{cameras=}")
    for camera in cameras:
        res = {}
        res["id"] = camera.get('camera_id')
        res["name"] = camera.get('camera_name')
        res["location"] = camera.get('location')
        res["type"] = "indoor"
        res["zone"] = "south"
        res["status"] = camera.get('status')
        res["hasAlerts"] = camera.get('alerts')
        res["active"] = camera.get('active')
        res["storage"] = camera.get('storage')
        result.append(res)

    print(f"{result=}")
    return jsonify(result), 200

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@camera_bp.route('/stream/<camera_id>', methods=['GET'])
def stream_camera(camera_id):
    camera = camera_collection.find_one({'camera_id': camera_id})
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404

    video_path = camera.get('path')
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file missing from server'}), 404

    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@camera_bp.route('/activate/<camera_id>', methods=['POST'])
def activate_camera(camera_id):
    camera = camera_collection.find_one({'camera_id': camera_id})
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404

    # Update the camera status in the database
    
    
    video_path = camera.get('path')
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Video file missing or path invalid'}), 404

    try:
        # Call your analysis function (custom to your pipeline)
        result = start_gemini_analysis(video_path).model_dump()
        result['camera_id'] = camera_id
        print(f"Analysis result: {result}")
        inserted_id = reports_collection.insert_one(result).inserted_id
        print(f"Inserted report with ID: {inserted_id}")
        camera_collection.update_one({'camera_id': camera_id}, {'$set': {'active': True}})
        return jsonify({'message': 'Analysis started', 'result': str(result)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@camera_bp.route('/analyze/<camera_id>', methods=['POST'])
def analyze_camera(camera_id):
    # Look up the camera in the MongoDB collection
    camera = camera_collection.find_one({'camera_id': camera_id})
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404

    video_path = camera.get('path')
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Video file missing or path invalid'}), 404

    try:
        # Call your analysis function (custom to your pipeline)
        result = start_analysis(video_path, camera_id)
        return jsonify({'message': 'Analysis started', 'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
