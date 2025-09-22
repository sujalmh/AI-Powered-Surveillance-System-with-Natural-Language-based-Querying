import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from app.services.metadata_store import store_video_metadata
from app.services.query_engine import send_response

import os


api_bp = Blueprint('api', __name__)

@api_bp.route('/upload', methods=['POST'])
def upload_video():
    print("[INFO] Received request for object detection")
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], video.filename)
    video.save(upload_path)

    metadata = store_video_metadata(upload_path, video.filename)

    return jsonify({'message': 'Video uploaded successfully', 'metadata': metadata}), 200



@api_bp.route('/query', methods=['POST'])
def handle_query():
    """Handles user queries related to video content."""
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
    
    try:
        response = send_response(query)
    except Exception as e:
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500
    
    if not response:
        return jsonify({"video": None, "message": "No relevant video found"})

    filename = os.path.basename(response["clip_path"])
    res_video = f"http://127.0.0.1:5000/videos/{filename}"
    reply = response["reply"]

    return jsonify({"video": res_video, "message": reply})
