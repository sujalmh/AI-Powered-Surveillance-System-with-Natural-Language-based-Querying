import os
from flask import Blueprint, send_from_directory, current_app, jsonify

main_bp = Blueprint('main', __name__)

@main_bp.route('/uploads/<filename>', methods=['GET'])
def uploaded_video(filename):
    """Serve uploaded video files."""
    try:
        return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@main_bp.route('/videos/<filename>', methods=['GET'])
def output_video(filename):
    """Serve output video files."""
    try:
        return send_from_directory(current_app.config["VIDEO_FOLDER"], filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
