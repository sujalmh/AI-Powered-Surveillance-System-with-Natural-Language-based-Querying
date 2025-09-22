from flask import Blueprint, send_from_directory, current_app, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
import uuid
from datetime import datetime
import os
import openai
from dotenv import load_dotenv
from app.ai.generate_query import ask_question
from app.ai.openai_search import get_relevant_report
load_dotenv()

chat_bp = Blueprint('chat', __name__)


# MongoDB connection
MONGODB_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGODB_URI)
db = client.get_database("chat")
sessions_col = db.sessions
feeds_collection = client["SurveillanceAI"]["camera_feeds"]

openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.chat_models import ChatOpenAI

# Use a lightweight model for fast explanation generation
nl_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def generate_natural_explanation(user_query: str, num_clips: int) -> str:
    explanation_prompt = f"""
    User asked: "{user_query}"
    There are {num_clips} relevant video clips.
    Write a natural language sentence summarizing this.
    """
    return nl_llm.predict(explanation_prompt).strip()

def get_video_path_from_camera_id(camera_id):
    print(feeds_collection)
    report = feeds_collection.find_one({"camera_id": camera_id})
    if not report:
        raise ValueError(f"No report found for camera_id: {camera_id}")
    print(f"Report: {report}")
    return report.get("path", "Unknown path"), report.get("location", "Unknown location"), report.get("camera_name", "Unknown camera name")


def generate_ai_response(session_messages):
    """
    Generate an AI response using OpenAI's ChatCompletion API.
    Expects `session_messages` to be a list of dicts with `role` and `content`.
    """
    
    # result = ask_question(session_messages)
    results = get_relevant_report(session_messages)
    print(f"Result from ask_question: {results}")
    if not results:
        return {"id": str(uuid.uuid4()), "role": "ai", "content": "Sorry, I couldn't find any relevant clips.", "timestamp": "1232:1342801:"}
    
    clips = []
    ai_messages = []
    for result in results.results:
        print(f"Result: {result}")

        try:
            video_path, location, camera_name = get_video_path_from_camera_id(result.camera_id)
            clips.append([video_path, location, camera_name])
        except ValueError as e:
            video_path = "Unknown path"
        ai_messages.append(result.relevance_reason)
    
    videos = []
    for clip in clips:
        # clip.replace("\\", "/")
        clip = "http://localhost:5000/chat/video/" + clip[0]
        videos.append({"id": str(uuid.uuid4()), "title": clip[2],"location": clip[1], "timestamp": "1232:1342801:", "url": clip})
    
    res = {"id": str(uuid.uuid4()), "role": "ai", "content": '\n\n'.join(ai_messages), "timestamp": "1232:1342801:", "videos": videos}
    print(f"AI response: {res}")
    return res

@chat_bp.route("/sessions", methods=["POST"])
def create_session():
    """
    Create a new chat session. Optionally include an initial message in the body:
    {
      "message": "your first question"
    }
    Returns the generated session_id.
    """
    data = request.get_json() or {}
    initial_text = data.get("message", None)
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    system_initial_message = {
        "id": str(uuid.uuid4()),
        "role": "system",
        "content": "Hello! I'm your AI video assistant. Ask me about any footage or events, and I'll find relevant clips for you.",
        "timestamp": now
    }
    # Build initial messages list
    messages = []
    messages.append(system_initial_message)
    if initial_text:
        messages.append({
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": initial_text,
            "timestamp": now
        })

    session_doc = {
        "session_id": session_id,
        "title": initial_text[:30] if initial_text else "New Chat",
        "messages": messages,
        "created_at": now,
        "updated_at": now
    }

    sessions_col.insert_one(session_doc)
    return jsonify({"session_id": session_id}), 201


@chat_bp.route("/sessions/<session_id>/messages", methods=["POST"])
def post_message(session_id):
    """
    Post a new user message to a session, generate AI response, and return it.
    Body: { "content": "your message text" }
    """
    data = request.get_json() or {}
    user_text = data.get("content")
    if not user_text:
        return jsonify({"error": "Missing 'content' in request body"}), 400

    now = datetime.utcnow()
    user_msg = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": user_text,
        "timestamp": now
    }

    # Append user message
    result = sessions_col.update_one(
        {"session_id": session_id},
        {"$push": {"messages": user_msg}, "$set": {"updated_at": now}}
    )
    if result.matched_count == 0:
        return jsonify({"error": "Session not found"}), 404

    # Fetch updated session
    session = sessions_col.find_one({"session_id": session_id})
    ai_msg = generate_ai_response(user_text)

    # Append AI message
    sessions_col.update_one(
        {"session_id": session_id},
        {"$push": {"messages": ai_msg}, "$set": {"updated_at": datetime.utcnow()}}
    )

    return jsonify({"aiResponse": ai_msg}), 200


@chat_bp.route("/sessions/<session_id>/messages", methods=["GET"])
def get_messages(session_id):
    """
    Retrieve all messages for a given session.
    """
    session = sessions_col.find_one({"session_id": session_id})
    if not session:
        return jsonify({"error": "Session not found"}), 404

    # Convert timestamps to ISO format
    messages = []
    for m in session.get("messages", []):
        if role := m.get("role") == "user":
            m["role"] = "user"
            messages.append({
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "timestamp": m["timestamp"]
            })
        elif role := m.get("role") == "ai":
            m["role"] = "ai"
            messages.append({
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "timestamp": m["timestamp"],
                "videos": m.get("videos", [])
            })
        

    return jsonify({"messages": messages}), 200


@chat_bp.route('/video/<path:filename>')
def serve_clip(filename):
    print(f"Requested video: {filename}")
    file_ = filename.split("/")[1]
    filename_ = filename.split("/")[2]
    clips_dir = os.path.join("S:\\hackathons\\hackfest-nitte\\backend-1\\", "cameras", file_)
    print(f"Serving video from: {clips_dir}")
    # return send_from_directory(current_app.config["CLIPS_FOLDER"], filename)
    return send_from_directory(clips_dir, filename_)
