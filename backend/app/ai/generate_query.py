from langchain.tools import Tool
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os


load_dotenv()
mongo_uri = os.getenv('MONGO_URI')

mongo_client = MongoClient(mongo_uri)
collection = mongo_client["SurveillanceAI"]["detected_objects"]

schema_description = """
MongoDB collection stores surveillance video frames. Each document has:

{{
  video_id: string,
  camera_id: string,
  frame_timestamp: ISODate,
  frame_number: int,
  detections: [
    {{
      object_id: string,
      class: string,
      confidence: float,
      bounding_box: {{
        x_min: int, y_min: int, x_max: int, y_max: int
      }},
      attributes: {{
        is_male: bool,
        has_bag: bool,
        has_backpack: bool,
        has_hat: bool,
        has_longsleeves: bool,
        has_longpants: bool,
        has_longhair: bool,
        top_color: string,
      }}
    }}
  ]
}}
"""


few_shot_examples = """
Example 1:
User: Find all frames with people wearing backpacks.
MongoDB query:
{{
  "detections": {{
    "$elemMatch": {{
      "class": "person",
      "attributes.has_backpack": true
    }}
  }}
}}

Example 2:
User: Show frames of women wearing long sleeves after 5pm on April 17, 2025.
MongoDB query:
{{
  "frame_timestamp": {{
    "$gte": datetime(2025, 4, 17, 17, 0, 0)
  }},
  "detections": {{
    "$elemMatch": {{
      "class": "person",
      "attributes.is_male": false,
      "attributes.has_longsleeves": true
    }}
  }}
}}
"""
import re

def safe_eval_query(query_str: str) -> dict:
    """Convert LLM-style booleans to Python and safely eval."""
    query_str = re.sub(r'\btrue\b', 'True', query_str, flags=re.IGNORECASE)
    query_str = re.sub(r'\bfalse\b', 'False', query_str, flags=re.IGNORECASE)
    query_str = re.sub(r'\bnull\b', 'None', query_str, flags=re.IGNORECASE)
    return eval(query_str, {"datetime": datetime})


from collections import defaultdict

FRAME_TOLERANCE = 5  # how many skipped frames are tolerated

feeds_collection = mongo_client["SurveillanceAI"]["camera_feeds"]

def get_video_path(camera_id: str) -> str:
    doc = feeds_collection.find_one({"camera_id": camera_id})
    if doc:
        return doc.get("path"), doc.get("location")
    return None, None

from moviepy.video.io.VideoFileClip import VideoFileClip
import os

FPS = 30  # Change if your video has different framerate

def save_clip(video_path, start_frame, end_frame, output_path):
    try:
        start_time = start_frame / FPS
        end_time = end_frame / FPS
        clip = VideoFileClip(video_path).subclip(start_time, end_time)
        clip.write_videofile(output_path, codec='libx264', audio=False)
        return output_path
    except Exception as e:
        return f"Clip error: {str(e)}"

def query_mongodb(query_str: str) -> str:
    try:
        query = safe_eval_query(query_str)
        cursor = collection.find(query).sort([
            ("video_id", 1),
            ("frame_number", 1)
        ])

        # Group frames by video/camera
        frame_groups = defaultdict(list)
        for doc in cursor:
            key = (doc["video_id"], doc["camera_id"])
            frame_groups[key].append({
                "frame_number": doc["frame_number"],
                "timestamp": doc["frame_timestamp"]
            })

        clips = []

        for (video_id, camera_id), frames in frame_groups.items():
            frames = sorted(frames, key=lambda f: f["frame_number"])
            current_clip = {
                "video_id": video_id,
                "camera_id": camera_id,
                "start_frame": frames[0]["frame_number"],
                "start_time": frames[0]["timestamp"],
                "end_frame": frames[0]["frame_number"],
                "end_time": frames[0]["timestamp"]
            }

            for i in range(1, len(frames)):
                prev = frames[i - 1]["frame_number"]
                curr = frames[i]["frame_number"]

                if curr - prev <= FRAME_TOLERANCE:
                    # Extend current clip
                    current_clip["end_frame"] = curr
                    current_clip["end_time"] = frames[i]["timestamp"]
                else:
                    # Save current clip and start new
                    clips.append(current_clip)
                    current_clip = {
                        "video_id": video_id,
                        "camera_id": camera_id,
                        "start_frame": curr,
                        "start_time": frames[i]["timestamp"],
                        "end_frame": curr,
                        "end_time": frames[i]["timestamp"]
                    }

            clips.append(current_clip)  # Add final clip

        if not clips:
            return None

        os.makedirs("clips", exist_ok=True)
        clip_paths = []

        for clip in clips:
            video_path, video_location = get_video_path(clip['camera_id'])
            if not video_path:
                clip_paths.append(f"Video path not found for video ID: {clip['video_id']}")
                continue

            output_filename = f"{clip['video_id']}_{clip['start_frame']}_{clip['end_frame']}.mp4"
            output_path = os.path.join("clips", output_filename)

            result = save_clip(video_path, clip["start_frame"], clip["end_frame"], output_path)
            clip_paths.append({"video": result, "location": video_location})

        return clip_paths


    except Exception as e:
        return f"Query Error: {str(e)}"

mongo_tool = Tool(
    name="MongoDBQueryTool",
    func=query_mongodb,
    description="Use to run a MongoDB query based on the frame/detection schema. Input must be a MongoDB query dictionary."
)

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_template = PromptTemplate.from_template(f"""
You are an assistant that helps convert user questions into MongoDB queries.

Schema:
{schema_description}

{few_shot_examples}

Now create a MongoDB query for the following input. Only return the query dictionary, don't explain anything.

User: {{input}}
MongoDB query:
""")

from langchain.chains import LLMChain

llm_chain = LLMChain(prompt=prompt_template, llm=llm)

def ask_question(user_query: str):
    mongo_query = llm_chain.run(user_query)
    print(f"MongoDB query: {mongo_query}")
    result = query_mongodb(mongo_query)
    print(f"Result: {result}")

    return result

# ask_question("Find a woman wearing long sleeves")

