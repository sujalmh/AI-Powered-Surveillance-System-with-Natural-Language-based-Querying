import openai
from typing import List, Dict, Union
import os
from openai import OpenAI, AuthenticationError, RateLimitError
import json
from dotenv import load_dotenv
load_dotenv()

import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from pymongo import MongoClient
MONGO_URI = os.environ.get("MONGO_URI")
client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGO_URI)

db = mongo_client["SurveillanceAI"]
collection = db["reports"]

from pydantic import BaseModel
from typing import List

class SearchResult(BaseModel):
    camera_id: str
    relevance_reason: str

class SearchResults(BaseModel):
    results: List[SearchResult]

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "video_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "camera_id": {"type": "string"},
                "video_clip": {"type": "string"}
            },
            "required": ["camera_id", "video_clip"],
            "additionalProperties": False
        }
    }
}

def get_all_reports():
    reports = list(collection.find())
    for report in reports:
        report["_id"] = str(report["_id"])  
        
    return reports

def get_relevant_report(user_query: str):
    reports = get_all_reports()
    print("All Reports:", reports)
    # Prepare prompt for OpenAI
    prompt = "Given the following reports, please find the one most relevant to the query and return it:\n\n"
    
    # Format reports into a JSON-like structure (use `str()` to keep it readable)
    formatted_reports = "\n\n".join([f"Report {i+1} camera_id:{report["camera_id"]}: {str(report)}" for i, report in enumerate(reports)])

    # Append user query
    prompt += f"Query: {user_query}\n\n"
    prompt += f"Reports:\n{formatted_reports}\n"

    # Send to OpenAI completion API
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",  # Or any other appropriate model
        messages=[
            {"role": "system", "content": "You are an AI assistant who helps find relevant reports."},
            {"role": "user", "content": prompt}
        ],
        response_format=SearchResults
    )

    # Extract the relevant report based on AI response
    relevant_report = completion.choices[0].message.parsed
    print("Relevant Report:", relevant_report)
    return relevant_report

