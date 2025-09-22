from google import genai
from flask import current_app
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)


from pydantic import BaseModel, Field
from typing import List, Optional

class PersonDetails(BaseModel):
    id: int
    gender: Optional[str] = Field(None, description="Gender of the person if identifiable")
    age_group: Optional[str] = Field(None, description="Approximate age group (e.g. child, adult, elderly)")
    clothing: Optional[str] = Field(None, description="Clothing description")
    activity: Optional[str] = Field(None, description="Activity being performed by this person")
    location: Optional[str] = Field(None, description="Where in the scene the person is located")

class ObjectDetails(BaseModel):
    name: str
    location: Optional[str] = Field(None, description="Position or area where the object is located")
    state: Optional[str] = Field(None, description="State or condition of the object (e.g., broken, active, unattended)")

class ActivityDetails(BaseModel):
    description: str
    participants: Optional[List[int]] = Field(None, description="List of person IDs involved")
    location: Optional[str] = Field(None, description="Where the activity is happening")

class AnomalyDetails(BaseModel):
    description: str
    timestamp: Optional[str] = Field(None, description="Time in the video when anomaly was detected")
    severity: Optional[str] = Field(None, description="Severity level of the anomaly")
    involved_persons: Optional[List[int]] = Field(None, description="IDs of persons involved if any")

class EnvironmentSummary(BaseModel):
    location_type: Optional[str] = Field(None, description="Type of environment, e.g., street, shop, corridor")
    time_of_day: Optional[str] = Field(None, description="Morning, Afternoon, Night, etc.")
    weather_condition: Optional[str] = Field(None, description="Weather if observable")

class SurveillanceReport(BaseModel):
    environment: EnvironmentSummary
    people: List[PersonDetails]
    objects: List[ObjectDetails]
    activities: List[ActivityDetails]
    anomalies: Optional[List[AnomalyDetails]] = Field(None, description="Any unusual or suspicious behavior detected")
    summary: str = Field(..., description="Overall summary of the video surveillance")

import time
def start_gemini_analysis(video_path: str):
    # Upload the video file to Gemini
    print("Uploading video file...")
    print(f"Video path: {video_path}")
    video_path = Path(video_path)

    uploaded_file = client.files.upload(file=str(video_path))
    file_id = uploaded_file.name  # file.name is the ID used by Gemini

    print(f"Uploaded file ID: {file_id}")

    # Poll until the file is ACTIVE
    for _ in range(10):  # wait up to ~10 seconds
        file_info = client.files.get(name=file_id)
        if file_info.state == "ACTIVE":
            print("File is ACTIVE and ready to use.")
            break
        print(f"Waiting for file to become ACTIVE... Current state: {file_info.state}")
        time.sleep(1)
    else:
        raise RuntimeError(f"File {file_id} did not become ACTIVE in time.")

    # Generate content using the uploaded video file
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=[uploaded_file, "Analyse this surveillance video, summarize the environment. Give details about the people, objects, and activities in the video. It could have anamolies, or anything that is not normal. Please provide a detailed report."],
        config={
            'response_mime_type': 'application/json',
            'response_schema': SurveillanceReport,
        },
    )
    print("Response received from Gemini.")
    print(f"Response: {response.parsed}")
    return response.parsed