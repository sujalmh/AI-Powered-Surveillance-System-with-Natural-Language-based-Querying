import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default_secret_key")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/default_db")
    UPLOAD_FOLDER = './uploads/'
    VIDEO_FOLDER = './videos/'
    UPLOADED_VIDEOS_FOLDER = './uploaded_videos/'
