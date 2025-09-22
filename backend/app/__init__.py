import os
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from pymongo import MongoClient
from app.routes.api_routes import api_bp
from app.routes.auth_routes import auth_bp
from app.routes.main_routes import main_bp
from app.routes.analytics_routes import analytics_bp
from app.routes.camera_routes import camera_bp

# Create JWT manager
jwt = JWTManager()

def create_app():
    app = Flask(__name__)

    # Configuration
    app.config["UPLOAD_FOLDER"] = './uploads/'
    app.config["VIDEO_FOLDER"] = './videos/'
    app.config["UPLOADED_VIDEOS_FOLDER"] = './uploaded_videos/'
    app.config["CLIPS_FOLDER"] = './clips/'
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["VIDEO_FOLDER"], exist_ok=True)
    os.makedirs(app.config["UPLOADED_VIDEOS_FOLDER"], exist_ok=True)

    # Load custom config
    app.config.from_object('instance.config.Config')

    # MongoDB setup using MongoClient
    client = MongoClient(app.config["MONGO_URI"])  # You will need to update the URI in the config file
    app.db = client.get_database('surveil')  # Specify your database name here

    # Initialize JWT
    jwt.init_app(app)
    print("[INFO] MongoDB and JWT initialized")

    # CORS setup
    CORS(app)
    print("[INFO] CORS enabled for all domains")

    # Register blueprints

    from app.routes.chat_routes import chat_bp

    app.register_blueprint(main_bp, url_prefix='/main')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(chat_bp, url_prefix="/chat")
    app.register_blueprint(analytics_bp, url_prefix='/analytics')
    app.register_blueprint(camera_bp, url_prefix='/cam')

#     from app.routes.main_routes import main_bp
#     from app.routes.api_routes import api_bp
#     from app.routes.auth_routes import auth_bp
#     from app.routes.camera_routes import camera_bp
#     app.register_blueprint(main_bp, url_prefix='/main')
#     app.register_blueprint(api_bp, url_prefix='/api')
#     app.register_blueprint(auth_bp, url_prefix='/auth')
#     app.register_blueprint(camera_bp, url_prefix='/cam')

    return app
