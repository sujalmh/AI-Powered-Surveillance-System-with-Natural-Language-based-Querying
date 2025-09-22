from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from app.models.db_models import create_user, get_user_by_email, check_password
from pymongo import MongoClient
from dotenv import load_dotenv
import os

MONGO_URI=os.environ.get("MONGO_URI")

auth_bp = Blueprint('auth', __name__)

client = MongoClient(MONGO_URI)
db = client['ai_surveillance']

@auth_bp.route('/register', methods=['POST'])
def register():
    email = request.json.get('email')
    password = request.json.get('password')

    # Pass db connection from app
    if get_user_by_email(db, email):
        return jsonify({"msg": "User already exists"}), 400

    create_user(db, email, password)
    return jsonify({"msg": "User created"}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')

    user = get_user_by_email(db, email)
    if user and check_password(user, password):
        token = create_access_token(identity=email)
        return jsonify(access_token=token), 200

    return jsonify({"msg": "Bad credentials"}), 401
