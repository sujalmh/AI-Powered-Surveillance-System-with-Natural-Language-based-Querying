from werkzeug.security import generate_password_hash, check_password_hash
import uuid

def save_chat(db, user_message, response, video_url):
    chat_data = {
        "chat_id": str(uuid.uuid4()),  # Unique ID
        "message": user_message,
        "response": response,
        "video_url": video_url
    }
    db.chats.insert_one(chat_data)
    return chat_data["chat_id"]

def create_user(db, email, password):
    hashed_password = generate_password_hash(password)
    db.users.insert_one({'email': email, 'password': hashed_password})

def get_user_by_email(db, email):
    return db.users.find_one({'email': email})

def check_password(user, password):
    return check_password_hash(user['password'], password)
