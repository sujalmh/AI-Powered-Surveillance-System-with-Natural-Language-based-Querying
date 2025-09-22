# In-memory "database" for users
users_db = {}

def add_user(username, email, password):
    users_db[username] = {'username': username, 'email': email, 'password': password}

def get_user_by_username_or_email(identifier):
    user = users_db.get(identifier)
    
    if not user:
        for user_obj in users_db.values():
            if user_obj['email'] == identifier:
                return user_obj
    
    return user