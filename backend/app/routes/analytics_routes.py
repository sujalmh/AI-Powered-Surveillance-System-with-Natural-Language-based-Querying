from flask import Blueprint, jsonify,request
from datetime import datetime, timedelta
from flask import current_app
from dotenv import load_dotenv
from pymongo import MongoClient
from collections import defaultdict
import os

load_dotenv()

analytics_bp = Blueprint('analytics', __name__)

client = MongoClient(os.getenv('MONGO_URI'))
db = client.get_database('SurveillanceAI')

@analytics_bp.route('/person-count-delta', methods=['GET'])
def person_count_delta():
    collection = db['detected_objects']

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    def get_person_count(start, end):
        docs = collection.find({
            "frame_timestamp": {"$gte": start, "$lt": end},
            "detections.class": "person"
        })
        count = 0
        for doc in docs:
            for det in doc.get("detections", []):
                if det.get("class") == "person":
                    count += 1
        return count

    today_count = get_person_count(today, tomorrow)
    yesterday_count = get_person_count(yesterday, today)

    if yesterday_count == 0 and today_count == 0:
        percent_change = 0.0
    elif yesterday_count == 0 and today_count > 0:
        percent_change = 100.0
    elif today_count == 0:
        percent_change = -100.0
    else:
        percent_change = ((today_count - yesterday_count) / yesterday_count) * 100

    return jsonify({
        "today_count": today_count,
        "yesterday_count": yesterday_count,
        "percent_change": round(percent_change, 2)
    })

@analytics_bp.route('/vehicle-count-delta', methods=['GET'])
def vehicle_count_delta():
    collection = db['detected_objects']

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    # Modify class filter for vehicle types
    vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

    def get_vehicle_count(start, end):
        docs = collection.find({
            "frame_timestamp": {"$gte": start, "$lt": end},
            "detections.class": {"$in": vehicle_classes}
        })
        count = 0
        for doc in docs:
            for det in doc.get("detections", []):
                if det.get("class") in vehicle_classes:
                    count += 1
        return count

    today_count = get_vehicle_count(today, tomorrow)
    yesterday_count = get_vehicle_count(yesterday, today)

    # Updated calculation logic
    if yesterday_count == 0 and today_count == 0:
        percent_change = 0.0
    elif yesterday_count == 0 and today_count > 0:
        percent_change = 100.0
    else:
        percent_change = ((today_count - yesterday_count) / yesterday_count) * 100

    return jsonify({
        "today_count": today_count,
        "yesterday_count": yesterday_count,
        "percent_change": round(percent_change, 2)
    })

@analytics_bp.route('/object-count-delta', methods=['GET'])
def overall_count_delta():
    collection = db['detected_objects']

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    def get_overall_count(start, end):
        docs = collection.find({
            "frame_timestamp": {"$gte": start, "$lt": end},
            "detections.0": {"$exists": True}  # Ensures detections array is not empty
        })
        count = 0
        for doc in docs:
            count += len(doc.get("detections", []))  # Count all objects in detections
        return count

    today_count = get_overall_count(today, tomorrow)
    yesterday_count = get_overall_count(yesterday, today)

    if yesterday_count == 0 and today_count == 0:
        percent_change = 0.0
    elif yesterday_count == 0 and today_count > 0:
        percent_change = 100.0
    else:
        percent_change = ((today_count - yesterday_count) / yesterday_count) * 100

    return jsonify({
        "today_count": today_count,
        "yesterday_count": yesterday_count,
        "percent_change": round(percent_change, 2)
    })
@analytics_bp.route('/person-weekly-delta', methods=['GET'])
def person_weekly_delta():
    collection = db['detected_objects']

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_this_week = today - timedelta(days=today.weekday())  # Monday this week
    start_of_last_week = start_of_this_week - timedelta(days=7)
    end_of_last_week = start_of_this_week

    def get_weekly_person_count(start, end):
        docs = collection.find({
            "frame_timestamp": {"$gte": start, "$lt": end},
            "detections.class": "person"
        })
        count = 0
        for doc in docs:
            for det in doc.get("detections", []):
                if det.get("class") == "person":
                    count += 1
        return count

    this_week_count = get_weekly_person_count(start_of_this_week, today)
    last_week_count = get_weekly_person_count(start_of_last_week, end_of_last_week)

    if last_week_count == 0:
        percent_change = float('inf') if this_week_count > 0 else 0.0
    else:
        percent_change = ((this_week_count - last_week_count) / last_week_count) * 100

    return jsonify({
        "this_week_count": this_week_count,
        "last_week_count": last_week_count,
        "percent_change": round(percent_change, 2) if percent_change != float('inf') else "Infinity"
    })

@analytics_bp.route('/trend', methods=['GET'])
def trend_data():
    range_param = request.args.get('range', '7d')  # default to 7d
    now = datetime.utcnow()

    collection = db['detected_objects']
    if range_param == '24h':
        start_time = now - timedelta(hours=24)
        group_format = "%H:00"  # hour of the day
    elif range_param == '30d':
        start_time = now - timedelta(days=30)
        group_format = "%d %b"
    else:  # default to 7d
        start_time = now - timedelta(days=7)
        group_format = "%d %b"

    docs = collection.find({"frame_timestamp": {"$gte": start_time}})

    grouped_data = defaultdict(lambda: defaultdict(int))

    for doc in docs:
        timestamp = doc.get("frame_timestamp")
        if not timestamp:
            continue

        label = timestamp.strftime(group_format)
        for det in doc.get("detections", []):
            obj_class = det.get("class", "unknown")
            grouped_data[label][obj_class] += 1

    # Sort labels chronologically
    sorted_labels = sorted(grouped_data.keys(), key=lambda x: datetime.strptime(x, group_format))
    trend_data = []

    for label in sorted_labels:
        entry = {"label": label}
        entry.update(grouped_data[label])
        trend_data.append(entry)

    return jsonify({
        "range": range_param,
        "trend_data": trend_data
    })
