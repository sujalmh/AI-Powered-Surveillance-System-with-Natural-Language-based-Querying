from __future__ import annotations

from typing import Any, Dict, List, Optional

from datetime import datetime, timedelta
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.db.mongo import chat_messages as chat_col, detections as detections_col, alerts as alerts_col
from backend.app.services.nl_parser import parse_nl_with_llm
from backend.app.services.unified_retrieval import UnifiedRetrieval
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Minimal color vocabulary for NL parsing (should align with detection color names)
KNOWN_COLORS = {
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
    "yellow": "Yellow",
    "black": "Black",
    "white": "White",
    "purple": "Purple",
    "orange": "Orange",
    "pink": "Pink",
    "brown": "Brown",
    "gray": "Gray",
    "grey": "Gray",
    "cyan": "Cyan",
    "magenta": "Magenta",
    "lime": "Lime",
    "navy": "Navy",
    "teal": "Teal",
    "violet": "Violet",
    "maroon": "Maroon",
    "silver": "Silver",
    "gold": "Gold",
    "coral": "Coral",
    "turquoise": "Turquoise",
    "salmon": "Salmon",
    "indigo": "Indigo",
}


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: str
    payload: Optional[Dict[str, Any]] = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]


class ChatSession(BaseModel):
    session_id: str
    last_message: Optional[str] = None
    last_message_time: Optional[str] = None
    message_count: int = 0


class ChatSendRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated session ID to keep message history")
    message: str = Field(..., description="Natural language query or command")
    # Optional: allow a direct structured filter override (bypass NL parsing)
    filter_override: Optional[Dict[str, Any]] = None
    limit: int = 50


class ChatSendResponse(BaseModel):
    session_id: str
    parsed_filter: Dict[str, Any]
    results: List[Dict[str, Any]]  # Unified hybrid results with video clips
    answer: str
    metadata: Optional[Dict[str, Any]] = None
    alert_created: Optional[Dict[str, Any]] = None
    # Additional fields for frontend compatibility
    semantic_results: Optional[List[Dict[str, Any]]] = None
    mode: Optional[str] = None


def iso_now() -> str:
    return datetime.utcnow().isoformat()


def save_message(session_id: str, role: str, content: str, payload: Optional[Dict[str, Any]] = None) -> None:
    chat_col.insert_one(
        {
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": iso_now(),
            "payload": payload,
        }
    )


def parse_simple_nl_to_filter(nl: str) -> Dict[str, Any]:
    """
    Very simple NL parsing as a baseline. Supports:
    - time: "last X minutes/hours" or absolute ISO-like phrases aren't handled here (use override)
    - object keywords: person/people, car/cars, vehicle/vehicles, bag/backpack
    - color keywords mapped to KNOWN_COLORS
    - camera by simple pattern "camera {id}" or "at <location words>" (location becomes string match)
    """
    txt = nl.lower()
    ask_color = bool(re.search(r"\b(color|colour|clothes|clothing|wearing|shirt|jacket|dress|pants|backpack|bag)\b", txt))

    # time: last X minutes/hours
    last_minutes = None
    m = re.search(r"last\s+(\d+)\s*(minute|minutes|min|mins)", txt)
    if m:
        last_minutes = int(m.group(1))
    m = re.search(r"last\s+(\d+)\s*(hour|hours|hr|hrs)", txt)
    if m:
        last_minutes = int(m.group(1)) * 60

    # object names guess
    object_name = None
    if re.search(r"\b(person|people)\b", txt):
        object_name = "person"
    elif re.search(r"\b(car|cars|vehicle|vehicles)\b", txt):
        object_name = "car"  # adjust as per your YOLO class names
    elif re.search(r"\b(bag|backpack)\b", txt):
        # Not a YOLO class by default; we will keep object as person but later filter by attributes if available
        object_name = "person"

    # If the question is about color and no explicit time was provided, default to last 60 minutes
    if last_minutes is None and ask_color:
        last_minutes = 60

    # color
    color = None
    for k, v in KNOWN_COLORS.items():
        if re.search(rf"\b{k}\b", txt):
            color = v
            break

    # camera: "camera 1"
    camera_id = None
    cm = re.search(r"camera\s+(\d+)", txt)
    if cm:
        camera_id = int(cm.group(1))

    # location hint: "at Main Entrance"
    location = None
    lm = re.search(r"(?:at|in)\s+([a-z][a-z0-9\s\-]+)$", txt.strip())
    if lm and not camera_id:
        # location becomes an exact match for now; could be used to map to camera IDs in future
        location = lm.group(1).strip().title()

    f: Dict[str, Any] = {}
    if camera_id is not None:
        f["camera_id"] = camera_id

    # time range
    now = datetime.utcnow()
    ts_filter: Dict[str, Any] = {}
    if last_minutes:
        start = now - timedelta(minutes=last_minutes)
        ts_filter["$gte"] = start.isoformat()
        ts_filter["$lte"] = now.isoformat()
    if ts_filter:
        f["timestamp"] = ts_filter

    if object_name:
        f["objects.object_name"] = object_name
    if color:
        f["objects.color"] = color

    # Note: location is not directly stored on detections; it is a property of cameras.
    # Basic baseline ignores location unless we add a join. For now we return it in parsed_filter for UI awareness.
    if location:
        f["__location_hint"] = location

    if ask_color:
        f["__ask_color"] = True

    return f


def run_detection_query(filter_query: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
    cur = detections_col.find(filter_query, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return list(cur)


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        # Fallback tolerant parser
        try:
            return datetime.strptime(ts.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return datetime.utcnow()


def _object_matches(obj: Dict[str, Any], parsed: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    name = parsed.get("objects.object_name")
    color = parsed.get("objects.color")
    if name and obj.get("object_name") != name:
        return False
    if color and obj.get("color") != color:
        return False
    return True


def merge_tracks(results: List[Dict[str, Any]], parsed: Dict[str, Any], max_gap_seconds: int = 3) -> List[Dict[str, Any]]:
    """
    Merge contiguous detections belonging to the same track (same camera_id + track_id)
    into single intervals when the time gap between consecutive hits is <= max_gap_seconds.
    Returns a flat list of merged intervals for all tracks.
    """
    # Collect timestamps per (camera_id, track_id) for matching objects
    per_track: Dict[tuple, List[datetime]] = {}
    meta: Dict[tuple, Dict[str, Any]] = {}

    for doc in results:
        cam = doc.get("camera_id")
        ts = _parse_ts(doc.get("timestamp", ""))
        for obj in doc.get("objects", []):
            if not _object_matches(obj, parsed):
                continue
            tid = obj.get("track_id", -1)
            if tid is None or tid < 0:
                # Skip untracked objects for merging
                continue
            key = (cam, tid)
            per_track.setdefault(key, []).append(ts)
            # Store representative metadata
            if key not in meta:
                meta[key] = {
                    "camera_id": cam,
                    "track_id": tid,
                    "object_name": obj.get("object_name"),
                    "color": obj.get("color"),
                }

    merged: List[Dict[str, Any]] = []
    for key, times in per_track.items():
        times.sort()
        if not times:
            continue
        start = times[0]
        prev = times[0]
        for t in times[1:]:
            gap = (t - prev).total_seconds()
            if gap <= max_gap_seconds:
                # Continue current segment
                prev = t
            else:
                # Close current segment and start a new one
                m = dict(meta[key])
                m["start"] = start.isoformat()
                m["end"] = prev.isoformat()
                m["duration_seconds"] = max(0, int((prev - start).total_seconds()))
                merged.append(m)
                start = t
                prev = t
        # Close last segment
        m = dict(meta[key])
        m["start"] = start.isoformat()
        m["end"] = prev.isoformat()
        m["duration_seconds"] = max(0, int((prev - start).total_seconds()))
        merged.append(m)

    # Sort merged intervals by start time descending (most recent first)
    merged.sort(key=lambda x: x.get("start", ""), reverse=True)
    return merged


def combine_track_segments(merged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine all contiguous segments for each (camera_id, track_id) into a single span:
    start = min(segment.start), end = max(segment.end). This produces one clip per person/track.
    """
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for seg in merged:
        key = (seg.get("camera_id"), seg.get("track_id"))
        s = _parse_ts(seg.get("start", ""))
        e = _parse_ts(seg.get("end", ""))
        if key not in by_key:
            by_key[key] = {
                "camera_id": seg.get("camera_id"),
                "track_id": seg.get("track_id"),
                "object_name": seg.get("object_name"),
                "color": seg.get("color"),
                "start": s,
                "end": e,
            }
        else:
            if s < by_key[key]["start"]:
                by_key[key]["start"] = s
            if e > by_key[key]["end"]:
                by_key[key]["end"] = e
    out: List[Dict[str, Any]] = []
    for v in by_key.values():
        start_dt = v["start"]
        end_dt = v["end"]
        out.append({
            "camera_id": v["camera_id"],
            "track_id": v["track_id"],
            "object_name": v.get("object_name"),
            "color": v.get("color"),
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_seconds": max(0, int((end_dt - start_dt).total_seconds())),
        })
    out.sort(key=lambda x: x.get("start", ""), reverse=True)
    return out


def coalesce_across_tracks(merged: List[Dict[str, Any]], join_gap_seconds: int = 10) -> List[Dict[str, Any]]:
    """
    Coalesce segments across different track_ids when they are temporally adjacent to form one final clip.
    Groups by (camera_id, object_name, color) and merges when next.start - prev.end <= join_gap_seconds.
    This handles tracker ID churn so many 2-second clips become one continuous clip like 09:19:20 → 09:20:20.
    """
    # Group by camera + object/color signature
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for seg in merged:
        key = (seg.get("camera_id"), seg.get("object_name"), seg.get("color"))
        groups.setdefault(key, []).append(seg)

    output: List[Dict[str, Any]] = []
    for key, segs in groups.items():
        # Sort by start time
        segs_sorted = sorted(segs, key=lambda s: s.get("start", ""))
        if not segs_sorted:
            continue
        # Rolling merge using join_gap_seconds
        cur_start = _parse_ts(segs_sorted[0]["start"])
        cur_end = _parse_ts(segs_sorted[0]["end"])
        cam, obj, col = key
        for seg in segs_sorted[1:]:
            s = _parse_ts(seg["start"])
            e = _parse_ts(seg["end"])
            gap = (s - cur_end).total_seconds()
            if gap <= join_gap_seconds:
                # Extend current window
                if e > cur_end:
                    cur_end = e
            else:
                # Emit current and start a new one
                output.append({
                    "camera_id": cam,
                    "object_name": obj,
                    "color": col,
                    "start": cur_start.isoformat(),
                    "end": cur_end.isoformat(),
                    "duration_seconds": max(0, int((cur_end - cur_start).total_seconds())),
                })
                cur_start, cur_end = s, e
        # Emit final
        output.append({
            "camera_id": cam,
            "object_name": obj,
            "color": col,
            "start": cur_start.isoformat(),
            "end": cur_end.isoformat(),
            "duration_seconds": max(0, int((cur_end - cur_start).total_seconds())),
        })
    # Sort combined spans by start descending
    output.sort(key=lambda x: x.get("start", ""), reverse=True)
    return output


def _guess_count_from_text(nl: str) -> Optional[int]:
    import re
    m = re.search(r"(>=|more than|at least)\s*(\d+)", nl.lower())
    if m:
        try:
            return int(m.group(2)) if m.group(2) else None
        except Exception:
            return None
    m2 = re.search(r"(\d+)\s*(people|persons|person|cars|car)", nl.lower())
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None

def _maybe_create_alert_from_nl(nl: str, parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Heuristic NL→alert creation: if the user asks to "alert"/"notify" etc., compile a basic rule and save it.
    """
    low = nl.lower()
    intent = any(k in low for k in ["alert", "notify", "create alert", "set alert", "make alert"])
    if not intent:
        return None
    obj = parsed.get("objects.object_name") or "person"
    color = parsed.get("objects.color")
    cam = parsed.get("camera_id")
    count = _guess_count_from_text(nl) or 1
    # event enter detection
    event = None
    if any(k in low for k in ["enter", "enters", "arrives", "comes in", "comes"]):
        event = {"name": "enter", "object": obj}
    # time-of-day: night
    tod = None
    if any(k in low for k in ["night", "after dark", "evening"]):
        tod = {"start": "20:00", "end": "06:00"}
    rule = {
        "cameras": [int(cam)] if cam is not None else None,
        "objects": [{"name": obj}],
        "color": color,
        "count": {">=": count},
        "event": event,
        "tod": tod,
        "cooldown_s": 30,
    }
    doc = {
        "name": f"NL: {nl[:60]}",
        "nl": nl,
        "rule": rule,
        "enabled": True,
        "actions": ["push_ws", "snapshot"],
        "created_at": iso_now(),
        "updated_at": iso_now(),
    }
    try:
        ins = alerts_col.insert_one(doc)
        return {"id": str(ins.inserted_id), "name": doc["name"], "enabled": True}
    except Exception:
        return None

def craft_answer(nl: str, parsed: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    # Count unique persons by distinct (camera_id, track_id) for matching objects.
    unique_ids = set()
    for doc in results:
        cam = doc.get("camera_id")
        for obj in doc.get("objects", []):
            if not _object_matches(obj, parsed):
                continue
            tid = obj.get("track_id", -1)
            if tid is None or tid < 0:
                # Skip untracked
                continue
            unique_ids.add((cam, tid))

    count = len(unique_ids)
    obj = parsed.get("objects.object_name")
    color = parsed.get("objects.color")
    cam = parsed.get("camera_id")
    parts = []
    if obj:
        parts.append(obj)
    if color:
        parts.append(color.lower())
    subject = " ".join(parts) if parts else "objects"
    where = f" on camera {cam}" if cam is not None else ""
    return f"Found {count} {subject}{where} for your query."


@router.get("/history", response_model=ChatHistoryResponse)
def history(session_id: str) -> ChatHistoryResponse:
    try:
        msgs = list(chat_col.find({"session_id": session_id}, {"_id": 0}).sort("created_at", 1))
        # Ensure required fields
        messages = [
            ChatMessage(
                role=m.get("role", "assistant"),
                content=m.get("content", ""),
                created_at=m.get("created_at", iso_now()),
                payload=m.get("payload"),
            )
            for m in msgs
        ]
        return ChatHistoryResponse(session_id=session_id, messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat history: {e}") from e


@router.get("/sessions", response_model=List[ChatSession])
def sessions(limit: int = 20) -> List[ChatSession]:
    """
    Return recent chat sessions with last message and count.
    """
    try:
        ids = chat_col.distinct("session_id")
        sessions: List[ChatSession] = []
        for sid in ids:
            # Latest message for this session
            last = chat_col.find({"session_id": sid}, {"_id": 0}).sort("created_at", -1).limit(1)
            last_msg = None
            last_time = None
            for m in last:
                last_msg = m.get("content")
                last_time = m.get("created_at")
            count = chat_col.count_documents({"session_id": sid})
            sessions.append(ChatSession(session_id=sid, last_message=last_msg, last_message_time=last_time, message_count=count))
        # Sort by last_message_time desc
        sessions.sort(key=lambda s: s.last_message_time or "", reverse=True)
        return sessions[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list chat sessions: {e}") from e


@router.post("/send", response_model=ChatSendResponse)
def send(req: ChatSendRequest) -> ChatSendResponse:
    try:
        logger.info(f"Received chat request. Session: {req.session_id}, Message: {req.message}")
        # Save user message
        save_message(req.session_id, "user", req.message, None)

        # Parse NL with LLM (structured) if no override provided; fallback to regex parser
        if req.filter_override is not None:
            parsed = req.filter_override
        else:
            try:
                parsed = parse_nl_with_llm(req.message)
                logger.info(f"LLM parsed query: {parsed}")
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}. Falling back to regex parser.")
                parsed = parse_simple_nl_to_filter(req.message)
                logger.info(f"Regex parsed query: {parsed}")

        # If LLM suggested a relative window and absolute timestamp not provided, expand to [now-last_minutes, now]
        tsf = parsed.get("timestamp")
        if "__last_minutes" in parsed and not tsf:
            try:
                minutes = int(parsed["__last_minutes"])
                now = datetime.utcnow()
                start = now - timedelta(minutes=minutes)
                parsed["timestamp"] = {"$gte": start.isoformat(), "$lte": now.isoformat()}
            except Exception:
                pass

        # Execute unified hybrid retrieval
        logger.info("Starting unified hybrid retrieval...")
        retrieval = UnifiedRetrieval()
        semantic_query = parsed.get("__semantic_query", req.message)
        
        search_result = retrieval.search(
            parsed_filter=parsed,
            semantic_query=semantic_query,
            limit=req.limit
        )
        logger.info(f"Retrieval complete. Metadata: {search_result.get('metadata')}")
        
        results = search_result["results"]
        answer = search_result["answer"]
        metadata = search_result["metadata"]

        # Maybe create alert from NL
        alert_info = _maybe_create_alert_from_nl(req.message, parsed)
        if alert_info:
            metadata["alert_created"] = alert_info

        # Save assistant message
        save_message(
            req.session_id,
            "assistant",
            answer,
            {
                "session_id": req.session_id,
                "parsed_filter": parsed,
                "results": results,
                "metadata": metadata,
                "answer": answer,
            },
        )

        # Build response compatible with frontend expectations
        response_data = {
            "session_id": req.session_id,
            "parsed_filter": parsed,
            "results": results,  # Unified results with clip_url
            "answer": answer,
            "metadata": metadata,
            "alert_created": alert_info,
        }
        
        # Add fields expected by frontend for video display
        # Frontend checks for semantic_results and mode fields
        query_type = metadata.get("query_type", "visual")
        has_action = parsed.get("action") is not None
        
        if has_action or query_type == "visual":
            # For visual queries, expose results as semantic_results for proper rendering
            response_data["semantic_results"] = results
            response_data["mode"] = "unstructured" if has_action else "hybrid"
        
        return ChatSendResponse(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}") from e
