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

    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def _to_int(tok: str) -> Optional[int]:
        tok = (tok or "").strip().lower()
        if not tok:
            return None
        if tok.isdigit():
            try:
                return int(tok)
            except Exception:
                return None
        return number_words.get(tok)
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

    # count constraints (very common in user requests)
    count_constraint: Optional[Dict[str, int]] = None

    # "no person" / "without person" style
    if re.search(r"\b(no|without|zero)\s+(person|people|persons)\b", txt):
        object_name = "person"
        count_constraint = {"eq": 0}
    elif re.search(r"\b(no|without|zero)\s+(car|cars|vehicle|vehicles)\b", txt):
        object_name = "car"
        count_constraint = {"eq": 0}
    else:
        # "more than X people"
        m_cnt = re.search(
            r"\b(?:more than|greater than|over)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(person|people|persons|car|cars|vehicle|vehicles)\b",
            txt,
        )
        if m_cnt:
            n = _to_int(m_cnt.group(1))
            noun = m_cnt.group(2)
            if n is not None:
                count_constraint = {"gt": int(n)}
                if noun in {"person", "people", "persons"}:
                    object_name = "person"
                elif noun in {"car", "cars", "vehicle", "vehicles"}:
                    object_name = "car"

        # "at least X people" / "X or more people" / "multiple people"
        if count_constraint is None:
            m_cnt = re.search(
                r"\b(?:at least|minimum of)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(person|people|persons|car|cars|vehicle|vehicles)\b",
                txt,
            )
            if m_cnt:
                n = _to_int(m_cnt.group(1))
                noun = m_cnt.group(2)
                if n is not None:
                    count_constraint = {"gte": int(n)}
                    if noun in {"person", "people", "persons"}:
                        object_name = "person"
                    elif noun in {"car", "cars", "vehicle", "vehicles"}:
                        object_name = "car"

        if count_constraint is None and re.search(r"\b(multiple|more than one|two or more)\s+(person|people|persons)\b", txt):
            object_name = "person"
            count_constraint = {"gte": 2}

        # "exactly X people" / "X people"
        if count_constraint is None:
            m_cnt = re.search(
                r"\b(?:exactly|equal to)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(person|people|persons|car|cars|vehicle|vehicles)\b",
                txt,
            )
            if m_cnt:
                n = _to_int(m_cnt.group(1))
                noun = m_cnt.group(2)
                if n is not None:
                    count_constraint = {"eq": int(n)}
                    if noun in {"person", "people", "persons"}:
                        object_name = "person"
                    elif noun in {"car", "cars", "vehicle", "vehicles"}:
                        object_name = "car"

        if count_constraint is None:
            m_cnt = re.search(
                r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(person|people|persons|car|cars|vehicle|vehicles)\b",
                txt,
            )
            if m_cnt:
                n = _to_int(m_cnt.group(1))
                noun = m_cnt.group(2)
                if n is not None:
                    count_constraint = {"eq": int(n)}
                    if noun in {"person", "people", "persons"}:
                        object_name = "person"
                    elif noun in {"car", "cars", "vehicle", "vehicles"}:
                        object_name = "car"

    # Result count hints: "only one clip", "top 3 clips"
    result_limit: Optional[int] = None
    if re.search(r"\bonly\s+one\s+(clip|clips|video|videos|result|results)\b", txt):
        result_limit = 1
    else:
        m_rl = re.search(r"\b(?:top|first|show|retrieve)\s+(\d+)\s+(clip|clips|video|videos|result|results)\b", txt)
        if m_rl:
            try:
                result_limit = max(1, int(m_rl.group(1)))
            except Exception:
                result_limit = None

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
    if count_constraint:
        f["count_constraint"] = count_constraint
    if result_limit is not None:
        f["__result_limit"] = int(result_limit)

    # Track raw/semantic payload for unified retrieval and answer generation
    f["__raw"] = nl
    f["__semantic_query"] = nl

    # Basic intent/query type inference for fallback mode
    if re.search(r"\b(how many|count|number of)\b", txt):
        f["query_type"] = "informational"
        f["__intent"] = "count"
    elif re.search(r"\b(track|where|went|follow)\b", txt):
        f["query_type"] = "visual"
        f["__intent"] = "track"
    else:
        f["query_type"] = "visual"
        f["__intent"] = "find"

    # Note: location is not directly stored on detections; it is a property of cameras.
    # Basic baseline ignores location unless we add a join. For now we return it in parsed_filter for UI awareness.
    if location:
        f["__location_hint"] = location

    if ask_color:
        f["__ask_color"] = True

    return f


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
    Optimized to use MongoDB aggregation.
    """
    try:
        pipeline = [
            {"$sort": {"created_at": -1}},
            {
                "$group": {
                    "_id": "$session_id",
                    "last_message": {"$first": "$content"},
                    "last_message_time": {"$first": "$created_at"},
                    "message_count": {"$sum": 1}
                }
            },
            {"$sort": {"last_message_time": -1}},
            {"$limit": limit},
            {
                "$project": {
                    "_id": 0,
                    "session_id": "$_id",
                    "last_message": 1,
                    "last_message_time": 1,
                    "message_count": 1
                }
            }
        ]
        docs = list(chat_col.aggregate(pipeline))
        return [ChatSession(**d) for d in docs]
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
        # Fix: empty dict {} is truthy in Python, so check for actual content
        has_absolute_ts = isinstance(tsf, dict) and (tsf.get("$gte") or tsf.get("$lte"))
        if "__last_minutes" in parsed and not has_absolute_ts:
            try:
                minutes = int(parsed["__last_minutes"])
                now = datetime.utcnow()
                start = now - timedelta(minutes=minutes)
                parsed["timestamp"] = {"$gte": start.isoformat(), "$lte": now.isoformat()}
            except Exception:
                pass

        # Default time window: if no time filter at all, default to last 24 hours
        # This prevents queries from searching the entire detection history
        if "timestamp" not in parsed and "__last_minutes" not in parsed:
            now = datetime.utcnow()
            start = now - timedelta(hours=24)
            parsed["timestamp"] = {"$gte": start.isoformat(), "$lte": now.isoformat()}
            parsed["__last_minutes"] = 1440

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

