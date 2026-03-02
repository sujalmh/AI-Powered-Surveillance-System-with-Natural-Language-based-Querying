from __future__ import annotations

from typing import Any, Dict, List, Optional

from datetime import datetime, timedelta, timezone
import re

_ALERT_CREATION_RE = re.compile(
    r'\b(?:create|set|make|add|new|schedule)\b.*\balerts?\b'
    r'|\balerts?\b.*\b(?:create|set|make|add|new|schedule)\b'
    r'|\balert\s+me\b'
    r'|\bnotify\s+me\b'
    r'|\bwarn\s+me\b'
    r'|\bsend\s+(?:an?\s+)?alerts?\b'
    r'|\btrigger\s+(?:an?\s+)?alerts?\b',
    re.IGNORECASE,
)
_ALERT_RETRIEVAL_RE = re.compile(
    r'\b(?:show|list|get|find|are\s+there|display|view)\b.*\balerts?\b'
    r'|\balerts?\b.*\b(?:show|list|get|find|display|view)\b',
    re.IGNORECASE,
)
_CONVERSATIONAL_RE = re.compile(
    r'^(?:hi+|hello+|hey+|howdy|yo+|sup|what\'?s\s+up|good\s+(?:morning|afternoon|evening|day)|'
    r'greetings|how\s+are\s+you|how\s+r\s+u|how\s+do\s+you\s+do|nice\s+to\s+meet\s+you|'
    r'who\s+are\s+you|what\s+(?:are|can)\s+you\s+do|what\s+is\s+this|'
    r'can\s+you\s+help|help\s+me|thanks?(?:\s+you)?|thank\s+you|bye+|goodbye|see\s+you|'
    r'lol+|haha+|testing|test|ping|okay|ok)[\s!?.]*$',
    re.IGNORECASE,
)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.async_utils import run_sync
from backend.app.db.mongo import chat_messages as chat_col, detections as detections_col, alerts as alerts_col
from backend.app.config import settings
from loguru import logger

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
    limit: int = 10


class ProcessingStep(BaseModel):
    name: str
    status: str  # "complete" | "in-progress" | "pending" | "error"
    details: str
    timestamp: Optional[str] = None


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
    processing_steps: Optional[List[ProcessingStep]] = None


class IntentClassifyRequest(BaseModel):
    message: str = Field(..., description="User's message to classify")


class IntentClassifyResponse(BaseModel):
    intent: str = Field(..., description="Classification: 'alert_creation' or 'search_retrieval'")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="Brief explanation of the classification")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


_answer_generator = None  # lazy singleton - avoids re-initializing heavy resources on every call


def _get_answer_generator():
    global _answer_generator
    if _answer_generator is None:
        from backend.app.services.answer_generator import AnswerGenerator
        _answer_generator = AnswerGenerator()
    return _answer_generator


def handle_conversational_response(session_id: str, message: str, parsed_filter: Optional[Dict[str, Any]] = None) -> "ChatSendResponse":
    """Generate, persist, and return a conversational reply without touching the retrieval pipeline."""
    answer = _get_answer_generator().generate_conversational(message)
    payload = {
        "session_id": session_id,
        "parsed_filter": parsed_filter or {},
        "results": [],
        "answer": answer,
        "metadata": {"query_type": "conversational"},
    }
    save_message(session_id, "assistant", answer, payload)
    return ChatSendResponse(
        session_id=session_id,
        parsed_filter=parsed_filter if parsed_filter is not None else {},
        results=[],
        answer=answer,
        metadata={"query_type": "conversational"},
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
    if re.search(r"\b(person|people|persons)\b", txt):
        object_name = "person"
    elif re.search(r"\b(car|cars|vehicle|vehicles)\b", txt):
        object_name = "car"  # adjust as per your YOLO class names
    elif re.search(r"\b(bag|backpack)\b", txt):
        # 'bag' / 'backpack' is a YOLO class in many models ("backpack").
        # Previously mapped silently to "person" which returned entirely
        # wrong clips.  Use "backpack" as the primary object name and let
        # the pipeline decide whether to soften via semantic search.
        object_name = "backpack"

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
                r"(?<!camera\s)(?<!cam\s)\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(person|people|persons|car|cars|vehicle|vehicles)\b",
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
    now = datetime.now(timezone.utc)
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

    # Action detection (running, walking, fighting, etc.)
    action_match = re.search(
        r"\b(running|walking|fighting|carrying|falling|standing|sitting|climbing|jumping|loitering)\b",
        txt,
    )
    if action_match:
        f["action"] = action_match.group(1)

    # Zone / area detection
    zone_match = re.search(
        r"\b(?:in|at|near|around)\s+(?:the\s+)?(entrance|exit|hallway|corridor|parking|gate|lobby|reception|stairway|loading\s+dock)\b",
        txt,
    )
    if zone_match:
        f["zone"] = zone_match.group(1).strip()

    # Informational query subtype: alerts / cameras
    if re.search(r"\b(show|list|get|find|any|are\s+there|display|view|recent)\b.*\balerts?\b", txt) or \
       re.search(r"\balerts?\b.*\b(show|list|get|find|display|view)\b", txt):
        f["query_type"] = "informational"
        f["__query_subtype"] = "alerts"
        f["__intent"] = "find"
    elif re.search(r"\b(camera|cameras)\b.*\b(status|active|online|offline|running|list)\b", txt) or \
         re.search(r"\b(status|active|online|offline|running|list)\b.*\b(camera|cameras)\b", txt) or \
         re.search(r"\bhow many cameras\b", txt):
        f["query_type"] = "informational"
        f["__query_subtype"] = "cameras"
        f["__intent"] = "find"
    elif re.search(r"\b(how many|count|number of)\b", txt):
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

    # Build a concise CLIP-optimized embedding text for vector search.
    # This is critical: CLIP works best with short visual descriptions.
    _obj = f.get("objects.object_name")
    _col = f.get("objects.color")
    _act = f.get("action")
    if _col and _act and _obj:
        f["__embedding_text"] = f"{_obj} wearing {str(_col).lower()} clothing {_act}"
    elif _col and _obj:
        f["__embedding_text"] = f"{_obj} wearing {str(_col).lower()} clothing"
    elif _act and _obj:
        f["__embedding_text"] = f"{_obj} {_act}"
    elif _obj:
        _cc = f.get("count_constraint")
        if _cc:
            _gt = _cc.get("gt", 0)
            _gte = _cc.get("gte", 0)
            if _gt > 1 or _gte > 1:
                f["__embedding_text"] = f"multiple {_obj}s"
            elif _gt == 1 or _gte == 2:
                f["__embedding_text"] = f"two or more {_obj}s"
            else:
                f["__embedding_text"] = _obj
        else:
            f["__embedding_text"] = _obj
    elif _act:
        f["__embedding_text"] = f"person {_act}"
    else:
        f["__embedding_text"] = nl

    # Run query expansion so downstream recall-improvement works even in
    # regex fallback mode (was previously only done by the LLM path).
    try:
        from backend.app.services.nl_parser import _expand_parsed_filter
        _expand_parsed_filter(f)
    except Exception:
        pass

    return f


def _guess_count_from_text(nl: str) -> Optional[int]:
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

def _is_alert_creation_intent(nl: str) -> bool:
    """Return True if the natural language input expresses alert creation intent."""
    return bool(_ALERT_CREATION_RE.search(nl)) and not bool(_ALERT_RETRIEVAL_RE.search(nl))


def _summarize_alert_rule(parsed: Dict[str, Any]) -> str:
    """Build a human-readable summary of the created alert rule."""
    parts = []
    obj = parsed.get("objects.object_name")
    if obj:
        parts.append(f"Object: **{obj}**")
    color = parsed.get("objects.color")
    if color:
        parts.append(f"Color: **{color}**")
    cam = parsed.get("camera_id")
    if cam is not None:
        parts.append(f"Camera: **{cam}**")
    count = parsed.get("count_constraint")
    if count:
        ops = {"eq": "exactly", "gt": "more than", "gte": "at least", "lt": "less than", "lte": "at most"}
        for op, val in count.items():
            parts.append(f"Count: {ops.get(op, op)} **{val}**")
    action = parsed.get("action")
    if action:
        parts.append(f"Behavior: **{action}**")
    zone = parsed.get("zone")
    if zone:
        parts.append(f"Zone: **{zone}**")
    return "\n".join(f"- {p}" for p in parts) if parts else "- General surveillance alert"


def _maybe_create_alert_from_nl(nl: str, parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Heuristic NL→alert creation: if the user asks to "alert"/"notify" etc., compile a basic rule and save it.
    """
    low = nl.lower()
    if not _is_alert_creation_intent(nl):
        return None
    obj = parsed.get("objects.object_name") or "person"
    color = parsed.get("objects.color")
    cam = parsed.get("camera_id")
    
    # Use count_constraint from parsed if available, otherwise guess
    count_constraint = parsed.get("count_constraint")
    count_dict = None
    if count_constraint:
        temp_dict = {}
        for k, v in count_constraint.items():
            if k == "eq":
                temp_dict["=="] = v
            elif k == "gt":
                temp_dict[">"] = v
            elif k == "gte":
                temp_dict[">="] = v
            elif k == "lt":
                temp_dict["<"] = v
            elif k == "lte":
                temp_dict["<="] = v
        if temp_dict:
            count_dict = temp_dict
    else:
        guessed = _guess_count_from_text(nl)
        count = guessed if guessed is not None else 1
        count_dict = {">=": count}

    # event enter detection
    event = None
    if any(k in low for k in ["enter", "enters", "arrives", "comes in", "comes"]):
        event = "enter"
    # time-of-day: night
    time_of_day = None
    if any(k in low for k in ["night", "after dark", "evening"]):
        time_of_day = {"start": "20:00", "end": "06:00"}
    
    behavior = parsed.get("action")
    zone = parsed.get("zone")
    area = {"zone_id": zone} if zone else None
    
    # Safely parse camera_id — may be non-numeric string from NLP
    cam_list = None
    if cam is not None:
        try:
            cam_list = [int(cam)]
        except (ValueError, TypeError):
            cam_list = None

    rule = {
        "cameras": cam_list,
        "objects": [{"name": obj}],
        "color": color,
        "count": count_dict,
        "event": event,
        "time_of_day": time_of_day,
        "behavior": behavior,
        "area": area,
    }
    
    # Only incorporate time window if the user EXPLICITLY mentioned a time range
    # (not the auto-injected default search window). Alerts should be persistent
    # monitoring rules unless the user asks for a specific schedule.
    if parsed.get("__time_explicit") and "timestamp" in parsed and isinstance(parsed["timestamp"], dict):
        ts = parsed["timestamp"]
        if "$gte" in ts and "$lte" in ts:
            rule["time"] = {"from": ts["$gte"], "to": ts["$lte"]}

    actions = ["store_clip", "snapshot", "push_ws"]
    
    severity = "info"
    if any(k in low for k in ["critical", "severe", "urgent"]):
        severity = "critical"
    elif any(k in low for k in ["high", "important"]):
        severity = "high"
    elif any(k in low for k in ["warning", "warn", "moderate"]):
        severity = "warning"
        
    cooldown_sec = 60.0
    m_cd = re.search(r"\bcooldown\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(sec|s|minute|m|hour|h)\b", low)
    if m_cd:
        try:
            val = float(m_cd.group(1))
            unit = m_cd.group(2)
            if unit.startswith("m"):
                val *= 60.0
            elif unit.startswith("h"):
                val *= 3600.0
            cooldown_sec = val
        except Exception as e:
            logger.opt(exception=True).debug("Failed to parse cooldown duration from {}, falling back to 60.0s: {}", m_cd.group(0), e)
    
    doc = {
        "name": f"NL: {nl[:60]}",
        "nl": nl,
        "rule": {k: v for k, v in rule.items() if v is not None},
        "enabled": True,
        "actions": actions,
        "severity": severity,
        "cooldown_sec": cooldown_sec,
        "created_at": iso_now(),
        "updated_at": iso_now(),
    }
    try:
        ins = alerts_col.insert_one(doc)
        return {"id": str(ins.inserted_id), "name": doc["name"], "enabled": True}
    except Exception:
        logger.exception("Failed to insert NL alert `{}`", doc.get("name"))
        return None



@router.get("/history", response_model=ChatHistoryResponse)
async def history(session_id: str) -> ChatHistoryResponse:
    def _block():
        try:
            msgs = list(chat_col.find({"session_id": session_id}, {"_id": 0}).sort("created_at", 1))
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
    return await run_sync(_block)


@router.get("/sessions", response_model=List[ChatSession])
async def sessions(limit: int = 20) -> List[ChatSession]:
    """Return recent chat sessions with last message and count (MongoDB aggregation)."""

    def _block():
        try:
            pipeline = [
                {"$sort": {"created_at": -1}},
                {"$group": {"_id": "$session_id", "last_message": {"$first": "$content"}, "last_message_time": {"$first": "$created_at"}, "message_count": {"$sum": 1}}},
                {"$sort": {"last_message_time": -1}},
                {"$limit": limit},
                {"$project": {"_id": 0, "session_id": "$_id", "last_message": 1, "last_message_time": 1, "message_count": 1}},
            ]
            docs = list(chat_col.aggregate(pipeline))
            return [ChatSession(**d) for d in docs]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list chat sessions: {e}") from e
    return await run_sync(_block)


@router.post("/classify-intent", response_model=IntentClassifyResponse, tags=["chat"])
async def classify_intent(req: IntentClassifyRequest):
    """
    Use LLM to intelligently classify user intent as either 'alert_creation' or 'search_retrieval'.
    This provides more accurate classification than regex patterns.
    """
    def _block():
        # Initialize LLM using same config as NL parser
        llm_cfg = settings.get_active_llm_config()
        provider = llm_cfg["provider"].strip().lower()
        model = llm_cfg["model"]
        api_key = llm_cfg["api_key"]
        
        llm = None
        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=model or "gpt-4o-mini", api_key=api_key, temperature=0.0)
            elif provider == "openrouter":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=model or "gpt-4o-mini",
                    base_url=settings.OPENROUTER_BASE_URL,
                    api_key=api_key,
                    temperature=0.0
                )
            elif provider == "ollama":
                from langchain_community.chat_models import ChatOllama
                llm = ChatOllama(model=model or "llama3.1", base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
            elif not provider and api_key:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=model or "gpt-4o-mini", api_key=api_key, temperature=0.0)
            else:
                # Fallback to regex if no LLM configured
                logger.warning("No LLM configured for intent classification, using regex fallback")
                is_creation = bool(_ALERT_CREATION_RE.search(req.message))
                is_retrieval = bool(_ALERT_RETRIEVAL_RE.search(req.message))
                intent = "alert_creation" if (is_creation and not is_retrieval) else "search_retrieval"
                return IntentClassifyResponse(
                    intent=intent,
                    confidence=0.7,
                    reasoning="Classified using regex patterns (no LLM configured)"
                )
        except Exception as e:
            logger.exception(f"Failed to initialize LLM: {e}")
            # Fallback to regex
            is_creation = bool(_ALERT_CREATION_RE.search(req.message))
            is_retrieval = bool(_ALERT_RETRIEVAL_RE.search(req.message))
            intent = "alert_creation" if (is_creation and not is_retrieval) else "search_retrieval"
            return IntentClassifyResponse(
                intent=intent,
                confidence=0.7,
                reasoning=f"Classified using regex fallback (LLM initialization failed: {str(e)[:100]})"
            )
        
        # Prepare classification prompt
        prompt = f"""You are an intent classifier for a surveillance video search system. 
Classify the following user message into one of two categories:

1. "alert_creation" - User wants to CREATE a new alert/notification rule that will monitor for future events
   Examples:
   - "Alert me when someone wears a red jacket"
   - "Notify me if a car is detected"
   - "Create an alert for people with bags"
   - "Warn me when someone enters at night"

2. "search_retrieval" - User wants to SEARCH/RETRIEVE existing footage or view past events
   Examples:
   - "Show me people wearing red jackets"
   - "Find cars from yesterday"
   - "When did someone with a bag appear?"
   - "List all alerts" (viewing existing alerts, not creating new ones)

User message: "{req.message}"

Respond with ONLY a JSON object in this exact format:
{{"intent": "alert_creation" or "search_retrieval", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            # Handle both string and list responses
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            content = str(content).strip()
            
            # Parse JSON response
            import json
            # Handle cases where LLM wraps response in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            return IntentClassifyResponse(
                intent=result.get("intent", "search_retrieval"),
                confidence=float(result.get("confidence", 0.8)),
                reasoning=result.get("reasoning", "Classified by LLM")
            )
        except Exception as e:
            logger.exception(f"LLM classification failed: {e}")
            # Fallback to regex
            is_creation = bool(_ALERT_CREATION_RE.search(req.message))
            is_retrieval = bool(_ALERT_RETRIEVAL_RE.search(req.message))
            intent = "alert_creation" if (is_creation and not is_retrieval) else "search_retrieval"
            return IntentClassifyResponse(
                intent=intent,
                confidence=0.6,
                reasoning=f"Classified using regex fallback (LLM parsing failed: {str(e)[:100]})"
            )
    
    return await run_sync(_block)


@router.post("/send", response_model=ChatSendResponse)
async def send(req: ChatSendRequest) -> ChatSendResponse:
    def _block():
        try:
            from backend.app.services.nl_parser import parse_nl_with_llm
            from backend.app.services.unified_retrieval import UnifiedRetrieval

            logger.info("Received chat request. Session: {}, Message length: {}", req.session_id, len(req.message))
            logger.debug("Full message payload omitted for PII; session: {}", req.session_id)
            # Save user message
            save_message(req.session_id, "user", req.message, None)

            # Fast path: conversational / greeting messages — skip the retrieval pipeline entirely
            if _CONVERSATIONAL_RE.match(req.message.strip()):
                return handle_conversational_response(req.session_id, req.message)

            # Parse NL with LLM (structured) if no override provided; fallback to regex parser
            if req.filter_override is not None:
                parsed = req.filter_override
            else:
                try:
                    parsed = parse_nl_with_llm(req.message)
                    logger.info("LLM parsed query: {}", parsed)
                except Exception as e:
                    logger.warning("LLM parsing failed: {}. Falling back to regex parser.", e)
                    parsed = parse_simple_nl_to_filter(req.message)
                    logger.opt(exception=True).info("Regex parsed query: {}", parsed)

            # If LLM (or regex fallback) classified this as a conversational query, handle it here
            if parsed.get("query_type") == "conversational":
                return handle_conversational_response(req.session_id, req.message, parsed)

            # If LLM suggested a relative window and absolute timestamp not provided, expand to [now-last_minutes, now]
            # Use UTC to keep timestamps consistent across services
            tsf = parsed.get("timestamp")
            # Check that tsf is a dict and contains $gte or $lte keys (empty dict is falsy in Python)
            has_absolute_ts = isinstance(tsf, dict) and (tsf.get("$gte") or tsf.get("$lte"))
            if "__last_minutes" in parsed and not has_absolute_ts:
                try:
                    minutes = int(parsed["__last_minutes"])
                    now = datetime.now(timezone.utc)
                    start = now - timedelta(minutes=minutes)
                    parsed["timestamp"] = {"$gte": start.isoformat(), "$lte": now.isoformat()}
                    parsed["__time_explicit"] = True  # user explicitly asked for a time range
                except Exception as e:
                    logger.debug("Failed to parse __last_minutes value {}: {}", parsed.get("__last_minutes"), e)
            elif has_absolute_ts:
                parsed["__time_explicit"] = True  # LLM returned absolute timestamps from user query

            # ── Check for alert creation intent BEFORE default time injection ──
            alert_info = _maybe_create_alert_from_nl(req.message, parsed)

            # Default time window: if no time filter at all, default to last 24 hours (UTC)
            # Applied AFTER alert check so auto-default doesn't leak into alert rules
            if "timestamp" not in parsed and "__last_minutes" not in parsed:
                now = datetime.now(timezone.utc)
                start = now - timedelta(hours=24)
                parsed["timestamp"] = {"$gte": start.isoformat(), "$lte": now.isoformat()}
                parsed["__last_minutes"] = 1440
            if alert_info:
                logger.info("Alert created from NL: {}", alert_info)
                rule_summary = _summarize_alert_rule(parsed)
                alert_name = alert_info.get("name", "Unnamed alert")
                answer = (
                    f"Alert created successfully!\n\n"
                    f"{alert_name}\n\n"
                    f"Monitoring conditions:\n{rule_summary}\n\n"
                    f"The alert is now active and will monitor your cameras in real-time. "
                    f"You can manage it from the Alerts page."
                )
                metadata: Dict[str, Any] = {"query_type": "alert_creation"}
                assistant_payload = {
                    "session_id": req.session_id,
                    "parsed_filter": parsed,
                    "results": [],
                    "answer": answer,
                    "metadata": metadata,
                    "alert_created": alert_info,
                }
                save_message(req.session_id, "assistant", answer, assistant_payload)
                return ChatSendResponse(
                    session_id=req.session_id,
                    parsed_filter=parsed,
                    results=[],
                    answer=answer,
                    metadata=metadata,
                    alert_created=alert_info,
                )

            # ── Execute unified hybrid retrieval ──
            logger.info("Starting unified hybrid retrieval...")
            retrieval = UnifiedRetrieval()
            semantic_query = parsed.get("__semantic_query", req.message)
            
            search_result = retrieval.search(
                parsed_filter=parsed,
                semantic_query=semantic_query,
                limit=req.limit
            )
            logger.info("Retrieval complete. Metadata: {}", search_result.get("metadata"))
            
            # Extract processing steps from metadata
            processing_steps = search_result.get("processing_steps", [])
            
            results = search_result["results"]
            answer = search_result["answer"]
            metadata = search_result.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            # Save assistant message with full payload for proper restoration from history
            assistant_payload = {
                "session_id": req.session_id,
                "parsed_filter": parsed,
                "results": results,  # Full results array for video display
                "answer": answer,
                "metadata": metadata,
                "semantic_results": results,  # Also store as semantic_results for frontend compatibility
                "mode": "unstructured" if parsed.get("action") else "hybrid",  # Include mode for proper frontend rendering
            }
            
            if processing_steps:
                assistant_payload["processing_steps"] = processing_steps
            
            save_message(
                req.session_id,
                "assistant",
                answer,
                assistant_payload,
            )

            # Build response compatible with frontend expectations
            response_data = {
                "session_id": req.session_id,
                "parsed_filter": parsed,
                "results": results,  # Unified results with clip_url
                "answer": answer,
                "metadata": metadata,
                "alert_created": None,
                "processing_steps": [ProcessingStep(**step) for step in processing_steps] if processing_steps else None,
            }
            
            # Add fields expected by frontend for video display
            # Frontend checks for semantic_results and mode fields
            query_type = metadata.get("query_type", "visual")
            has_action = parsed.get("action") is not None
            
            if has_action or query_type == "visual":
                response_data["semantic_results"] = results
                response_data["mode"] = "unstructured" if has_action else "hybrid"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}") from e
        return ChatSendResponse(**response_data)
    return await run_sync(_block)

