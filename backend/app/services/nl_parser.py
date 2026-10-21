from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import re

from loguru import logger
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.services import llm_client
from backend.app.services.llm_client import redact_secrets

# Optional LangChain imports are deferred to runtime; we keep fallbacks if unavailable.
# We will try OpenAI via langchain-openai if OPENAI_API_KEY is set, otherwise try Ollama via langchain-community.
# If neither is available, we raise and the caller can fallback to regex parser.

# Providers served by backend.app.services.llm_client (no LangChain involved).
_LLM_CLIENT_PROVIDERS = {
    "opencode", "openai_compatible", "openai_compat", "vllm",
    "lmstudio", "llamacpp", "tgi",
}

# Placeholders the parser prompt always carries.  An unfilled
# ``{reference_date}`` makes the model resolve relative dates against the
# session date instead of the archive epoch.
PROMPT_PLACEHOLDERS = ("query", "schema", "format_instructions", "reference_date")


class TimeFilter(BaseModel):
    last_minutes: Optional[int] = Field(default=None, description="If provided, relative window in minutes")
    from_iso: Optional[str] = Field(default=None, description="ISO8601 start time inclusive")
    to_iso: Optional[str] = Field(default=None, description="ISO8601 end time inclusive")


class ObjectSpec(BaseModel):
    name: str = Field(description="Object name, e.g., person, car")
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="Optional attributes for the object")


class NLFilter(BaseModel):
    query_type: str = Field(default="visual", description="Type of query: 'visual' (needs clips) or 'informational' (text only)")
    camera_id: Optional[int] = Field(default=None, description="If specified, numeric camera id to filter")
    time: Optional[TimeFilter] = Field(default=None, description="Time window filter")
    objects: Optional[List[ObjectSpec]] = Field(default=None, description="Requested objects")
    color: Optional[str] = Field(default=None, description="Requested dominant color if applicable (e.g., Red)")
    action: Optional[str] = Field(default=None, description="Action being performed (running, walking, fighting, etc.)")
    zone: Optional[str] = Field(default=None, description="Specific zone or area mentioned (entrance, exit, hallway, etc.)")
    count_constraint: Optional[Dict[str, int]] = Field(default=None, description="Count constraints: {'gt': 3} for 'more than 3', {'gte': 3} for 'at least 3', {'eq': 2} for 'exactly 2'")
    result_limit: Optional[int] = Field(default=None, description="Maximum number of results to return, e.g. 1 for 'show me the latest clip'")
    ask_color: Optional[bool] = Field(default=None, description="True if the user is asking about clothing color")
    location_hint: Optional[str] = Field(default=None, description="If user refers to a location name in NL")
    query_subtype: Optional[str] = Field(
        default=None,
        description="Optional subtype for informational/system queries: 'alerts' for alert logs, 'cameras' for camera devices/status",
    )
    semantic_query: str = Field(description="Natural language description for semantic/CLIP search - ALWAYS provide this")
    raw_query: str = Field(description="Original natural language input")


_NUMBER_WORDS: Dict[str, int] = {
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

# Synonym dictionaries for query expansion (improves recall without changing primary filter)
OBJECT_SYNONYMS: Dict[str, List[str]] = {
    "person": ["person", "people", "persons", "pedestrian", "human"],
    "people": ["person", "people", "persons", "pedestrian", "human"],
    "car": ["car", "cars", "vehicle", "vehicles", "automobile"],
    "vehicle": ["car", "cars", "vehicle", "vehicles", "automobile"],
    "dog": ["dog", "dogs", "canine"],
    "bicycle": ["bicycle", "bike", "bikes", "cyclist"],
    "truck": ["truck", "trucks", "lorry"],
}
ACTION_SYNONYMS: Dict[str, List[str]] = {
    "running": ["running", "jogging", "sprinting", "runs"],
    "walking": ["walking", "walks", "strolling"],
    "fighting": ["fighting", "altercation", "conflict", "violence", "fight", "brawl"],
    "carrying": ["carrying", "carries", "holding", "lifting"],
    "falling": ["falling", "falls", "fallen", "trip", "collapse"],
}
COLOR_SYNONYMS: Dict[str, List[str]] = {
    "red": ["red", "crimson", "maroon", "burgundy", "scarlet"],
    "blue": ["blue", "navy", "azure", "cobalt"],
    "green": ["green", "lime", "emerald", "teal"],
    "black": ["black", "dark"],
    "white": ["white", "light"],
    "yellow": ["yellow", "gold", "amber"],
}


def _expand_parsed_filter(f: Dict[str, Any]) -> None:
    """
    Add __expanded_terms to the parsed filter for downstream recall improvement.
    Does not change primary filters; expansion is used for optional $in or semantic broadening.
    """
    if not getattr(settings, "ENABLE_QUERY_EXPANSION", True):
        return
    try:
        expanded: Dict[str, Any] = {}
        obj = f.get("objects.object_name")
        if obj and isinstance(obj, str):
            obj_lower = obj.strip().lower()
            expanded["objects"] = OBJECT_SYNONYMS.get(obj_lower, [obj_lower])
        action = f.get("action")
        if action and isinstance(action, str):
            action_lower = action.strip().lower()
            expanded["action"] = ACTION_SYNONYMS.get(action_lower, [action_lower])
        col = f.get("objects.color")
        if col and isinstance(col, str):
            col_lower = col.strip().lower()
            expanded["color"] = COLOR_SYNONYMS.get(col_lower, [col_lower])
        if expanded:
            f["__expanded_terms"] = expanded
    except Exception:
        pass


def _word_to_int(tok: str) -> Optional[int]:
    t = (tok or "").strip().lower()
    if not t:
        return None
    if t.isdigit():
        try:
            return int(t)
        except Exception:
            return None
    return _NUMBER_WORDS.get(t)


def _infer_result_limit(nl: str) -> Optional[int]:
    low = (nl or "").lower()
    if re.search(r"\bonly\s+one\s+(clip|clips|video|videos|result|results)\b", low):
        return 1

    m = re.search(r"\b(?:top|first|show|retrieve)\s+(\d+)\s+(clip|clips|video|videos|result|results)\b", low)
    if m:
        try:
            return max(1, int(m.group(1)))
        except Exception:
            return None
    return None


def resolve_reference_date(
    data_epoch: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    """Return the ISO date used to ground relative time expressions.

    Offline/released-data runs set ``DATA_EPOCH_DATE`` so "today" resolves to
    the archive date rather than the wall-clock date.  Invalid values fall back
    to *now* instead of silently emitting a bad prompt.
    """
    raw = (settings.DATA_EPOCH_DATE if data_epoch is None else data_epoch) or ""
    raw = str(raw).strip()
    if raw:
        try:
            datetime.strptime(raw, "%Y-%m-%d")
            return raw
        except ValueError:
            pass
    reference = now or datetime.now()
    return reference.strftime("%Y-%m-%d")


def correct_query_type_for_count(
    query_type: str,
    count_constraint: Optional[Dict[str, Any]],
    query_subtype: Optional[str] = None,
) -> str:
    """Counts of scene objects are visual, not system-status informational.

    This is a small pure helper so the post-LLM correction is unit-testable
    without making a network call.
    """
    subtype = (query_subtype or "").strip().lower()
    if (
        str(query_type or "").strip().lower() == "informational"
        and count_constraint
        and subtype not in {"alerts", "cameras"}
    ):
        return "visual"
    return query_type


# Comparative phrases that turn a bare "eq" constraint into a range.
_COUNT_COMPARATIVE_RE = re.compile(
    r"\b(no more than|at most|less than|fewer than|more than|greater than|over|"
    r"at least|minimum of|minimum|exactly|equal to)\s+(\d+)",
    re.IGNORECASE,
)
_COUNT_COMPARATIVE_KEYS = {
    "more than": "gt", "greater than": "gt", "over": "gt",
    "at least": "gte", "minimum of": "gte", "minimum": "gte",
    "exactly": "eq", "equal to": "eq",
    "no more than": "lte", "at most": "lte",
    "less than": "lt", "fewer than": "lt",
}


def upgrade_count_constraint(
    count_constraint: Optional[Dict[str, Any]],
    query_text: str = "",
) -> Optional[Dict[str, Any]]:
    """Turn a bare ``{"eq": n}`` into the operator the question uses.

    A model that answers ``count_constraint: 5`` means "5" while the
    question may well say "more than 5" or "at least 5"; in that case the
    question wins.  "exactly N" keeps ``eq``.
    """
    if not count_constraint or not query_text:
        return count_constraint
    if set(count_constraint.keys()) != {"eq"}:
        return count_constraint
    match = _COUNT_COMPARATIVE_RE.search(query_text or "")
    if not match:
        return count_constraint
    key = _COUNT_COMPARATIVE_KEYS.get(match.group(1).strip().lower())
    if not key:
        return count_constraint
    try:
        number = int(match.group(2))
    except (TypeError, ValueError):
        return count_constraint
    return {key: number}


def _augment_count_constraint_from_query(nl: str, f: Dict[str, Any]) -> None:
    """
    Backstop heuristic for count constraints when LLM output misses them.
    Also infers objects.object_name when the LLM set count_constraint but forgot the object.
    """
    low = (nl or "").lower()

    def _set_obj_for_noun(noun: str) -> None:
        if noun in {"person", "people", "persons"}:
            f["objects.object_name"] = "person"
        elif noun in {"car", "cars", "vehicle", "vehicles"}:
            f["objects.object_name"] = "car"

    # If count_constraint is already set (by LLM), still ensure object name is present
    if f.get("count_constraint"):
        # A bare integer is read as "exactly N"; if the text says otherwise
        # ("more than 3", "at least 3"), the text wins.
        upgraded = upgrade_count_constraint(f.get("count_constraint"), low)
        if upgraded:
            f["count_constraint"] = upgraded
        if not f.get("objects.object_name"):
            # Try to infer the object noun from the NL query
            m = re.search(r"\b(person|people|persons|car|cars|vehicle|vehicles)\b", low)
            if m:
                _set_obj_for_noun(m.group(1))
        return

    # no/without/zero ...
    m = re.search(r"\b(no|without|zero)\s+(person|people|persons|car|cars|vehicle|vehicles)\b", low)
    if m:
        _set_obj_for_noun(m.group(2))
        f["count_constraint"] = {"eq": 0}
        return

    # more than / over / greater than
    m = re.search(
        r"\b(?:more than|greater than|over)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(person|people|persons|car|cars|vehicle|vehicles)\b",
        low,
    )
    if m:
        n = _word_to_int(m.group(1))
        if n is not None:
            _set_obj_for_noun(m.group(2))
            f["count_constraint"] = {"gt": int(n)}
            return

    # at least / minimum of
    m = re.search(
        r"\b(?:at least|minimum of)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(person|people|persons|car|cars|vehicle|vehicles)\b",
        low,
    )
    if m:
        n = _word_to_int(m.group(1))
        if n is not None:
            _set_obj_for_noun(m.group(2))
            f["count_constraint"] = {"gte": int(n)}
            return

    # multiple people
    if re.search(r"\b(multiple|more than one|two or more)\s+(person|people|persons)\b", low):
        f["objects.object_name"] = "person"
        f["count_constraint"] = {"gte": 2}
        return

    # exactly / equal to
    m = re.search(
        r"\b(?:exactly|equal to)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(person|people|persons|car|cars|vehicle|vehicles)\b",
        low,
    )
    if m:
        n = _word_to_int(m.group(1))
        if n is not None:
            _set_obj_for_noun(m.group(2))
            f["count_constraint"] = {"eq": int(n)}
            return

    # Crowd synonyms: "crowded", "busy", "packed" when zone is set (e.g. "was the lobby crowded?")
    if not f.get("count_constraint") and f.get("zone") and any(w in low for w in ("crowded", "busy", "packed", "crowd")):
        _set_obj_for_noun("person")
        f["count_constraint"] = {"gte": 5}


PROMPT_TEMPLATE = """You are an intelligent surveillance query parser. Convert the user's natural language query into a comprehensive JSON object.

CRITICAL INSTRUCTIONS:

0. REFERENCE DATE (MANDATORY FOR TIME PARSING):
   - The surveillance archive is centered on {reference_date}.
   - Resolve relative expressions ("today", "yesterday", "last night", "this evening") against {reference_date}.
   - If a date is written without a year, use the year of {reference_date}.
   - Never default a timestamp to 2023 or any other arbitrary year.
   - Emit absolute timestamps as ISO 8601 (e.g., 2026-03-01T22:00:00).

1. QUERY TYPE CLASSIFICATION (MANDATORY):
   - Set query_type = "visual" if the query is about:
     * Finding/showing objects, people, vehicles, events
     * Actions (running, walking, fighting, carrying)
     * Visual scenes or behaviors
     * Counts of objects/people in the video (e.g., "how many people were in the lobby?")
     * Examples: "show me people running", "find person in red shirt", "any cars entering?", "how many people at 10 PM?"
   
   - Set query_type = "informational" ONLY if the query is about:
     * System/status counts (cameras, alerts, total detections), not scene content
     * Non-visual information requests
     * Examples: "how many cameras are active?", "total detections today?", "system status?"
   - If a count_constraint is extracted for people/objects in the scene, query_type MUST be "visual".

2. QUERY SUBTYPE FOR SYSTEM QUERIES (OPTIONAL):
   - Use query_subtype ONLY for informational or system/status-style questions.
   - Set query_subtype = "alerts" if the user is asking about alerts, alert logs, or notifications, for example:
     * "any alerts found?", "have any alerts been triggered?", "show recent alerts", "what alerts fired today?"
   - Set query_subtype = "cameras" if the user is asking about camera DEVICES or their status (NOT detecting cameras in the scene), for example:
     * "any cameras found?", "how many cameras are active?", "camera status", "which cameras are running?"
   - For normal visual queries about detections (people, cars, etc.), leave query_subtype as null/omitted.

3. STRUCTURED FILTERS (extract if present):
   - time: Relative (last X minutes) or absolute timestamps
   - camera_id: Specific camera number (ONLY if numeric ID explicitly mentioned like "camera 5")
   - objects: Person, car, bag, etc. (IMPORTANT: use lowercase names like 'person', 'car', 'dog' etc.)
   - color: Clothing/object color (use title case: Red, Blue, Green, etc.)
   - action: Running, walking, fighting, carrying, falling, etc.
   - zone: entrance, exit, hallway, parking, gate, etc.
   - count_constraint: For "more than 3 people" use key "gt" with value 3, for "at least 3" use key "gte" with value 3, for "exactly 2" use key "eq" with value 2
   - result_limit: Maximum results to return. For "only one clip" use 1, for "top 5 results" use 5, for "the latest clip" use 1
   - location_hint: Camera location references or named areas. Examples:
     * "Lab camera" → location_hint: "Lab"
     * "Gate 3" → location_hint: "Gate 3"
     * "Main Entrance camera" → location_hint: "Main Entrance"
     * "from the parking lot" → location_hint: "parking lot"
     * Extract ANY location/area name mentioned with or without the word "camera"

4. SEMANTIC QUERY (ALWAYS REQUIRED):
   - Create a descriptive natural language string capturing the CONTEXTUAL meaning
   - Include urgency, behavior, scene description
   - Examples:
     * "person running fast near an exit with urgency"
     * "crowd gathering in unusual way"
     * "person in red shirt carrying large backpack"
   - If the query is informational, still provide a semantic version describing what's being asked

5. IMPORTANT:
   - ALWAYS fill both structured filters AND semantic_query
   - semantic_query must NEVER be null or empty
   - Only include fields that are actually present in the query
   - Be precise and avoid hallucinating information

User query:
{query}

JSON schema:
{schema}

Format instructions:
{format_instructions}
"""


def _get_lc_components():
    # Defer imports to runtime to avoid hard dependency if not configured.
    llm_cfg = settings.get_active_llm_config()
    provider = str(llm_cfg.get("provider") or "").strip().lower()
    model = llm_cfg.get("model") or ""
    api_key = llm_cfg.get("api_key") or ""

    # The opencode runtime and any OpenAI-compatible endpoint are handled by
    # backend.app.services.llm_client, which speaks the OpenAI
    # chat-completions contract and falls back to the opencode session API
    # and CLI.  It is used without LangChain so the backend keeps working
    # when LangChain is not installed.  This check must happen *before* the
    # LangChain import, otherwise the opencode provider would crash on a
    # machine without LangChain.
    if provider in _LLM_CLIENT_PROVIDERS:
        return None, None, None

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser
    except Exception as e:
        raise RuntimeError(
            "LangChain core not available: " + redact_secrets(str(e), [api_key])
        )

    llm = None
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model or "gpt-4o-mini",
                api_key=api_key,
                temperature=0.0,
            )
        except Exception as e:
            raise RuntimeError(
                "LangChain OpenAI not available or misconfigured: "
                + redact_secrets(str(e), [api_key])
            )
    elif provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model or "gpt-4o-mini",
                openai_api_base=settings.OPENROUTER_BASE_URL,
                openai_api_key=api_key or settings.OPENAI_API_KEY,
                temperature=0.0,
            )
        except Exception as e:
            raise RuntimeError(
                "LangChain OpenRouter not available or misconfigured: "
                + redact_secrets(str(e), [api_key, settings.OPENAI_API_KEY])
            )
    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model=model or "llama3.1", base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
        except Exception as e:
            raise RuntimeError(
                "LangChain Ollama not available or misconfigured: "
                + redact_secrets(str(e), [api_key])
            )
    elif not provider:
        # Default fallback logic
        if api_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=model or "gpt-4o-mini",
                    api_key=api_key,
                    temperature=0.0,
                )
            except Exception as e:
                raise RuntimeError(
                    "LangChain OpenAI default failed: " + redact_secrets(str(e), [api_key])
                )
        else:
            try:
                from langchain_community.chat_models import ChatOllama
                llm = ChatOllama(model=model or "llama3.1", base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
            except Exception as e:
                raise RuntimeError(
                    "LangChain Ollama default failed: " + redact_secrets(str(e), [api_key])
                )
    else:
        raise RuntimeError(
            f"Unsupported LLM provider: '{provider}'. Supported: openai, openrouter, "
            "ollama, opencode, openai_compatible, or leave empty."
        )

    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from langchain_core.output_parsers import PydanticOutputParser  # type: ignore

    parser = PydanticOutputParser(pydantic_object=NLFilter)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return llm, parser, prompt


_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
)
_DURATION_RE = re.compile(
    r"(\d+)\s*(second|sec|minute|min|hour|hr|day)s?", re.IGNORECASE
)
_DURATION_UNIT_MINUTES = {
    "second": 1.0 / 60.0,
    "sec": 1.0 / 60.0,
    "minute": 1.0,
    "min": 1.0,
    "hour": 60.0,
    "hr": 60.0,
    "day": 1440.0,
}


def _coerce_time(raw: Any) -> Optional[Dict[str, Any]]:
    """Accept the several time shapes a model may emit.

    ``{"from_iso", "to_iso"}`` is the schema, but models also answer with
    ``{"start", "end"}``, ``{"gte", "lte"}``, a bare number of minutes or a
    sentence such as ``"last 30 minutes"``.  Values that are not ISO
    timestamps are dropped instead of being forwarded as a broken range.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        match = _DURATION_RE.search(text)
        if match:
            minutes = int(round(int(match.group(1)) * _DURATION_UNIT_MINUTES[match.group(2).lower()]))
            return {"last_minutes": max(1, minutes)}
        if _ISO_TIMESTAMP_RE.match(text):
            return {"from_iso": text, "to_iso": text}
        return None
    if isinstance(raw, (int, float)):
        value = int(raw)
        if value >= 1_000_000:
            # A Unix epoch in seconds, not a number of minutes.
            try:
                stamp = datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
            except (OverflowError, OSError, ValueError):
                return None
            return {"from_iso": stamp, "to_iso": stamp}
        # "time": 30 -> the last 30 minutes.
        return {"last_minutes": max(1, value)}
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    for src_key, dst_key in (
        ("from_iso", "from_iso"), ("to_iso", "to_iso"),
        ("start", "from_iso"), ("end", "to_iso"),
        ("from", "from_iso"), ("to", "to_iso"),
        ("gte", "from_iso"), ("lte", "to_iso"),
        ("start_time", "from_iso"), ("end_time", "to_iso"),
    ):
        val = raw.get(src_key)
        if val in (None, ""):
            continue
        text = str(val).strip()
        if not _ISO_TIMESTAMP_RE.match(text):
            continue
        out[dst_key] = text
    for lm_key in ("last_minutes", "duration_minutes", "minutes", "last_minute"):
        lm = raw.get(lm_key)
        if lm is None:
            continue
        try:
            out["last_minutes"] = max(1, int(lm))
        except (TypeError, ValueError):
            continue
        break
    return out or None


# Aliases small models use for the count-constraint operators.
_COUNT_CONSTRAINT_KEYS = {
    "eq": "eq", "exact": "eq", "exactly": "eq", "equal": "eq", "equals": "eq", "is": "eq",
    "gte": "gte", "min": "gte", "minimum": "gte", "at_least": "gte", "atleast": "gte",
    "greater_or_equal": "gte",
    "gt": "gt", "greater_than": "gt", "more_than": "gt", "above": "gt", "over": "gt",
    "lte": "lte", "max": "lte", "maximum": "lte", "at_most": "lte", "atmost": "lte",
    "less_or_equal": "lte",
    "lt": "lt", "less_than": "lt", "below": "lt", "under": "lt",
}
_COUNT_PHRASE_RE = re.compile(
    r"(more than|greater than|over|at least|minimum of|minimum|exactly|equal to|no more than|"
    r"at most|less than|fewer than)\s*(\d+)",
    re.IGNORECASE,
)


def _coerce_count_constraint(raw: Any) -> Optional[Dict[str, Any]]:
    """Accept ``3``, ``{"eq": 3}``, ``{"gte": 3}`` or ``"more than 3"``."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return {"eq": int(raw)}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        match = _COUNT_PHRASE_RE.search(text)
        if match:
            phrase = match.group(1).lower()
            number = int(match.group(2))
            if phrase in ("more than", "greater than", "over"):
                return {"gt": number}
            if phrase in ("at least", "minimum of", "minimum"):
                return {"gte": number}
            if phrase in ("exactly", "equal to"):
                return {"eq": number}
            if phrase in ("no more than", "at most"):
                return {"lte": number}
            if phrase in ("less than", "fewer than"):
                return {"lt": number}
        digits = re.sub(r"[^0-9-]", "", text)
        try:
            return {"eq": int(digits)} if digits else None
        except ValueError:
            return None
    if isinstance(raw, dict):
        out: Dict[str, Any] = {}
        for key, val in raw.items():
            if val is None:
                continue
            canonical = _COUNT_CONSTRAINT_KEYS.get(str(key).strip().lower().replace(" ", "_"))
            if canonical is None:
                continue
            try:
                out[canonical] = int(val)
            except (TypeError, ValueError):
                continue
        return out or None
    return None


def _coerce_objects(raw: Any) -> List[Dict[str, Any]]:
    """Accept ``["person"]``, ``[{"name": "person"}]`` or a single string."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    elif isinstance(raw, dict):
        raw = [raw]
    elif not isinstance(raw, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            if name:
                out.append({"name": name})
        elif isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("object_name")
                or item.get("class")
                or item.get("object")
                or item.get("label")
            )
            if name:
                out.append({"name": str(name).strip()})
    return out


def _object_color(raw: Any) -> Optional[str]:
    """Read a colour that a model put inside the object entries."""
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    for item in items:
        if isinstance(item, dict):
            value = item.get("color") or item.get("colour")
            if value:
                return str(value).strip()
    return None


def _coerce_bool(raw: Any) -> Optional[bool]:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("true", "yes", "y", "1"):
            return True
        if text in ("false", "no", "n", "0"):
            return False
    return None


def _normalise_query_type(raw: Any) -> str:
    """Map the model's answer onto the two supported query types."""
    text = str(raw or "").strip().lower()
    if text in ("visual", "informational"):
        return text
    if any(word in text for word in ("info", "system", "status", "metadata")):
        return "informational"
    return "visual"


# Fields that may be dropped when the payload does not validate; the two
# required fields (semantic_query, raw_query) are never dropped.
_OPTIONAL_FILTER_FIELDS = (
    "query_subtype", "result_limit", "location_hint", "action", "zone", "color",
    "time", "count_constraint", "objects", "camera_id", "ask_color",
)


def coerce_to_nlfilter(data: Dict[str, Any], raw_query: str = "") -> "NLFilter":
    """Normalise a raw model JSON object into the ``NLFilter`` schema.

    Small models answer with the right *content* but a slightly different
    *shape* (``time.start`` instead of ``time.from_iso``, ``count_constraint``
    as a bare integer, ``objects`` as plain strings).  Rejecting those replies
    throws away correct parses, so the shape is normalised here and the values
    are then validated by Pydantic.  Fields that still fail validation are
    dropped rather than failing the whole parse.
    """
    data = data if isinstance(data, dict) else {}
    payload: Dict[str, Any] = {}
    for key in (
        "query_type", "query_subtype", "color", "action", "zone",
        "camera_id", "result_limit", "location_hint", "ask_color",
    ):
        if key in data and data[key] is not None:
            payload[key] = data[key]

    if "query_type" in payload:
        payload["query_type"] = _normalise_query_type(payload["query_type"])
    if "query_subtype" in payload:
        payload["query_subtype"] = str(payload["query_subtype"]).strip().lower() or None
        if payload["query_subtype"] is None:
            payload.pop("query_subtype")
    for key in ("color", "action", "zone", "location_hint"):
        if key in payload:
            payload[key] = str(payload[key]).strip()
    for key in ("camera_id", "result_limit"):
        if key in payload:
            try:
                payload[key] = int(payload[key])
            except (TypeError, ValueError):
                payload.pop(key, None)
    if "ask_color" in payload:
        coerced_bool = _coerce_bool(payload["ask_color"])
        if coerced_bool is None:
            payload.pop("ask_color")
        else:
            payload["ask_color"] = coerced_bool

    raw_objects = data.get("objects")
    objects = _coerce_objects(raw_objects)
    if objects:
        payload["objects"] = objects
    object_color = _object_color(raw_objects)
    if object_color and not payload.get("color"):
        payload["color"] = object_color
    time_filter = _coerce_time(data.get("time"))
    if time_filter:
        payload["time"] = time_filter
    count_constraint = _coerce_count_constraint(data.get("count_constraint"))
    if count_constraint:
        # The question decides whether a bare number is "exactly N" or a
        # range ("more than N" / "at least N").
        count_constraint = upgrade_count_constraint(count_constraint, str(raw_query or ""))
        payload["count_constraint"] = count_constraint

    # NLFilter requires these two; a model may omit them.  The original
    # question is the sensible default for both.
    semantic_query = str(data.get("semantic_query") or "").strip()
    if not semantic_query:
        semantic_query = str(data.get("query") or "").strip()
    if not semantic_query:
        semantic_query = str(raw_query or "").strip()
    payload["semantic_query"] = semantic_query
    payload["raw_query"] = str(data.get("raw_query") or "").strip() or str(raw_query or "").strip() or semantic_query

    try:
        return NLFilter.model_validate(payload)
    except Exception:
        minimal = {k: v for k, v in payload.items() if k not in _OPTIONAL_FILTER_FIELDS}
        minimal["semantic_query"] = semantic_query
        minimal["raw_query"] = payload["raw_query"]
        return NLFilter.model_validate(minimal)


def unfilled_placeholders(template: str, values: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return the placeholders of *template* that have no value.

    The check applies to the **template**, not to the rendered prompt: a
    ``{reference_date}`` that comes from the user's own text (a question
    about the placeholder itself) is not an unfilled placeholder.
    """
    values = values or {}
    return [
        name for name in llm_client.placeholders(template or "")
        if name not in values or values[name] is None
    ]


def build_parser_prompt(
    nl: str,
    schema: Any,
    format_instructions: str,
    reference_date: str,
) -> str:
    """Render ``PROMPT_TEMPLATE`` with all four placeholders filled.

    The prompt is always handed to the model with every placeholder
    substituted: an unfilled ``{reference_date}`` would make the model
    resolve "today"/"yesterday" against the session date instead of the
    archive epoch, and an unfilled ``{schema}`` would leave the model without
    the field list.  A missing value is a bug, so it raises instead of being
    forwarded.  The check is made on the template, so a user question that
    happens to contain "{reference_date}" is passed through unchanged.
    """
    values = {
        "query": nl or "",
        "schema": schema if isinstance(schema, str) else json.dumps(schema, ensure_ascii=False),
        "format_instructions": format_instructions or "",
        "reference_date": reference_date or "",
    }
    unknown = [
        name for name in llm_client.placeholders(PROMPT_TEMPLATE)
        if name not in PROMPT_PLACEHOLDERS
    ]
    if unknown:
        raise ValueError(
            "parser template uses unknown placeholders: " + ", ".join(sorted(unknown))
        )
    missing = unfilled_placeholders(PROMPT_TEMPLATE, values)
    if missing:
        raise ValueError(
            "parser prompt is missing values for: " + ", ".join(sorted(missing))
        )
    return llm_client.render_prompt(
        PROMPT_TEMPLATE, required=PROMPT_PLACEHOLDERS, **values
    )


def _llm_client_parse(
    nl: str,
    schema: Dict[str, Any],
    reference_date: str,
) -> "Any":
    """Parse with the opencode runtime / any OpenAI-compatible endpoint.

    Used when the configured provider is served by
    ``backend.app.services.llm_client`` instead of LangChain.  The reply is
    decoded with the tolerant JSON extractor (small models fence their JSON
    and add remarks) and normalised with :func:`coerce_to_nlfilter` before it
    is validated against the same ``NLFilter`` model, so the downstream
    filter construction is identical to the LangChain path.
    """
    from backend.app.services import llm_client

    format_instructions = (
        "Respond with ONE JSON object and nothing else. "
        "Do not wrap it in markdown fences and do not add explanations. "
        "It must validate against this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    prompt = build_parser_prompt(nl, schema, format_instructions, reference_date)
    backend = llm_client.build_backend()
    data = llm_client.chat_json(
        prompt, backend=backend, max_tokens=settings.NL_MAX_TOKENS
    )
    return coerce_to_nlfilter(data, raw_query=nl)


def parse_nl_with_llm(nl: str) -> Dict[str, Any]:
    llm, parser, prompt = _get_lc_components()
    # Use model schema and format instructions for robust structured output
    try:
        schema = NLFilter.model_json_schema()
    except Exception:
        # Pydantic v1 fallback
        schema = NLFilter.schema()  # type: ignore
    # ``parser`` is None for the opencode / OpenAI-compatible transports:
    # LangChain is bypassed there, so its format instructions must not be
    # asked for.
    format_instructions = parser.get_format_instructions() if parser is not None else ""
    # Ground relative dates in either the archived deployment epoch (offline
    # reproducibility) or the current date (live operation).
    reference_date = resolve_reference_date()
    if llm is None or parser is None or prompt is None:
        # opencode / OpenAI-compatible transport (no LangChain dependency).
        result: NLFilter = _llm_client_parse(nl, schema, reference_date)
    else:
        chain = prompt | llm | parser  # type: ignore
        result = chain.invoke({
            "query": nl,
            "schema": schema,
            "format_instructions": format_instructions,
            "reference_date": reference_date,
        })
    # Convert to backend filter shape
    f: Dict[str, Any] = {}
    if result.camera_id is not None:
        f["camera_id"] = int(result.camera_id)
    if result.time is not None:
        tsf: Dict[str, Any] = {}
        if result.time.last_minutes is not None:
            # Caller can expand to absolute range if needed; provide last_minutes hint here
            f["__last_minutes"] = int(result.time.last_minutes)
        if result.time.from_iso:
            tsf["$gte"] = result.time.from_iso
        if result.time.to_iso:
            tsf["$lte"] = result.time.to_iso
        if tsf:
            f["timestamp"] = tsf
    # objects
    if result.objects:
        # Use first object as main filter for compatibility with existing queries
        main = result.objects[0]
        if main and main.name:
            f["objects.object_name"] = main.name.lower()  # YOLO stores lowercase names
    # color
    if result.color:
        f["objects.color"] = result.color.strip().title()  # Normalize to title case (e.g., "Red")
        f["__ask_color"] = True
    # ask_color flag
    if result.ask_color:
        f["__ask_color"] = True
    # location hint
    if result.location_hint:
        f["__location_hint"] = result.location_hint
    # Always keep raw query for traceability
    f["__raw"] = nl
    # If neither absolute timestamps nor last_minutes present but ask_color is true, recommend default window
    if "__last_minutes" not in f and "timestamp" not in f and f.get("__ask_color"):
        f["__last_minutes"] = 60

    # Enhanced fields for hybrid search and downstream logic
    # Query type classification (visual vs informational)
    f["query_type"] = result.query_type
    
    # Intent: find | track | count (heuristic)
    nl_low = nl.lower()
    intent = "find"
    if "count" in nl_low or "how many" in nl_low:
        intent = "count"
    elif "track" in nl_low or "where" in nl_low or "went" in nl_low:
        intent = "track"
    f["__intent"] = intent

    # Optional query subtype for system/status queries (alerts, cameras, etc.)
    subtype_raw = getattr(result, "query_subtype", None)
    if subtype_raw:
        subtype = str(subtype_raw).strip().lower()
        if subtype in {"alert", "alerts"}:
            f["__query_subtype"] = "alerts"
        elif subtype in {"camera", "cameras"}:
            f["__query_subtype"] = "cameras"

    # New enhanced fields
    if result.action:
        f["action"] = result.action
    
    if result.zone:
        f["zone"] = result.zone
    
    if result.count_constraint:
        f["count_constraint"] = result.count_constraint
    _augment_count_constraint_from_query(nl, f)

    # A count of people/objects in the archive is a visual query, not a
    # status query.  The LLM prompt sometimes mislabels "how many ..." as
    # informational; correct it after the schema is materialized.
    f["query_type"] = correct_query_type_for_count(
        f.get("query_type", "visual"),
        f.get("count_constraint"),
        f.get("__query_subtype"),
    )

    # Regex backstop for camera status questions — match explicit constructs only
    status_terms = r"(online|offline|active|inactive|down|up|running|stopped|on|off|live)"
    camera_status_pattern = rf"\b(?:status\s+of\s+(?:the\s+)?(?:camera|cameras)\s*(?:is|are)?\s+{status_terms}|(?:camera|cameras)\s*(?:'s|is|are)\s+{status_terms})\b"
    if re.search(camera_status_pattern, nl_low):
        f["query_type"] = "informational"
        f["__query_subtype"] = "cameras"

    # Result limit: prefer LLM output, fall back to regex inference
    llm_limit = result.result_limit
    regex_limit = _infer_result_limit(nl)
    requested_limit = llm_limit if llm_limit is not None else regex_limit
    if requested_limit is not None:
        f["__result_limit"] = int(requested_limit)

    # Extract semantic query for CLIP-based search
    # semantic_query is now mandatory from LLM
    semantic_query = result.semantic_query
    f["__semantic_query"] = semantic_query
    
    # Legacy fields for backwards compatibility
    # Also build a more descriptive semantic query for CLIP when color+object are present
    obj = f.get("objects.object_name") or "person"
    col = f.get("objects.color")
    action = f.get("action")
    if col and action:
        f["__embedding_text"] = f"{obj} wearing {str(col).lower()} clothing {action}"
        f["__colors_norm"] = [str(col)]
    elif col:
        f["__embedding_text"] = f"{obj} wearing {str(col).lower()} clothing"
        f["__colors_norm"] = [str(col)]
    elif action:
        f["__embedding_text"] = f"{obj} {action}"
        f["__colors_norm"] = []
    else:
        f["__embedding_text"] = semantic_query
        f["__colors_norm"] = []

    # Query expansion: add synonym-based expanded terms for recall (optional downstream use)
    _expand_parsed_filter(f)

    logger.debug("[NL Parser] Input: {}", nl)
    logger.debug("[NL Parser] Parsed Filter: {}", f)
    return f
