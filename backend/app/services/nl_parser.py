from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import re

from pydantic import BaseModel, Field

from backend.app.config import settings

# Optional LangChain imports are deferred to runtime; we keep fallbacks if unavailable.
# We will try OpenAI via langchain-openai if OPENAI_API_KEY is set, otherwise try Ollama via langchain-community.
# If neither is available, we raise and the caller can fallback to regex parser.


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
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser
    except Exception as e:
        raise RuntimeError(f"LangChain core not available: {e}")

    llm_cfg = settings.get_active_llm_config()
    provider = llm_cfg["provider"].strip().lower()
    model = llm_cfg["model"]
    api_key = llm_cfg["api_key"]

    # The opencode runtime and any OpenAI-compatible endpoint are handled by
    # backend.app.services.llm_client, which speaks the OpenAI
    # chat-completions contract and falls back to the opencode session API
    # and CLI.  It is used without LangChain so the backend keeps working
    # when LangChain is not installed.
    if provider in {"opencode", "openai_compatible", "openai_compat", "vllm",
                    "lmstudio", "llamacpp", "tgi"}:
        return None, None, None

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
            raise RuntimeError(f"LangChain OpenAI not available or misconfigured: {e}")
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
            raise RuntimeError(f"LangChain OpenRouter not available or misconfigured: {e}")
    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model=model or "llama3.1", base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
        except Exception as e:
            raise RuntimeError(f"LangChain Ollama not available or misconfigured: {e}")
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
                raise RuntimeError(f"LangChain OpenAI default failed: {e}")
        else:
            try:
                from langchain_community.chat_models import ChatOllama
                llm = ChatOllama(model=model or "llama3.1", base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
            except Exception as e:
                raise RuntimeError(f"LangChain Ollama default failed: {e}")
    else:
        raise RuntimeError(f"Unsupported LLM provider: '{provider}'. Supported: openai, openrouter, ollama, or leave empty.")

    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from langchain_core.output_parsers import PydanticOutputParser  # type: ignore

    parser = PydanticOutputParser(pydantic_object=NLFilter)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return llm, parser, prompt


def _coerce_time(raw: Any) -> Optional[Dict[str, Any]]:
    """Accept the several time shapes a model may emit.

    ``{"from_iso", "to_iso"}`` is the schema, but models also answer with
    ``{"start", "end"}`` or ``{"gte", "lte"}``.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"from_iso": raw, "to_iso": raw}
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    for src_key, dst_key in (
        ("from_iso", "from_iso"), ("to_iso", "to_iso"),
        ("start", "from_iso"), ("end", "to_iso"),
        ("from", "from_iso"), ("to", "to_iso"),
        ("gte", "from_iso"), ("lte", "to_iso"),
    ):
        val = raw.get(src_key)
        if val:
            out[dst_key] = str(val)
    lm = raw.get("last_minutes")
    if lm is not None:
        try:
            out["last_minutes"] = int(lm)
        except (TypeError, ValueError):
            pass
    return out or None


def _coerce_count_constraint(raw: Any) -> Optional[Dict[str, Any]]:
    """Accept ``3``, ``{"eq": 3}`` or ``{"gte": 3}``."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return {"eq": int(raw)}
    if isinstance(raw, str):
        digits = re.sub(r"[^0-9-]", "", raw)
        try:
            return {"eq": int(digits)} if digits else None
        except ValueError:
            return None
    if isinstance(raw, dict):
        out: Dict[str, Any] = {}
        for key in ("eq", "gte", "lte", "gt", "lt"):
            val = raw.get(key)
            if val is None:
                continue
            try:
                out[key] = int(val)
            except (TypeError, ValueError):
                continue
        return out or None
    return None


def _coerce_objects(raw: Any) -> List[Dict[str, Any]]:
    """Accept ``["person"]``, ``[{"name": "person"}]`` or a single string."""
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            out.append({"name": item})
        elif isinstance(item, dict):
            name = item.get("name") or item.get("object_name") or item.get("class")
            if name:
                out.append({"name": str(name)})
    return out


def coerce_to_nlfilter(data: Dict[str, Any]) -> "NLFilter":
    """Normalise a raw model JSON object into the ``NLFilter`` schema.

    Small models answer with the right *content* but a slightly different
    *shape* (``time.start`` instead of ``time.from_iso``, ``count_constraint``
    as a bare integer, ``objects`` as plain strings).  Rejecting those replies
    throws away correct parses, so the shape is normalised here and the values
    are then validated by Pydantic.  Fields that still fail validation are
    dropped rather than failing the whole parse.
    """
    payload: Dict[str, Any] = {}
    for key in (
        "query_type", "query_subtype", "color", "action", "zone",
        "camera_id", "result_limit", "location_hint", "ask_color",
        "semantic_query",
    ):
        if key in data and data[key] is not None:
            payload[key] = data[key]
    objects = _coerce_objects(data.get("objects"))
    if objects:
        payload["objects"] = objects
    time_filter = _coerce_time(data.get("time"))
    if time_filter:
        payload["time"] = time_filter
    count_constraint = _coerce_count_constraint(data.get("count_constraint"))
    if count_constraint:
        payload["count_constraint"] = count_constraint
    # NLFilter requires these two; a model may omit them.
    payload.setdefault("semantic_query", data.get("semantic_query") or data.get("query") or "")
    payload.setdefault("raw_query", data.get("raw_query") or payload.get("semantic_query", ""))
    try:
        return NLFilter.model_validate(payload)
    except Exception:
        for key in ("query_subtype", "result_limit", "location_hint", "semantic_query",
                    "action", "zone", "color", "time", "count_constraint", "objects"):
            payload.pop(key, None)
        return NLFilter.model_validate(payload)


def _llm_client_parse(
    nl: str,
    schema: Dict[str, Any],
    reference_date: str,
) -> "Any":
    """Parse with the opencode runtime / any OpenAI-compatible endpoint.

    Used when the configured provider is served by
    ``backend.app.services.llm_client`` instead of LangChain.  The reply is
    decoded with the tolerant JSON extractor (small models fence their JSON
    and add remarks) and validated against the same ``NLFilter`` model, so the
    downstream filter construction is identical to the LangChain path.
    """
    from backend.app.services import llm_client

    format_instructions = (
        "Respond with ONE JSON object and nothing else. "
        "Do not wrap it in markdown fences and do not add explanations. "
        "It must validate against this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    prompt = (
        PROMPT_TEMPLATE
        .replace("{query}", nl)
        .replace("{schema}", json.dumps(schema, ensure_ascii=False))
        .replace("{format_instructions}", format_instructions)
        .replace("{reference_date}", reference_date)
    )
    backend = llm_client.build_backend()
    data = llm_client.chat_json(prompt, backend=backend)
    return NLFilter.model_validate(data)


def parse_nl_with_llm(nl: str) -> Dict[str, Any]:
    llm, parser, prompt = _get_lc_components()
    # Use model schema and format instructions for robust structured output
    try:
        schema = NLFilter.model_json_schema()
    except Exception:
        # Pydantic v1 fallback
        schema = NLFilter.schema()  # type: ignore
    format_instructions = parser.get_format_instructions() if parser is not None else ""
    # Ground relative dates in either the archived deployment epoch (offline
    # reproducibility) or the current date (live operation).
    reference_date = resolve_reference_date()
    if llm is None:
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

    print(f"[NL Parser] Input: {nl}")
    print(f"[NL Parser] Parsed Filter: {f}")
    return f
