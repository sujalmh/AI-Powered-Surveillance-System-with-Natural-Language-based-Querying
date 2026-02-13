from __future__ import annotations

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


PROMPT_TEMPLATE = """You are an intelligent surveillance query parser. Convert the user's natural language query into a comprehensive JSON object.

CRITICAL INSTRUCTIONS:

1. QUERY TYPE CLASSIFICATION (MANDATORY):
   - Set query_type = "visual" if the query is about:
     * Finding/showing objects, people, vehicles, events
     * Actions (running, walking, fighting, carrying)
     * Visual scenes or behaviors
     * Examples: "show me people running", "find person in red shirt", "any cars entering?"
   
   - Set query_type = "informational" if the query is about:
     * Counts, statistics, system status
     * Non-visual information requests
     * Examples: "how many cameras are active?", "total detections today?", "system status?"

2. STRUCTURED FILTERS (extract if present):
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

3. SEMANTIC QUERY (ALWAYS REQUIRED):
   - Create a descriptive natural language string capturing the CONTEXTUAL meaning
   - Include urgency, behavior, scene description
   - Examples:
     * "person running fast near an exit with urgency"
     * "crowd gathering in unusual way"
     * "person in red shirt carrying large backpack"
   - If the query is informational, still provide a semantic version describing what's being asked

4. IMPORTANT:
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
    
    llm = None
    if provider == "openai" or (not provider and api_key):
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
            # Community provider for Ollama
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model=model or "llama3.1", base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
        except Exception as e:
            raise RuntimeError(f"LangChain Ollama not available or misconfigured: {e}")
    else:
        # Try OpenAI if key present, otherwise Ollama as default
        if api_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=model or "gpt-4o-mini",
                    api_key=api_key,
                    temperature=0.0,
                )
            except Exception as e:
                raise RuntimeError(f"LangChain OpenAI not available or misconfigured: {e}")
        else:
            try:
                from langchain_community.chat_models import ChatOllama
                llm = ChatOllama(model=model or "llama3.1", base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
            except Exception as e:
                raise RuntimeError(f"LangChain Ollama not available or misconfigured: {e}")

    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from langchain_core.output_parsers import PydanticOutputParser  # type: ignore

    parser = PydanticOutputParser(pydantic_object=NLFilter)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return llm, parser, prompt


def parse_nl_with_llm(nl: str) -> Dict[str, Any]:
    llm, parser, prompt = _get_lc_components()
    # Use model schema and format instructions for robust structured output
    try:
        schema = NLFilter.model_json_schema()
    except Exception:
        # Pydantic v1 fallback
        schema = NLFilter.schema()  # type: ignore
    format_instructions = parser.get_format_instructions()
    chain = prompt | llm | parser  # type: ignore
    result: NLFilter = chain.invoke({"query": nl, "schema": schema, "format_instructions": format_instructions})
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

    # New enhanced fields
    if result.action:
        f["action"] = result.action
    
    if result.zone:
        f["zone"] = result.zone
    
    if result.count_constraint:
        f["count_constraint"] = result.count_constraint
    _augment_count_constraint_from_query(nl, f)

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

    print(f"[NL Parser] Input: {nl}")
    print(f"[NL Parser] Parsed Filter: {f}")
    return f
