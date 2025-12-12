from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    count_constraint: Optional[Dict[str, int]] = Field(default=None, description="Count constraints like {'gte': 3} for 'more than 3 people'")
    ask_color: Optional[bool] = Field(default=None, description="True if the user is asking about clothing color")
    location_hint: Optional[str] = Field(default=None, description="If user refers to a location name in NL")
    semantic_query: str = Field(description="Natural language description for semantic/CLIP search - ALWAYS provide this")
    raw_query: str = Field(description="Original natural language input")


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
   - objects: Person, car, bag, etc.
   - color: Clothing/object color (Red, Blue, etc.)
   - action: Running, walking, fighting, carrying, falling, etc.
   - zone: entrance, exit, hallway, parking, gate, etc.
   - count_constraint: For "more than 3 people" use key "gte" with value 3, for "exactly 2" use key "eq" with value 2
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

    provider = (settings.LLM_PROVIDER or "openai").strip().lower()
    llm = None
    if provider == "openai" or (not provider and settings.OPENAI_API_KEY):
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=settings.NL_DEFAULT_MODEL or "gpt-4o-mini",
                temperature=0.0,
            )
        except Exception as e:
            raise RuntimeError(f"LangChain OpenAI not available or misconfigured: {e}")
    elif provider == "ollama":
        try:
            # Community provider for Ollama
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model=settings.NL_DEFAULT_MODEL or "llama3", base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
        except Exception as e:
            raise RuntimeError(f"LangChain Ollama not available or misconfigured: {e}")
    else:
        # Try OpenAI if key present, otherwise Ollama as default
        if settings.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=settings.NL_DEFAULT_MODEL or "gpt-4o-mini",
                    temperature=0.0,
                )
            except Exception as e:
                raise RuntimeError(f"LangChain OpenAI not available or misconfigured: {e}")
        else:
            try:
                from langchain_community.chat_models import ChatOllama
                llm = ChatOllama(model=settings.NL_DEFAULT_MODEL or "llama3", base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
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
            f["objects.object_name"] = main.name
    # color
    if result.color:
        f["objects.color"] = result.color
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

    # Extract semantic query for CLIP-based search
    # semantic_query is now mandatory from LLM
    semantic_query = result.semantic_query
    f["__semantic_query"] = semantic_query
    
    # Legacy fields for backwards compatibility
    obj = f.get("objects.object_name") or "person"
    col = f.get("objects.color")
    if col:
        f["__embedding_text"] = f"{obj} wearing {str(col).lower()} clothing"
        f["__colors_norm"] = [str(col)]
    else:
        f["__embedding_text"] = semantic_query
        f["__colors_norm"] = []

    print(f"[NL Parser] Input: {nl}")
    print(f"[NL Parser] Parsed Filter: {f}")
    return f
