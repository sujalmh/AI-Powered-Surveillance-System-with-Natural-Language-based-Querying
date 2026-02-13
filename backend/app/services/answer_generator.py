from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from backend.app.config import settings

# Load env vars to ensure API keys are available
load_dotenv()


class AnswerGenerator:
    """
    LLM-based answer generator for surveillance queries.
    Generates dynamic natural language responses based on query type and results.
    """
    
    def __init__(self):
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM (OpenAI by default)."""
        llm_cfg = settings.get_active_llm_config()
        provider = llm_cfg["provider"].strip().lower()
        model = llm_cfg["model"]
        api_key = llm_cfg["api_key"]
        
        # Prioritize OpenAI
        if provider == "openai" or (not provider and api_key):
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=model or "gpt-4o-mini",
                    api_key=api_key,
                    temperature=0.3,  # Slightly creative but still factual
                )
            except Exception as e:
                print(f"Failed to initialize OpenAI: {e}")
                self.llm = None
        elif provider == "openrouter":
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=model or "gpt-4o-mini",
                    openai_api_base=settings.OPENROUTER_BASE_URL,
                    openai_api_key=api_key or settings.OPENAI_API_KEY,
                    temperature=0.3,
                )
            except Exception as e:
                print(f"Failed to initialize OpenRouter: {e}")
                self.llm = None
        elif provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
                self.llm = ChatOllama(
                    model=model or "llama3.1",
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=0.3
                )
            except Exception as e:
                print(f"Failed to initialize Ollama: {e}")
                self.llm = None
    
    def generate(
        self,
        query: str,
        query_type: str,
        results: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate natural language answer based on query and results.
        
        Args:
            query: Original user query
            query_type: 'visual' or 'informational'
            results: List of results (clips for visual, data for informational)
            parsed_filter: Parsed filter dict
            metadata: Additional metadata (counts, etc.)
        
        Returns:
            Natural language answer string
        """
        if not self.llm:
            # Fallback to template-based answer if LLM unavailable
            return self._fallback_answer(query_type, results, parsed_filter)
        
        # Build context for LLM
        context = self._build_context(query, query_type, results, parsed_filter, metadata)
        
        # Generate answer using LLM
        try:
            from langchain_core.messages import HumanMessage
            
            response = self.llm.invoke([HumanMessage(content=context)])
            return response.content.strip()
        except Exception as e:
            print(f"LLM answer generation failed: {e}")
            return self._fallback_answer(query_type, results, parsed_filter)
    
    def _build_context(
        self,
        query: str,
        query_type: str,
        results: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build comprehensive context with ALL raw retrieval data for LLM.
        MANDATORY: LLM must generate from this data only, no hardcoded responses.
        """
        
        # Extract ALL raw data from results
        clip_count = len(results)
        camera_ids = []
        timestamps_list = []
        objects_found = []
        actions_found = []
        attributes_found = []
        
        # Collect detailed data from results
        for r in results:
            if r.get('camera_id') is not None:
                camera_ids.append(r['camera_id'])
            if r.get('start'):
                timestamps_list.append(r['start'])
            elif r.get('timestamp'):
                timestamps_list.append(r['timestamp'])
            if r.get('object_name'):
                objects_found.append(r['object_name'])
            if r.get('action'):
                actions_found.append(r['action'])
            if r.get('color'):
                attributes_found.append(f"color: {r['color']}")
        
        # Format collected data
        unique_cameras = sorted(set(camera_ids))
        camera_list = ", ".join(f"Camera {c}" for c in unique_cameras) if unique_cameras else "No cameras"
        timestamps_formatted = timestamps_list[:10] if timestamps_list else []
        
        structured_count = metadata.get('structured_count', 0) if metadata else 0
        semantic_count = metadata.get('semantic_count', 0) if metadata else 0
        intent = parsed_filter.get('__intent', 'unknown')
        
        if query_type == "informational":
            prompt = f"""You are a surveillance system assistant. Generate a natural language answer STRICTLY from the data below.

USER QUERY: "{query}"

QUERY TYPE: Informational (text-only, NO video clips)

RAW RETRIEVAL DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Structured Query Results: {structured_count} detection records
Semantic Query Results: {semantic_count} semantic matches
Query Intent: {intent}

Filters Applied:
{self._format_filters_detailed(parsed_filter)}

Cameras Referenced: {camera_list}
Total Results: {clip_count}

Empty Result Status: {"YES - No data found" if clip_count == 0 else "NO - Data available"}

CRITICAL INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Generate answer ONLY from the raw data above
2. Use EXACT numbers provided (counts, cameras, etc.)
3. If empty results: explain what was searched for but not found
4. Be concise, factual, and helpful
5. Do NOT mention video clips (informational query)
6. Do NOT fabricate information not in the data

Generate Natural Language Answer:"""
        
        else:
            # Visual query: comprehensive data
            timestamps_display = "\n".join(f"  {i+1}. {ts}" for i, ts in enumerate(timestamps_formatted)) if timestamps_formatted else "  (No timestamps)"
            objects_display = ", ".join(set(objects_found)) if objects_found else "No specific objects"
            actions_display = ", ".join(set(actions_found)) if actions_found else "No specific actions"
            attributes_display = ", ".join(set(attributes_found)) if attributes_found else "No specific attributes"
            
            prompt = f"""You are a surveillance system assistant. Generate a natural language answer STRICTLY from the data below.

USER QUERY: "{query}"

QUERY TYPE: Visual (includes video clips)

RAW RETRIEVAL DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Clips Retrieved: {clip_count}

Camera IDs Involved:
{camera_list}

Timestamps of Clips:
{timestamps_display}

Objects Detected: {objects_display}
Actions Detected: {actions_display}
Attributes Found: {attributes_display}

Structured Fields Applied:
{self._format_filters_detailed(parsed_filter)}

Retrieval Counts:
- Structured MongoDB: {structured_count}
- Semantic FAISS: {semantic_count}

Empty Result: {"YES - No clips" if clip_count == 0 else "NO - Clips available"}

CRITICAL INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Generate answer ONLY from raw data above
2. Start with "I found {clip_count} clip(s)..." IF clips > 0
3. If empty: explain what was searched for
4. Mention specific camera IDs from the list
5. Include sample timestamps (first 2-3)
6. Reference objects/actions/attributes detected
7. Be natural, conversational, FACTUAL
8. Do NOT fabricate data not provided
9. Keep under 100 words

Generate Natural Language Answer:"""
        
        return prompt
    
    
    def _format_filters_detailed(self, parsed_filter: Dict[str, Any]) -> str:
        """Format all filters with complete detail for LLM context."""
        parts = []
        
        if parsed_filter.get("camera_id"):
            parts.append(f"  Camera ID: {parsed_filter['camera_id']}")
        if parsed_filter.get("objects.object_name"):
            parts.append(f"  Object Type: {parsed_filter['objects.object_name']}")
        if parsed_filter.get("objects.color"):
            parts.append(f"  Color: {parsed_filter['objects.color']}")
        if parsed_filter.get("action"):
            parts.append(f"  Action: {parsed_filter['action']}")
        if parsed_filter.get("zone"):
            parts.append(f"  Zone/Location: {parsed_filter['zone']}")
        if parsed_filter.get("count_constraint"):
            parts.append(f"  Count Filter: {parsed_filter['count_constraint']}")
        
        ts = parsed_filter.get("timestamp", {})
        if ts and "$gte" in ts and "$lte" in ts:
            try:
                start = datetime.fromisoformat(ts["$gte"])
                end = datetime.fromisoformat(ts["$lte"])
                duration_min = int((end - start).total_seconds() / 60)
                parts.append(f"  Time Range: {ts['$gte']} to {ts['$lte']} ({duration_min} min)")
            except:
                parts.append(f"  Time: {ts}")
        elif parsed_filter.get("__last_minutes"):
            parts.append(f"  Time Window: Last {parsed_filter['__last_minutes']} minutes")
        
        return "\n".join(parts) if parts else "  No specific filters"
    
    
    def _fallback_answer(
        self,
        query_type: str,
        results: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any]
    ) -> str:
        """
        Emergency fallback if LLM fails - still uses actual data.
        This should RARELY execute.
        """
        clip_count = len(results)
        
        if query_type == "informational":
            return f"Query processed. Found {clip_count} records. (LLM unavailable for detailed answer)"
        else:
            if clip_count == 0:
                obj = parsed_filter.get('objects.object_name', 'results')
                return f"No clips found for {obj}. (LLM unavailable for detailed answer)"
            
            cameras = sorted(set(r.get('camera_id') for r in results if r.get('camera_id')))
            cam_str = ", ".join(f"Camera {c}" for c in cameras) if cameras else ""
            loc = f" from {cam_str}" if cam_str else ""
            return f"Found {clip_count} clip(s){loc}. (LLM unavailable for detailed answer)"
