from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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
        self._current_cfg = {}
        self._ensure_llm()
    
    def _ensure_llm(self):
        """Ensure LLM is initialized and consistent with current DB settings."""
        llm_cfg = settings.get_active_llm_config()
        
        # Check if config has changed
        if (self.llm is not None and 
            llm_cfg.get("provider") == self._current_cfg.get("provider") and
            llm_cfg.get("model") == self._current_cfg.get("model") and
            llm_cfg.get("api_key") == self._current_cfg.get("api_key")):
            return

        self._current_cfg = llm_cfg
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM based on current config."""
        provider = self._current_cfg.get("provider", "").strip().lower()
        model = self._current_cfg.get("model")
        api_key = self._current_cfg.get("api_key")
        
        # Prioritize OpenAI
        if provider == "openai" or (not provider and api_key):
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=model or "gpt-4o-mini",
                    api_key=api_key,
                    temperature=0.3,
                )
            except Exception:
                logger.error("Failed to initialize OpenAI", exc_info=True)
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
            except Exception:
                logger.error("Failed to initialize OpenRouter", exc_info=True)
                self.llm = None
        elif provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
                self.llm = ChatOllama(
                    model=model or "llama3.1",
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=0.3
                )
            except Exception:
                logger.error("Failed to initialize Ollama", exc_info=True)
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
        self._ensure_llm()
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
        except Exception:
            logger.error("LLM answering generation failed", exc_info=True)
            return "I apologize, but I encountered an error while trying to generate an answer."
    
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
        alert_summaries = []
        camera_summaries = []
        
        # Collect detailed data from results
        for r in results:
            r_type = r.get("type")

            # Common camera/timestamp fields
            if r.get('camera_id') is not None:
                camera_ids.append(r['camera_id'])
            if r.get('start'):
                timestamps_list.append(r['start'])
            elif r.get('timestamp'):
                timestamps_list.append(r['timestamp'])

            # Visual detection style fields
            if r.get('object_name'):
                objects_found.append(r['object_name'])
            if r.get('action'):
                actions_found.append(r['action'])
            if r.get('color'):
                attributes_found.append(f"color: {r['color']}")

            # Alert-log specific summary
            if r_type == "alert_log":
                alert_summaries.append(
                    {
                        "alert_name": r.get("alert_name") or "",
                        "camera_id": r.get("camera_id"),
                        "triggered_at": r.get("triggered_at"),
                        "severity": r.get("severity", "info"),
                        "message": r.get("message", ""),
                    }
                )

            # Camera-status specific summary
            if r_type == "camera_status":
                camera_summaries.append(
                    {
                        "camera_id": r.get("camera_id"),
                        "location": r.get("location"),
                        "status": r.get("status"),
                        "running": r.get("running"),
                    }
                )
        
        # Format collected data
        unique_cameras = sorted(set(camera_ids))
        camera_list = ", ".join(f"Camera {c}" for c in unique_cameras) if unique_cameras else "No cameras"
        
        timestamps_formatted = []
        for ts_str in (timestamps_list[:10] if timestamps_list else []):
            try:
                dt = datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
                timestamps_formatted.append(dt.strftime("%Y-%m-%d %I:%M:%S %p"))
            except Exception:
                timestamps_formatted.append(str(ts_str))
        
        structured_count = metadata.get('structured_count', 0) if metadata else 0
        semantic_count = metadata.get('semantic_count', 0) if metadata else 0
        intent = parsed_filter.get('__intent', 'unknown')
        query_subtype = (metadata or {}).get("query_subtype", parsed_filter.get("__query_subtype"))
        
        if query_type == "informational":
            # Tailor informational context based on subtype when available
            if query_subtype == "alerts":
                prompt = f"""You are a surveillance system assistant. Generate a natural language answer STRICTLY from the data below.

USER QUERY: "{query}"

QUERY TYPE: Informational (alerts summary)

RAW ALERT LOG DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Alert Logs: {clip_count}
Query Intent: {intent}

Filters Applied:
{self._format_filters_detailed(parsed_filter)}

Alert Logs (up to first 10):
"""  # noqa: E501

                for idx, a in enumerate(alert_summaries[:10], start=1):
                    cam_str = f"Camera {a['camera_id']}" if a.get("camera_id") is not None else "Unknown camera"
                    trigger_ts = a.get('triggered_at')
                    try:
                        if trigger_ts:
                            dt = datetime.fromisoformat(str(trigger_ts).replace('Z', '+00:00'))
                            trigger_ts = dt.strftime("%Y-%m-%d %I:%M:%S %p")
                    except Exception as e:
                        logger.debug("Failed to parse trigger_ts %s for alert %s: %s", trigger_ts, a.get('alert_name') or idx, e)
                    prompt += f"- {idx}. [{a.get('severity', 'info')}] {a.get('alert_name') or 'Unnamed alert'} at {trigger_ts or 'unknown time'} on {cam_str}: {a.get('message')}\n"

                prompt += """

CRITICAL INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer ONLY from the alert log data above
2. Summarize whether any alerts fired, and roughly when and on which cameras
3. If no alerts: clearly state that no alerts were found in the searched time window
4. Be concise, factual, and avoid inventing alerts.

Generate Natural Language Answer:"""

            elif query_subtype == "cameras":
                running_count = sum(1 for c in camera_summaries if c.get("running"))
                total_cams = len(camera_summaries) if camera_summaries else len(unique_cameras)
                prompt = f"""You are a surveillance system assistant. Generate a natural language answer STRICTLY from the data below.

USER QUERY: "{query}"

QUERY TYPE: Informational (cameras/devices)

RAW CAMERA STATUS DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Cameras Matched: {total_cams}
Currently Running: {running_count}

Filters Applied:
{self._format_filters_detailed(parsed_filter)}

Camera Status (up to first 10):
"""  # noqa: E501

                for idx, c in enumerate(camera_summaries[:10], start=1):
                    status = c.get("status") or "unknown"
                    running = "running" if c.get("running") else "not running"
                    prompt += f"- {idx}. Camera {c.get('camera_id')} at '{c.get('location')}' is {status}, currently {running}.\n"  # noqa: E501

                prompt += """

CRITICAL INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer ONLY from the camera data above
2. Clearly state how many cameras are active/running vs total
3. Mention example camera IDs and locations when available
4. If no cameras found: say that no cameras matched the query
5. Be concise and factual.

Generate Natural Language Answer:"""

            else:
                # Generic informational answer (existing behaviour)
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
3. If empty: say clearly that no detections matched (e.g. "No person detections found in the last 24 hours") and briefly suggest checking that the camera was running and that it had a view during the searched time window
4. Mention specific camera IDs from the list when available
5. Include sample timestamps (first 2-3) when available
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
        if ts and isinstance(ts, dict) and "$gte" in ts and "$lte" in ts:
            try:
                start = datetime.fromisoformat(str(ts["$gte"]).replace('Z', '+00:00'))
                end = datetime.fromisoformat(str(ts["$lte"]).replace('Z', '+00:00'))
                start_fmt = start.strftime("%Y-%m-%d %I:%M:%S %p")
                end_fmt = end.strftime("%Y-%m-%d %I:%M:%S %p")
                duration_min = int((end - start).total_seconds() / 60)
                parts.append(f"  Time Range: {start_fmt} to {end_fmt} ({duration_min} min)")
            except Exception as e:
                logger.debug("Timestamp formatting failed for ts=%r", ts, exc_info=True)
                parts.append(f"  Time: {ts}")
        elif ts:
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    parts.append(f"  Time: {dt.strftime('%Y-%m-%d %I:%M:%S %p')}")
                else:
                    parts.append(f"  Time: {ts}")
            except Exception as e:
                logger.debug("Timestamp formatting failed for ts=%r", ts, exc_info=True)
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
                zone = parsed_filter.get('zone')
                if zone:
                    return f"No clips found for {obj} in {zone}. (LLM unavailable for detailed answer)"
                return f"No clips found for {obj}. (LLM unavailable for detailed answer)"

            cameras = sorted(set(r.get('camera_id') for r in results if r.get('camera_id')))
            cam_str = ", ".join(f"Camera {c}" for c in cameras) if cameras else ""
            zone = parsed_filter.get('zone')
            loc = f" from {cam_str}" if cam_str else ""
            if zone:
                loc = f" (zone: {zone}){loc}" if loc else f" (zone: {zone})"
            return f"Found {clip_count} clip(s){loc}. (LLM unavailable for detailed answer)"
