from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from loguru import logger
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from backend.app.config import settings

# Load env vars to ensure API keys are available
load_dotenv()


def _format_timestamp_ist(dt: datetime) -> str:
    """
    Convert datetime to IST (Asia/Kolkata) and format as readable string.
    Format: "Mar 2, 2026 14:30 IST"
    """
    try:
        # Ensure datetime is timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Convert to IST
        ist_dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))
        # Format: "Mar 2, 2026 14:30 IST"
        return ist_dt.strftime("%b %-d, %Y %H:%M IST")
    except Exception:
        # Fallback for Windows (no %-d support)
        try:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ist_dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))
            # Use %d and strip leading zero manually
            day = ist_dt.day
            return ist_dt.strftime(f"%b {day}, %Y %H:%M IST")
        except Exception as e:
            logger.warning(f"Failed to format timestamp to IST: {e}")
            return str(dt)


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
                logger.opt(exception=True).error("Failed to initialize OpenAI")
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
                logger.opt(exception=True).error("Failed to initialize OpenRouter")
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
                logger.opt(exception=True).error("Failed to initialize Ollama")
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
            logger.opt(exception=True).error("LLM answer generation failed — using data-driven fallback")
            # Instead of returning a generic apology that discards all
            # retrieval data, fall through to the template-based fallback
            # which constructs an answer from the actual results.
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
                timestamps_formatted.append(_format_timestamp_ist(dt))
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
                            trigger_ts = _format_timestamp_ist(dt)
                    except Exception as e:
                        logger.opt(exception=True).debug("Failed to parse trigger_ts {} for alert {}: {}", trigger_ts, a.get('alert_name') or idx, e)
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
                start_fmt = _format_timestamp_ist(start)
                end_fmt = _format_timestamp_ist(end)
                duration_min = int((end - start).total_seconds() / 60)
                parts.append(f"  Time Range: {start_fmt} to {end_fmt} ({duration_min} min)")
            except Exception:
                logger.debug("Timestamp formatting failed for ts={}", ts)
                parts.append(f"  Time: {ts}")
        elif ts:
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    parts.append(f"  Time: {_format_timestamp_ist(dt)}")
                else:
                    parts.append(f"  Time: {ts}")
            except Exception:
                logger.opt(exception=True).debug("Timestamp formatting failed for ts={}", ts)
                parts.append(f"  Time: {ts}")
        elif parsed_filter.get("__last_minutes"):
            parts.append(f"  Time Window: Last {parsed_filter['__last_minutes']} minutes")
        
        return "\n".join(parts) if parts else "  No specific filters"
    
    
    def generate_conversational(self, message: str) -> str:
        """
        Generate a natural, context-aware response for conversational / off-topic messages
        (greetings, general questions about capabilities, thanks, etc.).
        """
        self._ensure_llm()
        if not self.llm:
            return self._conversational_fallback(message)

        prompt = f"""You are a helpful AI assistant embedded in a surveillance monitoring system.
The user sent a conversational message (not a surveillance query).
Respond naturally and helpfully. If they're asking what you can do, give a concise overview of your capabilities.
Keep your answer brief (2-4 sentences max). Do not mention video clips, JSON, or technical internals.

Capabilities you have:
- Search for people, vehicles, or objects in recorded footage by description, color, action, or location
- Filter results by camera, time range, or zone
- Count detections or check for specific behaviors (running, carrying, fighting, etc.)
- Show triggered alerts and their history
- Check camera status (active/offline)
- Create alert rules that notify when certain conditions are detected in future footage

User message: "{message}"

Respond conversationally:"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception:
            logger.opt(exception=True).warning("LLM conversational response failed — using fallback")
            return self._conversational_fallback(message)

    def _conversational_fallback(self, message: str) -> str:
        """Simple rule-based fallback for when LLM is unavailable."""
        low = message.strip().lower().rstrip("!?.")
        if low in {"hi", "hello", "hey", "howdy", "yo", "sup"}:
            return "Hello! How can I help you with your surveillance system today?"
        if "thank" in low or low in {"thanks", "ty"}:
            return "You're welcome! Let me know if you need anything else."
        if re.search(r'\b(bye|goodbye)\b', low):
            return "Goodbye! Feel free to ask if you need anything."
        if "who are you" in low or "what are you" in low:
            return (
                "I'm your surveillance assistant. I can search footage for people, vehicles, and events; "
                "manage alert rules; and report on camera status."
            )
        return (
            "I'm your surveillance assistant. You can ask me to find footage, search by description, "
            "check alerts, or query camera status."
        )

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
