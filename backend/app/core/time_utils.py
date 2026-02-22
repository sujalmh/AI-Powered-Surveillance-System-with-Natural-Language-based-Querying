"""
Shared time parsing and MongoDB timestamp filter building.
Used by detections, alerts, and any endpoint that queries by time range.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string; accepts YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def normalize_iso(dt: datetime) -> str:
    """Convert datetime to UTC, drop tz, return naive ISO string for stored detection docs."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat()


def build_time_filter(
    from_ts: Optional[str] = None,
    to_ts: Optional[str] = None,
    last_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a MongoDB timestamp filter dict for the "timestamp" field.
    Returns {} if no time bounds; otherwise {"$gte": ..., "$lte": ...} as appropriate.
    """
    time_filter: Dict[str, Any] = {}
    if last_minutes is not None and last_minutes >= 1:
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=last_minutes)
        time_filter["$gte"] = normalize_iso(start)
        time_filter["$lte"] = normalize_iso(end)
    else:
        start = parse_iso(from_ts)
        end = parse_iso(to_ts)
        if start is not None:
            time_filter["$gte"] = normalize_iso(start)
        if end is not None:
            time_filter["$lte"] = normalize_iso(end)
    return time_filter
