from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from datetime import datetime, timedelta

from backend.app.db.mongo import detections as detections_col


router = APIRouter()


class DetectionObject(BaseModel):
    object_name: str
    track_id: int
    confidence: float
    color: Optional[str] = None
    person_attributes: Optional[Dict[str, Any]] = None


class DetectionDoc(BaseModel):
    camera_id: int
    timestamp: str
    objects: List[DetectionObject]


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Accept "YYYY-MM-DD" and "YYYY-MM-DDTHH:MM:SS" forms
        return datetime.fromisoformat(ts)
    except Exception:
        return None


@router.get("/", response_model=List[Dict[str, Any]])
def list_detections(
    camera_id: Optional[int] = None,
    object: Optional[str] = Query(None, description="object_name filter, e.g., person, car"),
    color: Optional[str] = None,
    from_ts: Optional[str] = Query(None, alias="from", description="ISO timestamp, e.g., 2025-10-23T10:00:00"),
    to_ts: Optional[str] = Query(None, alias="to", description="ISO timestamp, e.g., 2025-10-23T12:00:00"),
    last_minutes: Optional[int] = Query(None, ge=1, le=24 * 60, description="Override time range with last N minutes"),
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """
    Query raw detections with simple filters.
    """
    try:
        query: Dict[str, Any] = {}

        if camera_id is not None:
            query["camera_id"] = camera_id

        time_filter: Dict[str, Any] = {}
        if last_minutes:
            end = datetime.utcnow()
            start = end - timedelta(minutes=last_minutes)
            time_filter["$gte"] = start.isoformat()
            time_filter["$lte"] = end.isoformat()
        else:
            start = parse_iso(from_ts)
            end = parse_iso(to_ts)
            if start:
                time_filter["$gte"] = start.isoformat()
            if end:
                time_filter["$lte"] = end.isoformat()

        if time_filter:
            query["timestamp"] = time_filter

        # object/color nested filters
        obj_sub: Dict[str, Any] = {}
        if object:
            obj_sub["objects.object_name"] = object
        if color:
            obj_sub["objects.color"] = color
        query.update(obj_sub)

        cursor = detections_col.find(query, {"_id": 0}).sort("timestamp", -1).skip(skip).limit(limit)
        return list(cursor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query detections: {e}") from e


@router.get("/object-counts", response_model=List[Dict[str, Any]])
def object_counts(
    camera_id: Optional[int] = None,
    from_ts: Optional[str] = Query(None, alias="from"),
    to_ts: Optional[str] = Query(None, alias="to"),
    last_minutes: Optional[int] = Query(None, ge=1, le=24 * 60),
    top_n: int = Query(10, ge=1, le=50),
) -> List[Dict[str, Any]]:
    """
    Aggregate counts per object_name (simple analytics).
    """
    try:
        match: Dict[str, Any] = {}
        if camera_id is not None:
            match["camera_id"] = camera_id

        time_filter: Dict[str, Any] = {}
        if last_minutes:
            end = datetime.utcnow()
            start = end - timedelta(minutes=last_minutes)
            time_filter["$gte"] = start.isoformat()
            time_filter["$lte"] = end.isoformat()
        else:
            start = parse_iso(from_ts)
            end = parse_iso(to_ts)
            if start:
                time_filter["$gte"] = start.isoformat()
            if end:
                time_filter["$lte"] = end.isoformat()

        if time_filter:
            match["timestamp"] = time_filter

        pipeline = [
            {"$match": match},
            {"$unwind": "$objects"},
            {"$group": {"_id": "$objects.object_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": top_n},
            {"$project": {"object_name": "$_id", "_id": 0, "count": 1}},
        ]

        results = list(detections_col.aggregate(pipeline))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to aggregate object counts: {e}") from e


@router.get("/colors", response_model=List[Dict[str, Any]])
def color_counts(
    camera_id: Optional[int] = None,
    object: Optional[str] = None,
    from_ts: Optional[str] = Query(None, alias="from"),
    to_ts: Optional[str] = Query(None, alias="to"),
    last_minutes: Optional[int] = Query(None, ge=1, le=24 * 60),
    top_n: int = Query(10, ge=1, le=50),
) -> List[Dict[str, Any]]:
    """
    Aggregate counts per color (optionally per object).
    """
    try:
        match: Dict[str, Any] = {}
        if camera_id is not None:
            match["camera_id"] = camera_id

        time_filter: Dict[str, Any] = {}
        if last_minutes:
            end = datetime.utcnow()
            start = end - timedelta(minutes=last_minutes)
            time_filter["$gte"] = start.isoformat()
            time_filter["$lte"] = end.isoformat()
        else:
            start = parse_iso(from_ts)
            end = parse_iso(to_ts)
            if start:
                time_filter["$gte"] = start.isoformat()
            if end:
                time_filter["$lte"] = end.isoformat()

        if time_filter:
            match["timestamp"] = time_filter

        pipeline: List[Dict[str, Any]] = [{"$match": match}, {"$unwind": "$objects"}]
        if object:
            pipeline.append({"$match": {"objects.object_name": object}})
        pipeline.extend(
            [
                {"$group": {"_id": "$objects.color", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": top_n},
                {"$project": {"color": "$_id", "_id": 0, "count": 1}},
            ]
        )
        results = list(detections_col.aggregate(pipeline))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to aggregate color counts: {e}") from e
