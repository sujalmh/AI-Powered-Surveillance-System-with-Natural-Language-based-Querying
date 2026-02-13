"""
Dashboard API router.

Provides aggregated dashboard metrics including statistical anomaly detection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.anomaly_detector import AnomalyDetector

router = APIRouter()


@router.get("/anomalies")
def get_anomalies(
    date: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), defaults to today"),
    threshold: float = Query(2.0, description="Z-score threshold for anomaly flagging"),
    baseline_days: int = Query(7, description="Number of historical days for baseline"),
) -> Dict[str, Any]:
    """
    Compute or retrieve anomalies for a given date.

    Uses Z-score analysis comparing hourly detection counts against
    a rolling historical baseline.

    Anomaly types returned:
      - crowd_spike: person count significantly above normal
      - off_hours: detections during historically zero-activity hours
      - unusual_object: object types rarely seen on that camera
    """
    from datetime import datetime

    target = None
    if date:
        try:
            target = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {date}. Use YYYY-MM-DD.")

    detector = AnomalyDetector(
        baseline_days=baseline_days,
        z_threshold=threshold,
    )
    return detector.detect(target_date=target)
