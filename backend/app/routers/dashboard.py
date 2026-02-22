"""
Dashboard API router.

Provides aggregated dashboard metrics including statistical anomaly detection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.core.async_utils import run_sync
from backend.app.services.anomaly_detector import AnomalyDetector

router = APIRouter()


@router.get("/anomalies")
async def get_anomalies(
    date: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), defaults to today"),
    threshold: float = Query(2.0, gt=0, description="Z-score threshold for anomaly flagging"),
    baseline_days: int = Query(7, ge=1, description="Number of historical days for baseline"),
) -> Dict[str, Any]:
    """Compute or retrieve anomalies for a given date (Z-score vs historical baseline)."""

    def _block():
        from datetime import datetime
        target = None
        if date:
            try:
                target = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid date format: {date}. Use YYYY-MM-DD.")
        detector = AnomalyDetector(baseline_days=baseline_days, z_threshold=threshold)
        return detector.detect(target_date=target)
    return await run_sync(_block)
