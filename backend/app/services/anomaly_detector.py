"""
Statistical Anomaly Detection Service.

Uses Z-score spike analysis against a rolling historical baseline to
detect unusual patterns in surveillance detection data.

Anomaly types:
  - crowd_spike   : person count significantly above normal for that hour
  - off_hours     : detections during historically zero-activity hours
  - unusual_object: object types rarely seen on that camera
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from backend.app.db.mongo import detections, anomaly_events

logger = logging.getLogger(__name__)

# ──────────────────────────────── Constants ────────────────────────────────

DEFAULT_BASELINE_DAYS = 7
DEFAULT_Z_THRESHOLD = 2.0
SEVERITY_THRESHOLDS = {"high": 3.0, "medium": 2.0, "low": 1.5}


# ──────────────────────────────── Service ─────────────────────────────────

class AnomalyDetector:
    """Stateless detector — call detect() to compute anomalies for a date."""

    def __init__(
        self,
        baseline_days: int = DEFAULT_BASELINE_DAYS,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
    ):
        self.baseline_days = baseline_days
        self.z_threshold = z_threshold

    # ── public ────────────────────────────────────────────────────────────

    def detect(self, target_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run anomaly detection for *target_date* (defaults to today).

        Returns:
            {
                "date": "2026-02-13",
                "count": <int>,
                "events": [<AnomalyEvent>, ...],
                "baseline_days": <int>,
                "z_threshold": <float>,
            }
        """
        now = datetime.utcnow()
        target = target_date or now
        day_start = target.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        # 1. Build hourly baseline from historical data
        baseline = self._build_hourly_baseline(day_start)

        # 2. Get today's hourly counts
        today_counts = self._get_hourly_counts(day_start, day_end)

        # 3. Compare and flag anomalies
        events: List[Dict[str, Any]] = []

        # ── Crowd spike & off-hours detection ──
        events.extend(self._detect_count_anomalies(today_counts, baseline))

        # ── Unusual object detection per camera ──
        events.extend(self._detect_unusual_objects(day_start, day_end))

        # Sort by severity (high first), then by hour
        severity_order = {"high": 0, "medium": 1, "low": 2}
        events.sort(key=lambda e: (severity_order.get(e["severity"], 9), e.get("hour", 0)))

        result = {
            "date": day_start.strftime("%Y-%m-%d"),
            "count": len(events),
            "events": events,
            "baseline_days": self.baseline_days,
            "z_threshold": self.z_threshold,
        }

        # Cache to DB for dashboard quick-reads
        self._cache_result(result)

        return result

    # ── private: baseline ─────────────────────────────────────────────────

    def _build_hourly_baseline(self, day_start: datetime) -> Dict[int, Dict[str, float]]:
        """
        Compute mean and std_dev of person_count per hour-of-day
        over the last N days (excluding *day_start* itself).

        Returns: {hour: {"mean": float, "std": float, "n": int}, ...}
        """
        baseline_start = day_start - timedelta(days=self.baseline_days)

        pipeline = [
            {
                "$match": {
                    "timestamp": {
                        "$gte": baseline_start.isoformat(),
                        "$lt": day_start.isoformat(),
                    }
                }
            },
            # Parse hour from ISO timestamp string
            {
                "$addFields": {
                    "ts_date": {"$dateFromString": {"dateString": "$timestamp"}},
                }
            },
            {
                "$addFields": {
                    "hour": {"$hour": "$ts_date"},
                    "day_str": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ts_date"}},
                }
            },
            # Sum person_count per (day, hour)
            {
                "$group": {
                    "_id": {"day": "$day_str", "hour": "$hour"},
                    "total_persons": {"$sum": {"$ifNull": ["$person_count", 0]}},
                }
            },
            # Now aggregate across days per hour
            {
                "$group": {
                    "_id": "$_id.hour",
                    "values": {"$push": "$total_persons"},
                    "n": {"$sum": 1},
                    "mean": {"$avg": "$total_persons"},
                }
            },
        ]

        try:
            results = list(detections.aggregate(pipeline, allowDiskUse=True))
        except Exception as e:
            logger.error(f"Baseline aggregation failed: {e}")
            return {}

        baseline: Dict[int, Dict[str, float]] = {}
        for doc in results:
            hour = doc["_id"]
            values = doc.get("values", [])
            mean = doc.get("mean", 0.0)
            n = doc.get("n", 0)
            # Compute std dev manually
            if n >= 2:
                variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0
            baseline[hour] = {"mean": mean, "std": std, "n": n}

        logger.info(f"Built baseline from {self.baseline_days} days: {len(baseline)} hours with data")
        return baseline

    # ── private: today's counts ───────────────────────────────────────────

    def _get_hourly_counts(
        self, day_start: datetime, day_end: datetime
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get person_count totals per (camera_id, hour) for the target day.

        Returns: {hour: {"total": int, "cameras": {cam_id: count, ...}}, ...}
        """
        pipeline = [
            {
                "$match": {
                    "timestamp": {
                        "$gte": day_start.isoformat(),
                        "$lt": day_end.isoformat(),
                    }
                }
            },
            {
                "$addFields": {
                    "ts_date": {"$dateFromString": {"dateString": "$timestamp"}},
                }
            },
            {
                "$addFields": {
                    "hour": {"$hour": "$ts_date"},
                }
            },
            {
                "$group": {
                    "_id": {"hour": "$hour", "camera_id": "$camera_id"},
                    "person_total": {"$sum": {"$ifNull": ["$person_count", 0]}},
                }
            },
        ]

        try:
            results = list(detections.aggregate(pipeline, allowDiskUse=True))
        except Exception as e:
            logger.error(f"Today hourly counts failed: {e}")
            return {}

        hourly: Dict[int, Dict[str, Any]] = {}
        for doc in results:
            hour = doc["_id"]["hour"]
            cam = doc["_id"].get("camera_id")
            count = doc.get("person_total", 0)
            if hour not in hourly:
                hourly[hour] = {"total": 0, "cameras": {}}
            hourly[hour]["total"] += count
            if cam is not None:
                hourly[hour]["cameras"][cam] = count

        return hourly

    # ── private: anomaly detectors ────────────────────────────────────────

    def _detect_count_anomalies(
        self,
        today: Dict[int, Dict[str, Any]],
        baseline: Dict[int, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Detect crowd spikes and off-hours activity."""
        events: List[Dict[str, Any]] = []

        for hour, data in today.items():
            actual = data["total"]
            if actual == 0:
                continue

            base = baseline.get(hour)

            if base is None or base["n"] == 0:
                # Off-hours: activity where historically there is none
                events.append({
                    "type": "off_hours",
                    "hour": hour,
                    "actual_count": actual,
                    "baseline_mean": 0,
                    "z_score": None,
                    "severity": "high",
                    "cameras": data.get("cameras", {}),
                    "description": (
                        f"Activity detected at {hour:02d}:00 "
                        f"({actual} persons) — no historical baseline for this hour"
                    ),
                })
                continue

            mean = base["mean"]
            std = base["std"]

            # Avoid division by zero — if std==0 but actual != mean, it's unusual
            if std == 0:
                if actual > mean:
                    z = float("inf")
                else:
                    continue  # identical to historical, not anomalous
            else:
                z = (actual - mean) / std

            if z >= self.z_threshold:
                severity = self._z_to_severity(z)
                events.append({
                    "type": "crowd_spike",
                    "hour": hour,
                    "actual_count": actual,
                    "baseline_mean": round(mean, 1),
                    "baseline_std": round(std, 1),
                    "z_score": round(z, 2),
                    "severity": severity,
                    "cameras": data.get("cameras", {}),
                    "description": (
                        f"Crowd spike at {hour:02d}:00 — "
                        f"{actual} persons (baseline ~{mean:.0f}±{std:.0f}, z={z:.1f})"
                    ),
                })

        return events

    def _detect_unusual_objects(
        self, day_start: datetime, day_end: datetime
    ) -> List[Dict[str, Any]]:
        """
        Detect object types that appear today but were rarely seen
        in the baseline period on the same camera.
        """
        baseline_start = day_start - timedelta(days=self.baseline_days)
        events: List[Dict[str, Any]] = []

        # Get today's object type distribution per camera
        pipeline_today = [
            {
                "$match": {
                    "timestamp": {"$gte": day_start.isoformat(), "$lt": day_end.isoformat()}
                }
            },
            {"$unwind": "$objects"},
            {
                "$group": {
                    "_id": {
                        "camera_id": "$camera_id",
                        "object_name": "$objects.object_name",
                    },
                    "count": {"$sum": 1},
                }
            },
        ]

        # Get baseline object type distribution per camera
        pipeline_baseline = [
            {
                "$match": {
                    "timestamp": {"$gte": baseline_start.isoformat(), "$lt": day_start.isoformat()}
                }
            },
            {"$unwind": "$objects"},
            {
                "$group": {
                    "_id": {
                        "camera_id": "$camera_id",
                        "object_name": "$objects.object_name",
                    },
                    "count": {"$sum": 1},
                }
            },
        ]

        try:
            today_objects = list(detections.aggregate(pipeline_today, allowDiskUse=True))
            baseline_objects = list(detections.aggregate(pipeline_baseline, allowDiskUse=True))
        except Exception as e:
            logger.error(f"Unusual object detection failed: {e}")
            return events

        # Build baseline lookup: (camera_id, object_name) -> count
        base_counts: Dict[tuple, int] = {}
        for doc in baseline_objects:
            key = (doc["_id"].get("camera_id"), doc["_id"].get("object_name"))
            base_counts[key] = doc.get("count", 0)

        # Check today's objects against baseline
        for doc in today_objects:
            cam = doc["_id"].get("camera_id")
            obj_name = doc["_id"].get("object_name")
            today_count = doc.get("count", 0)

            if not obj_name or obj_name == "person":
                # Person anomalies handled by count detector
                continue

            baseline_count = base_counts.get((cam, obj_name), 0)

            # Flag if object was never/rarely seen: today > 3 instances and baseline < 2
            if today_count >= 3 and baseline_count < 2:
                events.append({
                    "type": "unusual_object",
                    "hour": None,
                    "camera_id": cam,
                    "object_name": obj_name,
                    "actual_count": today_count,
                    "baseline_count": baseline_count,
                    "z_score": None,
                    "severity": "medium",
                    "description": (
                        f"Unusual object '{obj_name}' on Camera {cam}: "
                        f"{today_count} today vs {baseline_count} in past {self.baseline_days} days"
                    ),
                })

        return events

    # ── helpers ────────────────────────────────────────────────────────────

    def _z_to_severity(self, z: float) -> str:
        if z >= SEVERITY_THRESHOLDS["high"]:
            return "high"
        elif z >= SEVERITY_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    def _cache_result(self, result: Dict[str, Any]) -> None:
        """Upsert today's anomaly result to MongoDB for fast dashboard reads."""
        try:
            anomaly_events.update_one(
                {"date": result["date"]},
                {"$set": result},
                upsert=True,
            )
        except Exception as e:
            logger.error(f"Failed to cache anomaly result: {e}")
