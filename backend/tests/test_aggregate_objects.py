"""aggregate_objects must keep a single representative colour."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.retrieval_utils import aggregate_objects  # noqa: E402


def _obj(color, conf=0.9):
    return {"object_name": "person", "track_id": 1, "confidence": conf, "color": color}


def test_colour_stays_scalar_when_frames_disagree():
    agg = aggregate_objects([_obj("Black"), _obj("Brown"), _obj("Black")])
    assert len(agg) == 1
    assert "," not in agg[0]["color"]
    assert agg[0]["color"] == "Black"  # dominant


def test_full_colour_set_is_preserved():
    agg = aggregate_objects([_obj("Black"), _obj("Brown"), _obj("Blue")])
    assert agg[0]["colors"] == ["Black", "Brown", "Blue"]
    assert agg[0]["color"] in agg[0]["colors"]


def test_ties_prefer_the_first_seen_colour():
    agg = aggregate_objects([_obj("Brown"), _obj("Black")])
    assert agg[0]["color"] == "Brown"


def test_distinct_tracks_stay_separate():
    a = {"object_name": "person", "track_id": 1, "confidence": 0.9, "color": "Brown"}
    b = {"object_name": "person", "track_id": 2, "confidence": 0.8, "color": "Black"}
    agg = aggregate_objects([a, b])
    assert len(agg) == 2
    assert {o["color"] for o in agg} == {"Brown", "Black"}


def test_missing_track_ids_fall_back_to_class_grouping():
    a = {"object_name": "person", "track_id": -1, "color": "Brown"}
    b = {"object_name": "person", "track_id": -1, "color": "Brown"}
    agg = aggregate_objects([a, b])
    assert len(agg) == 1


def test_confidence_is_averaged():
    agg = aggregate_objects([_obj("Black", 0.8), _obj("Black", 0.6)])
    assert abs(agg[0]["confidence"] - 0.7) < 1e-9


def test_single_instance_is_unchanged():
    agg = aggregate_objects([_obj("Red")])
    assert agg[0]["color"] == "Red"
    assert agg[0]["colors"] == ["Red"]
