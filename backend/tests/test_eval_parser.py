"""Offline tests for the parser-quality evaluation script.

No LLM call, no network access and no model import is required: the module
under test only pulls in the standard library at import time, so every test
here exercises the *pure* comparison logic (field agreement, time-window
comparison, aggregation, report writers) with a synthetic fixture.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts import eval_parser as ep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _query(qid: str, category: str, text: str, parsed: Dict[str, Any], split: str = "test") -> Dict[str, Any]:
    return {
        "qid": qid,
        "category": category,
        "split": split,
        "group": "2026-03-01T17:00",
        "text": text,
        "parsed": parsed,
    }


def _report_from(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a report from synthetic rows (no LLM involved)."""
    meta = {
        "cache_path": "backend/benchmarks/parser_cache/test-model.json",
        "n_calls": len(rows),
        "n_from_cache": 0,
        "models_used": {"test-model": len(rows)},
    }
    return ep.build_report(
        rows=rows,
        model="test-model",
        provider="opencode",
        meta=meta,
        queries_path=ROOT / "backend" / "benchmarks" / "queries.jsonl",
        reference_date="2026-03-01",
        n_total_queries=len(rows),
    )


def _entry(parsed: Dict[str, Any], seconds: float = 1.0, ok: bool = True,
           error: str | None = None) -> Dict[str, Any]:
    return {
        "ok": ok,
        "parsed": parsed,
        "error": error,
        "seconds": seconds,
        "model": "test-model",
    }


# ---------------------------------------------------------------------------
# Scalar comparators
# ---------------------------------------------------------------------------

def test_camera_id_exact_match():
    assert ep.compare_camera_id(0, 0) is True
    assert ep.compare_camera_id(0, "0") is True
    assert ep.compare_camera_id(0, 1) is False
    assert ep.compare_camera_id(0, None) is False


def test_colour_zone_action_are_case_insensitive():
    assert ep.compare_ci_exact("Olive", "olive") is True
    assert ep.compare_ci_exact("Red", "red") is True
    assert ep.compare_ci_exact("corridor", "Corridor") is True
    assert ep.compare_ci_exact("smoke", "Smoke") is True
    assert ep.compare_ci_exact("Olive", "Red") is False
    assert ep.compare_ci_exact(None, "Red") is False


def test_object_class_set_agreement():
    # Same class, different surface form.
    assert ep.compare_object_class("person", "people") is True
    assert ep.compare_object_class("person", "persons") is True
    assert ep.compare_object_class("person", "pedestrian") is True
    # Different class.
    assert ep.compare_object_class("person", "car") is False
    # Missing side never agrees.
    assert ep.compare_object_class("person", None) is False
    assert ep.compare_object_class(None, "person") is False


def test_canonical_object_classes_handles_lists_and_commas():
    assert ep.canonical_object_classes("person") == ["person"]
    assert ep.canonical_object_classes("people") == ["person"]
    assert ep.canonical_object_classes(["person", "car"]) == ["person", "car"]
    assert ep.canonical_object_classes("people, car") == ["person", "car"]
    assert ep.canonical_object_classes(None) == []


def test_count_constraint_agreement_rule():
    # Same numeric target -> agreement (the reference stores no operator).
    assert ep.compare_count_constraint(3, {"gte": 3}) is True
    assert ep.compare_count_constraint(3, {"eq": 3}) is True
    assert ep.compare_count_constraint(3, 3) is True
    # Different target -> disagreement.
    assert ep.compare_count_constraint(3, {"gte": 4}) is False
    # A constraint on one side only is never an agreement.
    assert ep.compare_count_constraint(None, {"gt": 2}) is False
    assert ep.compare_count_constraint(3, None) is False


def test_count_constraint_values():
    assert ep.count_constraint_values({"gt": 3}) == [3]
    assert ep.count_constraint_values(4) == [4]
    assert ep.count_constraint_values("more than 5") == [5]
    assert ep.count_constraint_values(None) == []
    assert ep.count_constraint_values(True) == []


def test_intent_comparison_is_case_insensitive():
    assert ep.compare_intent("track", "track") is True
    assert ep.compare_intent("track", "TRACK") is True
    assert ep.compare_intent("track", "find") is False


# ---------------------------------------------------------------------------
# compare_field outcomes
# ---------------------------------------------------------------------------

def test_compare_field_outcomes():
    agree = ep.compare_field("color", "Red", "Red")
    assert agree["status"] == "agree" and agree["agree"] is True

    mismatch = ep.compare_field("color", "Red", "Blue")
    assert mismatch["status"] == "mismatch" and mismatch["agree"] is False

    missing = ep.compare_field("color", "Red", None)
    assert missing["status"] == "missing_in_parse" and missing["agree"] is False

    spurious = ep.compare_field("color", None, "Blue")
    assert spurious["status"] == "spurious_in_parse" and spurious["agree"] is None

    absent = ep.compare_field("color", None, None)
    assert absent["status"] == "absent" and absent["agree"] is None


# ---------------------------------------------------------------------------
# Time-window comparison
# ---------------------------------------------------------------------------

REF = ("2026-03-01T17:01:00", "2026-03-01T17:03:00")


def test_time_window_superset_overlaps():
    """The parsed window contains the reference window."""
    detail = ep.compare_time_window(REF, ("2026-03-01T17:00:00", "2026-03-01T17:04:00"))
    assert detail["overlap"] is True
    assert detail["abs_error_start_s"] == 60.0
    assert detail["abs_error_end_s"] == 60.0
    assert 0.0 < detail["iou"] < 1.0


def test_time_window_subset_overlaps():
    """The parsed window is contained in the reference window."""
    detail = ep.compare_time_window(REF, ("2026-03-01T17:01:30", "2026-03-01T17:02:00"))
    assert detail["overlap"] is True
    assert detail["abs_error_start_s"] == 30.0
    assert detail["abs_error_end_s"] == 60.0
    assert 0.0 < detail["iou"] < 1.0


def test_time_window_disjoint():
    """A window that does not touch the reference window."""
    detail = ep.compare_time_window(REF, ("2026-03-01T18:00:00", "2026-03-01T18:05:00"))
    assert detail["overlap"] is False
    assert detail["iou"] == 0.0
    assert detail["abs_error_start_s"] == pytest.approx(3540.0)


def test_time_window_point_inside_reference():
    """A degenerate (single-instant) window inside the reference overlaps."""
    detail = ep.compare_time_window(REF, ("2026-03-01T17:02:00", "2026-03-01T17:02:00"))
    assert detail["overlap"] is True
    assert detail["iou"] == 0.0


def test_time_window_outside_by_less_than_tolerance():
    """Just outside the reference window: no overlap, but within tolerance."""
    detail = ep.compare_time_window(REF, ("2026-03-01T17:03:30", "2026-03-01T17:04:00"))
    assert detail["overlap"] is False
    assert detail["overlap_with_tolerance"] is True


def test_time_window_missing_is_a_disagreement():
    outcome = ep.compare_field("time_window", REF, (None, None))
    assert outcome["status"] == "missing_in_parse"
    assert outcome["agree"] is False


def test_time_window_partial_is_flagged():
    detail = ep.compare_time_window(REF, ("2026-03-01T17:02:00", None))
    assert detail["partial_parsed_window"] is True
    assert detail["abs_error_end_s"] is None


def pytest_approx_unused() -> None:  # pragma: no cover - placeholder removed
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_field_counts_and_rates():
    outcomes = [
        ep.compare_field("color", "Red", "Red"),      # agree
        ep.compare_field("color", "Red", "Blue"),     # mismatch
        ep.compare_field("color", "Red", None),       # missing in parse
        ep.compare_field("color", None, "Blue"),      # spurious
        ep.compare_field("color", None, None),        # absent
    ]
    stat = ep.aggregate_field(outcomes)
    assert stat["n_evaluated"] == 5
    assert stat["n_reference"] == 3
    assert stat["n_parsed"] == 3
    assert stat["n_both"] == 2
    assert stat["n_agree"] == 1
    assert stat["n_mismatch"] == 1
    assert stat["n_missing_in_parse"] == 1
    assert stat["n_spurious_in_parse"] == 1
    assert stat["agreement_rate"] == 1 / 3
    assert stat["agreement_rate_among_both"] == 0.5
    assert stat["agreement_rate_pct"] == 33.3


def test_aggregate_field_rate_undefined_without_reference():
    stat = ep.aggregate_field([ep.compare_field("count_constraint", None, {"gt": 2})])
    assert stat["n_reference"] == 0
    assert stat["agreement_rate"] is None
    assert stat["agreement_rate_pct"] is None
    assert stat["n_spurious_in_parse"] == 1


def test_category_aggregation():
    query_a = _query(
        "q_count_01", "count", "how many people around 17:01?",
        {"time_start": "2026-03-01T17:01:00", "time_end": "2026-03-01T17:03:00",
         "camera_id": 0, "count_query": True, "count_target_in_text": None},
    )
    query_b = _query(
        "q_find_01", "find", "find a person wearing Olive clothing between 17:01 and 17:03.",
        {"time_start": "2026-03-01T17:01:00", "time_end": "2026-03-01T17:03:00",
         "camera_id": 0, "objects.object_name": "person", "objects.color": "Olive",
         "attributes": ["coat or jacket"]},
    )
    parse_a = {"timestamp": {"$gte": "2026-03-01T17:01:00", "$lte": "2026-03-01T17:03:00"},
               "objects.object_name": "person", "__intent": "count"}
    parse_b = {"timestamp": {"$gte": "2026-03-01T17:01:00", "$lte": "2026-03-01T17:03:00"},
               "objects.object_name": "person", "objects.color": "Olive",
               "__intent": "find"}

    records = [ep.evaluate_query(query_a, parse_a), ep.evaluate_query(query_b, parse_b)]
    assert records[0]["fields"]["intent"]["status"] == "agree"
    assert records[1]["fields"]["color"]["status"] == "agree"

    overall = ep.aggregate_fields(records)
    assert overall["object_name"]["n_reference"] == 1
    assert overall["object_name"]["agreement_rate"] == 1.0
    assert overall["color"]["n_reference"] == 1
    assert overall["color"]["agreement_rate"] == 1.0
    assert overall["color"]["n_parsed"] == 1

    # The count query has no reference colour, the find query has no zone:
    # the per-category tables must not mix the two.
    by_category = {
        category: ep.aggregate_fields([r for r in records if r["category"] == category])
        for category in {r["category"] for r in records}
    }
    assert by_category["count"]["color"]["n_reference"] == 0
    assert by_category["find"]["color"]["agreement_rate"] == 1.0
    assert by_category["find"]["zone"]["n_reference"] == 0


def test_window_stats_median_and_overlap():
    query = _query(
        "q1", "count", "q",
        {"time_start": "2026-03-01T17:01:00", "time_end": "2026-03-01T17:03:00"},
    )
    records = []
    for start, end in (("2026-03-01T17:01:00", "2026-03-01T17:03:00"),
                       ("2026-03-01T17:01:30", "2026-03-01T17:02:30"),
                       ("2026-03-01T18:00:00", "2026-03-01T18:01:00")):
        rec = ep.evaluate_query(query, {"timestamp": {"$gte": start, "$lte": end}})
        records.append(rec)
    stats = ep.window_stats(records)
    assert stats["n_evaluated"] == 3
    assert stats["window_overlap_rate"] == 2 / 3
    assert stats["median_abs_error_start_s"] == 30.0
    assert stats["median_abs_error_s_r"] == 30.0


def test_latency_stats():
    stat = ep.latency_stats([1.0, 2.0, 3.0, 10.0])
    assert stat["n"] == 4
    assert stat["mean_seconds"] == 4.0
    assert stat["median_seconds"] == 2.5
    assert stat["max_seconds"] == 10.0
    assert 2.0 < stat["p95_seconds"] < 10.0
    empty = ep.latency_stats([])
    assert empty["n"] == 0 and empty["mean_seconds"] is None


def test_camera_explicit_stats_only_counts_named_cameras():
    named = _query("q1", "count", "how many people on camera 0?",
                   {"camera_id": 0, "time_start": "2026-03-01T17:01:00",
                    "time_end": "2026-03-01T17:03:00"})
    unnamed = _query("q2", "count", "how many people?",
                     {"camera_id": 0, "time_start": "2026-03-01T17:01:00",
                      "time_end": "2026-03-01T17:03:00"})
    records = [
        ep.evaluate_query(named, {"camera_id": 0}),
        ep.evaluate_query(unnamed, {}),
    ]
    stat = ep.camera_explicit_stats(records)
    assert stat["n_named_in_text"] == 1
    assert stat["n_agree"] == 1
    assert stat["agreement_rate"] == 1.0


# ---------------------------------------------------------------------------
# Report assembly and writers
# ---------------------------------------------------------------------------

def _synthetic_rows() -> List[Dict[str, Any]]:
    q_count = _query(
        "q_count_01", "count", "how many people were visible around 17:01:00?",
        {"time_start": "2026-03-01T17:01:00", "time_end": "2026-03-01T17:02:30",
         "camera_id": 0, "count_query": True, "count_target_in_text": None},
    )
    q_find = _query(
        "q_find_01", "find", "find a person wearing Olive clothing between 17:01 and 17:03.",
        {"time_start": "2026-03-01T17:01:00", "time_end": "2026-03-01T17:03:00",
         "camera_id": 0, "objects.object_name": "person", "objects.color": "Olive",
         "attributes": ["coat or jacket"]},
        split="tune",
    )
    q_track = _query(
        "q_track_01", "track", "track the person wearing Olive clothing from 17:01 to 17:04.",
        {"time_start": "2026-03-01T17:01:00", "time_end": "2026-03-01T17:04:42",
         "camera_id": 0, "objects.object_name": "person", "objects.color": "Olive",
         "intent": "track"},
    )
    q_broken = _query(
        "q_action_01", "action", "is there smoke around 17:02:19?",
        {"time_start": "2026-03-01T17:01:34", "time_end": "2026-03-01T17:03:04",
         "camera_id": 0, "action": "smoke", "event": "smoke"},
    )
    return [
        {"query": q_count, "entry": _entry({
            "timestamp": {"$gte": "2026-03-01T17:01:00", "$lte": "2026-03-01T17:02:00"},
            "objects.object_name": "person", "__intent": "count",
        }, seconds=2.5)},
        {"query": q_find, "entry": _entry({
            "timestamp": {"$gte": "2026-03-01T17:01:00", "$lte": "2026-03-01T17:03:00"},
            "objects.object_name": "person", "objects.color": "Olive", "__intent": "find",
        }, seconds=3.0)},
        {"query": q_track, "entry": _entry({
            "timestamp": {"$gte": "2026-03-01T17:01:00", "$lte": "2026-03-01T17:04:42"},
            "objects.object_name": "person", "objects.color": "Olive", "__intent": "track",
        }, seconds=4.0)},
        {"query": q_broken, "entry": _entry({}, seconds=0.5, ok=False,
                                           error="LLMError: no JSON object in LLM reply")},
    ]


def test_build_report_counts_failures_and_splits():
    report = _report_from(_synthetic_rows())
    assert report["totals"]["n"] == 4
    assert report["totals"]["n_parse_ok"] == 3
    assert report["totals"]["n_parse_failed"] == 1
    assert report["dataset"]["splits"] == {"test": 3, "tune": 1}
    assert report["failed_query_ids"] == ["q_action_01"]
    assert report["latency"]["n"] == 4
    assert report["latency"]["mean_seconds"] == 2.5
    assert report["field_agreement_overall"]["color"]["agreement_rate"] == 1.0
    assert report["field_agreement_overall"]["intent"]["n_agree"] == 3
    # The failed query cannot agree with anything.
    assert report["field_agreement_overall"]["action"]["n_agree"] == 0
    assert report["ambiguous_query_ids"], "the broken query must be reported"
    assert report["limitations"], "the report must state its limitations"


def test_build_report_limitations_use_computed_numbers():
    report = _report_from(_synthetic_rows())
    joined = " ".join(report["limitations"])
    assert "single-camera" in joined
    assert "not human-annotated" in joined
    assert "agreement rates" in joined
    assert "4-camera deployment" in joined
    # The count-constraint limitation quotes the computed number of
    # reference parses that state a count target (0 here).
    assert "0 of the 4 reference parses" in joined


def test_conflicts_are_reported_separately_from_dropped_fields():
    query = _query(
        "q_find_01", "find", "find a person wearing Olive clothing",
        {"time_start": "2026-03-01T17:01:00", "time_end": "2026-03-01T17:03:00",
         "camera_id": 0, "objects.object_name": "person", "objects.color": "Olive"},
    )
    rows = [{"query": query, "entry": _entry({
        "camera_id": 0,
        "timestamp": {"$gte": "2026-03-01T17:01:00", "$lte": "2026-03-01T17:03:00"},
        "objects.object_name": "person",
        "objects.color": "Red",       # the reference says Olive -> conflict
        "__intent": "find",
    }, seconds=1.0)}]
    report = _report_from(rows)
    assert report["conflict_query_ids"] == ["q_find_01"]
    assert report["ambiguous_query_ids"] == ["q_find_01"]
    assert report["dropped_field_query_ids"] == []
    assert report["dropped_field_breakdown"]["color"] == 0
    assert report["field_agreement_overall"]["color"]["agreement_rate"] == 0.0
    assert report["field_agreement_overall"]["color"]["n_mismatch"] == 1


def test_writers_emit_consistent_files(tmp_path):
    report = _report_from(_synthetic_rows())
    json_path = ep.write_json_report(report, tmp_path / "parser_results.json")
    md_path = ep.write_markdown_report(report, tmp_path / "parser_results.md")
    assert json_path.exists() and md_path.exists()

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["totals"]["n"] == 4
    assert written["scope"]["model_requested"] == "test-model"

    md = md_path.read_text(encoding="utf-8")
    assert "# NL parser evaluation" in md
    assert "## Known limitations" in md
    assert "## Per-field agreement (all queries)" in md
    assert "## Time window" in md
    assert "## Latency" in md
    assert "not human-annotated" in md
    assert "single-camera" in md

    # Every number printed in the markdown must exist in the JSON.
    json_numbers = _collect_numbers(json_path.read_text(encoding="utf-8"))
    md_numbers = _collect_numbers(md)
    missing = [n for n in md_numbers if n not in json_numbers]
    assert not missing, f"markdown numbers missing from the JSON: {missing}"


def _collect_numbers(text: str) -> set:
    """Numeric tokens of a text, ignoring ids, timestamps and code blocks."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # fenced code blocks
    text = re.sub(r"[A-Za-z_][A-Za-z_0-9]*\d[A-Za-z_0-9]*", " ", text)  # q ids
    text = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?", " ", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text)
    return {round(float(token), 2) for token in re.findall(r"\d+(?:\.\d+)?", text)}
