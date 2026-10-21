"""Tests for the parser's routing-intent rule and the end-to-end harness.

The intent (count | track | find) is a deterministic rule over the query text,
not a model output: it decides which branch the retrieval pipeline routes the
query to, so it must be reproducible without calling the model.  The
evaluation harness uses the same helper, which is why these tests pin it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services import nl_parser  # noqa: E402
from backend.scripts import eval_end_to_end as e2e  # noqa: E402


@pytest.mark.parametrize(
    "query,expected",
    [
        ("how many people are visible around 17:01:00?", "count"),
        ("count the people in the scene at 17:01:30.", "count"),
        ("show clips with exactly 3 people between 17:01 and 17:02", "count"),
        ("track the person wearing Olive clothing from 17:01 to 17:04.", "track"),
        ("where does the person in Red clothing move between 18:04 and 18:05?", "track"),
        ("where did the person go?", "track"),
        ("show me a person wearing olive clothing between 17:01 and 17:03.", "find"),
        ("show all red-clothed people in the corridor between 17:01 and 17:05.", "find"),
        ("", "find"),
    ],
)
def test_infer_intent_routing(query, expected):
    assert nl_parser.infer_intent(query) == expected


def test_infer_intent_is_case_insensitive():
    assert nl_parser.infer_intent("HOW MANY people?") == "count"
    assert nl_parser.infer_intent("Track the person") == "track"


def test_infer_intent_tolerates_none():
    assert nl_parser.infer_intent(None) == "find"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end harness
# ---------------------------------------------------------------------------

def _control_row(qid="q_find_01"):
    path = ROOT / "backend" / "benchmarks" / "queries_control.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["qid"] == qid:
                return row
    raise AssertionError(f"{qid} not found")


def test_normalise_uses_the_parsed_time_window():
    row = _control_row()
    parsed = {
        "timestamp": {"$gte": "2026-03-01T17:01:00", "$lte": "2026-03-01T17:03:00"},
        "objects.color": "Olive",
    }
    out = e2e.normalise_for_eval({**row, "parsed": parsed})
    # The parser's window must drive the branch, not the benchmark's window.
    assert out["time_start"] == "2026-03-01T17:01:00"
    assert out["time_end"] == "2026-03-01T17:03:00"
    assert out["parsed"]["objects.color"] == "Olive"


def test_normalise_falls_back_to_the_benchmark_window_without_a_parse():
    row = _control_row()
    out = e2e.normalise_for_eval({**row, "parsed": {}})
    assert out["time_start"] == row["time_start"]
    assert out["time_end"] == row["time_end"]


def test_normalise_carries_the_event_and_the_attributes():
    row = _control_row()
    parsed = {
        "timestamp": {"$gte": "2026-03-01T17:01:00", "$lte": "2026-03-01T17:03:00"},
        "action": "smoke",
        "attributes": ["coat or jacket"],
    }
    out = e2e.normalise_for_eval({**row, "parsed": parsed})
    assert out["parsed"]["event"] == "smoke"
    assert out["parsed"]["attributes"] == ["coat or jacket"]


def test_normalise_coerces_a_bare_integer_count_constraint():
    row = _control_row("qc_count_01")
    parsed = {"count_constraint": 3}
    out = e2e.normalise_for_eval({**row, "parsed": parsed})
    assert out["parsed"]["count_constraint"] == {"eq": 3}
    assert out["parsed"]["count_target_in_text"] == 3


def test_oracle_normalisation_uses_the_reference_filter():
    row = _control_row("q_find_01")
    out = e2e.normalise_oracle(row)
    assert out["parsed"]["objects.color"] == row["parsed"]["objects.color"]
    # The oracle keeps the benchmark's own window.
    assert out["time_start"] == row["time_start"]


def test_oracle_normalisation_recovers_the_count_constraint():
    row = _control_row("qc_count_01")
    out = e2e.normalise_oracle(row)
    assert out["parsed"]["count_constraint"] == {"eq": row["parsed"]["count_target_in_text"]}


def test_cache_key_depends_on_the_query_text_and_the_model():
    a = e2e.cache_key("q1", "show a person", "opencode-go/beam-free")
    b = e2e.cache_key("q1", "show a person", "opencode-go/minimax-m3.1-flash-free")
    c = e2e.cache_key("q1", "show a person in red", "opencode-go/beam-free")
    d = e2e.cache_key("q2", "show a person", "opencode-go/beam-free")
    assert a != b and a != c and a != d
    assert a == e2e.cache_key("q1", "show a person", "opencode-go/beam-free")
