"""Build a protocol-matched control from the released-subset queries.

The raw released-subset benchmark (``queries.jsonl``) is deliberately
metadata-derived, but two of its categories are *not* protocol-matched to the
full-deployment C1 study:

* count questions are ``how many`` questions with no target count, so the
  structured branch has no numeric target to rank; and
* action questions carry explicit time windows, so the structured branch has a
  strong time cue even though the paper's action override intentionally
  suppresses structured evidence.

This script rewrites only those two categories, using the same released
metadata and the same tune/test sizes:

* count -> "show clips with exactly N people between ..." with an explicit
  ``count_constraint``; relevance is unchanged (clips whose caption count is N
  in the window).
* action -> event queries ("smoke", "fighting") with the full released time
  range and no explicit time cue; relevance is every clip whose VLM caption
  contains the event.  The motion-energy proxy rows are omitted because they
  are not semantic actions and would not test the paper's action override.

The find/track/filter rows are copied unchanged.  The control is a post-hoc
fairness check, not the primary benchmark and not human-annotated.

Run:
    python backend/benchmarks/build_subset_control.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "backend" / "benchmarks"
SCENE_META = ROOT / "data" / "index" / "faiss" / "meta.json"

OUT_QUERIES = BENCH / "queries_control.jsonl"
OUT_SPLITS = BENCH / "splits_control.json"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_clip_interval(clip_id: str) -> Optional[tuple[str, str]]:
    match = re.match(r"(.+?)__to__(.+)", clip_id)
    if not match:
        return None
    return match.group(1), match.group(2)


def main() -> int:
    queries = read_jsonl(BENCH / "queries.jsonl")
    scene = json.loads(SCENE_META.read_text(encoding="utf-8"))

    # Full released range, used by event-only action queries so that the
    # structured branch has no informative time cue.
    frame_times = sorted(str(x["frame_ts"]) for x in scene if x.get("frame_ts"))
    full_start, full_end = frame_times[0], frame_times[-1]

    event_clips: Dict[str, List[str]] = defaultdict(list)
    for frame in scene:
        caption = str(frame.get("caption", "")).lower()
        clip_path = frame.get("clip_path")
        if not clip_path:
            continue
        clip_id = Path(str(clip_path)).stem
        for event in ("smoke", "fighting"):
            if event in caption and clip_id not in event_clips[event]:
                event_clips[event].append(clip_id)

    out: List[Dict[str, Any]] = []
    action_sources: List[Dict[str, Any]] = []
    for q in queries:
        cat = q["category"]
        if cat == "action":
            parsed = q.get("parsed") or {}
            if parsed.get("event") in {"smoke", "fighting"}:
                action_sources.append(q)
            continue
        if cat != "count":
            row = dict(q)
            row["control_note"] = "copied from released benchmark"
            out.append(row)
            continue

        if cat == "count":
            target = q.get("ground_truth_count")
            if target is None:
                raise RuntimeError(f"count query {q['qid']} lacks ground_truth_count")
            text = (
                f"show clips with exactly {target} people between "
                f"{_fmt_time(q['time_start'])} and {_fmt_time(q['time_end'])}"
            )
            parsed = dict(q.get("parsed") or {})
            parsed["count_query"] = True
            parsed["count_target_in_text"] = int(target)
            parsed["count_constraint"] = {"eq": int(target)}
            row = dict(q)
            row.update(
                {
                    "qid": q["qid"].replace("q_count_", "qc_count_"),
                    "text": text,
                    "parsed": parsed,
                    "control_note": (
                        "count target made explicit from the anchor clip; "
                        "relevance set unchanged"
                    ),
                }
            )
            out.append(row)
            continue

    # Action control: two event-only templates for every source event row.
    # This replaces the 15 event + 15 motion rows with 30 semantic-action rows
    # (the motion proxy is not a semantic action and does not exercise the
    # paper's action override).
    for k, q in enumerate(action_sources):
        event = (q.get("parsed") or {}).get("event")
        relevant = list(event_clips.get(str(event), []))
        if not relevant:
            continue
        for variant in range(2):
            idx = k * 2 + variant
            row = dict(q)
            row.update(
                {
                    "qid": f"qc_action_{idx + 1:02d}",
                    "text": _event_text(str(event), idx),
                    "time_start": full_start,
                    "time_end": full_end,
                    "relevant": relevant,
                    "n_relevant": len(relevant),
                    "parsed": {
                        "time_start": full_start,
                        "time_end": full_end,
                        "camera_id": 0,
                        "event": event,
                        "action": event,
                    },
                    "anchor_clip": None,
                    "anchor_ts": None,
                    "group": f"action_{event}",
                    "control_note": (
                        "event-only query without an explicit time cue; relevance "
                        "is every released clip whose VLM caption contains the event"
                    ),
                }
            )
            out.append(row)

    # Rebalance action rows to 10 tune / 20 test across the two events.
    action_rows = [r for r in out if r["category"] == "action"]
    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        by_event[str(row["parsed"]["event"])].append(row)
    tune_remaining = 10
    for event in sorted(by_event):
        rows = sorted(by_event[event], key=lambda r: r["qid"])
        n_tune = int(round(len(rows) / 3.0))
        n_tune = min(n_tune, tune_remaining)
        for j, row in enumerate(rows):
            row["split"] = "tune" if j < n_tune else "test"
        tune_remaining -= n_tune
    if tune_remaining != 0:
        # Give any remainder to the first event's lowest-qid test rows.
        rows = sorted(action_rows, key=lambda r: r["qid"])
        for row in rows:
            if tune_remaining <= 0:
                break
            if row["split"] == "test":
                row["split"] = "tune"
                tune_remaining -= 1

    # Ensure the control has the same 50/100 split sizes as the raw benchmark.
    split_counts = Counter(r["split"] for r in out)
    if split_counts != {"tune": 50, "test": 100}:
        raise RuntimeError(f"unexpected control split sizes: {dict(split_counts)}")

    OUT_QUERIES.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
        encoding="utf-8",
    )
    OUT_SPLITS.write_text(
        json.dumps(
            {
                "n_tune": split_counts["tune"],
                "n_test": split_counts["test"],
                "total": len(out),
                "seed": 20260301,
                "source": "backend/benchmarks/queries.jsonl",
                "tune": [r["qid"] for r in out if r["split"] == "tune"],
                "test": [r["qid"] for r in out if r["split"] == "test"],
                "note": (
                    "Post-hoc protocol-matched control. Count targets are made "
                    "explicit; action queries are event-only with no time cue; "
                    "find/track/filter are copied. Not human-annotated."
                ),
                "categories": dict(Counter(r["category"] for r in out)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT_QUERIES} ({len(out)} rows) and {OUT_SPLITS}")
    print("categories:", dict(Counter(r["category"] for r in out)))
    print("splits:", dict(split_counts))
    return 0


def _fmt_time(iso_ts: str) -> str:
    # 2026-03-01T17:01:00.052839 -> 17:01
    return iso_ts[11:16]


def _event_text(event: str, index: int) -> str:
    templates = {
        "smoke": [
            "show clips of smoke",
            "find scenes with smoke",
            "where is smoke visible?",
            "show me smoke on camera 0",
            "which clips contain smoke?",
        ],
        "fighting": [
            "show clips of people fighting",
            "find scenes with fighting",
            "where are people fighting?",
            "show me fighting on camera 0",
            "which clips contain fighting?",
        ],
    }
    choices = templates[event]
    return choices[index % len(choices)]


if __name__ == "__main__":
    raise SystemExit(main())
