"""Deterministic benchmark generator (R1-1, R1-8, R1-10, R1-14).

Creates under backend/benchmarks/:
- queries.jsonl: 150 NL queries, 30 per category (count/find/track/action/filter)
  fields: qid, category, text, relevant (clip ids), video_id, time_bucket
- alert_utterances.jsonl: 120 utterances (60 clean + 60 stress)
  fields: uid, text, stress, expected_event, expected_zone, expected_count, expected_tod
- person_pairs.jsonl: 50 calibration pairs (identity-disjoint placeholder)
  fields: pid, identity_a, identity_b, same, camera_a, camera_b
- splits.json: tune/test split (50 tune / 100 test stratified)
- README.md: construction, annotation, agreement plan, release note

No real footage needed: clip/video ids are stable synthetic ids so eval scripts
and tests run offline. Replace relevant/expected with human labels when available.
Seed fixed at 0 for reproducibility.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
rng = random.Random(0)

CATEGORIES = ["count", "find", "track", "action", "filter"]

COUNT_T = [
    "how many cars in Zone A between {h1}:00 and {h2}:00",
    "count people in Zone B after {h1} PM",
    "how many persons entered Gate in the last hour (cam {c})",
    "number of vehicles in Parking between {h1}:00-{h2}:00",
]
FIND_T = [
    "red jacket near the entrance",
    "person with black bag in Zone A",
    "white car near Gate cam {c}",
    "person wearing blue shirt Zone B",
]
TRACK_T = [
    "track person in red coat across cam {c} and cam {c2}",
    "follow white van leaving Parking",
    "track loitering person near Entrance",
    "person moving from Zone A to Zone B",
]
ACTION_T = [
    "person running near gate",
    "fight near Entrance cam {c}",
    "person loitering in Zone B for a while",
    "crowd gathering in Zone A",
]
FILTER_T = [
    "persons in Zone A between 22:00 and 23:00",
    "cars in Parking after 10 PM cam {c}",
    "all persons with bags yesterday Zone C",
    "snapshots of crowd in Entrance last hour",
]


def _mk(qid: str, cat: str, text: str, i: int) -> dict:
    return {
        "qid": qid,
        "category": cat,
        "text": text,
        "relevant": [f"clip_{cat}_{(i % 8) + 1:02d}a", f"clip_{cat}_{(i % 8) + 1:02d}b"],
        "video_id": f"vid_{(i % 6) + 1}",
        "time_bucket": f"2026-10-0{(i % 5) + 1}",
    }


def gen_queries() -> list:
    out = []
    templates = {"count": COUNT_T, "find": FIND_T, "track": TRACK_T,
                 "action": ACTION_T, "filter": FILTER_T}
    for cat in CATEGORIES:
        for i in range(30):
            t = templates[cat][i % len(templates[cat])]
            text = t.format(h1=18 + (i % 5), h2=20 + (i % 4), c=(i % 4),
                            c2=((i + 1) % 4))
            out.append(_mk(f"q_{cat}_{i + 1:02d}", cat, text, i))
    return out


CLEAN_ALERTS = [
    ("alert me when more than five people gather in Zone A after 10 PM", "crowd", "Zone A", 5, "22:00-06:00"),
    ("notify me if someone loiters in Zone B for over a minute", "loiter", "Zone B", 1, None),
    ("warn me on fight near Entrance", "fight", "Entrance", 2, None),
    ("count cars entering Parking after 9 PM", "count", "Parking", 1, "21:00-06:00"),
]


def gen_alerts() -> list:
    out = []
    for i in range(60):
        t, ev, z, c, tod = CLEAN_ALERTS[i % len(CLEAN_ALERTS)]
        out.append({"uid": f"a_clean_{i + 1:02d}", "text": t, "stress": "clean",
                    "expected_event": ev, "expected_zone": z,
                    "expected_count": c, "expected_tod": tod})
    stress_pool = [
        ("alert if a few people gather over there", "ambiguous"),
        ("warn me if Zone A or Zone B but not at night, actually during...", "conflicting"),
        ("alert me when people gather", "missing"),
        ("notify me to follow him home", "unsupported"),
        ("alert on the back entrance thing tonight", "ambiguous"),
        ("count vehicles, or maybe persons, after sometime", "conflicting"),
    ]
    for i in range(60):
        t, k = stress_pool[i % len(stress_pool)]
        out.append({"uid": f"a_stress_{i + 1:02d}", "text": t, "stress": k,
                    "expected_event": None, "expected_zone": None,
                    "expected_count": None, "expected_tod": None})
    return out


def gen_pairs() -> list:
    out = []
    for i in range(50):
        same = (i % 2 == 0)
        out.append({
            "pid": f"p_{i + 1:02d}",
            "identity_a": f"id_{(i % 25) + 1:03d}",
            "identity_b": f"id_{(i % 25) + 1:03d}" if same else f"id_{((i + 13) % 25) + 1:03d}",
            "same": same,
            "camera_a": (i % 4), "camera_b": ((i + 1) % 4),
        })
    return out


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    queries = gen_queries()
    (HERE / "queries.jsonl").write_text(
        "\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8")
    alerts = gen_alerts()
    (HERE / "alert_utterances.jsonl").write_text(
        "\n".join(json.dumps(a) for a in alerts) + "\n", encoding="utf-8")
    pairs = gen_pairs()
    (HERE / "person_pairs.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8")
    tune = [q["qid"] for q in queries if int(q["qid"].split("_")[-1]) <= 10]
    test = [q["qid"] for q in queries if int(q["qid"].split("_")[-1]) > 10]
    (HERE / "splits.json").write_text(
        json.dumps({"tune": tune, "test": test,
                    "note": "50 tune / 100 test stratified; replace with human labels"},
                   indent=2), encoding="utf-8")
    print(f"wrote {len(queries)} queries, {len(alerts)} alerts, {len(pairs)} pairs")


if __name__ == "__main__":
    main()
