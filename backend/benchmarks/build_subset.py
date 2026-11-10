"""Build the released-subset benchmark from shipped metadata (R1-1, R1-9, R1-14).

This is the reproducible replacement for ``generate.py``'s placeholder
queries.  It reads only:

  data/index/faiss/meta.json        242 scene frames, camera_0
  data/index/faiss_person/meta.json 1,778 person descriptors, camera_0
  data/clips/camera_0/*.mp4         clip intervals (parsed from filenames)
  data/snapshots/camera_0/*.jpg     motion-energy proxy

and writes:

  backend/benchmarks/queries.jsonl
  backend/benchmarks/splits.json
  backend/benchmarks/person_pairs.jsonl
  results/subset_stats.json

All labels are metadata-derived, not human-annotated.  See README.md for the
provenance and the explicit list of claims this subset cannot support.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from backend.app.eval import subset

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 20260301
COLORS = ["Olive", "Red", "Brown"]
OPTIONAL_ATTRS = [
    ("carrying bag", ", carrying a bag"),
    ("long sleeves", " with long sleeves"),
    ("female", " who appears female"),
    ("male", " who appears male"),
    ("coat or jacket", ", wearing a coat or jacket"),
]
DATA_START = dt.datetime(2026, 3, 1, 17, 1, 0, 52839)
DATA_END = dt.datetime(2026, 3, 1, 18, 56, 30)
CAL_BLOCK_END = dt.datetime(2026, 3, 1, 17, 9, 0)
CAL_BLOCK2_START = dt.datetime(2026, 3, 1, 18, 55, 0)


def clamp_window(start: dt.datetime, end: dt.datetime) -> Tuple[dt.datetime, dt.datetime]:
    return max(start, DATA_START), min(end, DATA_END)


def iso(ts: dt.datetime) -> str:
    return ts.isoformat(timespec="microseconds")


def overlaps(clip: Dict[str, Any], start: dt.datetime, end: dt.datetime) -> float:
    return subset.overlap_seconds(clip["start"], clip["end"], start, end)


def unique_sorted(ids: Iterable[str], clips: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(set(ids), key=lambda cid: clips[cid]["start"])


def clip_ids_for_frames(
    frames: Sequence[Dict[str, Any]], predicate, start: dt.datetime, end: dt.datetime
) -> List[str]:
    out = []
    for fr in frames:
        if not (start <= fr["frame_ts"] <= end):
            continue
        if predicate(fr):
            out.append(fr["clip_id"])
    return sorted(set(out))


def detect_attr(det: Dict[str, Any], attr: str) -> bool:
    text = det["attribute_text"].lower()
    if attr == "carrying bag":
        return "carrying bag" in text
    if attr == "long sleeves":
        return "long sleeve" in text
    if attr == "female":
        return "female" in text
    if attr == "male":
        return "male" in text and "female" not in text
    if attr == "coat or jacket":
        return "coat or jacket" in text
    return False


def dets_matching(
    dets: Sequence[Dict[str, Any]],
    clip_of: Dict[int, str],
    color: Optional[str],
    attrs: Sequence[str],
    start: dt.datetime,
    end: dt.datetime,
) -> List[Dict[str, Any]]:
    out = []
    for det in dets:
        if not (start <= det["timestamp"] <= end):
            continue
        if color and det["color"].lower() != color.lower():
            continue
        if attrs and not all(detect_attr(det, a) for a in attrs):
            continue
        out.append(det)
    return out


def q(
    qid: str,
    category: str,
    text: str,
    start: dt.datetime,
    end: dt.datetime,
    relevant: Sequence[str],
    clips: Dict[str, Dict[str, Any]],
    parsed: Dict[str, Any],
    label_source: str,
    **extra: Any,
) -> Dict[str, Any]:
    rel = unique_sorted(relevant, clips)
    row = {
        "qid": qid,
        "category": category,
        "text": text,
        "time_start": iso(start),
        "time_end": iso(end),
        "camera_id": 0,
        "parsed": parsed,
        "relevant": rel,
        "n_relevant": len(rel),
        "label_source": label_source,
        "group": subset.five_min_bucket(start + (end - start) / 2),
        "split": None,
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# COUNT
# ---------------------------------------------------------------------------

def make_count_queries(
    frames: Sequence[Dict[str, Any]], clips: Dict[str, Dict[str, Any]], rng: random.Random
) -> List[Dict[str, Any]]:
    by_count: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for fr in frames:
        if fr["count"] is not None:
            by_count[fr["count"]].append(fr)
    for v in by_count.values():
        v.sort(key=lambda f: f["frame_ts"])

    anchors: List[Dict[str, Any]] = []
    used_clips: set = set()
    cursor = {c: 0 for c in by_count}
    counts = sorted(by_count)
    # Round-robin over count values so the 30 queries are not all 5-person
    # frames (the modal caption count is 5).
    guard = 0
    while len(anchors) < 30 and guard < 5000:
        guard += 1
        progressed = False
        for c in counts:
            if len(anchors) >= 30:
                break
            lst = by_count[c]
            while cursor[c] < len(lst):
                fr = lst[cursor[c]]
                cursor[c] += 1
                if fr["clip_id"] in used_clips:
                    continue
                if any(
                    abs((fr["frame_ts"] - a["frame_ts"]).total_seconds()) < 110
                    for a in anchors
                ):
                    continue
                anchors.append(fr)
                used_clips.add(fr["clip_id"])
                progressed = True
                break
        if not progressed:
            break
    if len(anchors) < 30:
        # Relax the temporal spacing but keep unique clips.
        for c in counts:
            for fr in by_count[c]:
                if len(anchors) >= 30:
                    break
                if fr["clip_id"] in used_clips:
                    continue
                anchors.append(fr)
                used_clips.add(fr["clip_id"])
    anchors = sorted(anchors[:30], key=lambda f: f["frame_ts"])
    if len(anchors) < 30:
        raise RuntimeError(f"only {len(anchors)} defensible count anchors")

    templates = [
        "how many people were visible around {t}?",
        "count the people in the scene at {t}.",
        "how many persons are on camera 0 at {t}?",
        "what is the person count around {t}?",
        "how many people were present between {a} and {b}?",
    ]
    out = []
    for i, anchor in enumerate(anchors):
        target = int(anchor["count"])
        start, end = clamp_window(anchor["frame_ts"] - dt.timedelta(seconds=90),
                                  anchor["frame_ts"] + dt.timedelta(seconds=90))
        rel = clip_ids_for_frames(
            frames, lambda fr: fr["count"] == target, start, end
        )
        a, b = subset.fmt_hms(start), subset.fmt_hms(end)
        t = subset.fmt_hms(anchor["frame_ts"], seconds=True)
        text = templates[i % len(templates)].format(t=t, a=a, b=b)
        parsed = {
            "time_start": iso(start),
            "time_end": iso(end),
            "camera_id": 0,
            "count_query": True,
            "count_target_in_text": None,
        }
        out.append(
            q(
                f"q_count_{i + 1:02d}",
                "count",
                text,
                start,
                end,
                rel,
                clips,
                parsed,
                "caption_person_count",
                anchor_clip=anchor["clip_id"],
                anchor_ts=iso(anchor["frame_ts"]),
                ground_truth_count=target,
            )
        )
    return out


# ---------------------------------------------------------------------------
# FIND
# ---------------------------------------------------------------------------

def make_find_queries(
    dets: Sequence[Dict[str, Any]],
    clip_of: Dict[int, str],
    clips: Dict[str, Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    candidates = [d for d in dets if d["color"] in COLORS]
    # One anchor per clip, then 30 time-ordered bins so the queries cover the
    # whole released window instead of clustering in the dense 17:01 burst.
    per_clip: Dict[str, Dict[str, Any]] = {}
    for d in sorted(candidates, key=lambda x: (x["timestamp"], x["faiss_id"])):
        cid = clip_of.get(d["faiss_id"])
        if cid in clips and cid not in per_clip:
            per_clip[cid] = d
    pool = sorted(per_clip.values(), key=lambda x: (x["timestamp"], x["faiss_id"]))
    if len(pool) < 30:
        raise RuntimeError(f"only {len(pool)} defensible find anchors")
    anchors: List[Dict[str, Any]] = []
    for k, idxs in enumerate(np.array_split(np.arange(len(pool)), 30)):
        desired = COLORS[k % len(COLORS)]
        chosen = None
        for j in idxs:
            if pool[int(j)]["color"] == desired:
                chosen = pool[int(j)]
                break
        if chosen is None:
            chosen = pool[int(idxs[0])]
        anchors.append(chosen)
    anchors = sorted(anchors, key=lambda d: (d["timestamp"], d["faiss_id"]))

    templates = [
        "find a person wearing {color_l} clothing between {a} and {b}.",
        "locate someone in {color_l} clothing between {a} and {b}.",
        "show me a person wearing {color_l} clothing{attr} between {a} and {b}.",
        "which clips contain a {color_l}-clothed person{attr} between {a} and {b}?",
        "find a {color_l} person{attr} on camera 0 from {a} to {b}.",
    ]
    out = []
    for i, anchor in enumerate(anchors):
        color = anchor["color"]
        start, end = clamp_window(anchor["timestamp"] - dt.timedelta(seconds=120),
                                  anchor["timestamp"] + dt.timedelta(seconds=120))
        # Pick at most one optional attribute that is actually present in the
        # anchor text and yields at least one matching clip in the window.
        attrs: List[str] = []
        attr_phrase = ""
        for key, phrase in OPTIONAL_ATTRS:
            if not detect_attr(anchor, key):
                continue
            matches = dets_matching(dets, clip_of, color, [key], start, end)
            if len({clip_of[d["faiss_id"]] for d in matches if clip_of.get(d["faiss_id"]) in clips}) >= 2:
                attrs = [key]
                attr_phrase = phrase
                break
        matches = dets_matching(dets, clip_of, color, attrs, start, end)
        rel = [clip_of[d["faiss_id"]] for d in matches if clip_of.get(d["faiss_id"]) in clips]
        if len(set(rel)) < 2:
            start, end = clamp_window(anchor["timestamp"] - dt.timedelta(seconds=180),
                                      anchor["timestamp"] + dt.timedelta(seconds=180))
            matches = dets_matching(dets, clip_of, color, attrs, start, end)
            rel = [clip_of[d["faiss_id"]] for d in matches if clip_of.get(d["faiss_id"]) in clips]
        a, b = subset.fmt_hms(start), subset.fmt_hms(end)
        text = templates[i % len(templates)].format(
            color=color, color_l=color.lower(), attr=attr_phrase, a=a, b=b
        )
        parsed = {
            "time_start": iso(start),
            "time_end": iso(end),
            "camera_id": 0,
            "objects.object_name": "person",
            "objects.color": color,
            "attributes": list(attrs),
        }
        out.append(
            q(
                f"q_find_{i + 1:02d}",
                "find",
                text,
                start,
                end,
                rel,
                clips,
                parsed,
                "person_attribute_metadata",
                anchor_detection=anchor["detection_id"],
                anchor_ts=iso(anchor["timestamp"]),
            )
        )
    return out


# ---------------------------------------------------------------------------
# TRACK
# ---------------------------------------------------------------------------

def color_runs(
    dets: Sequence[Dict[str, Any]],
    clip_of: Dict[int, str],
    clips: Dict[str, Dict[str, Any]],
    max_gap: float = 30.0,
) -> List[Dict[str, Any]]:
    clip_colors: Dict[str, set] = defaultdict(set)
    for d in dets:
        cid = clip_of.get(d["faiss_id"])
        if cid in clips:
            clip_colors[cid].add(d["color"])
    runs: List[Dict[str, Any]] = []
    for color in COLORS:
        ordered = [
            (cid, clips[cid]["start"], clips[cid]["end"])
            for cid in sorted(clips, key=lambda c: clips[c]["start"])
            if color in clip_colors.get(cid, set())
        ]
        cur: List[Tuple[str, dt.datetime, dt.datetime]] = []
        for item in ordered:
            if not cur:
                cur = [item]
                continue
            if (item[1] - cur[-1][2]).total_seconds() <= max_gap:
                cur.append(item)
            else:
                if len(cur) >= 2:
                    runs.append({"color": color, "clips": cur})
                cur = [item]
        if len(cur) >= 2:
            runs.append({"color": color, "clips": cur})
    runs.sort(key=lambda r: (r["clips"][0][1], r["color"]))
    for i, r in enumerate(runs):
        r["run_id"] = f"trk_{r['color'].lower()}_{i + 1:03d}"
    return runs


def make_track_queries(
    dets: Sequence[Dict[str, Any]],
    clip_of: Dict[int, str],
    clips: Dict[str, Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    runs = color_runs(dets, clip_of, clips)
    tasks: List[Dict[str, Any]] = []
    for r in runs:
        tasks.append({"run": r, "clips": r["clips"]})
    # Long runs can support a second, shorter tracking window.  This keeps
    # every query a genuine >=2-clip tracking task while reaching 30 queries.
    extra_candidates = []
    for r in runs:
        if len(r["clips"]) >= 4:
            mid = len(r["clips"]) // 2
            for part in (r["clips"][:mid], r["clips"][mid:]):
                if len(part) >= 2:
                    extra_candidates.append({"run": r, "clips": part})
    extra_candidates.sort(key=lambda t: (-len(t["clips"]), t["clips"][0][1]))
    for cand in extra_candidates:
        if len(tasks) >= 30:
            break
        # Do not duplicate the exact full-run clip set.
        if cand["clips"] == cand["run"]["clips"]:
            continue
        tasks.append(cand)
    if len(tasks) < 30:
        # Fall back to full runs again with a different phrasing only if the
        # release genuinely has fewer than 30 distinct >=2-clip runs.
        for r in runs:
            if len(tasks) >= 30:
                break
            tasks.append({"run": r, "clips": r["clips"], "repeat": True})
    tasks = tasks[:30]
    if len(tasks) < 30:
        raise RuntimeError(f"only {len(tasks)} defensible track tasks")

    templates = [
        "track the person wearing {color} clothing from {a} to {b}.",
        "follow the {color_l}-clothed person between {a} and {b} on camera 0.",
        "where does the person in {color} clothing move between {a} and {b}?",
        "show the {color_l} track from {a} to {b}.",
        "which clips continue the {color_l} person's track between {a} and {b}?",
    ]
    out = []
    for i, task in enumerate(tasks):
        color = task["run"]["color"]
        clip_list = task["clips"]
        start, end = clamp_window(clip_list[0][1], clip_list[-1][2])
        rel = [c[0] for c in clip_list if overlaps(clips[c[0]], start, end) > 0]
        a, b = subset.fmt_hms(start), subset.fmt_hms(end)
        text = templates[i % len(templates)].format(color=color, color_l=color.lower(), a=a, b=b)
        parsed = {
            "time_start": iso(start),
            "time_end": iso(end),
            "camera_id": 0,
            "objects.object_name": "person",
            "objects.color": color,
            "intent": "track",
        }
        out.append(
            q(
                f"q_track_{i + 1:02d}",
                "track",
                text,
                start,
                end,
                rel,
                clips,
                parsed,
                "single_camera_color_continuity_proxy",
                track_proxy_id=task["run"]["run_id"],
                n_proxy_clips=len(rel),
            )
        )
    return out


# ---------------------------------------------------------------------------
# FILTER
# ---------------------------------------------------------------------------

def make_filter_queries(
    frames: Sequence[Dict[str, Any]],
    dets: Sequence[Dict[str, Any]],
    clip_of: Dict[int, str],
    clips: Dict[str, Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    # Candidate 4-minute windows, stepped by 2 minutes.
    windows = []
    start = dt.datetime(2026, 3, 1, 17, 1, 0)
    while start < DATA_END:
        windows.append((start, min(start + dt.timedelta(minutes=4), DATA_END)))
        start += dt.timedelta(minutes=2)

    candidates: List[Dict[str, Any]] = []
    for wi, (ws, we) in enumerate(windows):
        for zone in ("corridor", None):
            for color in COLORS:
                rel = []
                for det in dets:
                    if not (ws <= det["timestamp"] <= we):
                        continue
                    if det["color"].lower() != color.lower():
                        continue
                    cid = clip_of.get(det["faiss_id"])
                    if cid not in clips:
                        continue
                    if zone:
                        zone_ok = any(
                            fr["zone"] == zone and ws <= fr["frame_ts"] <= we
                            for fr in clips[cid]["frames"]
                        )
                        if not zone_ok:
                            continue
                    rel.append(cid)
                rel = sorted(set(rel))
                if 1 <= len(rel) <= 12:
                    candidates.append(
                        {
                            "window_index": wi,
                            "start": ws,
                            "end": we,
                            "zone": zone,
                            "color": color,
                            "relevant": rel,
                            "n": len(rel),
                        }
                    )

    # Greedy selection: prefer corridor queries every third slot, cycle
    # colours, and require a 4-minute gap between selected window starts.
    selected: List[Dict[str, Any]] = []
    window_use: Counter = Counter()
    window_colors: Dict[int, set] = defaultdict(set)
    used_candidates: set = set()
    color_counts = Counter()
    while len(selected) < 30:
        want_zone = "corridor" if len(selected) % 3 == 0 else None
        best = None
        for cand in candidates:
            key = (cand["window_index"], cand["zone"], cand["color"])
            if key in used_candidates:
                continue
            if window_use[cand["window_index"]] >= 2:
                continue
            if cand["color"] in window_colors[cand["window_index"]]:
                continue
            if want_zone and cand["zone"] != want_zone:
                continue
            score = (
                -color_counts[cand["color"]],
                -cand["n"],
                cand["start"],
            )
            if best is None or score > best[0]:
                best = (score, cand)
        if best is None and want_zone:
            # Relax the zone preference for this slot.
            for cand in candidates:
                key = (cand["window_index"], cand["zone"], cand["color"])
                if key in used_candidates:
                    continue
                if window_use[cand["window_index"]] >= 2:
                    continue
                if cand["color"] in window_colors[cand["window_index"]]:
                    continue
                score = (-color_counts[cand["color"]], -cand["n"], cand["start"])
                if best is None or score > best[0]:
                    best = (score, cand)
        if best is None:
            break
        cand = best[1]
        selected.append(cand)
        key = (cand["window_index"], cand["zone"], cand["color"])
        used_candidates.add(key)
        window_use[cand["window_index"]] += 1
        window_colors[cand["window_index"]].add(cand["color"])
        color_counts[cand["color"]] += 1
    if len(selected) < 30:
        raise RuntimeError(f"only {len(selected)} defensible filter tasks")
    selected.sort(key=lambda c: (c["start"], c["color"]))

    templates = [
        "show all {color_l}-clothed people{zone} between {a} and {b}.",
        "filter to {color} clothing{zone} on camera 0 from {a} to {b}.",
        "list clips with a person in {color}{zone} between {a} and {b}.",
        "restrict results to {color_l} clothing{zone} between {a} and {b}.",
        "which clips show {color_l} clothing{zone} from {a} to {b}?",
    ]
    out = []
    for i, cand in enumerate(selected):
        zone_phrase = " in the corridor" if cand["zone"] == "corridor" else ""
        a, b = subset.fmt_hms(cand["start"]), subset.fmt_hms(cand["end"])
        text = templates[i % len(templates)].format(
            color=cand["color"], color_l=cand["color"].lower(), zone=zone_phrase, a=a, b=b
        )
        parsed = {
            "time_start": iso(cand["start"]),
            "time_end": iso(cand["end"]),
            "camera_id": 0,
            "zone": cand["zone"],
            "objects.object_name": "person",
            "objects.color": cand["color"],
        }
        out.append(
            q(
                f"q_filter_{i + 1:02d}",
                "filter",
                text,
                cand["start"],
                cand["end"],
                cand["relevant"],
                clips,
                parsed,
                "scene_zone_and_person_color_metadata",
                zone=cand["zone"],
            )
        )
    return out


# ---------------------------------------------------------------------------
# ACTION
# ---------------------------------------------------------------------------

def make_action_queries(
    frames: Sequence[Dict[str, Any]],
    clips: Dict[str, Dict[str, Any]],
    motion: Dict[str, Dict[str, float]],
    motion_threshold: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    event_frames = [fr for fr in frames if fr["event"] in {"fighting", "smoke"}]
    event_frames.sort(key=lambda f: f["frame_ts"])
    event_tasks = []
    for fr in event_frames:
        start, end = clamp_window(fr["frame_ts"] - dt.timedelta(seconds=45),
                                  fr["frame_ts"] + dt.timedelta(seconds=45))
        rel = clip_ids_for_frames(
            frames, lambda f, ev=fr["event"]: f["event"] == ev, start, end
        )
        event_tasks.append(
            {"event": fr["event"], "start": start, "end": end, "relevant": rel,
             "anchor_clip": fr["clip_id"], "anchor_ts": fr["frame_ts"]}
        )
    if len(event_tasks) < 15:
        raise RuntimeError(f"only {len(event_tasks)} caption event frames")

    high_motion = [
        (cid, v["mean"])
        for cid, v in motion.items()
        if v["mean"] >= motion_threshold
    ]
    high_motion.sort(key=lambda x: (clips[x[0]]["start"], x[0]))
    motion_tasks = []
    last_ts = None
    for cid, val in high_motion:
        ts = clips[cid]["start"]
        if last_ts is not None and (ts - last_ts).total_seconds() < 60:
            continue
        start, end = clamp_window(ts - dt.timedelta(seconds=30),
                                  ts + dt.timedelta(seconds=35))
        rel = [
            c
            for c in clips
            if overlaps(clips[c], start, end) > 0
            and motion.get(c, {}).get("mean", 0.0) >= motion_threshold
        ]
        motion_tasks.append(
            {"start": start, "end": end, "relevant": rel, "anchor_clip": cid,
             "motion": val}
        )
        last_ts = ts
        if len(motion_tasks) >= 15:
            break
    if len(motion_tasks) < 15:
        raise RuntimeError(f"only {len(motion_tasks)} high-motion anchors")

    tasks: List[Dict[str, Any]] = []
    for i in range(15):
        tasks.append(("event", event_tasks[i]))
        tasks.append(("motion", motion_tasks[i]))

    event_templates = {
        "fighting": [
            "where is someone fighting around {t}?",
            "show clips with fighting near {t}.",
            "is there a fight on camera 0 around {t}?",
        ],
        "smoke": [
            "is there smoke around {t}?",
            "show clips containing smoke near {t}.",
            "where does smoke appear on camera 0 around {t}?",
        ],
    }
    motion_templates = [
        "where is there a lot of movement between {a} and {b}?",
        "which clips show the most motion between {a} and {b}?",
        "show the highest-motion clips from {a} to {b}.",
    ]
    out = []
    for i, (kind, task) in enumerate(tasks):
        a, b = subset.fmt_hms(task["start"]), subset.fmt_hms(task["end"])
        t = subset.fmt_hms(task["start"] + (task["end"] - task["start"]) / 2, seconds=True)
        if kind == "event":
            ev = task["event"]
            tmpl = event_templates[ev][i % len(event_templates[ev])]
            text = tmpl.format(t=t)
            parsed = {
                "time_start": iso(task["start"]),
                "time_end": iso(task["end"]),
                "camera_id": 0,
                "action": ev,
                "event": ev,
            }
            source = "vlm_caption_event_weak_label"
            extra = {"event": ev, "anchor_ts": iso(task["anchor_ts"])}
        else:
            text = motion_templates[i % len(motion_templates)].format(a=a, b=b)
            parsed = {
                "time_start": iso(task["start"]),
                "time_end": iso(task["end"]),
                "camera_id": 0,
                "action": "motion_energy",
                "motion_threshold": motion_threshold,
            }
            source = "motion_energy_proxy"
            extra = {"motion_energy": task["motion"]}
        out.append(
            q(
                f"q_action_{i + 1:02d}",
                "action",
                text,
                task["start"],
                task["end"],
                task["relevant"],
                clips,
                parsed,
                source,
                action_kind=kind,
                **extra,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Splits and person pairs
# ---------------------------------------------------------------------------

def assign_splits(queries: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    tune: List[str] = []
    test: List[str] = []
    for cat in ["count", "find", "track", "action", "filter"]:
        cat_qs = [x for x in queries if x["category"] == cat]
        if cat == "action":
            # 5 event + 5 motion in tune, 10+10 in test.
            event = [x for x in cat_qs if x["action_kind"] == "event"]
            motion = [x for x in cat_qs if x["action_kind"] == "motion"]
            rng.shuffle(event)
            rng.shuffle(motion)
            tune.extend(x["qid"] for x in event[:5] + motion[:5])
            test.extend(x["qid"] for x in event[5:] + motion[5:])
        else:
            shuffled = list(cat_qs)
            rng.shuffle(shuffled)
            tune.extend(x["qid"] for x in shuffled[:10])
            test.extend(x["qid"] for x in shuffled[10:])
    for row in queries:
        row["split"] = "tune" if row["qid"] in tune else "test"
    return {
        "seed": SEED,
        "tune": sorted(tune),
        "test": sorted(test),
        "n_tune": len(tune),
        "n_test": len(test),
        "stratified_by": "category; action additionally by event/motion subfamily",
        "note": (
            "Queries are split 10 tune / 20 test per category (50/100 total). "
            "Query text and labels are derived only from released metadata; "
            "no human relevance labels exist for this subset."
        ),
    }


def build_person_pairs(
    dets: Sequence[Dict[str, Any]],
    track_of: Sequence[int],
    tracks: Sequence[Dict[str, Any]],
    clip_of: Dict[int, str],
    threshold: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Assign whole tracks to calibration/test.  Track-disjoint by construction.
    order = sorted(
        tracks, key=lambda t: (t["start"], t["track_id"])
    )
    cal_tracks = {t["track_id"] for i, t in enumerate(order) if i % 4 == 0}
    rows: List[Dict[str, Any]] = []
    pid = 0
    stats = Counter()
    for split_name, allowed in (("calibration", cal_tracks), ("test", set(t["track_id"] for t in order) - cal_tracks)):
        track_members: Dict[int, List[int]] = defaultdict(list)
        for i, tid in enumerate(track_of):
            if tid in allowed:
                track_members[tid].append(i)
        for tid, members in sorted(track_members.items()):
            if len(members) < 2:
                continue
            # Positive pairs: different timestamps, sample up to 8 per track.
            pos = []
            for _ in range(40):
                if len(pos) >= 8:
                    break
                a, b = rng.sample(members, 2)
                if abs((dets[a]["timestamp"] - dets[b]["timestamp"]).total_seconds()) < 0.5:
                    continue
                pos.append((a, b))
            for a, b in pos:
                pid += 1
                rows.append(
                    {
                        "pid": f"pair_{pid:06d}",
                        "split": split_name,
                        "query_row_index": int(a),
                        "gallery_row_index": int(b),
                        "query_faiss_id": int(dets[a]["faiss_id"]),
                        "gallery_faiss_id": int(dets[b]["faiss_id"]),
                        "query_detection_id": dets[a]["detection_id"],
                        "gallery_detection_id": dets[b]["detection_id"],
                        "query_object_index": int(dets[a]["object_index"]),
                        "gallery_object_index": int(dets[b]["object_index"]),
                        "same": True,
                        "track_a": f"{split_name}_t{tid:05d}",
                        "track_b": f"{split_name}_t{tid:05d}",
                        "dt_seconds": abs((dets[a]["timestamp"] - dets[b]["timestamp"]).total_seconds()),
                        "camera_id": 0,
                        "color_a": dets[a]["color"],
                        "color_b": dets[b]["color"],
                    }
                )
                stats["positives"] += 1
                # Negative with a nearby different track.
                candidates = [
                    j
                    for j in range(len(dets))
                    if track_of[j] != tid
                    and track_of[j] in allowed
                    and abs((dets[a]["timestamp"] - dets[j]["timestamp"]).total_seconds()) <= 30.0
                ]
                if not candidates:
                    candidates = [
                        j
                        for j in range(len(dets))
                        if track_of[j] != tid and track_of[j] in allowed
                    ]
                if candidates:
                    j = rng.choice(candidates)
                    pid += 1
                    rows.append(
                        {
                            "pid": f"pair_{pid:06d}",
                            "split": split_name,
                            "query_row_index": int(a),
                            "gallery_row_index": int(j),
                            "query_faiss_id": int(dets[a]["faiss_id"]),
                            "gallery_faiss_id": int(dets[j]["faiss_id"]),
                            "query_detection_id": dets[a]["detection_id"],
                            "gallery_detection_id": dets[j]["detection_id"],
                            "query_object_index": int(dets[a]["object_index"]),
                            "gallery_object_index": int(dets[j]["object_index"]),
                            "same": False,
                            "track_a": f"{split_name}_t{tid:05d}",
                            "track_b": f"{split_name}_t{track_of[j]:05d}",
                            "dt_seconds": abs((dets[a]["timestamp"] - dets[j]["timestamp"]).total_seconds()),
                            "camera_id": 0,
                            "color_a": dets[a]["color"],
                            "color_b": dets[j]["color"],
                        }
                    )
                    stats["negatives"] += 1
    summary = {
        "threshold": threshold,
        "calibration_tracks": len(cal_tracks),
        "test_tracks": len(set(t["track_id"] for t in order) - cal_tracks),
        "calibration_detections": sum(1 for i, t in enumerate(track_of) if t in cal_tracks),
        "test_detections": sum(1 for i, t in enumerate(track_of) if t not in cal_tracks),
        "positives": stats["positives"],
        "negatives": stats["negatives"],
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    frames = subset.load_scene_frames()
    clips = subset.build_clip_table(frames)
    # Sorted metadata is used for query construction (preserves the published
    # query file); index-order metadata is used for anything paired with FAISS
    # vectors, because the sidecar and index share insertion order.
    dets = subset.load_person_meta()
    dets_idx = subset.load_person_meta(sort_by_time=False)
    clip_of_list = subset.assign_detections_to_clips(dets_idx, clips)
    clip_of = {d["faiss_id"]: cid for d, cid in zip(dets_idx, clip_of_list) if cid}
    missing = [d for d, cid in zip(dets_idx, clip_of_list) if not cid]

    vectors = subset.load_index_vectors(subset.PERSON_INDEX, 1152)
    visual, text = subset.split_person_vectors(vectors)

    # Calibration block for the track threshold: the sparse 17:01-17:09
    # burst plus the final 18:55-18:56 burst.  It is disjoint from the main
    # 18:04-18:27 test burst and is never used for retrieval queries.
    cal_mask = np.array(
        [
            d["timestamp"] < CAL_BLOCK_END or d["timestamp"] >= CAL_BLOCK2_START
            for d in dets_idx
        ]
    )
    cal_dets = [d for d, m in zip(dets_idx, cal_mask) if m]
    cal_visual = visual[cal_mask]
    track_threshold, threshold_rows = subset.tune_track_threshold(cal_dets, cal_visual)
    track_of, tracks, edges = subset.build_tracks(
        dets_idx, visual, max_gap=3.0, threshold=track_threshold
    )

    motion = subset.compute_motion_energy(clips)
    cal_motion = [
        v["mean"]
        for cid, v in motion.items()
        if clips[cid]["start"] < dt.datetime(2026, 3, 1, 17, 51, 0)
    ]
    motion_threshold = float(np.percentile(cal_motion, 75))

    queries: List[Dict[str, Any]] = []
    queries.extend(make_count_queries(frames, clips, rng))
    queries.extend(make_find_queries(dets, clip_of, clips, rng))
    queries.extend(make_track_queries(dets, clip_of, clips, rng))
    queries.extend(make_filter_queries(frames, dets, clip_of, clips, rng))
    queries.extend(make_action_queries(frames, clips, motion, motion_threshold, rng))

    if len(queries) != 150:
        raise RuntimeError(f"expected 150 queries, built {len(queries)}")
    if len({x["text"] for x in queries}) != 150:
        raise RuntimeError("duplicate query text")
    for row in queries:
        if not row["relevant"]:
            raise RuntimeError(f"{row['qid']} has no relevant clips")

    splits = assign_splits(queries, rng)
    pairs, pair_summary = build_person_pairs(
        dets_idx, track_of, tracks, clip_of, track_threshold, rng
    )

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "queries.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in queries) + "\n",
        encoding="utf-8",
    )
    (HERE / "splits.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )
    (HERE / "person_pairs.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in pairs) + "\n",
        encoding="utf-8",
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    cat_counts = Counter(x["category"] for x in queries)
    source_counts = Counter(x["label_source"] for x in queries)
    rel_counts = [len(x["relevant"]) for x in queries]
    stats = {
        "dataset": "released single-camera subset",
        "camera_ids": sorted({fr["camera_id"] for fr in frames}),
        "scene_frames": len(frames),
        "scene_clips": len(clips),
        "scene_vector_dim": 512,
        "scene_model": "OpenCLIP ViT-B/32 laion2b_s34b_b79k",
        "person_descriptors": len(dets),
        "person_vector_dim": 1152,
        "person_visual_dim": 768,
        "person_text_dim": 384,
        "person_models": "SigLIP-base-patch16-224 + all-MiniLM-L12-v2",
        "time_start": iso(min(fr["frame_ts"] for fr in frames)),
        "time_end": iso(max(fr["frame_ts"] for fr in frames)),
        "clips_on_disk": len(list(subset.CLIP_DIR.glob("*.mp4"))),
        "clips_indexed": len(clips),
        "clips_not_indexed": sorted(
            set(p.stem for p in subset.CLIP_DIR.glob("*.mp4")) - set(clips)
        ),
        "person_detections_outside_indexed_clips": len(missing),
        "counts_per_category": dict(cat_counts),
        "label_sources": dict(source_counts),
        "relevant_clip_count": {
            "min": min(rel_counts),
            "median": float(np.median(rel_counts)),
            "mean": float(np.mean(rel_counts)),
            "max": max(rel_counts),
        },
        "splits": {
            "tune": splits["n_tune"],
            "test": splits["n_test"],
            "by_category": {
                cat: {
                    "tune": sum(1 for x in queries if x["category"] == cat and x["split"] == "tune"),
                    "test": sum(1 for x in queries if x["category"] == cat and x["split"] == "test"),
                }
                for cat in ["count", "find", "track", "action", "filter"]
            },
        },
        "track_proxy": {
            "method": "greedy visual-only cosine on normalized SigLIP 768-d + max gap 3 s + max duration 60 s + max size 20",
            "calibration_block": f"{iso(CAL_BLOCK_END - dt.timedelta(minutes=8))}..{iso(CAL_BLOCK_END)} and {iso(CAL_BLOCK2_START)}..",
            "threshold_grid": threshold_rows,
            "selected_threshold": track_threshold,
            "selected_rule": "strictest threshold within 5% of best calibration objective",
            "tracks": len(tracks),
            "tracks_with_ge2": sum(1 for t in tracks if t["size"] >= 2),
            "track_link_edges": len(edges),
            **pair_summary,
        },
        "motion_energy": {
            "definition": "mean absolute grayscale difference between consecutive released snapshots (96x96), averaged per clip",
            "calibration_threshold": motion_threshold,
            "calibration_quantile": 0.75,
            "clips_above_threshold": sum(1 for v in motion.values() if v["mean"] >= motion_threshold),
            "clips_scored": len(motion),
        },
        "cannot_reproduce": [
            "4 cameras / 312k person descriptors",
            "human-annotated relevance and inter-annotator agreement",
            "cross-camera identity tracking or generalisation",
            "the paper's full-deployment P@5/F1 values",
            "semantic running/fighting labels beyond the released VLM captions",
        ],
        "notes": [
            "All labels are derived from released metadata; none are human annotations.",
            "Action queries use VLM caption event labels (fighting/smoke) and a motion-energy proxy; no running/fighting label is invented.",
            "Track queries use single-camera colour-continuity proxies; cross-camera tracking is untestable.",
        ],
    }
    (RESULTS / "subset_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(
        f"wrote {len(queries)} queries ({dict(cat_counts)}), "
        f"{len(pairs)} person pairs, {len(tracks)} tracks (threshold={track_threshold:.2f})"
    )
    print(f"scene clips={len(clips)} frames={len(frames)} detections={len(dets)}")
    print(f"motion threshold={motion_threshold:.6f}; stats -> {RESULTS / 'subset_stats.json'}")


if __name__ == "__main__":
    main()
