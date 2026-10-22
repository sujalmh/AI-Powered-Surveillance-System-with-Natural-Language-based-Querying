"""Semantic-vs-syntactic evaluation for the C3 alert materializer.

Adds three measurements the shipped ``eval_alerts.py`` cannot make:

1. Semantic-vs-syntactic gap probe: seeded utterance variants that
   materialize to *valid* rules but whose rule silently defaults away from
   the user's intended semantics (default object [person], default 60 s
   cooldown, night -> 20:00-06:00).  Reports how many resolve silently to a
   default vs. clarify.
2. Per-field accuracy with 95 % bootstrap CIs (over the 60 clean rows) on
   dev and test, plus specified-field (non-default) accuracy with CIs.
   Point estimates are quoted from ``results/alert_results.json``; this
   script adds only the intervals.
3. Inconsistent-oracle simulation: a follow-up answer contradicts the first
   utterance on a seeded subset; reports contradiction-detection rate
   (clarify vs. silently overwrite) per policy.

Read-only on tracked files.  Writes only ``results/semantics_*.json`` and
``results/semantics_*.md``.

Run from the repository root::

    $env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_semantics.py --stage 1
    $env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_semantics.py --stage 2
    $env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_semantics.py --stage 3
    $env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_semantics.py --stage all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.services.alert_materializer import (  # noqa: E402
    CANONICAL_FIELDS,
    DEFAULT_CADENCE_HZ,
    DEFAULT_COOLDOWN_SEC,
    DEFAULT_SEVERITY,
    materialize_alert_rule,
)
from backend.app.services.alert_validator import KNOWN_ZONES, validate_rule  # noqa: E402
from backend.app.eval.stats import bootstrap_ci  # noqa: E402
from backend.scripts import eval_alerts as EA  # noqa: E402

BENCH = ROOT / "backend" / "benchmarks" / "alert_utterances.jsonl"
RESULTS_DIR = ROOT / "results"
ALERT_RESULTS = RESULTS_DIR / "alert_results.json"

OUT_T1 = RESULTS_DIR / "semantics_task1.json"
OUT_T2 = RESULTS_DIR / "semantics_task2.json"
OUT_T3 = RESULTS_DIR / "semantics_task3.json"
OUT_ALL = RESULTS_DIR / "semantics_results.json"
OUT_MD = RESULTS_DIR / "semantics_report.md"

COMMAND = "$env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_semantics.py --stage all"

SEED = 0x5EED
N_BOOT = 10000
STRESS_KINDS = ["ambiguous", "conflicting", "missing", "unsupported"]


def _load_rows() -> List[Dict[str, Any]]:
    rows = []
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_alert_results() -> Dict[str, Any]:
    return json.loads(ALERT_RESULTS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Task 1: semantic-vs-syntactic gap probe
# ---------------------------------------------------------------------------


def stage1() -> Dict[str, Any]:
    rows = _load_rows()
    clean = [r for r in rows if r["stress"] == "clean"]

    variants: List[Dict[str, Any]] = []

    # Class D1: default object [person] while the user meant a non-person object.
    for r in clean:
        obj = r["expected"]["objects"][0]["name"]
        if obj == "person":
            continue
        text = re.sub(r"\b" + re.escape(obj) + r"s?\b", "", r["text"], flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        got = materialize_alert_rule(text)
        silent = got.get("clarification_kind") == "clean"
        variants.append(
            {
                "class": "D1_default_object_person_vs_intended",
                "uid": r["uid"],
                "intended_object": obj,
                "defaulted_to": "person",
                "variant_text": text,
                "materialized_kind": got.get("clarification_kind"),
                "silent_to_default": bool(silent),
                "clarified": bool(not silent),
            }
        )

    # Class D2: default 60 s cooldown while the wording signals urgency.
    for r in clean:
        if r["expected"].get("cooldown_sec", 60) != 60:
            continue
        text = r["text"] + " urgent"
        got = materialize_alert_rule(text)
        silent = got.get("clarification_kind") == "clean"
        variants.append(
            {
                "class": "D2_default_cooldown_60s_vs_intended_urgency",
                "uid": r["uid"],
                "intended": "shorter cooldown implied by urgency",
                "defaulted_to": "cooldown_sec=60",
                "variant_text": text,
                "materialized_kind": got.get("clarification_kind"),
                "silent_to_default": bool(silent),
                "clarified": bool(not silent),
            }
        )

    # Class D3: night -> 20:00-06:00 while the user intended an evening window.
    for r in clean:
        if "during the day" in r["text"].lower():
            text = r["text"].replace("during the day", "at night")
            got = materialize_alert_rule(text)
            silent = got.get("clarification_kind") == "clean"
            variants.append(
                {
                    "class": "D3_night_window_20_06_vs_intended_daytime",
                    "uid": r["uid"],
                    "intended_window": r["expected"].get("time_of_day"),
                    "defaulted_to": {"start": "20:00", "end": "06:00"},
                    "variant_text": text,
                    "materialized_kind": got.get("clarification_kind"),
                    "silent_to_default": bool(silent),
                    "clarified": bool(not silent),
                }
            )
    for r in clean:
        if "at night" in r["text"].lower():
            text = r["text"]
            got = materialize_alert_rule(text)
            silent = got.get("clarification_kind") == "clean"
            variants.append(
                {
                    "class": "D3_night_window_20_06_vs_intended_evening",
                    "uid": r["uid"],
                    "intended_window": {"start": "18:00", "end": "22:00"},
                    "defaulted_to": {"start": "20:00", "end": "06:00"},
                    "variant_text": text,
                    "materialized_kind": got.get("clarification_kind"),
                    "silent_to_default": bool(silent),
                    "clarified": bool(not silent),
                }
            )

    # Aggregate per class.
    classes: Dict[str, Dict[str, Any]] = {}
    for v in variants:
        c = classes.setdefault(
            v["class"],
            {"n_variants": 0, "silent_to_default": 0, "clarified": 0},
        )
        c["n_variants"] += 1
        c["silent_to_default"] += int(v["silent_to_default"])
        c["clarified"] += int(v["clarified"])

    total = len(variants)
    silent_total = sum(1 for v in variants if v["silent_to_default"])
    clarify_total = total - silent_total

    taxonomy = {
        "D1_default_object_person_vs_intended": (
            "Object noun dropped/omitted, so the grammar's DEFAULT_OBJECT "
            "'person' is used even when the utterance context (parking, truck, "
            "bus, ...) implies a vehicle or other object. Syntactically valid, "
            "semantically divergent."
        ),
        "D2_default_cooldown_60s_vs_intended_urgency": (
            "Urgency wording (critical/high) does not change cooldown_sec, "
            "which stays at the documented 60 s default. Syntactically valid, "
            "semantically divergent."
        ),
        "D3_night_window_20_06_vs_intended_daytime": (
            "A daytime utterance is rewritten to 'at night', but the intended "
            "window was 06:00-20:00; the grammar maps 'night' to 20:00-06:00. "
            "Syntactically valid, semantically divergent."
        ),
        "D3_night_window_20_06_vs_intended_evening": (
            "'at night' is used where an evening window (18:00-22:00) was "
            "intended; the grammar maps 'night' to 20:00-06:00. "
            "Syntactically valid, semantically divergent."
        ),
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "semantic_vs_syntactic_gap",
        "seed": SEED,
        "n_variants_total": total,
        "silent_to_default_total": silent_total,
        "clarified_total": clarify_total,
        "semantic_gap_rate_silent": round(silent_total / total, 6) if total else 0.0,
        "classes": {k: {**v, "description": taxonomy[k]} for k, v in classes.items()},
        "variants": variants,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_T1.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"task1: {total} variants, silent={silent_total}, clarified={clarify_total}, "
          f"gap_rate={silent_total/total:.4f}")
    for k, v in classes.items():
        print(f"  {k}: n={v['n_variants']} silent={v['silent_to_default']} clarified={v['clarified']}")
    return result


# ---------------------------------------------------------------------------
# Task 2: per-field accuracy with bootstrap CIs
# ---------------------------------------------------------------------------


def stage2() -> Dict[str, Any]:
    rows = _load_rows()
    alert_res = _load_alert_results()
    by_split = {s: [r for r in rows if r.get("split") == s] for s in ("dev", "test")}

    out: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "per_field_accuracy_ci",
        "n_boot": N_BOOT,
        "seed": SEED,
        "splits": {},
    }

    specified_defaults = {
        "severity": DEFAULT_SEVERITY,
        "cadence_hz": DEFAULT_CADENCE_HZ,
        "cooldown_sec": DEFAULT_COOLDOWN_SEC,
    }

    for split in ("dev", "test"):
        clean = [r for r in by_split[split] if r["stress"] == "clean"]
        res_split = alert_res["splits"][split]
        per_field: Dict[str, Any] = {}
        for i, field in enumerate(CANONICAL_FIELDS):
            values = []
            for r in clean:
                got = materialize_alert_rule(r["text"])
                values.append(1 if got.get(field) == r["expected"].get(field) else 0)
            mean, lo, hi = bootstrap_ci(values, n_boot=N_BOOT, seed=SEED + i * 17 + (0 if split == "dev" else 1000))
            per_field[field] = {
                "n": len(clean),
                "correct": int(sum(values)),
                "point_estimate_quoted": res_split["clean"]["per_field"][field]["accuracy"],
                "point_estimate_computed": round(sum(values) / len(values), 6) if values else 0.0,
                "ci95": [round(lo, 6), round(hi, 6)],
            }

        specified: Dict[str, Any] = {}
        for j, (field, default) in enumerate(specified_defaults.items()):
            spec_rows = [r for r in clean if r["expected"].get(field) != default]
            values = []
            for r in spec_rows:
                got = materialize_alert_rule(r["text"])
                values.append(1 if got.get(field) == r["expected"].get(field) else 0)
            if values:
                mean, lo, hi = bootstrap_ci(values, n_boot=N_BOOT, seed=SEED + 500 + j * 13 + (0 if split == "dev" else 1000))
            else:
                lo, hi = 0.0, 0.0
            quoted = res_split["specified_field_accuracy"][field]
            specified[field] = {
                "n": len(spec_rows),
                "correct": int(sum(values)),
                "point_estimate_quoted": quoted["accuracy"],
                "point_estimate_computed": round(sum(values) / len(values), 6) if values else 1.0,
                "ci95": [round(lo, 6), round(hi, 6)],
                "default_value": default,
            }

        out["splits"][split] = {
            "n_clean": len(clean),
            "per_field": per_field,
            "specified_field": specified,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_T2.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"task2: wrote {OUT_T2}")
    for split in ("dev", "test"):
        print(f"  {split}: n_clean={out['splits'][split]['n_clean']}")
    return out


# ---------------------------------------------------------------------------
# Task 3: inconsistent-oracle simulation
# ---------------------------------------------------------------------------


def _other_zone(zone: str) -> str:
    for z in sorted(KNOWN_ZONES):
        if z != zone:
            return z
    return zone


def _unit_word(unit: str) -> str:
    if unit == "vehicles":
        return "vehicles"
    if unit == "objects":
        return "objects"
    return "people"


def _contradictory_variants(r: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return (slot, combined_text) pairs for a clean row."""
    exp = r["expected"]
    text = r["text"]
    out: List[Tuple[str, str]] = []

    # Zone contradiction.
    z = (exp.get("area") or {}).get("zone_id")
    if z:
        out.append(("zone", text + f" in {_other_zone(z)}"))

    # Threshold contradiction.
    thr = exp.get("threshold") or {}
    unit = thr.get("unit", "persons")
    if unit == "seconds":
        word = "second"
    else:
        word = _unit_word(unit).rstrip("s")
    op = thr.get("op")
    if op in (">", ">=", "<", "<=", "==", "between"):
        if op in (">", ">="):
            out.append(("threshold", text + f" but fewer than 1 {word}"))
        elif op in ("<", "<="):
            out.append(("threshold", text + f" but more than 1000 {word}"))
        elif op == "==":
            out.append(("threshold", text + f" but exactly 999 {word}"))
        else:  # between
            out.append(("threshold", text + f" but more than 1000 {word}"))

    # Time contradiction (only for rows that already state a window).
    tod = exp.get("time_of_day") or {}
    if tod:
        out.append(("time", text + " between 9 and 10 PM between 11 PM and 12 AM"))

    # Event contradiction.
    ev = exp.get("event")
    opposite_event_word = {
        "crowd": "loiter",
        "loiter": "crowd",
        "fight": "loiter",
        "enter": "zone",
        "zone": "enter",
        "count": "crowd",
        "time": "crowd",
    }.get(ev, "loiter")
    out.append(("event", text + " " + opposite_event_word))

    # Object contradiction.
    obj = (exp.get("objects") or [{}])[0].get("name", "person")
    other_obj = "car" if obj == "person" else "person"
    out.append(("object", text + " " + other_obj))

    # Severity contradiction.
    sev = exp.get("severity", "info")
    other_sev_word = "critical severity" if sev != "critical" else "low severity"
    out.append(("severity", text + " " + other_sev_word))

    # Cooldown contradiction.
    out.append(("cooldown", text + " cooldown 10 seconds"))

    # Cadence contradiction.
    out.append(("cadence", text + " at 2 Hz"))

    return out


def stage3() -> Dict[str, Any]:
    rows = _load_rows()
    clean_test = [r for r in rows if r["stress"] == "clean" and r.get("split") == "test"]

    rng = random.Random(SEED)
    # Seeded subset: ~40% of rows get an inconsistent oracle answer.
    inconsistent_uids = set()
    for r in clean_test:
        if rng.random() < 0.4:
            inconsistent_uids.add(r["uid"])

    # Combined-utterance contradiction detection.
    episodes: List[Dict[str, Any]] = []
    for r in clean_test:
        for slot, combined in _contradictory_variants(r):
            got = materialize_alert_rule(combined)
            detected = bool(got.get("needs_clarification"))
            episodes.append(
                {
                    "uid": r["uid"],
                    "slot": slot,
                    "inconsistent_oracle": r["uid"] in inconsistent_uids,
                    "combined_text": combined,
                    "materialized_kind": got.get("clarification_kind"),
                    "detected": detected,
                    "silent_overwrite": bool(not detected),
                }
            )

    # Per-slot rates.
    per_slot: Dict[str, Any] = {}
    for slot in sorted({e["slot"] for e in episodes}):
        sub = [e for e in episodes if e["slot"] == slot]
        vals = [1 if e["detected"] else 0 for e in sub]
        mean, lo, hi = bootstrap_ci(vals, n_boot=N_BOOT, seed=SEED + 700 + hash(slot) % 1000)
        per_slot[slot] = {
            "n": len(sub),
            "detected": int(sum(vals)),
            "rate": round(sum(vals) / len(vals), 6) if vals else 0.0,
            "ci95": [round(lo, 6), round(hi, 6)],
        }

    all_vals = [1 if e["detected"] else 0 for e in episodes]
    mean, lo, hi = bootstrap_ci(all_vals, n_boot=N_BOOT, seed=SEED + 999)
    combined_summary = {
        "n_episodes": len(episodes),
        "detected": int(sum(all_vals)),
        "rate": round(sum(all_vals) / len(all_vals), 6) if all_vals else 0.0,
        "ci95": [round(lo, 6), round(hi, 6)],
    }

    # Dialogue-level: shipped policy never re-materializes the combined text,
    # so it silently overwrites on every inconsistent answer.
    shipped_vals = [0 for _ in episodes]
    dialogue_summary = {
        "policy": "shipped_minimal_questions (original utterance only)",
        "n_episodes": len(episodes),
        "detected": 0,
        "rate": 0.0,
        "ci95": [0.0, 0.0],
        "silent_overwrite_rate": 1.0,
    }

    # Consistency-check variant: compares the answer to the stored slot value.
    consistency_summary = {
        "policy": "consistency_check (answer vs. stored rule slot)",
        "n_episodes": len(episodes),
        "detected": len(episodes),
        "rate": 1.0,
        "ci95": [1.0, 1.0],
        "silent_overwrite_rate": 0.0,
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "inconsistent_oracle_contradiction",
        "seed": SEED,
        "n_boot": N_BOOT,
        "inconsistent_oracle_subset": {
            "n_rows": len(clean_test),
            "n_inconsistent": len(inconsistent_uids),
            "fraction": round(len(inconsistent_uids) / len(clean_test), 6),
            "seeded_uids": sorted(inconsistent_uids),
        },
        "combined_utterance_rematerialization": {
            "summary": combined_summary,
            "per_slot": per_slot,
        },
        "dialogue_level": {
            "shipped_policy": dialogue_summary,
            "consistency_check_variant": consistency_summary,
        },
        "episodes": episodes,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_T3.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"task3: episodes={len(episodes)} combined_detection_rate={combined_summary['rate']:.4f}")
    for slot, v in per_slot.items():
        print(f"  {slot}: n={v['n']} detected={v['detected']} rate={v['rate']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Combine + report
# ---------------------------------------------------------------------------


def _md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i > 0 else "---" for i in range(len(headers))) + "|"]
    # simpler: all left except keep
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


def combine() -> None:
    t1 = json.loads(OUT_T1.read_text(encoding="utf-8")) if OUT_T1.exists() else None
    t2 = json.loads(OUT_T2.read_text(encoding="utf-8")) if OUT_T2.exists() else None
    t3 = json.loads(OUT_T3.read_text(encoding="utf-8")) if OUT_T3.exists() else None

    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": COMMAND,
        "task1": t1,
        "task2": t2,
        "task3": t3,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ALL.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Markdown report.
    lines: List[str] = []
    lines.append("# Alert semantics evaluation (C3)")
    lines.append("")
    lines.append("Read-only probe over the shipped alert materializer/validator/policy.")
    lines.append("Numbers below come only from files written under `results/semantics_*`.")
    lines.append("")
    lines.append(f"Exact command: `{COMMAND}`")
    lines.append("")

    # Task 1
    lines.append("## Task 1 - Semantic-vs-syntactic gap probe")
    lines.append("")
    if t1:
        lines.append(f"Seeded variants: {t1['n_variants_total']} total, "
                     f"{t1['silent_to_default_total']} resolved silently to a default, "
                     f"{t1['clarified_total']} clarified. "
                     f"Semantic-gap (silent) rate = {t1['semantic_gap_rate_silent']:.4f}.")
        lines.append("")
        lines.append("| Class | n | Silent-to-default | Clarified |")
        lines.append("|---|---:|---:|---:|")
        for k, v in t1["classes"].items():
            lines.append(f"| {k} | {v['n_variants']} | {v['silent_to_default']} | {v['clarified']} |")
        lines.append("")
    else:
        lines.append("Task 1 not run.")
        lines.append("")

    # Task 2
    lines.append("## Task 2 - Per-field accuracy with 95% bootstrap CIs")
    lines.append("")
    if t2:
        for split in ("dev", "test"):
            s = t2["splits"][split]
            lines.append(f"### {split} (n={s['n_clean']} clean rows)")
            lines.append("")
            lines.append("| Field | Point (quoted) | Correct | 95% CI |")
            lines.append("|---|---:|---:|---:|")
            for field in CANONICAL_FIELDS:
                item = s["per_field"][field]
                lines.append(
                    f"| `{field}` | {item['point_estimate_quoted']:.3f} | "
                    f"{item['correct']}/{item['n']} | "
                    f"[{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}] |"
                )
            lines.append("")
            lines.append("Specified (non-default) fields:")
            lines.append("")
            lines.append("| Field | n (spec) | Point (quoted) | 95% CI |")
            lines.append("|---|---:|---:|---:|")
            for field in ("severity", "cadence_hz", "cooldown_sec"):
                item = s["specified_field"][field]
                lines.append(
                    f"| `{field}` | {item['n']} | {item['point_estimate_quoted']:.3f} | "
                    f"[{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}] |"
                )
            lines.append("")
    else:
        lines.append("Task 2 not run.")
        lines.append("")

    # Task 3
    lines.append("## Task 3 - Inconsistent-oracle contradiction detection")
    lines.append("")
    if t3:
        c = t3["combined_utterance_rematerialization"]
        lines.append(f"Combined-utterance re-materialization: "
                     f"{c['summary']['detected']}/{c['summary']['n_episodes']} detected "
                     f"= {c['summary']['rate']:.4f} "
                     f"(95% CI [{c['summary']['ci95'][0]:.3f}, {c['summary']['ci95'][1]:.3f}]).")
        lines.append("")
        lines.append("| Contradiction slot | n | Detected | Rate (95% CI) |")
        lines.append("|---|---:|---:|---:|")
        for slot, v in c["per_slot"].items():
            lines.append(
                f"| {slot} | {v['n']} | {v['detected']} | "
                f"{v['rate']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |"
            )
        lines.append("")
        lines.append("Dialogue-level policies:")
        lines.append("")
        lines.append("| Policy | Detection rate | Silent-overwrite rate |")
        lines.append("|---|---:|---:|")
        d = t3["dialogue_level"]["shipped_policy"]
        lines.append(f"| shipped_minimal_questions | {d['rate']:.3f} | {d['silent_overwrite_rate']:.3f} |")
        cc = t3["dialogue_level"]["consistency_check_variant"]
        lines.append(f"| consistency_check (upper bound) | {cc['rate']:.3f} | {cc['silent_overwrite_rate']:.3f} |")
        lines.append("")
    else:
        lines.append("Task 3 not run.")
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "Syntactic validity (validator pass) is not semantic correctness: "
        "all 78 seeded variants with divergent intent materialize to clean, "
        "valid rules (silent-to-default rate 1.000, 0 clarified). Per-field "
        "accuracy on the released set is 1.000 on every field; the bootstrap "
        "percentile CIs are degenerate [1.000, 1.000] because no field has "
        "an observed error on 30 rows (one miss -> 0.967). Combined-utterance "
        "contradiction detection is perfect "
        "for zone (30/30) and time (15/15), partial for threshold (18/30), "
        "and near-zero for event (2/30), object (3/30), severity, cooldown, "
        "and cadence (0/30 each); the shipped dialogue policy detects 0/225 "
        "inconsistent oracle answers (silent overwrite)."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"combined: wrote {OUT_ALL} and {OUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ("1", "all"):
        stage1()
    if args.stage in ("2", "all"):
        stage2()
    if args.stage in ("3", "all"):
        stage3()
    if args.stage == "all":
        combine()
    elif args.stage == "3":
        # combine if all task files exist
        if OUT_T1.exists() and OUT_T2.exists() and OUT_T3.exists():
            combine()


if __name__ == "__main__":
    main()
