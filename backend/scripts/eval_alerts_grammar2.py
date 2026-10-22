"""Grammar-2 evaluation: fix two-zone conjunctions and until/invalid-clock drops.

This script does NOT modify the shipped materializer.  It runtime-monkeypatches
``backend.app.services.alert_materializer`` to mirror the exact source patch in
``results/grammar2_results.md``, then measures:

* silent-drop rate on a seeded adversarial set (>= 200 inputs), before/after
* shipped 120-row benchmark expectations (must stay 30/30 clean, 30/30 stress)
* fuzz property tests (crashes / soundness / targeted false-materializations)

Writes ``results/grammar2_results.json`` and ``results/grammar2_results.md``.
Run from repository root::

    $env:PYTHONIOENCODING='utf-8'
    python backend/scripts/eval_alerts_grammar2.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backend.app.services.alert_materializer as M  # noqa: E402
from backend.app.services.alert_materializer import (  # noqa: E402
    _zones,
)
import backend.app.services.alert_validator as V  # noqa: E402
import backend.scripts.eval_alerts as EA  # noqa: E402

BENCH = ROOT / "backend" / "benchmarks" / "alert_utterances.jsonl"
RESULTS_DIR = ROOT / "results"
OUT_JSON = RESULTS_DIR / "grammar2_results.json"
OUT_MD = RESULTS_DIR / "grammar2_results.md"

# ---------------------------------------------------------------------------
# Save originals (for the "before" measurement and for restoration).
# ---------------------------------------------------------------------------
_ORIG_TIME_WINDOW = M._time_window
_ORIG_CONFLICT = M._conflict
_ORIG_MATERIALIZE = M.materialize_alert_rule


def _has_invalid_clock(text: str) -> bool:
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", text):
        if int(m.group(2)) > 59 or int(m.group(1)) > 23:
            return True
    return False


def _patched_time_window(text: str):
    r = _ORIG_TIME_WINDOW(text)
    if r is not None:
        return r
    m = re.search(r"\b(?:until|till)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    if m:
        h, minute, mer = m.groups()
        pair = M._parse_clock(h, minute, mer)
        if pair:
            hour = M._apply_meridiem(pair[0], mer) if mer else pair[0]
            mins = pair[1]
            if 0 <= hour <= 23 and 0 <= mins <= 59:
                result = M._range((0, 0), (hour, mins))
                if result:
                    return result
    return None


def _patched_conflict(text, objects, zones):
    if len(set(zones)) > 1:
        return "multiple zones specified ({})".format(", ".join(zones))
    if _has_invalid_clock(text):
        return "invalid clock time"
    return _ORIG_CONFLICT(text, objects, zones)


def _single_until_re():
    return re.compile(r"\b(?:until|till)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", re.I)


def _time_stated(text: str) -> bool:
    if _ORIG_TIME_WINDOW(text) is not None:
        return True
    if _has_invalid_clock(text):
        return True
    if _single_until_re().search(text):
        return True
    return False


def _silent_drops(text: str, rule: Dict[str, Any]) -> List[str]:
    drops: List[str] = []
    zones = _zones(text)
    if len(zones) > 1 and not bool(rule.get("needs_clarification")):
        drops.append("zone_conjunction")
    if _time_stated(text) and not bool(rule.get("needs_clarification")):
        tod = rule.get("time_of_day") or {}
        if not tod:
            drops.append("time_constraint")
    qs = M._quantifiers(text)
    if qs and not bool(rule.get("needs_clarification")) and rule.get("threshold") is None:
        drops.append("threshold")
    return drops


# ---------------------------------------------------------------------------
# Seeded adversarial set (>= 200 inputs).
# ---------------------------------------------------------------------------
def _build_adversarial(seed: int = 0xA11CE) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    zones = ["Zone A", "Zone B", "Zone C", "Gate", "Entrance", "Parking"]
    nums = [3, 5, 7, 10, 12]
    valid_clocks = ["9 PM", "10:30 PM", "8 AM", "22:00", "14:30", "11 PM"]
    bad_clocks = ["10:70", "25:00", "12:99", "7:60", "99:99", "24:00"]
    rows: List[Dict[str, Any]] = []

    for _ in range(80):
        z1, z2 = rng.sample(zones, 2)
        n = rng.choice(nums)
        tmpl = rng.choice([
            "alert me if more than {n} people gather in {z1} and {z2}",
            "notify if people crowd in {z1} and {z2}",
            "warn if more than {n} persons enter {z1} and {z2}",
            "if more than {n} people gather in {z1} and {z2} after 8 PM",
            "detect loitering of more than {n} people in {z1} and {z2}",
        ])
        rows.append({"category": "two_zone", "text": tmpl.format(n=n, z1=z1, z2=z2)})

    for _ in range(80):
        z = rng.choice(zones)
        n = rng.choice(nums)
        clk = rng.choice(valid_clocks)
        tmpl = rng.choice([
            "alert if more than {n} people gather in {z} until {clk}",
            "notify if people crowd in {z} until {clk}",
            "warn if more than {n} persons enter {z} until {clk}",
            "if more than {n} people gather in {z} until {clk}",
        ])
        rows.append({"category": "until", "text": tmpl.format(n=n, z=z, clk=clk)})

    for _ in range(80):
        z = rng.choice(zones)
        n = rng.choice(nums)
        bad = rng.choice(bad_clocks)
        good = rng.choice(["10:00", "12:00", "08:00"])
        tmpl = rng.choice([
            "alert if more than {n} people gather in {z} between {bad} and {good}",
            "warn if more than {n} persons crowd in {z} from 8:00 to {bad}",
            "notify if people gather in {z} between {good} and {bad}",
            "if more than {n} people gather in {z} at {bad}",
        ])
        rows.append({"category": "invalid_clock", "text": tmpl.format(n=n, z=z, bad=bad, good=good)})

    seen = set()
    out = []
    for r in rows:
        key = r["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> None:
    adv = _build_adversarial()
    assert len(adv) >= 200, len(adv)

    # --- Before measurement: globals original ---
    M._time_window = _ORIG_TIME_WINDOW
    M._conflict = _ORIG_CONFLICT
    before_drops = 0
    before_any = 0
    before_examples = []
    for item in adv:
        rule = _ORIG_MATERIALIZE(item["text"])
        drops = _silent_drops(item["text"], rule)
        if drops:
            before_any += 1
            before_drops += len(drops)
            if len(before_examples) < 10:
                before_examples.append({"text": item["text"], "drops": drops})

    # --- Apply patch ---
    M._time_window = _patched_time_window
    M._conflict = _patched_conflict
    # expose helper for parity
    M._has_invalid_clock = _has_invalid_clock

    # --- After measurement ---
    after_drops = 0
    after_any = 0
    after_examples = []
    for item in adv:
        rule = M.materialize_alert_rule(item["text"])
        drops = _silent_drops(item["text"], rule)
        if drops:
            after_any += 1
            after_drops += len(drops)
            if len(after_examples) < 10:
                after_examples.append({"text": item["text"], "drops": drops})

    # --- Seeded case proofs ---
    seeded_cases = [
        "more than 5 people gather in Zone A and Zone B",
        "more than 5 people in Zone A until 10 PM",
        "more than 5 people in Zone A between 10:70 and 12:00",
    ]
    # reset globals to original for before, then patched for after per case
    seeded_proof = []
    for t in seeded_cases:
        M._time_window = _ORIG_TIME_WINDOW
        M._conflict = _ORIG_CONFLICT
        before_rule = _ORIG_MATERIALIZE(t)
        M._time_window = _patched_time_window
        M._conflict = _patched_conflict
        after_rule = M.materialize_alert_rule(t)
        seeded_proof.append({
            "text": t,
            "before": {
                "needs_clarification": bool(before_rule.get("needs_clarification")),
                "clarification_kind": before_rule.get("clarification_kind"),
                "zone": (before_rule.get("area") or {}).get("zone_id"),
                "time_of_day": before_rule.get("time_of_day"),
                "silent_drops": _silent_drops(t, before_rule),
            },
            "after": {
                "needs_clarification": bool(after_rule.get("needs_clarification")),
                "clarification_kind": after_rule.get("clarification_kind"),
                "zone": (after_rule.get("area") or {}).get("zone_id"),
                "time_of_day": after_rule.get("time_of_day"),
                "silent_drops": _silent_drops(t, after_rule),
                "clarification": after_rule.get("clarification"),
            },
        })

    # --- Patch other modules' imported references for eval_alerts scoring ---
    EA.materialize_alert_rule = M.materialize_alert_rule
    V.classify_stress  # touch

    # --- Benchmark verification (shipped 120 rows) ---
    rows = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
    bench_mismatches = 0
    bench_details = []
    for r in rows:
        got = M.materialize_alert_rule(r["text"])
        if r["stress"] == "clean":
            exp = r["expected"]
            fields = ["event", "objects", "area", "threshold", "time_of_day",
                      "severity", "cooldown_sec", "side_effects", "cadence_hz"]
            ok = (not got.get("needs_clarification")) and all(
                got.get(f) == exp.get(f) for f in fields
            )
            if not ok:
                bench_mismatches += 1
                bench_details.append({"uid": r["uid"], "type": "clean", "got_kind": got.get("clarification_kind")})
        else:
            ok = bool(got.get("needs_clarification")) and got.get("clarification_kind") == r["stress"]
            if not ok:
                bench_mismatches += 1
                bench_details.append({"uid": r["uid"], "type": "stress", "got_kind": got.get("clarification_kind")})

    # --- eval_alerts scoring (mentally-equivalent, in-memory, no file writes) ---
    by_split = {sp: [r for r in rows if r.get("split") == sp] for sp in ("dev", "test")}
    scored_test = EA._score(by_split["test"])
    scored_dev = EA._score(by_split["dev"])

    # --- Fuzz property tests (patched) ---
    fuzz = EA.fuzz_property_tests()

    # --- Category breakdown for adversarial set ---
    cat_summary = {}
    for item in adv:
        cat_summary.setdefault(item["category"], {"n": 0, "before_drops": 0, "after_drops": 0})
        cat_summary[item["category"]]["n"] += 1
    # recompute per-cat drops
    M._time_window = _ORIG_TIME_WINDOW
    M._conflict = _ORIG_CONFLICT
    for item in adv:
        r = _ORIG_MATERIALIZE(item["text"])
        if _silent_drops(item["text"], r):
            cat_summary[item["category"]]["before_drops"] += 1
    M._time_window = _patched_time_window
    M._conflict = _patched_conflict
    for item in adv:
        r = M.materialize_alert_rule(item["text"])
        if _silent_drops(item["text"], r):
            cat_summary[item["category"]]["after_drops"] += 1

    total = len(adv)
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adversarial": {
            "seed": 0xA11CE,
            "n": total,
            "before": {
                "inputs_with_silent_drop": before_any,
                "silent_drop_rate": round(before_any / total, 6),
                "total_drop_instances": before_drops,
            },
            "after": {
                "inputs_with_silent_drop": after_any,
                "silent_drop_rate": round(after_any / total, 6),
                "total_drop_instances": after_drops,
            },
            "by_category": cat_summary,
            "before_examples": before_examples,
            "after_examples": after_examples,
        },
        "seeded_cases": seeded_proof,
        "benchmark": {
            "rows": len(rows),
            "mismatches": bench_mismatches,
            "mismatch_details": bench_details,
            "test_clean_rule_correctness": scored_test["clean"]["rule_correctness"],
            "test_clean_n": scored_test["clean"]["n"],
            "test_stress_clarified": scored_test["stress"]["clarified"],
            "test_stress_n": scored_test["stress"]["n"],
            "test_stress_false_materializations": scored_test["stress"]["false_materializations"],
            "test_clarify_precision": scored_test["clarify"]["precision"],
            "test_clarify_recall": scored_test["clarify"]["recall"],
            "test_clarify_f1": scored_test["clarify"]["f1"],
        },
        "fuzz": {
            "crashes": fuzz["crashes"],
            "n_soundness_violations": fuzz["n_soundness_violations"],
            "n_targeted_false": fuzz["n_targeted_false"],
            "clarified_noise": fuzz["clarified_noise"],
            "materialized_noise": fuzz["materialized_noise"],
        },
        "eval_alerts_test": {
            "clean_rule_correct": scored_test["clean"]["rule_correct"],
            "clean_n": scored_test["clean"]["n"],
            "stress_clarified": scored_test["stress"]["clarified"],
            "stress_n": scored_test["stress"]["n"],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- Markdown report ---
    patch_diff = (ROOT / "placeholder").exists() and "" or ""
    lines: List[str] = []
    lines.append("# Grammar-2 alert materializer evaluation")
    lines.append("")
    lines.append("Runtime monkeypatch of `backend/app/services/alert_materializer.py` (NOT applied to source).")
    lines.append("")
    lines.append("## Silent-drop rate on seeded adversarial set")
    lines.append("")
    lines.append("| Metric | Before | After |")
    lines.append("|---|---:|---:|")
    lines.append(
        "| Inputs with a silent constraint drop | "
        f"{before_any}/{total} | {after_any}/{total} |"
    )
    lines.append(
        "| Silent-drop rate | "
        f"{before_any/total:.4f} | {after_any/total:.4f} |"
    )
    lines.append(
        "| Drop instances | "
        f"{before_drops} | {after_drops} |"
    )
    lines.append("")
    lines.append("By category:")
    lines.append("")
    lines.append("| Category | n | before drops | after drops |")
    lines.append("|---|---:|---:|---:|")
    for cat, info in cat_summary.items():
        lines.append(f"| {cat} | {info['n']} | {info['before_drops']} | {info['after_drops']} |")
    lines.append("")
    lines.append("## Seeded case proofs")
    lines.append("")
    for sp in seeded_proof:
        lines.append(f"- `{sp['text']}`")
        lines.append(
            f"  - before: clarify={sp['before']['needs_clarification']} "
            f"kind={sp['before']['clarification_kind']} zone={sp['before']['zone']} "
            f"time={sp['before']['time_of_day']} drops={sp['before']['silent_drops']}"
        )
        lines.append(
            f"  - after:  clarify={sp['after']['needs_clarification']} "
            f"kind={sp['after']['clarification_kind']} zone={sp['after']['zone']} "
            f"time={sp['after']['time_of_day']} drops={sp['after']['silent_drops']}"
        )
        if sp['after'].get('clarification'):
            lines.append(f"  - after clarification: {sp['after']['clarification']}")
    lines.append("")
    lines.append("## Shipped 120-row benchmark (patched)")
    lines.append("")
    lines.append(f"- mismatches: {bench_mismatches}")
    lines.append(f"- test clean rule correctness: {scored_test['clean']['rule_correct']}/{scored_test['clean']['n']}")
    lines.append(f"- test stress clarified: {scored_test['stress']['clarified']}/{scored_test['stress']['n']}")
    lines.append(f"- test clarify P/R/F1: {scored_test['clarify']['precision']}/{scored_test['clarify']['recall']}/{scored_test['clarify']['f1']}")
    lines.append("")
    lines.append("## Fuzz (patched)")
    lines.append("")
    lines.append(f"- crashes: {fuzz['crashes']}")
    lines.append(f"- soundness violations: {fuzz['n_soundness_violations']}")
    lines.append(f"- targeted false materializations: {fuzz['n_targeted_false']}")
    lines.append("")
    lines.append("## Exact source patch (unapplied unified diff)")
    lines.append("")
    lines.append("```diff")
    # read diff from scratchpad if available
    diff_path = Path("C:/Users/sujal/AppData/Local/Temp/scratchpad/patch.diff")
    if diff_path.exists():
        lines.append(diff_path.read_text(encoding="utf-8").rstrip("\n"))
    lines.append("```")
    lines.append("")
    lines.append("## New tests to add")
    lines.append("")
    lines.append("- `test_two_zone_conjunction_clarifies`: assert `materialize_alert_rule('more than 5 people in Zone A and Zone B').needs_clarification is True` and clarification contains 'Zone A' and 'Zone B'.")
    lines.append("- `test_until_single_materializes_window`: assert time_of_day == {'start':'00:00','end':'22:00'} for '... until 10 PM' and needs_clarification False.")
    lines.append("- `test_invalid_clock_clarifies`: assert needs_clarification True for '... between 10:70 and 12:00'.")
    lines.append("- `test_no_silent_drop_property`: for seeded adversarial set, after patch silent_drop_rate == 0.")
    lines.append("- `test_benchmark_regression`: shipped 120 rows unchanged (30/30, 30/30).")
    lines.append("")
    lines.append("## Exact commands")
    lines.append("")
    lines.append("```")
    lines.append("$env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_grammar2.py")
    lines.append("```")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON.relative_to(ROOT), "and", OUT_MD.relative_to(ROOT))
    print(f"adversarial: before {before_any}/{total} after {after_any}/{total}")
    print(f"benchmark mismatches: {bench_mismatches}")
    print(f"fuzz: crashes {fuzz['crashes']} soundness {fuzz['n_soundness_violations']} targeted {fuzz['n_targeted_false']}")


if __name__ == "__main__":
    main()
