"""C3 probe #2: attack the alert materializer / clarification policy (read-only).

This script does **not** modify any tracked file, the benchmark, or the
existing ``results/`` files.  It only *reads* the shipped code and writes new
artifacts ``results/c3probe2_results.json`` and ``results/c3probe2_report.md``.

Three attack lines are run (see the task brief):

(a) **Bug hunt** -- expanded grammar-aware fuzzing (stacked comparators, unit
    confusion, between-bounds inversion, unicode/whitespace abuse,
    contradictory-but-grammatical inputs) looking for crashes, *validator*
    soundness violations (a materialized rule that ``validate_rule`` rejects)
    and *semantic* soundness violations (a materialized rule that admits
    counts the user excluded).  Every finding is shrunk to a minimal
    reproducer.
(b) **Policy frontier** -- variants of ``minimal_questions``; the variant is
    selected on the DEV split only and the frozen variant is evaluated once
    on the TEST split.
(c) **Paraphrase robustness** -- programmatic perturbations of the 60 clean
    benchmark rows (no benchmark edit), rule-correctness under perturbation
    vs clean.

The proposed fix is **not** applied to the source tree.  It is proved at
runtime with monkeypatches that mirror the source patch exactly (the patch
text is embedded in this file and reproduced in the report).

Run from the repository root::

    $env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_probe2.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.services import alert_materializer as AM  # noqa: E402
from backend.app.services.alert_materializer import (  # noqa: E402
    CANONICAL_FIELDS,
    DEFAULT_CADENCE_HZ,
    DEFAULT_COOLDOWN_SEC,
    DEFAULT_SEVERITY,
    DEFAULT_SIDE_EFFECTS,
    materialize_alert_rule,
)
from backend.app.services.alert_validator import (  # noqa: E402
    KNOWN_EVENTS,
    KNOWN_ZONES,
    validate_rule,
)
from backend.app.services.clarification_policy import (  # noqa: E402
    NAIVE_BASELINE_FIELDS,
    minimal_questions,
)
from backend.scripts import eval_alerts as EA  # noqa: E402

BENCH = ROOT / "backend" / "benchmarks" / "alert_utterances.jsonl"
RESULTS_DIR = ROOT / "results"
OUT_JSON = RESULTS_DIR / "c3probe2_results.json"
OUT_MD = RESULTS_DIR / "c3probe2_report.md"
COMMAND = "$env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_probe2.py"

# Seeds for the new (probe-local) fuzzers.  The shipped fuzz seed
# (0xF422) is untouched and is re-run as a regression invariant.
GRAMMAR_FUZZ_SEED = 0xC3F022
GRAMMAR_FUZZ_N = 20000
PRETURB_SEED = 0xC3B022

STRESS_KINDS = ["ambiguous", "conflicting", "missing", "unsupported"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _acc(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def _load_rows() -> List[Dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(BENCH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise SystemExit(f"{BENCH}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _materialize(text: str) -> Dict[str, Any]:
    """Call the *module attribute* so the runtime monkeypatch is exercised."""
    return AM.materialize_alert_rule(text)


def _interval(threshold: Optional[Dict[str, Any]]) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """Integer interval [lo, hi] admitted by a threshold (None = unbounded).

    ``None`` is returned when the threshold cannot be mapped onto an integer
    interval (e.g. a non-integer bound).
    """
    if not isinstance(threshold, dict):
        return None
    op = threshold.get("op")
    value = threshold.get("value")
    if op == "between":
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            return None
        lo, hi = value
        if not (isinstance(lo, int) and isinstance(hi, int)):
            return None
        return (lo, hi)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if op == ">":
        return (value + 1, None)
    if op == ">=":
        return (value, None)
    if op == "<":
        return (None, value - 1)
    if op == "<=":
        return (None, value)
    if op == "==":
        return (value, value)
    return None


def _subset(inner: Optional[Tuple[Optional[int], Optional[int]]],
            outer: Optional[Tuple[Optional[int], Optional[int]]]) -> Optional[bool]:
    """True when ``inner`` admits no value outside ``outer`` (None if unknown).

    A half-unbounded ``inner`` is NOT a subset of a bounded ``outer``: the
    unbounded side admits values the outer bound excludes.
    """
    if inner is None or outer is None:
        return None
    ilo, ihi = inner
    olo, ohi = outer
    if olo is not None and (ilo is None or ilo < olo):
        return False
    if ohi is not None and (ihi is None or ihi > ohi):
        return False
    return True


# ---------------------------------------------------------------------------
# 1. baseline replication (must reproduce the shipped green numbers)
# ---------------------------------------------------------------------------

def baseline_replication(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_split = {s: [r for r in rows if r.get("split") == s] for s in ("dev", "test")}
    scored = {s: EA._score(copy.deepcopy(by_split[s])) for s in ("dev", "test")}
    overall = EA._score(copy.deepcopy(rows))
    clean_all = [r for r in rows if r["stress"] == "clean"]
    clean_test = [r for r in by_split["test"] if r["stress"] == "clean"]
    compositional = EA.compositional_heldout(copy.deepcopy(clean_all))
    oracle = EA.oracle_turns_simulation(copy.deepcopy(clean_test))
    fuzz = EA.fuzz_property_tests()

    invariants = {
        "test_clean_rule_correct_30_of_30": (
            scored["test"]["clean"]["rule_correct"] == 30
            and scored["test"]["clean"]["n"] == 30
        ),
        "test_stress_clarified_30_of_30": (
            scored["test"]["stress"]["clarified"] == 30
            and scored["test"]["stress"]["n"] == 30
        ),
        "test_false_materializations_zero": scored["test"]["stress"]["false_materializations"] == 0,
        "held_out_9_of_9": compositional["held_out"]["correct"] == 9,
        "seen_51_of_51": compositional["seen"]["correct"] == 51,
        "fuzz_crashes_zero": fuzz["crashes"] == 0,
        "fuzz_soundness_violations_zero": fuzz["n_soundness_violations"] == 0,
        "fuzz_targeted_false_materializations_zero": fuzz["n_targeted_false"] == 0,
        "oracle_policy_mean_1p0778": abs(oracle["policy"]["mean_turns"] - 1.0778) < 1e-9,
        "oracle_policy_max_turns_2": oracle["policy"]["max_turns"] == 2,
        "oracle_naive_mean_2p5778": abs(oracle["naive_per_field"]["mean_turns"] - 2.5778) < 1e-9,
        "oracle_naive_max_turns_4": oracle["naive_per_field"]["max_turns"] == 4,
        "oracle_both_correct_75_of_90": (
            oracle["policy"]["correct"] == 75
            and oracle["naive_per_field"]["correct"] == 75
            and oracle["n_episodes"] == 90
        ),
        "oracle_both_valid_90_of_90": (
            oracle["policy"]["valid"] == 90 and oracle["naive_per_field"]["valid"] == 90
        ),
    }
    return {
        "splits": scored,
        "overall": overall,
        "compositional_heldout": compositional,
        "oracle_turns": oracle,
        "fuzz": fuzz,
        "invariants": invariants,
        "all_invariants_hold": all(invariants.values()),
    }


# ---------------------------------------------------------------------------
# 2. attack (a): expanded grammar-aware fuzzing
# ---------------------------------------------------------------------------

# -- adversarial corpus -----------------------------------------------------
# expectation kinds:
#   "clarify"     -> the utterance must NOT be materialized
#   "materialize" -> the utterance must be materialized and validate cleanly
#   "interval"    -> the utterance must be materialized and its threshold must
#                    admit no count outside ``interval`` (semantic soundness)
CORPUS: List[Dict[str, Any]] = [
    # --- F1: conjunctive (two-sided) bounds ------------------------------
    {"uid": "f1_01", "family": "two_sided_bounds", "expect": "interval",
     "interval": (6, 9),
     "lower": ("more than five", 5, False), "upper": ("fewer than ten", 10, False),
     "text": "alert me if more than five but fewer than ten people gather in Zone A"},
    {"uid": "f1_02", "family": "two_sided_bounds", "expect": "interval",
     "interval": (5, 10),
     "lower": ("at least five", 5, True), "upper": ("at most ten", 10, True),
     "text": "notify me if at least five and at most ten people gather in Zone A"},
    {"uid": "f1_03", "family": "two_sided_bounds", "expect": "interval",
     "interval": (6, 9),
     "lower": ("more than five", 5, False), "upper": ("fewer than ten", 10, False),
     "text": "alert me if more than five people but fewer than ten people gather in Zone A"},
    {"uid": "f1_04", "family": "two_sided_bounds", "expect": "interval",
     "interval": (6, 9),
     "lower": ("more than five", 5, False), "upper": ("less than ten", 10, False),
     "text": "warn me when the number of people in Zone A is more than five but less than ten"},
    {"uid": "f1_05", "family": "two_sided_bounds", "expect": "interval",
     "interval": (3, 3),
     "lower": ("more than two", 2, False), "upper": ("fewer than four", 4, False),
     "text": "alert me if more than two but fewer than four cars are in the Parking lot"},
    {"uid": "f1_06", "family": "two_sided_bounds", "expect": "interval",
     "interval": (301, 599),
     "lower": ("more than five minutes", 300, False), "upper": ("less than ten minutes", 600, False),
     "required": ["for"],
     "text": "notify me if someone loiters in Zone A for more than five minutes but less than ten minutes"},
    {"uid": "f1_07", "family": "stacked_lower_bounds", "expect": "interval",
     "interval": (6, None),
     "lower": ("more than five", 5, False),
     "text": "alert me if more than five more than three people gather in Zone A"},
    {"uid": "f1_08", "family": "stacked_upper_bounds", "expect": "interval",
     "interval": (None, 7),
     "upper": ("fewer than eight", 8, False),
     "text": "alert me if fewer than ten fewer than eight people gather in Zone A"},
    {"uid": "f1_09", "family": "stacked_upper_bounds", "expect": "interval",
     "interval": (None, 7),
     "upper": ("fewer than eight", 8, False),
     "text": "alert me if fewer than ten people fewer than eight people gather in Zone A"},
    {"uid": "f1_10", "family": "stacked_lower_bounds", "expect": "interval",
     "interval": (6, None),
     "lower": ("more than five", 5, False),
     "text": "alert me if more than five people more than three people gather in Zone A"},
    # --- F2: between-bounds inversion (contradictory rule) ---------------
    {"uid": "f2_01", "family": "between_inversion", "expect": "clarify",
     "text": "alert me when between ten and three people gather in Zone A"},
    {"uid": "f2_02", "family": "between_inversion", "expect": "clarify",
     "text": "notify me if the number of people in Zone A is between ten and three"},
    {"uid": "f2_03", "family": "between_inversion", "expect": "clarify",
     "text": "alert me when the count of cars in the Parking lot is between nine and two"},
    # --- F3: degenerate time window ---------------------------------------
    {"uid": "f3_01", "family": "degenerate_window", "expect": "clarify",
     "text": "alert me when more than five people gather in Zone A between 3 AM and 3 AM"},
    {"uid": "f3_02", "family": "degenerate_window", "expect": "clarify",
     "text": "alert me when more than five people gather in Zone A from 9 PM to 9 PM"},
    {"uid": "f3_03", "family": "degenerate_window", "expect": "clarify",
     "text": "alert me when more than five people gather in Zone A before 12 AM"},
    {"uid": "f3_04", "family": "degenerate_window", "expect": "clarify",
     "text": "alert me when more than five people gather in Zone A between 12 AM and 12 AM"},
    # --- F4: cadence outside the representable range ----------------------
    {"uid": "f4_01", "family": "cadence_out_of_range", "expect": "clarify",
     "text": "alert me every 0.05 seconds when more than five people gather in Zone A"},
    {"uid": "f4_02", "family": "cadence_out_of_range", "expect": "clarify",
     "text": "alert me at 20 hz when more than five people gather in Zone A"},
    {"uid": "f4_03", "family": "cadence_out_of_range", "expect": "clarify",
     "text": "alert me at 99 hz when more than five people gather in Zone A"},
    # --- F5: unicode / whitespace abuse (no crash, no invalid rule) ------
    {"uid": "f5_01", "family": "unicode", "expect": "materialize",
     "text": "alert me when more than 5\u00a0people gather in Zone A"},
    {"uid": "f5_02", "family": "unicode", "expect": "materialize",
     "text": "alert me when more than five people gather in Zone A\u200d"},
    {"uid": "f5_03", "family": "unicode", "expect": "clarify",
     "text": "alert me when more than five people gather in Zone\u200bA"},
    {"uid": "f5_04", "family": "unicode", "expect": "materialize",
     "text": "alert me when more than \u0665 people gather in Zone A"},
    {"uid": "f5_05", "family": "unicode", "expect": "materialize",
     "text": "alert me when more than \uff15 people gather in Zone A"},
    {"uid": "f5_06", "family": "unicode", "expect": "materialize",
     "text": "  alert me when more than five people gather in Zone A\u00a0\u00a0"},
    {"uid": "f5_07", "family": "unicode", "expect": "materialize",
     "text": "alert me when more than five people gather in Zone\u00a0A"},
    # --- F6: contradictory but grammatical --------------------------------
    {"uid": "f6_01", "family": "contradiction", "expect": "clarify",
     "text": "alert me if more than five but fewer than three people gather in Zone A"},
    {"uid": "f6_02", "family": "contradiction", "expect": "clarify",
     "text": "alert me if people gather in Zone A or Zone B"},
    {"uid": "f6_03", "family": "contradiction", "expect": "clarify",
     "text": "warn me if someone loiters in Zone A for more than five minutes but less than one minute"},
    {"uid": "f6_04", "family": "contradiction", "expect": "clarify",
     "text": "notify me if more than ten people and exactly two people are in Zone B"},
    {"uid": "f6_05", "family": "contradiction", "expect": "clarify",
     "text": "warn me between 9 and 11 PM but also before 8 PM"},
    # --- F7: unit confusion ----------------------------------------------
    {"uid": "f7_01", "family": "unit_confusion", "expect": "materialize",
     "text": "alert me when more than five objects gather in Zone A"},
    {"uid": "f7_02", "family": "unit_confusion", "expect": "materialize",
     "text": "alert me when more than five trucks gather in Zone A"},
    {"uid": "f7_03", "family": "unit_confusion", "expect": "materialize",
     "text": "alert me when more than five people gather in Zone A for 2 hours"},
    # --- F8: zone alias leaks an object ("car park" -> car) --------------
    {"uid": "f8_01", "family": "zone_alias_object_leak", "expect": "materialize",
     "objects": ["person"],
     "text": "notify me during the day in the car park"},
    {"uid": "f8_02", "family": "zone_alias_object_leak", "expect": "materialize",
     "objects": ["person"],
     "text": "alert me when more than five people gather in the car park"},
    {"uid": "f8_03", "family": "zone_alias_object_leak", "expect": "materialize",
     "objects": ["car"],
     "text": "alert me when a car enters the car park"},
]

# Residual (known, NOT fixed by the proposed patch) -- reported honestly.
RESIDUAL: List[Dict[str, Any]] = [
    {"uid": "r_01", "family": "two_zone_conjunction",
     "text": "alert me when more than five people gather in Zone A and more than three people gather in Zone B",
     "note": "conjunction of two (zone, threshold) pairs; the grammar holds one zone, "
             "so the second clause is silently dropped"},
    {"uid": "r_02", "family": "dropped_time_constraint",
     "text": "alert me when more than five people gather in Zone A until 3 AM",
     "note": "'until' is not in the grammar, so the time constraint is silently dropped"},
    {"uid": "r_03", "family": "dropped_time_constraint",
     "text": "alert me when more than five people gather in Zone A between 10:70 and 11 PM",
     "note": "invalid clock minute -> the whole time window is silently dropped"},
]


def _stated_interval(case: Dict[str, Any], text: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """Interval stated by the *candidate* text (None when a bound phrase is gone).

    The bound phrases are re-read from the candidate so that the shrinker
    cannot delete one side of the conjunction and keep the property.
    """
    low = text.lower()
    lo = hi = None
    if "lower" in case:
        phrase, value, inclusive = case["lower"]
        if phrase not in low:
            return None
        lo = value if inclusive else value + 1
    if "upper" in case:
        phrase, value, inclusive = case["upper"]
        if phrase not in low:
            return None
        hi = value if inclusive else value - 1
    if lo is None and hi is None:
        return None
    return (lo, hi)


def _check_utterance(text: str) -> Dict[str, Any]:
    """Materialize + validate one utterance, catching crashes."""
    try:
        rule = _materialize(text)
    except Exception as exc:  # noqa: BLE001 - crash-freedom is the property
        return {"crash": f"{type(exc).__name__}: {exc}", "rule": None, "valid": None}
    if rule.get("needs_clarification"):
        return {"crash": None, "rule": rule, "valid": None}
    try:
        ok, errors, _ = validate_rule(rule)
    except Exception as exc:  # noqa: BLE001
        return {"crash": f"validate {type(exc).__name__}: {exc}", "rule": rule, "valid": None}
    return {"crash": None, "rule": rule, "valid": bool(ok), "errors": errors if not ok else []}


def corpus_suite() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    for case in CORPUS:
        res = _check_utterance(case["text"])
        rule = res["rule"]
        materialized = bool(rule) and not rule.get("needs_clarification")
        entry: Dict[str, Any] = {
            "uid": case["uid"],
            "family": case["family"],
            "text": case["text"],
            "expect": case["expect"],
            "crash": res["crash"],
            "materialized": materialized,
            "clarification_kind": (rule or {}).get("clarification_kind"),
            "validator_ok": res["valid"],
            "threshold": (rule or {}).get("threshold"),
            "count": (rule or {}).get("count"),
        }
        failure: Optional[str] = None
        if res["crash"]:
            failure = "crash"
        elif materialized and res["valid"] is False:
            failure = "validator_soundness"
        elif case["expect"] == "clarify" and materialized:
            failure = "false_materialization"
        elif case["expect"] == "materialize" and not materialized:
            failure = "false_clarification"
        elif case["expect"] == "materialize" and "objects" in case:
            got_objects = [o.get("name") for o in ((rule or {}).get("objects") or [])]
            if got_objects != case["objects"]:
                failure = "object_leak"
            entry["objects"] = got_objects
        elif case["expect"] == "interval":
            got = _interval((rule or {}).get("threshold"))
            want = _stated_interval(case, case["text"])
            sub = _subset(got, want)
            if got is None or sub is False:
                failure = "semantic_soundness"
            entry["implied_interval"] = list(got) if got else None
            entry["stated_interval"] = list(want) if want else None
        entry["failure"] = failure
        rows.append(entry)
        if failure:
            violations.append(entry)
    by_family: Dict[str, int] = {}
    for row in rows:
        by_family[row["family"]] = by_family.get(row["family"], 0) + int(bool(row["failure"]))
    return {
        "n_cases": len(rows),
        "n_violations": len(violations),
        "violations_by_family": by_family,
        "cases": rows,
        "violations": violations,
    }


def _grammar_fuzz(n: int = GRAMMAR_FUZZ_N, seed: int = GRAMMAR_FUZZ_SEED) -> Dict[str, Any]:
    """Larger, grammar-aware fuzz: noise + incoherent-by-construction inputs."""
    rng = random.Random(seed)
    leads = ["alert me when", "notify me if", "warn me when", "alert me whenever",
             "notify me when", "warn me if", "alert me", ""]
    events = ["people gather", "a crowd forms", "someone loiters", "a fight breaks out",
              "a car enters", "someone enters", "people are inside", "a person remains in",
              "the count of people is high", "people crowd into", "persons assemble"]
    zones = ["in Zone A", "in Zone B", "in Zone C", "at the main gate",
             "at the front entrance", "in the Parking lot"]
    good_bounds = ["more than five people", "at least ten people", "fewer than three people",
                   "exactly six people", "between two and eight people", "five or more people",
                   "no more than four people"]
    # (phrase, stated integer interval, stated unit family)
    two_sided = [
        ("more than five but fewer than ten people", (6, 9), "persons"),
        ("at least five and at most ten people", (5, 10), "persons"),
        ("more than five people but fewer than ten people", (6, 9), "persons"),
    ]
    bad_between = ["between ten and three people", "between nine and two people"]
    bad_window = ["between 3 AM and 3 AM", "from 9 PM to 9 PM", "before 12 AM"]
    bad_cadence = ["every 0.05 seconds", "at 20 hz", "at 99 hz"]
    times = ["after 10 PM", "between 9 and 11 PM", "at night", "during the day",
             "between 22:00 and 23:30", "in the morning", ""]
    tails = ["", ", cooldown 2 minutes", ", notification only", ", save a clip",
             " with critical severity", " every 30 seconds", ", high priority"]
    noise = ["\u00a0", "\u200b", "\u200d", "!!!", "...", "xkq", "\ufffd", "  ",
             "\u0665", "\uff15", "\t"]

    crashes: List[str] = []
    validator_violations: List[Dict[str, Any]] = []
    semantic_violations: List[Dict[str, Any]] = []
    false_materializations: List[Dict[str, Any]] = []
    materialized = 0
    clarified = 0
    n_two_sided = 0
    n_incoherent = 0
    count_bound_not_applicable = 0
    unit_conflict_residual = 0

    for i in range(n):
        mode = rng.random()
        lead = rng.choice(leads)
        event = rng.choice(events)
        zone = rng.choice(zones)
        tail = rng.choice(tails)
        time_clause = rng.choice(times)
        noise_tokens = " ".join(rng.choice(noise) for _ in range(rng.randint(0, 2)))
        stated_interval: Optional[Tuple[Optional[int], Optional[int]]] = None
        stated_unit: Optional[str] = None
        expect_clarify = False

        if mode < 0.35:
            bound = rng.choice(good_bounds)
        elif mode < 0.50:
            phrase, stated_interval, stated_unit = rng.choice(two_sided)
            bound = phrase
            n_two_sided += 1
        elif mode < 0.60:
            bound = rng.choice(bad_between)
            expect_clarify = True
            n_incoherent += 1
        elif mode < 0.70:
            bound = rng.choice(good_bounds)
            time_clause = rng.choice(bad_window)
            expect_clarify = True
            n_incoherent += 1
        elif mode < 0.80:
            bound = rng.choice(good_bounds)
            tail = " " + rng.choice(bad_cadence)
            expect_clarify = True
            n_incoherent += 1
        else:
            bound = rng.choice(good_bounds + [p for p, _, _ in two_sided] + bad_between)

        text = " ".join(
            p for p in [lead, event, bound, zone, time_clause, tail, noise_tokens] if p
        ).strip()
        if rng.random() < 0.25:
            text = text.upper()
        if rng.random() < 0.25:
            text = text + "!!!"

        res = _check_utterance(text)
        if res["crash"]:
            crashes.append(f"#{i} {text!r}: {res['crash']}")
            continue
        rule = res["rule"]
        if rule.get("needs_clarification"):
            clarified += 1
            continue
        materialized += 1
        if res["valid"] is False:
            validator_violations.append({
                "index": i, "text": text, "errors": res.get("errors", []),
                "threshold": rule.get("threshold"), "time_of_day": rule.get("time_of_day"),
                "cadence_hz": rule.get("cadence_hz"),
            })
            continue
        if expect_clarify:
            false_materializations.append({
                "index": i, "text": text, "reason": "incoherent-by-construction utterance materialized",
                "threshold": rule.get("threshold"), "time_of_day": rule.get("time_of_day"),
                "cadence_hz": rule.get("cadence_hz"),
            })
            continue
        if stated_interval is not None:
            thr = rule.get("threshold") or {}
            thr_unit = thr.get("unit")
            if thr_unit == "seconds":
                # The request is a dwell-time rule: a people-count bound does
                # not apply to it (the materializer correctly uses the
                # duration threshold).  Not a bound-truncation violation.
                count_bound_not_applicable += 1
            elif thr_unit != stated_unit:
                # The stated bound counts a different object family than the
                # rule (e.g. "a car enters more than five but fewer than ten
                # people"): a known residual of the proposed patch.
                unit_conflict_residual += 1
            else:
                got = _interval(thr)
                sub = _subset(got, stated_interval)
                if got is None or sub is False:
                    semantic_violations.append({
                        "index": i, "text": text, "implied": list(got) if got else None,
                        "stated": list(stated_interval), "threshold": thr,
                    })
    return {
        "seed": seed,
        "n": n,
        "n_two_sided_generated": n_two_sided,
        "n_incoherent_generated": n_incoherent,
        "crashes": len(crashes),
        "crash_samples": crashes[:10],
        "validator_soundness_violations": len(validator_violations),
        "validator_violation_samples": validator_violations[:10],
        "semantic_soundness_violations": len(semantic_violations),
        "semantic_violation_samples": semantic_violations[:10],
        "count_bound_not_applicable": count_bound_not_applicable,
        "unit_conflict_residual": unit_conflict_residual,
        "false_materializations": len(false_materializations),
        "false_materialization_samples": false_materializations[:10],
        "materialized": materialized,
        "clarified": clarified,
    }


def _shrink(text: str, prop: Callable[[str], bool]) -> str:
    """Greedy token-deletion shrinker: smallest token sequence keeping ``prop``."""
    tokens = text.split()
    changed = True
    while changed and len(tokens) > 1:
        changed = False
        for i in range(len(tokens)):
            cand = tokens[:i] + tokens[i + 1:]
            joined = " ".join(cand)
            if prop(joined):
                tokens = cand
                changed = True
                break
    return " ".join(tokens)


def minimal_reproducers() -> List[Dict[str, Any]]:
    """Shrink every corpus violation to a minimal reproducer."""
    out: List[Dict[str, Any]] = []
    for case in CORPUS:
        text = case["text"]
        res = _check_utterance(text)
        rule = res["rule"]
        materialized = bool(rule) and not rule.get("needs_clarification")
        if res["crash"]:
            out.append({"uid": case["uid"], "family": case["family"], "kind": "crash",
                        "reproducer": text, "detail": res["crash"]})
            continue
        if materialized and res["valid"] is False:
            shrunk = _shrink(text, lambda t: (
                (lambda r: bool(r) and not r.get("needs_clarification")
                 and _check_utterance(t)["valid"] is False)(_materialize(t))))
            out.append({
                "uid": case["uid"], "family": case["family"], "kind": "validator_soundness",
                "reproducer": shrunk,
                "detail": "; ".join(_check_utterance(shrunk).get("errors", [])),
            })
            continue
        if case["expect"] == "interval" and materialized:
            def _too_wide(t: str) -> bool:
                want = _stated_interval(case, t)
                if want is None:
                    return False
                if any(tok not in t.lower() for tok in case.get("required", [])):
                    return False
                r = _materialize(t)
                if r.get("needs_clarification"):
                    return False
                got = _interval(r.get("threshold"))
                if got is None:
                    return True
                return _subset(got, want) is not True

            if _too_wide(text):
                shrunk = _shrink(text, _too_wide)
                r = _materialize(shrunk)
                out.append({
                    "uid": case["uid"], "family": case["family"], "kind": "semantic_soundness",
                    "reproducer": shrunk,
                    "stated_interval": list(_stated_interval(case, shrunk) or []),
                    "implied_interval": list(_interval(r.get("threshold")) or []),
                    "detail": f"threshold {r.get('threshold')} admits counts outside "
                              f"{list(_stated_interval(case, shrunk) or [])}",
                })
            continue
        if case["expect"] == "clarify" and materialized:
            shrunk = _shrink(text, lambda t: (lambda r: bool(r) and not r.get("needs_clarification"))(_materialize(t)))
            out.append({
                "uid": case["uid"], "family": case["family"], "kind": "false_materialization",
                "reproducer": shrunk,
                "detail": "incoherent utterance materialized as " + json.dumps(_materialize(shrunk).get("threshold")),
            })
            continue
        if case["expect"] == "materialize" and "objects" in case:
            want_objects = case["objects"]

            def _leaks(t: str) -> bool:
                r = _materialize(t)
                if r.get("needs_clarification"):
                    return False
                got = [o.get("name") for o in (r.get("objects") or [])]
                return got != want_objects

            if _leaks(text):
                shrunk = _shrink(text, _leaks)
                r = _materialize(shrunk)
                out.append({
                    "uid": case["uid"], "family": case["family"], "kind": "object_leak",
                    "reproducer": shrunk,
                    "detail": f"objects {[o.get('name') for o in (r.get('objects') or [])]} "
                              f"(expected {want_objects}); the zone alias leaks an object",
                })
            continue
    return out


# ---------------------------------------------------------------------------
# 3. the proposed fix (source patch text + runtime monkeypatch)
# ---------------------------------------------------------------------------

PATCH_TEXT = """--- a/backend/app/services/alert_materializer.py
+++ b/backend/app/services/alert_materializer.py
@@ -28,10 +28,11 @@
 from typing import Any, Dict, List, Optional, Tuple
 
 from backend.app.services.alert_validator import (
     KNOWN_EVENTS,
     KNOWN_ZONES,
     UNSUPPORTED_HINTS,
+    validate_rule,
 )
 
 CANONICAL_FIELDS = [
@@ -122,6 +123,14 @@
     (\"person\", r\"\\b(?:persons?|people|someone|somebody|anyone|individuals?|intruders?|pedestrians?)\\b\"),
 ]
 
+# Zone aliases are masked before the object scan so that a zone phrase such
+# as \"car park\" does not contribute a spurious object (\"car\").
+_ZONE_SPAN_RE = re.compile(
+    r\"\\b(?:zone\\s*[- ]?[abc]|gate|entrance|\"
+    r\"parking\\s+lot|car\\s+park|parking\\s+area|parking)\\b\",
+    re.IGNORECASE,
+)
+
 _VEHICLE_OBJECTS = {\"car\", \"truck\", \"bus\", \"van\", \"vehicle\", \"motorcycle\"}
@@ -425,9 +434,10 @@ def _objects(text: str) -> List[str]:
     \"\"\"Object mentions in the utterance, in order of appearance.\"\"\"
     hits: List[Tuple[int, str]] = []
+    scan = _ZONE_SPAN_RE.sub(\" \", text)
     for name, pattern in _OBJECT_PATTERNS:
-        for match in re.finditer(pattern, text, re.IGNORECASE):
+        for match in re.finditer(pattern, scan, re.IGNORECASE):
             hits.append((match.start(), name))
     hits.sort(key=lambda item: item[0])
@@ -188,6 +198,13 @@
     rf\"backpacks?|bags?|objects?)\\b\",
     re.IGNORECASE,
 )
 
+# Duration bounds with their own unit, so that minutes and seconds are not
+# compared as raw numbers.
+_SECONDS_BOUND_RE = re.compile(
+    rf\"\\b(?P<cmp>{_COMPARATOR_RE})\\s+(?P<num>{_NUM_RE})\\s*(?P<unit>{_TIME_UNIT_RE})\\b\",
+    re.IGNORECASE,
+)
+
 _DURATION_RE = re.compile(
     rf\"\\bfor\\s+(?:(?P<cmp>{_COMPARATOR_RE})\\s+)?\"
     rf\"(?P<num>{_NUM_RE}|a|an)\\s*(?P<unit>{_TIME_UNIT_RE})\\b\",
     re.IGNORECASE,
@@ -521,6 +529,44 @@ def _op_from_comparator(cmp_token: str) -> str:
     return ">="
 
 
+def _combine_bounds(bounds: List[Dict[str, Any]], unit: str) -> Optional[Dict[str, Any]]:
+    \"\"\"Combine conjunctive bounds of one unit family into a single threshold.
+
+    A request may state several bounds of the same unit (\"more than five but
+    fewer than ten people\").  Keeping only one of them silently widens the
+    rule: the created alert then fires in configurations the user excluded.
+    This helper returns the conjunction of ``bounds`` as one threshold
+    (``between`` for a two-sided range, the strictest bound otherwise) and
+    ``None`` when the bounds cannot be combined.
+    \"\"\"
+    lowers = [b for b in bounds if b[\"op\"] in {\">\", \">=\"}]
+    uppers = [b for b in bounds if b[\"op\"] in {\"<\", \"<=\"}]
+    exact = [b for b in bounds if b[\"op\"] == \"==\"]
+    if exact:
+        return {\"op\": \"==\", \"value\": exact[0][\"value\"], \"unit\": unit}
+    if lowers and uppers:
+        lo_b = max(lowers, key=lambda b: (b[\"value\"], b[\"op\"] == \">\"))
+        hi_b = min(uppers, key=lambda b: (b[\"value\"], b[\"op\"] == \"<\"))
+        lo, hi = lo_b[\"value\"], hi_b[\"value\"]
+        # Counts and durations are integers: tighten strict bounds into
+        # inclusive integer bounds so the range is represented exactly.
+        if lo_b[\"op\"] == \">\" and isinstance(lo, int) and not isinstance(lo, bool):
+            lo += 1
+        if hi_b[\"op\"] == \"<\" and isinstance(hi, int) and not isinstance(hi, bool):
+            hi -= 1
+        if lo > hi:
+            return None
+        return {\"op\": \"between\", \"value\": [lo, hi], \"unit\": unit}
+    if lowers:
+        b = max(lowers, key=lambda b: (b[\"value\"], b[\"op\"] == \">\"))
+        return {\"op\": b[\"op\"], \"value\": b[\"value\"], \"unit\": unit}
+    if uppers:
+        b = min(uppers, key=lambda b: (b[\"value\"], b[\"op\"] == \"<\"))
+        return {\"op\": b[\"op\"], \"value\": b[\"value\"], \"unit\": unit}
+    return None
+
+
 def _quantifiers(text: str) -> List[Dict[str, Any]]:
     \"\"\"All comparator+number mentions, for conflict detection and extraction.\"\"\"
     out: List[Dict[str, Any]] = []
@@ -603,6 +653,16 @@ def _extract_count(text: str, objects: List[str]) -> Optional[Dict[str, Any]]:
     if unit == \"seconds\":
         # A duration used as a count is not meaningful for crowd/count events;
         # keep it as seconds and let the caller reject/clarify.
         return {\"op\": chosen[\"op\"], \"value\": value, \"unit\": \"seconds\"}
+
+    # Keep every bound stated for the same unit family: a two-sided request
+    # (\"more than five but fewer than ten people\") must not lose one side.
+    family = [q for q in quantifiers if (q[\"unit\"] or unit) == unit]
+    if len(family) > 1:
+        combined = _combine_bounds(family, unit)
+        if combined is not None:
+            return combined
     return {\"op\": chosen[\"op\"], \"value\": value, \"unit\": unit}
 
 
@@ -625,7 +685,34 @@ def _extract_duration(text: str) -> Optional[Dict[str, Any]]:
     cmp_token = match.group(\"cmp\")
     op = _op_from_comparator(cmp_token) if cmp_token else \">=\"
-    return {\"op\": op, \"value\": seconds, \"unit\": \"seconds\"}
+    bounds = [{\"op\": op, \"value\": seconds}]
+    for other in _SECONDS_BOUND_RE.finditer(text):
+        val = _num(other.group(\"num\"))
+        if val is None:
+            continue
+        factor = _TIME_UNIT_SEC.get(other.group(\"unit\").lower())
+        if not factor:
+            continue
+        secs = val * factor
+        if float(secs).is_integer():
+            secs = int(secs)
+        other_op = _op_from_comparator(other.group(\"cmp\"))
+        if (other_op, secs) == (op, seconds):
+            continue
+        bounds.append({\"op\": other_op, \"value\": secs})
+    if len(bounds) > 1:
+        combined = _combine_bounds(bounds, \"seconds\")
+        if combined is not None:
+            return combined
+        lowers = [b for b in bounds if b[\"op\"] in {\">\", \">=\"}]
+        if lowers:
+            strictest = max(lowers, key=lambda b: (b[\"value\"], b[\"op\"] == \">\"))
+            return {\"op\": strictest[\"op\"], \"value\": strictest[\"value\"], \"unit\": \"seconds\"}
+    return {\"op\": op, \"value\": seconds, \"unit\": \"seconds\"}
 
 
 def _threshold_for_event(
@@ -760,6 +847,24 @@ def _conflict(text: str, objects: List[str], zones: List[str]) -> Optional[str]:
         if eq_values and (upper is not None and upper < next(iter(eq_values))):
             return \"exact threshold above upper bound\"
 
+    # An inverted \"between X and Y <count-unit>\" range contradicts itself.
+    match = _BETWEEN_COUNT_RE.search(text)
+    if match:
+        lo = _num(match.group(\"lo\"))
+        hi = _num(match.group(\"hi\"))
+        if lo is not None and hi is not None and lo > hi:
+            return \"between bounds are inverted (lower exceeds upper)\"
+    if re.search(r\"\\b(?:count|number\\s+of|vehicle\\s+count)\\b\", text):
+        match = re.search(
+            rf\"\\bbetween\\s+(?P<lo>{_NUM_RE})\\s+and\\s+(?P<hi>{_NUM_RE})\\b\", text
+        )
+        if match and not re.search(
+            rf\"\\bbetween\\s+{match.group('lo')}\\s+and\\s+{match.group('hi')}\\s*(?:am|pm)\\b\",
+            text,
+        ):
+            lo = _num(match.group(\"lo\"))
+            hi = _num(match.group(\"hi\"))
+            if lo is not None and hi is not None and lo > hi:
+                return \"between bounds are inverted (lower exceeds upper)\"
+
     # Conflicting zones / object sets expressed as disjunctions.
     if re.search(r\"\\bor\\b\", text):
         if len(set(zones)) > 1:
@@ -952,6 +1057,20 @@ def materialize_alert_rule(
         \"clarification_kind\": kind,
     }
+
+    # Safety gate: a rule the validator rejects must never be reported as
+    # \"clean\" (the live chat path refuses to insert it and the dialogue would
+    # assume the rule is complete).  Such a request is contradictory for the
+    # typed grammar, so it is clarified instead of materialized.
+    if kind == \"clean\":
+        ok, errors, _warnings = validate_rule(rule)
+        if not ok:
+            kind = \"conflicting\"
+            detail = \"the requested rule is not representable (\" + \"; \".join(errors) + \")\"
+            rule[\"needs_clarification\"] = True
+            rule[\"clarification\"] = _clarification_reason(kind, detail)
+            rule[\"clarification_kind\"] = kind
     return rule
"""

_ORIG_MATERIALIZE = AM.materialize_alert_rule
_ORIG_EXTRACT_COUNT = AM._extract_count
_ORIG_EXTRACT_DURATION = AM._extract_duration
_ORIG_CONFLICT = AM._conflict
_ORIG_OBJECTS = AM._objects

_SECONDS_BOUND_RE = re.compile(
    rf"\b(?P<cmp>{AM._COMPARATOR_RE})\s+(?P<num>{AM._NUM_RE})\s*(?P<unit>{AM._TIME_UNIT_RE})\b",
    re.IGNORECASE,
)

_ZONE_SPAN_RE = re.compile(
    r"\b(?:zone\s*[- ]?[abc]|gate|entrance|"
    r"parking\s+lot|car\s+park|parking\s+area|parking)\b",
    re.IGNORECASE,
)


def _patched_objects(text: str) -> List[str]:
    """Mirror of the patched ``_objects`` (see PATCH_TEXT)."""
    scan = _ZONE_SPAN_RE.sub(" ", text)
    hits: List[Tuple[int, str]] = []
    for name, pattern in AM._OBJECT_PATTERNS:
        for match in re.finditer(pattern, scan, re.IGNORECASE):
            hits.append((match.start(), name))
    hits.sort(key=lambda item: item[0])
    out: List[str] = []
    for _, name in hits:
        if name not in out:
            out.append(name)
    return out


def _patched_combine_bounds(bounds: List[Dict[str, Any]], unit: str) -> Optional[Dict[str, Any]]:
    """Mirror of the patched ``_combine_bounds`` (see PATCH_TEXT)."""
    lowers = [b for b in bounds if b["op"] in {">", ">="}]
    uppers = [b for b in bounds if b["op"] in {"<", "<="}]
    exact = [b for b in bounds if b["op"] == "=="]
    if exact:
        return {"op": "==", "value": exact[0]["value"], "unit": unit}
    if lowers and uppers:
        lo_b = max(lowers, key=lambda b: (b["value"], b["op"] == ">"))
        hi_b = min(uppers, key=lambda b: (b["value"], b["op"] == "<"))
        lo, hi = lo_b["value"], hi_b["value"]
        if lo_b["op"] == ">" and isinstance(lo, int) and not isinstance(lo, bool):
            lo += 1
        if hi_b["op"] == "<" and isinstance(hi, int) and not isinstance(hi, bool):
            hi -= 1
        if lo > hi:
            return None
        return {"op": "between", "value": [lo, hi], "unit": unit}
    if lowers:
        b = max(lowers, key=lambda b: (b["value"], b["op"] == ">"))
        return {"op": b["op"], "value": b["value"], "unit": unit}
    if uppers:
        b = min(uppers, key=lambda b: (b["value"], b["op"] == "<"))
        return {"op": b["op"], "value": b["value"], "unit": unit}
    return None


def _patched_extract_count(text: str, objects: List[str]) -> Optional[Dict[str, Any]]:
    """Mirror of the patched ``_extract_count`` (see PATCH_TEXT)."""
    default_unit = AM._unit_for_objects(objects)
    match = AM._BETWEEN_COUNT_RE.search(text)
    if match:
        low = AM._num(match.group("lo"))
        high = AM._num(match.group("hi"))
        if low is not None and high is not None:
            unit = AM._unit_from_token(match.group("unit")) or default_unit
            return {"op": "between", "value": [low, high], "unit": unit}
    if re.search(r"\b(?:count|number\s+of|vehicle\s+count)\b", text):
        match = re.search(rf"\bbetween\s+(?P<lo>{AM._NUM_RE})\s+and\s+(?P<hi>{AM._NUM_RE})\b", text)
        if match and not re.search(
            rf"\bbetween\s+{match.group('lo')}\s+and\s+{match.group('hi')}\s*(?:am|pm)\b", text
        ):
            low = AM._num(match.group("lo"))
            high = AM._num(match.group("hi"))
            if low is not None and high is not None:
                return {"op": "between", "value": [low, high], "unit": default_unit}
    match = re.search(rf"\b(?:exceeds?|is\s+above|is\s+over)\s+(?P<num>{AM._NUM_RE})\b", text)
    if match:
        value = AM._num(match.group("num"))
        if value is not None:
            return {"op": ">", "value": value, "unit": default_unit}

    quantifiers = AM._quantifiers(text)
    if not quantifiers:
        return None

    chosen = quantifiers[0]
    for q in quantifiers:
        if q["unit"] == default_unit:
            chosen = q
            break

    unit = chosen["unit"] or default_unit
    value = chosen["value"]
    if unit == "seconds":
        return {"op": chosen["op"], "value": value, "unit": "seconds"}

    family = [q for q in quantifiers if (q["unit"] or unit) == unit]
    if len(family) > 1:
        combined = _patched_combine_bounds(family, unit)
        if combined is not None:
            return combined
    return {"op": chosen["op"], "value": value, "unit": unit}


def _patched_extract_duration(text: str) -> Optional[Dict[str, Any]]:
    """Mirror of the patched ``_extract_duration`` (see PATCH_TEXT)."""
    match = AM._DURATION_RE.search(text)
    if not match:
        return None
    token = match.group("num")
    value = 1.0 if token in {"a", "an"} else AM._num(token)
    if value is None:
        return None
    unit = AM._TIME_UNIT_SEC.get(match.group("unit").lower())
    if not unit:
        return None
    seconds = value * unit
    if float(seconds).is_integer():
        seconds = int(seconds)
    cmp_token = match.group("cmp")
    op = AM._op_from_comparator(cmp_token) if cmp_token else ">="
    bounds = [{"op": op, "value": seconds}]
    for m2 in _SECONDS_BOUND_RE.finditer(text):
        val = AM._num(m2.group("num"))
        if val is None:
            continue
        factor = AM._TIME_UNIT_SEC.get(m2.group("unit").lower())
        if not factor:
            continue
        secs = val * factor
        if float(secs).is_integer():
            secs = int(secs)
        b_op = AM._op_from_comparator(m2.group("cmp"))
        if (b_op, secs) == (op, seconds):
            continue
        bounds.append({"op": b_op, "value": secs})
    if len(bounds) > 1:
        combined = _patched_combine_bounds(bounds, "seconds")
        if combined is not None:
            return combined
        strictest = _strictest_lower(bounds)
        if strictest is not None:
            return {"op": strictest["op"], "value": strictest["value"], "unit": "seconds"}
    return {"op": op, "value": seconds, "unit": "seconds"}


def _strictest_lower(bounds: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    lowers = [b for b in bounds if b["op"] in {">", ">="}]
    if not lowers:
        return None
    b = max(lowers, key=lambda b: (b["value"], b["op"] == ">"))
    return {"op": b["op"], "value": b["value"]}


def _inverted_between(text: str) -> Optional[str]:
    """Detect a self-contradictory ``between X and Y <count-unit>`` range."""
    m = AM._BETWEEN_COUNT_RE.search(text)
    if m:
        lo = AM._num(m.group("lo"))
        hi = AM._num(m.group("hi"))
        if lo is not None and hi is not None and lo > hi:
            return "between bounds are inverted (lower exceeds upper)"
    if re.search(r"\b(?:count|number\s+of|vehicle\s+count)\b", text):
        m = re.search(rf"\bbetween\s+(?P<lo>{AM._NUM_RE})\s+and\s+(?P<hi>{AM._NUM_RE})\b", text)
        if m and not re.search(
            rf"\bbetween\s+{m.group('lo')}\s+and\s+{m.group('hi')}\s*(?:am|pm)\b", text
        ):
            lo = AM._num(m.group("lo"))
            hi = AM._num(m.group("hi"))
            if lo is not None and hi is not None and lo > hi:
                return "between bounds are inverted (lower exceeds upper)"
    return None


def _patched_conflict(text: str, objects: List[str], zones: List[str]) -> Optional[str]:
    """Mirror of the patched ``_conflict`` (see PATCH_TEXT)."""
    reason = _ORIG_CONFLICT(text, objects, zones)
    if reason:
        return reason
    return _inverted_between(text)


def _patched_materialize(utterance: str, parsed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Mirror of the patched ``materialize_alert_rule`` (see PATCH_TEXT)."""
    rule = _ORIG_MATERIALIZE(utterance, parsed)
    if not rule.get("needs_clarification"):
        ok, errors, _warnings = validate_rule(rule)
        if not ok:
            rule["needs_clarification"] = True
            rule["clarification_kind"] = "conflicting"
            rule["clarification"] = (
                "Conflicting conditions (the requested rule is not representable: "
                + "; ".join(errors) + "). Please give one consistent zone, object set, "
                "numeric threshold, and time window."
            )
    return rule


@contextmanager
def patched_materializer():
    """Install the proposed fix as a runtime monkeypatch (source is untouched)."""
    saved = {
        "AM._extract_count": AM._extract_count,
        "AM._extract_duration": AM._extract_duration,
        "AM._conflict": AM._conflict,
        "AM._objects": AM._objects,
        "AM.materialize_alert_rule": AM.materialize_alert_rule,
        "EA.materialize_alert_rule": EA.materialize_alert_rule,
        "CP.materialize_alert_rule": sys.modules["backend.app.services.clarification_policy"].materialize_alert_rule,
    }
    AM._extract_count = _patched_extract_count
    AM._extract_duration = _patched_extract_duration
    AM._conflict = _patched_conflict
    AM._objects = _patched_objects
    AM.materialize_alert_rule = _patched_materialize
    EA.materialize_alert_rule = _patched_materialize
    sys.modules["backend.app.services.clarification_policy"].materialize_alert_rule = _patched_materialize
    try:
        yield
    finally:
        AM._extract_count = saved["AM._extract_count"]
        AM._extract_duration = saved["AM._extract_duration"]
        AM._conflict = saved["AM._conflict"]
        AM._objects = saved["AM._objects"]
        AM.materialize_alert_rule = saved["AM.materialize_alert_rule"]
        EA.materialize_alert_rule = saved["EA.materialize_alert_rule"]
        sys.modules["backend.app.services.clarification_policy"].materialize_alert_rule = saved["CP.materialize_alert_rule"]


# ---------------------------------------------------------------------------
# 4. attack (b): clarification-policy frontier
# ---------------------------------------------------------------------------

TURN_CONVENTION = (
    "One turn = one assistant message. A message may bundle several slot "
    "questions (this is the convention the shipped harness already uses: "
    "`minimal_questions` returns a LIST of questions and the simulated oracle "
    "answers every asked slot inside that one message). The dialogue ends when "
    "the partial rule is complete (`_partial_complete`) or the policy returns "
    "no question. `turns` = number of assistant messages; `slots_asked` = "
    "number of slot questions asked across the dialogue (reported separately "
    "because a bundled question costs one turn but one slot). The protocol is "
    "fixed across all variants and identical to the shipped harness "
    "(seed 0xC3A1, reps=3, k=rep+1 masked slots of {event, zone, threshold})."
)

_EVENT_Q = ("What should I detect? Choose one: crowd, loiter, fight, enter, "
            "zone occupancy, count, or time-of-day watch.")
_ZONE_Q = ("Which canonical zone? Choose one: Zone A, Zone B, Zone C, Gate, "
           "Entrance, or Parking.")
_THRESHOLD_Q = ("What numeric threshold? For example 'more than five people' or "
                "'at least two minutes'.")


def _policy_v0(utt: str, partial: Dict[str, Any]) -> List[Dict[str, str]]:
    """Baseline: the shipped minimal-question policy.

    The shipped harness stops the dialogue as soon as the partial rule is
    complete, so the policy is never consulted again.  The same stop rule is
    applied here so the baseline reproduces the shipped numbers exactly.
    """
    if EA._partial_complete(partial):
        return []
    return list(minimal_questions(utt, rule=partial))


def _policy_v1(utt: str, partial: Dict[str, Any]) -> List[Dict[str, str]]:
    """Never ask for the threshold (event defaults always apply)."""
    if EA._partial_complete(partial):
        return []
    return [q for q in minimal_questions(utt, rule=partial) if q["slot"] != "threshold"]


def _policy_v2(utt: str, partial: Dict[str, Any]) -> List[Dict[str, str]]:
    """Always ask for the threshold when it is unknown.

    The threshold is asked even when the partial rule is already "complete"
    in the harness sense (the documented event default is *assumed*, never
    confirmed).  All slot questions are bundled into one message, so the
    extra question costs no extra turn.
    """
    qs: List[Dict[str, str]] = []
    if not EA._partial_complete(partial):
        qs = list(minimal_questions(utt, rule=partial))
    if partial.get("threshold") is None and not any(q["slot"] == "threshold" for q in qs):
        qs = qs + [{"slot": "threshold", "question": _THRESHOLD_Q}]
    return qs


def _policy_v3(utt: str, partial: Dict[str, Any]) -> List[Dict[str, str]]:
    """Bundle the (possibly needed) threshold with the event question.

    The threshold is asked only when the event class requires it, but when the
    event is still unknown the threshold question is bundled into the same
    message as the event question (no second round).
    """
    if partial.get("event") not in KNOWN_EVENTS:
        qs = [{"slot": "event", "question": _EVENT_Q}]
        if not (partial.get("area") or {}).get("zone_id"):
            qs.append({"slot": "zone", "question": _ZONE_Q})
        if partial.get("threshold") is None:
            qs.append({"slot": "threshold", "question": _THRESHOLD_Q})
        return qs
    if EA._partial_complete(partial):
        return []
    return list(minimal_questions(utt, rule=partial))


def _policy_v4(utt: str, partial: Dict[str, Any]) -> List[Dict[str, str]]:
    """V2 + V3 (always-threshold and bundle coincide)."""
    if partial.get("event") not in KNOWN_EVENTS:
        return _policy_v3(utt, partial)
    return _policy_v2(utt, partial)


POLICIES: Dict[str, Callable[[str, Dict[str, Any]], List[Dict[str, str]]]] = {
    "v0_baseline_minimal_questions": _policy_v0,
    "v1_never_ask_threshold": _policy_v1,
    "v2_always_ask_threshold": _policy_v2,
    "v3_bundle_threshold_with_event": _policy_v3,
    "v4_v2_plus_v3": _policy_v4,
}


def simulate_dialogue(
    policy: Callable[[str, Dict[str, Any]], List[Dict[str, str]]],
    rows: List[Dict[str, Any]],
    seed: int = EA.ORACLE_SEED,
    reps: int = EA.ORACLE_REPS,
    include_naive: bool = True,
) -> Dict[str, Any]:
    """Same protocol as ``eval_alerts.oracle_turns_simulation`` but with a policy.

    The dialogue is *policy driven*: the policy returns the next assistant
    message (a list of slot questions) or ``[]`` to stop.  The shipped
    baseline (``v0``) reproduces the shipped harness exactly because it
    applies the same "stop as soon as the partial rule is complete" rule.
    """
    rng = random.Random(seed)
    episodes: List[Dict[str, Any]] = []
    for r in rows:
        expected = r["expected"]
        for rep in range(reps):
            k = rep + 1
            masked = rng.sample(EA._required_slots(), k)
            partial = EA._fresh_partial(expected, masked)
            turns = 0
            slots_asked: List[str] = []
            while turns < 10:
                qs = policy(r["text"], partial)
                if not qs:
                    break
                turns += 1
                for q in qs:
                    slots_asked.append(q["slot"])
                    if q["slot"] in ("event", "zone", "threshold"):
                        EA._fill_from_expected(partial, expected, q["slot"])
            assembled = EA._assemble_for_validation(expected, partial)
            ok, _, _ = validate_rule(assembled)
            ep = {
                "uid": r["uid"],
                "masked": sorted(masked),
                "turns": turns,
                "slots_asked": sorted(set(slots_asked)),
                "n_slot_questions": len(slots_asked),
                "correct": bool(EA._dialogue_correct(partial, expected, masked)),
                "valid": bool(ok),
            }
            if include_naive:
                naive_partial = EA._fresh_partial(expected, masked)
                naive_turns = 0
                for field in NAIVE_BASELINE_FIELDS:
                    if EA._partial_complete(naive_partial):
                        break
                    naive_turns += 1
                    slot = {"area": "zone"}.get(field, field)
                    if slot in ("event", "zone", "threshold") and slot in masked:
                        EA._fill_from_expected(naive_partial, expected, slot)
                naive_assembled = EA._assemble_for_validation(expected, naive_partial)
                naive_ok, _, _ = validate_rule(naive_assembled)
                ep["naive_turns"] = naive_turns
                ep["naive_correct"] = bool(EA._dialogue_correct(naive_partial, expected, masked))
                ep["naive_valid"] = bool(naive_ok)
            episodes.append(ep)
    n = len(episodes)
    turns = [e["turns"] for e in episodes]
    out: Dict[str, Any] = {
        "n_episodes": n,
        "seed": seed,
        "reps": reps,
        "mean_turns": round(sum(turns) / n, 4) if n else 0.0,
        "max_turns": max(turns) if turns else 0,
        "correct": sum(1 for e in episodes if e["correct"]),
        "correct_rate": _acc(sum(1 for e in episodes if e["correct"]), n),
        "valid": sum(1 for e in episodes if e["valid"]),
        "mean_slot_questions": round(
            sum(e["n_slot_questions"] for e in episodes) / n, 4) if n else 0.0,
        "episodes": episodes,
    }
    if include_naive:
        nt = [e["naive_turns"] for e in episodes]
        out["naive_per_field"] = {
            "mean_turns": round(sum(nt) / n, 4) if n else 0.0,
            "max_turns": max(nt) if nt else 0,
            "correct": sum(1 for e in episodes if e["naive_correct"]),
            "valid": sum(1 for e in episodes if e["naive_valid"]),
        }
    return out


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math_choose(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return round(min(1.0, 2.0 * tail), 6)


def math_choose(n: int, k: int) -> float:
    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)
    return result


def cluster_bootstrap(
    episodes_a: List[Dict[str, Any]],
    episodes_b: List[Dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 0xC3B022,
) -> Dict[str, Any]:
    """Row-clustered bootstrap of the paired differences (correctness, turns)."""
    by_uid_a: Dict[str, List[Dict[str, Any]]] = {}
    by_uid_b: Dict[str, List[Dict[str, Any]]] = {}
    for e in episodes_a:
        by_uid_a.setdefault(e["uid"], []).append(e)
    for e in episodes_b:
        by_uid_b.setdefault(e["uid"], []).append(e)
    uids = sorted(set(by_uid_a) & set(by_uid_b))
    rng = random.Random(seed)
    d_corr: List[float] = []
    d_turn: List[float] = []
    for _ in range(n_boot):
        pick = [uids[rng.randrange(len(uids))] for _ in uids]
        ca = cb = ta = tb = 0.0
        for uid in pick:
            for e in by_uid_a[uid]:
                ca += 1.0 if e["correct"] else 0.0
                ta += e["turns"]
            for e in by_uid_b[uid]:
                cb += 1.0 if e["correct"] else 0.0
                tb += e["turns"]
        n_e = sum(len(by_uid_a[u]) for u in pick)
        d_corr.append(cb / n_e - ca / n_e)
        d_turn.append(tb / n_e - ta / n_e)
    d_corr.sort()
    d_turn.sort()

    def _ci(vals: List[float]) -> List[float]:
        lo = vals[int(0.025 * (len(vals) - 1))]
        hi = vals[int(0.975 * (len(vals) - 1))]
        return [round(lo, 6), round(hi, 6)]

    return {
        "n_boot": n_boot,
        "seed": seed,
        "cluster": "test row (3 episodes per row)",
        "delta_correctness_frozen_minus_baseline_ci95": _ci(d_corr),
        "delta_turns_frozen_minus_baseline_ci95": _ci(d_turn),
    }


# ---------------------------------------------------------------------------
# 5. attack (c): paraphrase robustness on the 60 clean rows
# ---------------------------------------------------------------------------

_SYNONYMS: List[Tuple[str, str]] = [
    (r"\balert me\b", "notify me"),
    (r"\balert me\b", "warn me"),
    (r"\bnotify me\b", "alert me"),
    (r"\bwarn me\b", "alert me"),
    (r"\bwhen\b", "whenever"),
    (r"\bwhen\b", "if"),
    (r"\bgather\b", "congregate"),
    (r"\bgather\b", "assemble"),
    (r"\bgathers\b", "crowd"),
    (r"\bpeople\b", "persons"),
    (r"\bpeople\b", "individuals"),
    (r"\bmore than\b", "greater than"),
    (r"\bmore than\b", "over"),
    (r"\bmore than\b", "in excess of"),
    (r"\bat least\b", "no fewer than"),
    (r"\bfewer than\b", "less than"),
    (r"\bat most\b", "no more than"),
    (r"\bexactly\b", "equal to"),
    (r"\bloiters\b", "lingers"),
    (r"\bloiter\b", "linger"),
    (r"\bfight\b", "scuffle"),
    (r"\bfight\b", "brawl"),
    (r"\benters\b", "walks into"),
    (r"\binside\b", "present in"),
    (r"\bat night\b", "overnight"),
    (r"\bduring the day\b", "daytime"),
    (r"\bduring the day\b", "in the day"),
    (r"\bthe Parking lot\b", "the car park"),
    (r"\bthe main gate\b", "the front gate"),
    (r"\bthe front entrance\b", "the entrance"),
    (r"\bZone ([ABC])\b", r"zone \1"),
    (r"\bcooldown (\d+)", r"cooldown of \1"),
    (r"\bsave a clip\b", "store a clip"),
    (r"\bnotification only\b", "push only"),
    (r"\bsend a snapshot only\b", "snapshot only"),
]

_FILLERS = ["please", "kindly", "hey", "ok", "really", "quickly", "thanks"]

_ZONE_PHRASE_RE = re.compile(
    r"\b(?:in|at|near|inside|within)\s+(?:zone\s*[- ]?[abc]\b|"
    r"(?:the\s+)?(?:main|front)?\s*gate\b|"
    r"(?:the\s+)?(?:main|front)?\s*entrance\b|"
    r"(?:the\s+)?(?:parking\s+lot|car\s+park|parking\s+area|parking)\b)",
    re.IGNORECASE,
)
_PREDICATE_VERBS = [
    "gather", "congregate", "assemble", "cluster", "crowd", "loiter", "linger",
    "fight", "enter", "walks into", "come into", "crosses into", "stay in",
    "remain in", "are inside", "is inside", "present in", "occupied",
]


def _perturb_case(text: str) -> List[Dict[str, str]]:
    variants: List[Dict[str, str]] = []
    variants.append({"kind": "case_punct", "text": text.upper()})
    variants.append({"kind": "case_punct", "text": text.lower()})
    variants.append({"kind": "case_punct", "text": text.rstrip(".!?") + "!"})
    variants.append({"kind": "case_punct", "text": text.replace(",", ";")})
    variants.append({"kind": "case_punct", "text": text + "  \t"})
    tokens = text.split()
    for pos, filler in ((0, _FILLERS[0]), (0, _FILLERS[1]),
                        (min(2, len(tokens)), _FILLERS[2]),
                        (len(tokens), _FILLERS[3])):
        cand = tokens[:pos] + [filler] + tokens[pos:]
        variants.append({"kind": "filler", "text": " ".join(cand)})
    for pattern, repl in _SYNONYMS:
        new_text, count = re.subn(pattern, repl, text, count=1)
        if count:
            variants.append({"kind": "synonym", "text": new_text, "rule": f"{pattern} -> {repl}"})
    # word order: move the zone phrase in front of the predicate verb
    m = _ZONE_PHRASE_RE.search(text)
    if m:
        zone_span = m.group(0)
        before = text[:m.start()]
        best = None
        for verb in _PREDICATE_VERBS:
            idx = before.rfind(verb)
            if idx >= 0 and (best is None or idx > best[0]):
                best = (idx, verb)
        if best is not None:
            idx, verb = best
            moved = text[:idx] + zone_span + " " + text[idx:m.start()] + text[m.end():]
            variants.append({"kind": "word_order", "text": moved, "rule": "zone phrase moved before predicate"})
    return variants


def paraphrase_robustness(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean = [r for r in rows if r["stress"] == "clean"]
    results: List[Dict[str, Any]] = []
    by_kind: Dict[str, Dict[str, int]] = {}
    clean_ok = 0
    for r in clean:
        got = _materialize(r["text"])
        ok = (not got.get("needs_clarification")) and all(
            got.get(f) == r["expected"].get(f) for f in CANONICAL_FIELDS)
        clean_ok += int(ok)
        for var in _perturb_case(r["text"]):
            g = _materialize(var["text"])
            v_ok = (not g.get("needs_clarification")) and all(
                g.get(f) == r["expected"].get(f) for f in CANONICAL_FIELDS)
            kind = var["kind"]
            slot = by_kind.setdefault(kind, {"n": 0, "correct": 0})
            slot["n"] += 1
            slot["correct"] += int(v_ok)
            results.append({
                "uid": r["uid"], "split": r["split"], "kind": kind,
                "rule": var.get("rule"),
                "text": var["text"], "correct": bool(v_ok),
                "clarification_kind": g.get("clarification_kind"),
                "mismatch_fields": [
                    f for f in CANONICAL_FIELDS if g.get(f) != r["expected"].get(f)
                ] if not v_ok else [],
            })
    n = len(results)
    return {
        "clean_rows": len(clean),
        "clean_rule_correct": clean_ok,
        "clean_rule_correctness": _acc(clean_ok, len(clean)),
        "n_perturbed": n,
        "perturbed_rule_correct": sum(1 for x in results if x["correct"]),
        "perturbed_rule_correctness": _acc(sum(1 for x in results if x["correct"]), n),
        "by_perturbation_kind": {
            k: {"n": v["n"], "correct": v["correct"], "accuracy": _acc(v["correct"], v["n"])}
            for k, v in sorted(by_kind.items())
        },
        "by_split": {
            s: {
                "n": sum(1 for x in results if x["split"] == s),
                "correct": sum(1 for x in results if x["split"] == s and x["correct"]),
            }
            for s in ("dev", "test")
        },
        "failures": [x for x in results if not x["correct"]][:40],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = _load_rows()
    by_split = {s: [r for r in rows if r.get("split") == s] for s in ("dev", "test")}
    clean_dev = [r for r in by_split["dev"] if r["stress"] == "clean"]
    clean_test = [r for r in by_split["test"] if r["stress"] == "clean"]

    out: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": COMMAND,
        "python": sys.version.split()[0],
        "probe": "C3 probe #2 (bug hunt / policy frontier / paraphrase robustness)",
        "read_only": "no tracked file, benchmark, or existing results/ file was modified",
        "dataset": {
            "path": "backend/benchmarks/alert_utterances.jsonl",
            "sha256": hashlib.sha256(BENCH.read_bytes()).hexdigest(),
            "rows": len(rows),
        },
    }

    # ---- 1. baseline replication -------------------------------------
    print("[1/6] replicating the shipped baseline ...")
    base = baseline_replication(rows)
    out["baseline_replication"] = base
    print("      invariants hold:", base["all_invariants_hold"])
    for k, v in base["invariants"].items():
        if not v:
            print("      FAILED:", k)

    # ---- 2. attack (a): bug hunt -------------------------------------
    print("[2/6] attack (a): expanded grammar-aware fuzz ...")
    corpus = corpus_suite()
    gfuzz = _grammar_fuzz()
    repros = minimal_reproducers()
    residual = []
    for case in RESIDUAL:
        res = _check_utterance(case["text"])
        rule = res["rule"]
        residual.append({
            **case,
            "materialized": bool(rule) and not rule.get("needs_clarification"),
            "validator_ok": res["valid"],
            "threshold": (rule or {}).get("threshold"),
            "count": (rule or {}).get("count"),
            "time_of_day": (rule or {}).get("time_of_day"),
        })
    out["attack_a"] = {
        "corpus": corpus,
        "grammar_fuzz": gfuzz,
        "minimal_reproducers": repros,
        "residual_unfixed": residual,
    }
    print(f"      corpus violations: {corpus['n_violations']}/{corpus['n_cases']}")
    print(f"      grammar fuzz: validator violations {gfuzz['validator_soundness_violations']}, "
          f"semantic violations {gfuzz['semantic_soundness_violations']}, "
          f"false materializations {gfuzz['false_materializations']}, crashes {gfuzz['crashes']}")

    # ---- 3. attack (b): policy frontier (DEV selection) --------------
    print("[3/6] attack (b): policy frontier (dev selection) ...")
    dev_variants: Dict[str, Any] = {}
    for name, pol in POLICIES.items():
        sim = simulate_dialogue(pol, clean_dev)
        dev_variants[name] = sim
    selection_rule = (
        "Pre-registered on the DEV split only: maximise rule-correctness; "
        "tie-break on the lower mean number of turns; final tie-break on the "
        "lower mean number of slot questions. The winner is frozen and "
        "evaluated ONCE on the TEST split."
    )
    ranked = sorted(
        dev_variants.items(),
        key=lambda kv: (-kv[1]["correct"], kv[1]["mean_turns"], kv[1]["mean_slot_questions"]),
    )
    frozen_name = ranked[0][0]
    out["attack_b"] = {
        "turn_convention": TURN_CONVENTION,
        "selection_rule": selection_rule,
        "dev_variants": {
            k: {kk: vv for kk, vv in v.items() if kk != "episodes"} for k, v in dev_variants.items()
        },
        "dev_episodes": {k: v["episodes"] for k, v in dev_variants.items()},
        "dev_ranking": [
            {"name": k, "correct": v["correct"], "mean_turns": v["mean_turns"],
             "max_turns": v["max_turns"], "valid": v["valid"],
             "mean_slot_questions": v["mean_slot_questions"]}
            for k, v in ranked
        ],
        "frozen_variant": frozen_name,
    }
    print(f"      dev winner: {frozen_name}")

    # ---- 4. attack (b): frozen variant on TEST (one evaluation) -------
    print(f"[4/6] attack (b): frozen variant {frozen_name} on TEST ...")
    frozen_test = simulate_dialogue(POLICIES[frozen_name], clean_test)
    baseline_test = simulate_dialogue(POLICIES["v0_baseline_minimal_questions"], clean_test)
    b_only = sum(1 for a, b in zip(baseline_test["episodes"], frozen_test["episodes"])
                 if a["correct"] and not b["correct"])
    c_only = sum(1 for a, b in zip(baseline_test["episodes"], frozen_test["episodes"])
                 if b["correct"] and not a["correct"])
    boot = cluster_bootstrap(baseline_test["episodes"], frozen_test["episodes"])
    out["attack_b"]["frozen_test"] = {k: v for k, v in frozen_test.items() if k != "episodes"}
    out["attack_b"]["frozen_test_episodes"] = frozen_test["episodes"]
    out["attack_b"]["baseline_test_reference"] = {
        k: v for k, v in baseline_test.items() if k != "episodes"}
    out["attack_b"]["paired_test"] = {
        "baseline_correct_only": b_only,
        "frozen_correct_only": c_only,
        "mcnemar_exact_p": _mcnemar_exact(b_only, c_only),
        "bootstrap": boot,
    }
    print(f"      frozen test: {frozen_test['correct']}/{frozen_test['n_episodes']} correct, "
          f"mean turns {frozen_test['mean_turns']} (baseline {baseline_test['correct']}/"
          f"{baseline_test['n_episodes']}, mean turns {baseline_test['mean_turns']})")

    # ---- 5. attack (c): paraphrase robustness ------------------------
    print("[5/6] attack (c): paraphrase robustness ...")
    para = paraphrase_robustness(rows)
    out["attack_c"] = para
    print(f"      clean {para['clean_rule_correct']}/{para['clean_rows']}; "
          f"perturbed {para['perturbed_rule_correct']}/{para['n_perturbed']} "
          f"({para['perturbed_rule_correctness']:.3f})")

    # ---- 6. fix proof (runtime monkeypatch) --------------------------
    print("[6/6] proof of the proposed fix (runtime monkeypatch, source untouched) ...")
    with patched_materializer():
        patched_base = baseline_replication(rows)
        patched_corpus = corpus_suite()
        patched_gfuzz = _grammar_fuzz()
        patched_para = paraphrase_robustness(rows)
        patched_repro = []
        cases_by_uid = {c["uid"]: c for c in CORPUS}
        for rep in repros:
            res = _check_utterance(rep["reproducer"])
            rule = res["rule"]
            materialized = bool(rule) and not rule.get("needs_clarification")
            if rep["kind"] == "semantic_soundness":
                case = cases_by_uid.get(rep["uid"], {})
                want = _stated_interval(case, rep["reproducer"])
                got = _interval((rule or {}).get("threshold"))
                resolved = _subset(got, want) is True
            elif rep["kind"] == "object_leak":
                case = cases_by_uid.get(rep["uid"], {})
                got_objects = [o.get("name") for o in ((rule or {}).get("objects") or [])]
                resolved = got_objects == case.get("objects")
            else:
                resolved = not materialized
            patched_repro.append({
                **rep,
                "now_materialized": materialized,
                "now_validator_ok": res["valid"],
                "now_clarification_kind": (rule or {}).get("clarification_kind"),
                "now_threshold": (rule or {}).get("threshold"),
                "now_implied_interval": list(_interval((rule or {}).get("threshold")) or []),
                "resolved": resolved,
            })
    out["fix"] = {
        "patch": PATCH_TEXT,
        "patch_applied_to_source": False,
        "monkeypatched_baseline": {
            "invariants": patched_base["invariants"],
            "all_invariants_hold": patched_base["all_invariants_hold"],
            "test_clean": {
                "rule_correct": patched_base["splits"]["test"]["clean"]["rule_correct"],
                "n": patched_base["splits"]["test"]["clean"]["n"],
                "rule_correctness": patched_base["splits"]["test"]["clean"]["rule_correctness"],
            },
            "test_stress": {
                "clarified": patched_base["splits"]["test"]["stress"]["clarified"],
                "n": patched_base["splits"]["test"]["stress"]["n"],
                "false_materializations": patched_base["splits"]["test"]["stress"]["false_materializations"],
                "per_kind": patched_base["splits"]["test"]["stress"]["per_kind"],
            },
            "compositional": patched_base["compositional_heldout"],
            "oracle_turns": {
                "policy": patched_base["oracle_turns"]["policy"],
                "naive_per_field": patched_base["oracle_turns"]["naive_per_field"],
                "n_episodes": patched_base["oracle_turns"]["n_episodes"],
            },
            "fuzz": {
                k: v for k, v in patched_base["fuzz"].items()
                if k not in ("soundness_violations", "targeted_false_materializations")
            },
        },
        "monkeypatched_corpus": {
            "n_cases": patched_corpus["n_cases"],
            "n_violations": patched_corpus["n_violations"],
            "violations_by_family": patched_corpus["violations_by_family"],
            "violations": patched_corpus["violations"],
        },
        "monkeypatched_grammar_fuzz": {
            k: v for k, v in patched_gfuzz.items() if not k.endswith("_samples")
        },
        "monkeypatched_paraphrase": {
            "clean_rule_correct": patched_para["clean_rule_correct"],
            "clean_rows": patched_para["clean_rows"],
            "n_perturbed": patched_para["n_perturbed"],
            "perturbed_rule_correct": patched_para["perturbed_rule_correct"],
            "perturbed_rule_correctness": patched_para["perturbed_rule_correctness"],
            "by_perturbation_kind": patched_para["by_perturbation_kind"],
            "failures": patched_para["failures"][:20],
        },
        "monkeypatched_reproducers": patched_repro,
    }
    print(f"      patched invariants hold: {patched_base['all_invariants_hold']}")
    print(f"      patched corpus violations: {patched_corpus['n_violations']}")
    print(f"      patched fuzz: validator {patched_gfuzz['validator_soundness_violations']}, "
          f"semantic {patched_gfuzz['semantic_soundness_violations']}, "
          f"false-mat {patched_gfuzz['false_materializations']}, crashes {patched_gfuzz['crashes']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(_write_markdown(out), encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


def _write_markdown(o: Dict[str, Any]) -> str:
    L: List[str] = []
    base = o["baseline_replication"]
    a = o["attack_a"]
    b = o["attack_b"]
    c = o["attack_c"]
    fx = o["fix"]

    L.append("# C3 probe #2 — attack report (alert materializer / clarification policy)")
    L.append("")
    L.append(f"**Command:** `{o['command']}` (repository root, Python {o['python']}, UTF-8).")
    L.append("")
    L.append("**Read-only probe.** No tracked file, no benchmark row and no pre-existing "
             "`results/` artifact was modified; the proposed fix is pasted below and "
             "proved with runtime monkeypatches only.")
    L.append("")
    L.append(f"**Dataset:** `{o['dataset']['path']}` — {o['dataset']['rows']} rows, "
             f"sha256 `{o['dataset']['sha256'][:16]}…` (unchanged).")
    L.append("")

    L.append("## 0. Baseline replication (shipped numbers reproduced)")
    L.append("")
    L.append("| Invariant | Result |")
    L.append("|---|---|")
    for k, v in base["invariants"].items():
        L.append(f"| `{k}` | {'PASS' if v else 'FAIL'} |")
    L.append("")

    L.append("## 1. Attack (a) — bug hunt (expanded grammar-aware fuzz)")
    L.append("")
    L.append(f"Adversarial corpus: **{a['corpus']['n_violations']}/{a['corpus']['n_cases']}** "
             "cases violate an expectation. "
             f"Randomized grammar-aware fuzz: N={a['grammar_fuzz']['n']} (seed "
             f"{a['grammar_fuzz']['seed']:#x}).")
    L.append("")
    L.append("| Property | Value |")
    L.append("|---|---:|")
    gf = a["grammar_fuzz"]
    L.append(f"| Crashes (materialize + validate) | {gf['crashes']} |")
    L.append(f"| Validator soundness violations (materialized ∧ ¬`validate_rule`) | "
             f"{gf['validator_soundness_violations']} |")
    L.append(f"| Semantic soundness violations (rule admits excluded counts) | "
             f"{gf['semantic_soundness_violations']} |")
    L.append(f"| False materializations of incoherent-by-construction inputs | "
             f"{gf['false_materializations']} |")
    L.append("")
    L.append("### Corpus violations by family")
    L.append("")
    L.append("| Family | Violations |")
    L.append("|---|---:|")
    for fam, n in sorted(a["corpus"]["violations_by_family"].items()):
        L.append(f"| `{fam}` | {n} |")
    L.append("")
    L.append("### Minimal seeded reproducers")
    L.append("")
    for rep in a["minimal_reproducers"]:
        L.append(f"- **{rep['kind']}** (`{rep['family']}`): `{rep['reproducer']}`")
        L.append(f"  - {rep['detail']}")
    L.append("")
    L.append("### Residual findings NOT fixed by the proposed patch")
    L.append("")
    for r in a["residual_unfixed"]:
        L.append(f"- `{r['family']}`: `{r['text']}` — {r['note']} "
                 f"(materialized: {r['materialized']}, threshold: {r['threshold']})")
    L.append("")

    L.append("## 2. Attack (b) — clarification-policy frontier")
    L.append("")
    L.append(f"**Turn counting (fixed for every variant):** {b['turn_convention']}")
    L.append("")
    L.append(f"**Selection rule (DEV only):** {b['selection_rule']}")
    L.append("")
    L.append("### DEV split (selection)")
    L.append("")
    L.append("| Variant | Correct | Mean turns | Max turns | Valid | Mean slot questions |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for row in b["dev_ranking"]:
        L.append(f"| `{row['name']}` | {row['correct']}/{len(b['dev_episodes'][row['name']])} | "
                 f"{row['mean_turns']:.2f} | {row['max_turns']} | {row['valid']} | "
                 f"{row['mean_slot_questions']:.2f} |")
    L.append("")
    L.append(f"**Frozen variant: `{b['frozen_variant']}`**")
    L.append("")
    L.append("### TEST split (single evaluation of the frozen variant)")
    L.append("")
    ft = b["frozen_test"]
    bt = b["baseline_test_reference"]
    L.append("| Policy | Mean turns | Max turns | Correct | Valid | Mean slot questions |")
    L.append("|---|---:|---:|---:|---:|---:|")
    L.append(f"| Baseline `minimal_questions` (shipped) | {bt['mean_turns']:.2f} | "
             f"{bt['max_turns']} | {bt['correct']}/{bt['n_episodes']} | {bt['valid']} | "
             f"{bt['mean_slot_questions']:.2f} |")
    L.append(f"| Frozen `{b['frozen_variant']}` | {ft['mean_turns']:.2f} | {ft['max_turns']} | "
             f"{ft['correct']}/{ft['n_episodes']} | {ft['valid']} | "
             f"{ft['mean_slot_questions']:.2f} |")
    L.append("")
    pt = b["paired_test"]
    L.append(f"Paired (n={ft['n_episodes']} episodes): baseline-correct-only "
             f"{pt['baseline_correct_only']}, frozen-correct-only {pt['frozen_correct_only']}; "
             f"exact McNemar p = {pt['mcnemar_exact_p']}; row-clustered bootstrap 95% CI on "
             f"Δcorrectness (frozen−baseline) "
             f"{pt['bootstrap']['delta_correctness_frozen_minus_baseline_ci95']}, on "
             f"Δturns {pt['bootstrap']['delta_turns_frozen_minus_baseline_ci95']}.")
    L.append("")

    L.append("## 3. Attack (c) — paraphrase robustness (60 clean rows, no benchmark edit)")
    L.append("")
    L.append(f"Clean rule-correctness: **{c['clean_rule_correct']}/{c['clean_rows']}** "
             f"({c['clean_rule_correctness']:.3f}).")
    L.append(f"Perturbed rule-correctness: **{c['perturbed_rule_correct']}/{c['n_perturbed']}** "
             f"({c['perturbed_rule_correctness']:.3f}).")
    L.append("")
    L.append("| Perturbation | n | Correct | Accuracy |")
    L.append("|---|---:|---:|---:|")
    for kind, v in c["by_perturbation_kind"].items():
        L.append(f"| `{kind}` | {v['n']} | {v['correct']} | {v['accuracy']:.3f} |")
    L.append("")
    if c["failures"]:
        L.append("Failures (first 40):")
        L.append("")
        for f in c["failures"]:
            L.append(f"- `{f['uid']}` ({f['kind']}, {f['rule']}): `{f['text']}` "
                     f"→ mismatched {f['mismatch_fields']}")
        L.append("")

    L.append("## 4. Proposed fix (source patch — NOT applied)")
    L.append("")
    L.append("```diff")
    L.append(fx["patch"].rstrip("\n"))
    L.append("```")
    L.append("")
    L.append("### Runtime-monkeypatched proof (source tree untouched)")
    L.append("")
    mb = fx["monkeypatched_baseline"]
    L.append("| Invariant | Result |")
    L.append("|---|---|")
    for k, v in mb["invariants"].items():
        L.append(f"| `{k}` | {'PASS' if v else 'FAIL'} |")
    L.append("")
    L.append(f"Adversarial corpus violations after the fix: "
             f"**{fx['monkeypatched_corpus']['n_violations']}/"
             f"{fx['monkeypatched_corpus']['n_cases']}**; grammar-aware fuzz after the fix: "
             f"validator violations {fx['monkeypatched_grammar_fuzz']['validator_soundness_violations']}, "
             f"semantic violations {fx['monkeypatched_grammar_fuzz']['semantic_soundness_violations']}, "
             f"false materializations {fx['monkeypatched_grammar_fuzz']['false_materializations']}, "
             f"crashes {fx['monkeypatched_grammar_fuzz']['crashes']}.")
    mp = fx.get("monkeypatched_paraphrase")
    if mp:
        L.append("")
        L.append(f"Paraphrase robustness after the fix: clean "
                 f"{mp['clean_rule_correct']}/{mp['clean_rows']}, perturbed "
                 f"{mp['perturbed_rule_correct']}/{mp['n_perturbed']} "
                 f"({mp['perturbed_rule_correctness']:.3f}).")
    L.append("")
    L.append("| Reproducer | Resolved after fix | New clarification kind |")
    L.append("|---|---|---|")
    for rep in fx["monkeypatched_reproducers"]:
        L.append(f"| `{rep['reproducer']}` | {'yes' if rep['resolved'] else 'NO'} | "
                 f"{rep.get('now_clarification_kind')} |")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
