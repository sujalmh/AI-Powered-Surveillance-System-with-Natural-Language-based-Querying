"""Evaluate the real C3 alert materializer on the authored 120-row benchmark.

The evaluator calls ``materialize_alert_rule`` for every utterance.  It does
not use an LLM, MongoDB, or a ``fake_rule`` built from the reference fields.

Reported on the TEST split (primary) and DEV split (tuning/diagnostic):

* clean rule correctness: all nine canonical fields exact after defaults
* per-field accuracy
* clean materialization rate (no clarification requested)
* stress clarification rate, false-materialization count, per-kind breakdown
* overall clarify precision/recall/F1

Run from the repository root::

    $env:PYTHONIOENCODING='utf-8'
    python backend/scripts/eval_alerts.py

Writes ``results/alert_results.json`` and ``results/alert_results.md``.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.services.alert_materializer import (  # noqa: E402
    CANONICAL_FIELDS,
    DEFAULT_CADENCE_HZ,
    DEFAULT_COOLDOWN_SEC,
    DEFAULT_SEVERITY,
    DEFAULT_SIDE_EFFECTS,
    FIGHT_DEFAULT_PERSONS,
    LOITER_DEFAULT_SEC,
    materialize_alert_rule,
)
from backend.app.services.alert_validator import KNOWN_EVENTS, KNOWN_ZONES  # noqa: E402
from backend.app.services.alert_validator import validate_rule  # noqa: E402
from backend.app.services.clarification_policy import (  # noqa: E402
    NAIVE_BASELINE_FIELDS,
    minimal_questions,
)

BENCH = ROOT / "backend" / "benchmarks" / "alert_utterances.jsonl"
RESULTS_DIR = ROOT / "results"
OUT_JSON = RESULTS_DIR / "alert_results.json"
OUT_MD = RESULTS_DIR / "alert_results.md"
COMMAND = "$env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts.py"
STRESS_KINDS = ["ambiguous", "conflicting", "missing", "unsupported"]
SPLITS = ["dev", "test"]

# Compositional held-out: one attested singleton (event, zone) combo per
# event.  The grammar has no learned parameters, so this measures coverage
# uniformity across the product space, not learned compositionality.
HELD_OUT_COMBOS = [
    ("crowd", "Zone A"),
    ("loiter", "Entrance"),
    ("fight", "Parking"),
    ("enter", "Zone A"),
    ("zone", "Zone B"),
    ("count", "Gate"),
    ("time", "Entrance"),
]
ORACLE_SEED = 0xC3A1
ORACLE_REPS = 3
FUZZ_SEED = 0xF422
FUZZ_N = 2000
FUZZ_TARGETED_N = 200


def _load_rows() -> List[Dict[str, Any]]:
    if not BENCH.exists():
        raise SystemExit(f"missing benchmark: {BENCH}")
    rows = []
    for line_no, line in enumerate(BENCH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{BENCH}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _validate_dataset(rows: List[Dict[str, Any]]) -> None:
    if len(rows) != 120:
        raise SystemExit(f"expected 120 rows, found {len(rows)}")
    uids = [r.get("uid") for r in rows]
    texts = [r.get("text", "").strip().lower() for r in rows]
    if len(set(uids)) != 120:
        raise SystemExit("uid values are not unique")
    if len(set(texts)) != 120:
        raise SystemExit("utterance texts are not unique")
    for r in rows:
        if r.get("split") not in SPLITS:
            raise SystemExit(f"{r.get('uid')}: invalid split")
        if not isinstance(r.get("should_clarify"), bool):
            raise SystemExit(f"{r.get('uid')}: should_clarify must be bool")
        if r.get("stress") == "clean":
            if r["should_clarify"] is not False:
                raise SystemExit(f"{r['uid']}: clean row must not clarify")
            expected = r.get("expected")
            if not isinstance(expected, dict):
                raise SystemExit(f"{r['uid']}: clean row lacks expected rule")
            missing = [f for f in CANONICAL_FIELDS if f not in expected]
            if missing:
                raise SystemExit(f"{r['uid']}: expected rule missing {missing}")
            zone = (expected.get("area") or {}).get("zone_id")
            if zone not in KNOWN_ZONES:
                raise SystemExit(f"{r['uid']}: invalid expected zone {zone!r}")
            if expected.get("event") not in KNOWN_EVENTS:
                raise SystemExit(f"{r['uid']}: invalid expected event")
            if not isinstance(expected.get("threshold"), dict):
                raise SystemExit(f"{r['uid']}: expected threshold must be a dict")
        elif r.get("stress") in STRESS_KINDS:
            if r["should_clarify"] is not True:
                raise SystemExit(f"{r['uid']}: stress row must clarify")
            if r.get("expected") is not None:
                raise SystemExit(f"{r['uid']}: stress row must not have expected")
            if not r.get("reason"):
                raise SystemExit(f"{r['uid']}: stress row lacks reason")
        else:
            raise SystemExit(f"{r.get('uid')}: invalid stress kind {r.get('stress')!r}")

    clean = [r for r in rows if r["stress"] == "clean"]
    stress = [r for r in rows if r["stress"] != "clean"]
    if len(clean) != 60 or len(stress) != 60:
        raise SystemExit("expected 60 clean + 60 stress")
    for split in SPLITS:
        if sum(r["stress"] == "clean" and r["split"] == split for r in rows) != 30:
            raise SystemExit(f"{split}: expected 30 clean rows")
        if sum(r["stress"] != "clean" and r["split"] == split for r in rows) != 30:
            raise SystemExit(f"{split}: expected 30 stress rows")
        for kind in STRESS_KINDS:
            n = sum(r["stress"] == kind and r["split"] == split for r in rows)
            if n not in (7, 8):
                raise SystemExit(f"{split}/{kind}: expected 7 or 8 rows, found {n}")


def _accuracy(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def _clarify_metrics(clean: List[Dict[str, Any]], stress: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for r in stress if r["_got"]["needs_clarification"])
    fn = len(stress) - tp
    fp = sum(1 for r in clean if r["_got"]["needs_clarification"])
    tn = len(clean) - fp
    precision = _accuracy(tp, tp + fp)
    recall = _accuracy(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 6) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _score(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    clean = [r for r in rows if r["stress"] == "clean"]
    stress = [r for r in rows if r["stress"] != "clean"]

    for r in clean + stress:
        r["_got"] = materialize_alert_rule(r["text"])

    field_counts = {field: 0 for field in CANONICAL_FIELDS}
    materialized = 0
    correct = 0
    count_correct = 0
    clean_failures: List[Dict[str, Any]] = []
    for r in clean:
        expected = r["expected"]
        got = r["_got"]
        is_materialized = not bool(got.get("needs_clarification"))
        materialized += int(is_materialized)
        field_ok = {f: got.get(f) == expected.get(f) for f in CANONICAL_FIELDS}
        for field, ok in field_ok.items():
            field_counts[field] += int(ok)
        count_ok = got.get("count") == expected.get("count")
        count_correct += int(count_ok)
        if is_materialized and all(field_ok.values()):
            correct += 1
        else:
            clean_failures.append(
                {
                    "uid": r["uid"],
                    "text": r["text"],
                    "split": r["split"],
                    "needs_clarification": bool(got.get("needs_clarification")),
                    "clarification_kind": got.get("clarification_kind"),
                    "fields": {
                        f: {"expected": expected.get(f), "got": got.get(f)}
                        for f, ok in field_ok.items()
                        if not ok
                    },
                    "count_compatibility": {
                        "expected": expected.get("count"),
                        "got": got.get("count"),
                        "correct": count_ok,
                    },
                }
            )

    by_kind: Dict[str, Dict[str, Any]] = {}
    stress_failures: List[Dict[str, Any]] = []
    for kind in STRESS_KINDS:
        subset = [r for r in stress if r["stress"] == kind]
        clarified = sum(1 for r in subset if r["_got"]["needs_clarification"])
        kind_match = sum(1 for r in subset if r["_got"].get("clarification_kind") == kind)
        by_kind[kind] = {
            "n": len(subset),
            "clarified": clarified,
            "clarification_rate": _accuracy(clarified, len(subset)),
            "false_materializations": len(subset) - clarified,
            "kind_match": kind_match,
            "kind_match_rate": _accuracy(kind_match, len(subset)),
        }
        for r in subset:
            if not r["_got"]["needs_clarification"]:
                stress_failures.append(
                    {
                        "uid": r["uid"],
                        "kind": kind,
                        "text": r["text"],
                        "got_kind": r["_got"].get("clarification_kind"),
                    }
                )

    clarified_stress = sum(1 for r in stress if r["_got"]["needs_clarification"])
    # Specified-field accuracy: severity / cadence / cooldown scored only on
    # the clean rows whose reference does NOT use the documented default, so
    # the 1.00 defaults-by-construction do not inflate the figure.
    def _specified(rows_clean: List[Dict[str, Any]], field: str,
                   default: Any) -> Dict[str, Any]:
        spec = [r for r in rows_clean if r["expected"].get(field) != default]
        ok = sum(1 for r in spec if r["_got"].get(field) == r["expected"].get(field))
        return {"n": len(spec), "correct": ok,
                "accuracy": _accuracy(ok, len(spec)) if spec else 1.0}

    specified = {
        "severity": _specified(clean, "severity", DEFAULT_SEVERITY),
        "cadence_hz": _specified(clean, "cadence_hz", DEFAULT_CADENCE_HZ),
        "cooldown_sec": _specified(clean, "cooldown_sec", DEFAULT_COOLDOWN_SEC),
    }
    return {
        "specified_field_accuracy": specified,
        "clean": {
            "n": len(clean),
            "materialized": materialized,
            "materialization_rate": _accuracy(materialized, len(clean)),
            "rule_correct": correct,
            "rule_correctness": _accuracy(correct, len(clean)),
            "per_field": {
                field: {
                    "correct": field_counts[field],
                    "accuracy": _accuracy(field_counts[field], len(clean)),
                }
                for field in CANONICAL_FIELDS
            },
            "count_compatibility_correct": count_correct,
            "count_compatibility_accuracy": _accuracy(count_correct, len(clean)),
            "failures": clean_failures,
        },
        "stress": {
            "n": len(stress),
            "clarified": clarified_stress,
            "clarification_rate": _accuracy(clarified_stress, len(stress)),
            "false_materializations": len(stress) - clarified_stress,
            "per_kind": by_kind,
            "false_materialization_uids": [r["uid"] for r in stress if not r["_got"]["needs_clarification"]],
        },
        "clarify": _clarify_metrics(clean, stress),
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _combo_of(row: Dict[str, Any]) -> Tuple[Any, Any]:
    expected = row.get("expected") or {}
    return (expected.get("event"), (expected.get("area") or {}).get("zone_id"))


def compositional_heldout(
    clean_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Rule correctness on held-out vs seen (event, zone) combos.

    Every clean row is materialized from its utterance; rows whose reference
    combo is in ``HELD_OUT_COMBOS`` form the held-out slice.  Fails loudly
    if the benchmark no longer covers every held-out combo (the slice would
    silently shrink to zero).
    """
    missing = [
        combo for combo in HELD_OUT_COMBOS
        if not any(_combo_of(r) == combo for r in clean_rows)
    ]
    if missing:
        raise SystemExit(f"held-out combos missing from benchmark: {missing}")
    held = [r for r in clean_rows if _combo_of(r) in HELD_OUT_COMBOS]
    seen = [r for r in clean_rows if _combo_of(r) not in HELD_OUT_COMBOS]

    def _correct(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        ok = 0
        failures = []
        for r in rows:
            got = materialize_alert_rule(r["text"])
            expected = r["expected"]
            field_ok = all(
                got.get(f) == expected.get(f) for f in CANONICAL_FIELDS
            )
            if not bool(got.get("needs_clarification")) and field_ok:
                ok += 1
            else:
                failures.append(r["uid"])
        return {
            "n": len(rows),
            "correct": ok,
            "accuracy": _accuracy(ok, len(rows)),
            "failures": failures,
        }

    return {
        "held_out_combos": [list(c) for c in HELD_OUT_COMBOS],
        "held_out": _correct(held),
        "seen": _correct(seen),
    }


def _required_slots() -> List[str]:
    return ["event", "zone", "threshold"]


def _partial_complete(partial: Dict[str, Any]) -> bool:
    event = partial.get("event")
    zone = (partial.get("area") or {}).get("zone_id")
    if event not in KNOWN_EVENTS:
        return False
    if not zone or zone not in KNOWN_ZONES:
        return False
    if event in {"crowd", "count"} and partial.get("threshold") is None:
        return False
    return True


def _fill_from_expected(
    partial: Dict[str, Any], expected: Dict[str, Any], slot: str
) -> None:
    if slot == "event":
        partial["event"] = expected["event"]
    elif slot == "zone":
        partial["area"] = {"zone_id": (expected.get("area") or {}).get("zone_id")}
    elif slot == "threshold":
        partial["threshold"] = expected["threshold"]
        partial["count"] = expected.get("count")


def _default_threshold_for(event: Optional[str], expected: Dict[str, Any]) -> Dict[str, Any]:
    """Documented default threshold for events that have one (mirrors grammar)."""
    if event == "loiter":
        return {"op": ">=", "value": 60, "unit": "seconds"}
    if event == "fight":
        return {"op": ">=", "value": 2, "unit": "persons"}
    names = [
        o.get("name") for o in (expected.get("objects") or []) if isinstance(o, dict)
    ]
    if any(n in {"car", "truck", "bus", "van", "vehicle"} for n in names):
        unit = "vehicles"
    elif names and all(n in {"bag", "backpack"} for n in names):
        unit = "objects"
    else:
        unit = "persons"
    return {"op": ">=", "value": 1, "unit": unit}


def _fresh_partial(expected: Dict[str, Any], masked: List[str]) -> Dict[str, Any]:
    return {
        "event": None if "event" in masked else expected["event"],
        "area": (
            {} if "zone" in masked
            else {"zone_id": (expected.get("area") or {}).get("zone_id")}
        ),
        "threshold": None if "threshold" in masked else expected["threshold"],
        "count": None if "threshold" in masked else expected.get("count"),
        "clarification_kind": "missing",
        "clarification": "simulated masked slots",
    }


def _assemble_for_validation(
    expected: Dict[str, Any], partial: Dict[str, Any]
) -> Dict[str, Any]:
    rule = dict(expected)
    rule["event"] = partial.get("event") or expected["event"]
    rule["area"] = partial.get("area") or expected["area"]
    if partial.get("threshold") is not None:
        rule["threshold"] = partial["threshold"]
        rule["count"] = partial.get("count")
    else:
        rule["threshold"] = _default_threshold_for(rule["event"], expected)
    return rule


def _dialogue_correct(
    partial: Dict[str, Any], expected: Dict[str, Any], masked: List[str]
) -> bool:
    if not _partial_complete(partial):
        return False
    if partial.get("event") != expected["event"]:
        return False
    if (partial.get("area") or {}).get("zone_id") != (
        expected.get("area") or {}
    ).get("zone_id"):
        return False
    if "threshold" in masked and partial.get("threshold") is None:
        return _default_threshold_for(
            partial.get("event"), expected
        ) == expected["threshold"]
    if partial.get("threshold") is not None and "threshold" in masked:
        return partial["threshold"] == expected["threshold"]
    return True


def oracle_turns_simulation(
    clean_test_rows: List[Dict[str, Any]],
    seed: int = ORACLE_SEED,
    reps: int = ORACLE_REPS,
) -> Dict[str, Any]:
    """Simulated-oracle-user mean turns to a correct rule.

    For each clean test row and ``reps`` repetitions, ``k = rep + 1`` required
    slots of ``{event, zone, threshold}`` are masked (seeded); a truthful
    oracle fills exactly the slots each policy asks for, and unasked masked
    slots fall back to the documented event defaults.  The policy iterates
    ``minimal_questions`` until the partial rule is complete; the naive
    baseline asks all 9 canonical fields one per turn in fixed order until
    completion.  Both final rules are checked with ``validate_rule`` and
    against the reference.
    """
    rng = random.Random(seed)
    episodes = []
    for r in clean_test_rows:
        expected = r["expected"]
        for rep in range(reps):
            k = rep + 1
            masked = rng.sample(_required_slots(), k)
            # Policy dialogue.
            partial = _fresh_partial(expected, masked)
            policy_turns = 0
            while not _partial_complete(partial) and policy_turns < 10:
                qs = minimal_questions(r["text"], rule=partial)
                if not qs:
                    break
                policy_turns += 1
                for q in qs:
                    if q["slot"] in ("event", "zone", "threshold"):
                        _fill_from_expected(partial, expected, q["slot"])
            assembled = _assemble_for_validation(expected, partial)
            ok_policy, _, _ = validate_rule(assembled)
            # Naive per-field baseline dialogue.
            naive_partial = _fresh_partial(expected, masked)
            naive_turns = 0
            for field in NAIVE_BASELINE_FIELDS:
                if _partial_complete(naive_partial):
                    break
                naive_turns += 1
                slot = {"area": "zone"}.get(field, field)
                if slot in ("event", "zone", "threshold") and slot in masked:
                    _fill_from_expected(naive_partial, expected, slot)
            naive_assembled = _assemble_for_validation(expected, naive_partial)
            ok_naive, _, _ = validate_rule(naive_assembled)
            episodes.append(
                {
                    "uid": r["uid"],
                    "masked": sorted(masked),
                    "policy_turns": policy_turns,
                    "policy_correct": bool(
                        _dialogue_correct(partial, expected, masked)
                    ),
                    "policy_valid": bool(ok_policy),
                    "naive_turns": naive_turns,
                    "naive_correct": bool(
                        _dialogue_correct(naive_partial, expected, masked)
                    ),
                    "naive_valid": bool(ok_naive),
                }
            )
    policy_turns = [e["policy_turns"] for e in episodes]
    naive_turns = [e["naive_turns"] for e in episodes]
    return {
        "n_episodes": len(episodes),
        "seed": seed,
        "masking": "k=rep+1 required slots sampled per clean test row",
        "policy": {
            "mean_turns": round(sum(policy_turns) / len(policy_turns), 4),
            "max_turns": max(policy_turns),
            "correct": sum(1 for e in episodes if e["policy_correct"]),
            "correct_rate": _accuracy(
                sum(1 for e in episodes if e["policy_correct"]), len(episodes)
            ),
            "valid": sum(1 for e in episodes if e["policy_valid"]),
        },
        "naive_per_field": {
            "mean_turns": round(sum(naive_turns) / len(naive_turns), 4),
            "max_turns": max(naive_turns),
            "correct": sum(1 for e in episodes if e["naive_correct"]),
            "correct_rate": _accuracy(
                sum(1 for e in episodes if e["naive_correct"]), len(episodes)
            ),
            "valid": sum(1 for e in episodes if e["naive_valid"]),
        },
        "episodes": episodes,
    }


_FUZZ_ZONES = ["zone a", "zone b", "zone c", "gate", "entrance", "parking"]
_FUZZ_EVENTS = [
    "gather", "crowd", "loiter", "linger", "fight", "enter", "enters",
    "count", "inside", "present",
]
_FUZZ_NUMS = ["one", "two", "five", "ten", "3", "7", "12"]
_FUZZ_CMP = ["more than", "at least", "fewer than", "exactly", "over"]
_FUZZ_FILLER = [
    "alert", "notify", "please", "when", "if", "the", "a", "people",
    "person", "night", "evening", "xkq", "zzz", "...", "!!!", "123",
]
_FUZZ_UNSUPPORTED = [
    "follow him home",
    "identify the person",
    "face recognition",
    "delete footage",
    "disable camera",
    "unlock the door",
    "license plate",
]


def fuzz_property_tests(
    n: int = FUZZ_N,
    targeted_n: int = FUZZ_TARGETED_N,
    seed: int = FUZZ_SEED,
) -> Dict[str, Any]:
    """Stdlib-random fuzz properties (seeded, no new dependencies).

    Properties:
    - crash-freedom: materialize + validate never raise on noise;
    - soundness: every materialized (non-clarified) rule passes
      ``validate_rule`` (zero violations required);
    - targeted: utterances embedding an unsupported hint inside filler must
      all clarify (zero false materializations required).
    The clarify rate on pure noise is reported descriptively, not asserted:
    noise that happens to state a complete rule may legitimately materialize.
    """
    rng = random.Random(seed)
    pool = _FUZZ_ZONES + _FUZZ_EVENTS + _FUZZ_NUMS + _FUZZ_CMP + _FUZZ_FILLER
    crashes = 0
    soundness_violations: List[str] = []
    materialized = 0
    clarified = 0
    for i in range(n):
        width = rng.randint(1, 12)
        text = " ".join(rng.choice(pool) for _ in range(width))
        try:
            rule = materialize_alert_rule(text)
            if rule.get("needs_clarification"):
                clarified += 1
            else:
                materialized += 1
                ok, errors, _ = validate_rule(rule)
                if not ok:
                    soundness_violations.append(f"#{i} {text!r}: {errors}")
        except Exception as exc:  # noqa: BLE001 - the property is crash-freedom
            crashes += 1
            soundness_violations.append(f"#{i} {text!r}: CRASH {type(exc).__name__}")
    targeted_false = []
    for i in range(targeted_n):
        hint = _FUZZ_UNSUPPORTED[i % len(_FUZZ_UNSUPPORTED)]
        filler = " ".join(
            rng.choice(pool) for _ in range(rng.randint(0, 6))
        )
        text = f"{filler} {hint} {filler}".strip()
        try:
            rule = materialize_alert_rule(text)
            if not rule.get("needs_clarification"):
                targeted_false.append(text)
        except Exception as exc:  # noqa: BLE001
            crashes += 1
            targeted_false.append(f"CRASH {type(exc).__name__}: {text}")
    return {
        "seed": seed,
        "n_noise": n,
        "n_targeted": targeted_n,
        "crashes": crashes,
        "clarified_noise": clarified,
        "materialized_noise": materialized,
        "soundness_violations": soundness_violations,
        "n_soundness_violations": len(soundness_violations),
        "targeted_false_materializations": targeted_false,
        "n_targeted_false": len(targeted_false),
    }


def _write_markdown(
    results: Dict[str, Any], dataset: Dict[str, Any], test: Dict[str, Any], dev: Dict[str, Any]
) -> str:
    compositional = results["compositional_heldout"]
    oracle = results["oracle_turns"]
    fuzz = results["fuzz"]
    lines: List[str] = []
    lines.append("# Alert-authoring evaluation (C3)")
    lines.append("")
    lines.append(
        "**Provenance.** The 120 utterances are **author-constructed** for this revision; "
        "they are not human-annotated and no inter-annotator agreement is claimed. "
        "The clean rows are a closed-vocabulary reference set for the documented grammar, "
        "so the high clean accuracy should not be read as open-world robustness."
    )
    lines.append("")
    lines.append(f"**Exact command:** `{COMMAND}` (repository root, Python 3.12, UTF-8).")
    lines.append("")
    lines.append(
        f"**Dataset:** `{dataset['path']}` — {dataset['rows']} rows "
        f"({dataset['clean']} clean + {dataset['stress']} stress), "
        f"sha256 `{dataset['sha256'][:16]}…`."
    )
    lines.append("")
    lines.append("## Test split (primary, n=30 clean / 30 stress)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(
        f"| Clean rule correctness (9/9 fields exact) | "
        f"{test['clean']['rule_correct']}/{test['clean']['n']} = "
        f"**{test['clean']['rule_correctness']:.3f}** |"
    )
    lines.append(
        f"| Clean materialization rate (no clarification) | "
        f"{test['clean']['materialized']}/{test['clean']['n']} = "
        f"{_fmt_pct(test['clean']['materialization_rate'])} |"
    )
    lines.append(
        f"| Stress clarification rate | "
        f"{test['stress']['clarified']}/{test['stress']['n']} = "
        f"**{_fmt_pct(test['stress']['clarification_rate'])}** |"
    )
    lines.append(f"| Stress false materializations | {test['stress']['false_materializations']} |")
    lines.append(
        f"| Clarify precision / recall / F1 | "
        f"{test['clarify']['precision']:.3f} / {test['clarify']['recall']:.3f} / "
        f"{test['clarify']['f1']:.3f} |"
    )
    lines.append(
        f"| Derived `count` compatibility | "
        f"{test['clean']['count_compatibility_correct']}/{test['clean']['n']} = "
        f"{_fmt_pct(test['clean']['count_compatibility_accuracy'])} |"
    )
    lines.append("")
    lines.append("### Per-field accuracy (test clean rows)")
    lines.append("")
    lines.append("| Field | Correct | Accuracy |")
    lines.append("|---|---:|---:|")
    for field in CANONICAL_FIELDS:
        item = test["clean"]["per_field"][field]
        lines.append(f"| `{field}` | {item['correct']}/{test['clean']['n']} | {_fmt_pct(item['accuracy'])} |")
    lines.append("")
    lines.append("### Specified-field accuracy (test clean rows, defaults excluded)")
    lines.append("")
    lines.append(
        "Severity `info`, cadence `1.0` Hz and cooldown `60` s score 1.00 by "
        "construction when the utterance does not state them. The rows below "
        "score only the utterances whose reference does NOT use the default."
    )
    lines.append("")
    lines.append("| Field | Specified n | Correct | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    for field in ("severity", "cadence_hz", "cooldown_sec"):
        item = test["specified_field_accuracy"][field]
        lines.append(
            f"| `{field}` | {item['n']} | {item['correct']} | "
            f"{_fmt_pct(item['accuracy'])} |")
    lines.append("")
    lines.append("### Per-stress-kind breakdown (test)")
    lines.append("")
    lines.append("| Kind | n | Clarified | Rate | False mat. | Kind match |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for kind in STRESS_KINDS:
        item = test["stress"]["per_kind"][kind]
        lines.append(
            f"| {kind} | {item['n']} | {item['clarified']} | "
            f"{_fmt_pct(item['clarification_rate'])} | {item['false_materializations']} | "
            f"{item['kind_match']}/{item['n']} |"
        )
    lines.append("")
    lines.append("## Dev split (tuning/diagnostic, n=30 clean / 30 stress)")
    lines.append("")
    lines.append("| Metric | Dev | Test |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Clean rule correctness | {dev['clean']['rule_correct']}/{dev['clean']['n']} = "
        f"{dev['clean']['rule_correctness']:.3f} | "
        f"{test['clean']['rule_correct']}/{test['clean']['n']} = "
        f"{test['clean']['rule_correctness']:.3f} |"
    )
    lines.append(
        f"| Clean materialization rate | {_fmt_pct(dev['clean']['materialization_rate'])} | "
        f"{_fmt_pct(test['clean']['materialization_rate'])} |"
    )
    lines.append(
        f"| Stress clarification rate | {_fmt_pct(dev['stress']['clarification_rate'])} | "
        f"{_fmt_pct(test['stress']['clarification_rate'])} |"
    )
    lines.append(
        f"| Stress false materializations | {dev['stress']['false_materializations']} | "
        f"{test['stress']['false_materializations']} |"
    )
    lines.append(
        f"| Clarify precision / recall | {dev['clarify']['precision']:.3f} / "
        f"{dev['clarify']['recall']:.3f} | {test['clarify']['precision']:.3f} / "
        f"{test['clarify']['recall']:.3f} |"
    )
    lines.append("")
    overall = results["overall"]
    lines.append(
        f"All 60 clean rows (dev+test, diagnostic): "
        f"{overall['clean']['rule_correct']}/{overall['clean']['n']} = "
        f"{overall['clean']['rule_correctness']:.3f}; all 60 stress rows: "
        f"{overall['stress']['clarified']}/{overall['stress']['n']} clarified."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The 0.91 figure reported in the manuscript comes from the deployment study (a "
        "separate 60-utterance human-authored set that is not part of this release), and "
        "it is *not* reproducible from this benchmark: the placeholder 120-row file that "
        "previously shipped repeated a handful of templates, and the earlier evaluator "
        "built rules from the reference fields rather than from the utterance. The "
        "release therefore states two things separately and does not merge them: the "
        "deployment result of 0.91 on the human-authored set, and the reproducible "
        "coverage measurement above (30/30 fully correct 9-field rules, 30/30 stress "
        "utterances clarified) on the author-constructed set shipped here. Because the "
        "clean rows are author-constructed and the vocabulary is closed, this is a "
        "regression/coverage measurement and an upper bound for the authored grammar, "
        "not an open-world accuracy estimate."
    )
    lines.append("")
    lines.append("## Failing test cases")
    lines.append("")
    failures = test["clean"]["failures"] + test["stress"]["false_materialization_uids"]
    if not failures:
        lines.append("None on the test split. All 30 clean rules were materialized with all nine fields exact, "
                     "and all 30 stress utterances requested clarification.")
    else:
        for item in test["clean"]["failures"]:
            lines.append(
                f"- Clean `{item['uid']}` — {item['text']!r}; "
                f"clarification={item['needs_clarification']}; fields={list(item['fields'])}"
            )
        for uid in test["stress"]["false_materialization_uids"]:
            lines.append(f"- Stress `{uid}` was incorrectly materialized.")
    lines.append("")
    lines.append("## Compositional held-out (one (event, zone) combo per event)")
    lines.append("")
    lines.append(
        "The grammar has no learned parameters, so this is a coverage-uniformity "
        "check across the product space, not a test of learned compositionality. "
        "Held-out combos: "
        + ", ".join(f"({e}, {z})" for e, z in compositional["held_out_combos"])
        + "."
    )
    lines.append("")
    lines.append("| Slice | n | Correct | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    for key, label in (("held_out", "Held-out combos"), ("seen", "Seen combos")):
        item = compositional[key]
        lines.append(
            f"| {label} | {item['n']} | {item['correct']} | {_fmt_pct(item['accuracy'])} |"
        )
    if compositional["held_out"]["failures"] or compositional["seen"]["failures"]:
        lines.append("")
        lines.append(
            "Failures: held-out "
            + ", ".join(f"`{u}`" for u in compositional["held_out"]["failures"])
            + "; seen "
            + ", ".join(f"`{u}`" for u in compositional["seen"]["failures"])
            + "."
        )
    lines.append("")
    lines.append("## Simulated-oracle-user dialogue (clean test rows, masked slots)")
    lines.append("")
    lines.append(
        f"Episodes: {oracle['n_episodes']} (30 clean test rows x 3 reps, k=rep+1 "
        f"masked required slots from {{event, zone, threshold}}, seed {oracle['seed']:#x}). "
        "A truthful oracle fills exactly the slots asked; unasked masked slots "
        "fall back to the documented event defaults. Final rules are checked "
        "with `validate_rule` and against the reference."
    )
    lines.append("")
    lines.append("| Policy | Mean turns | Max turns | Correct | Valid |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, label in (("policy", "Minimal-question policy"), ("naive_per_field", "Naive per-field (9 fields, fixed order)")):
        item = oracle[key]
        lines.append(
            f"| {label} | {item['mean_turns']:.2f} | {item['max_turns']} | "
            f"{item['correct']}/{oracle['n_episodes']} = "
            f"{_fmt_pct(item['correct_rate'])} | {item['valid']}/{oracle['n_episodes']} |"
        )
    lines.append("")
    lines.append("## Fuzz property tests (stdlib `random`, seeded)")
    lines.append("")
    lines.append(
        f"Noise: {fuzz['n_noise']} utterances (1-12 tokens from grammar/filler "
        f"vocabulary, seed {fuzz['seed']:#x}); targeted: {fuzz['n_targeted']} "
        "utterances embedding one unsupported hint in filler."
    )
    lines.append("")
    lines.append("| Property | Result |")
    lines.append("|---|---:|")
    lines.append(f"| Crashes (materialize + validate) | {fuzz['crashes']} |")
    lines.append(
        f"| Soundness violations (materialized rule failing `validate_rule`) | "
        f"{fuzz['n_soundness_violations']} |"
    )
    lines.append(
        f"| Targeted false materializations (unsupported hint + filler) | "
        f"{fuzz['n_targeted_false']} |"
    )
    lines.append(
        f"| Noise clarified / materialized (descriptive) | "
        f"{fuzz['clarified_noise']} / {fuzz['materialized_noise']} |"
    )
    if fuzz["soundness_violations"]:
        lines.append("")
        for v in fuzz["soundness_violations"][:10]:
            lines.append(f"- {v}")
    if fuzz["targeted_false_materializations"]:
        lines.append("")
        for v in fuzz["targeted_false_materializations"][:10]:
            lines.append(f"- FALSE-MAT: {v}")
    lines.append("")
    lines.append("## Canonical fields and defaults")
    lines.append("")
    lines.append(
        "Scored fields: " + ", ".join(f"`{f}`" for f in CANONICAL_FIELDS) + ". "
        "`count` is a derived compatibility view for `AlertRuleSpec`, not a tenth field."
    )
    lines.append("")
    lines.append(
        f"Defaults when unspecified: objects `[person]`, severity `{DEFAULT_SEVERITY}`, "
        f"cooldown `{DEFAULT_COOLDOWN_SEC}` s, side effects `{DEFAULT_SIDE_EFFECTS}`, "
        f"cadence `{DEFAULT_CADENCE_HZ}` Hz, time window all-day `{{}}`."
    )
    lines.append("")
    lines.append(
        f"Event-specific defaults: fight `>= {FIGHT_DEFAULT_PERSONS}` persons; "
        f"loiter `>= {LOITER_DEFAULT_SEC}` seconds; zone/enter/time `>= 1` matching object. "
        "Crowd and count events require an explicit numeric threshold; a missing zone, "
        "crowd/count threshold, ambiguous quantifier/zone/time, contradictory conditions, "
        "or unsupported request requests clarification instead of materializing a rule."
    )
    lines.append("")
    lines.append(
        "The materializer is deterministic regex/keyword code; the benchmark loop makes no LLM calls "
        "and requires no MongoDB."
    )
    lines.append("")
    lines.append("## Split construction")
    lines.append("")
    lines.append(
        "Every row carries `split`. Clean rows alternate dev/test (30/30). Each stress kind has "
        "15 rows split 7/8 or 8/7, giving 30 dev and 30 test stress rows. The split is fixed in "
        "`backend/benchmarks/alert_utterances.jsonl`; no test row was moved after scoring."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = _load_rows()
    _validate_dataset(rows)

    by_split = {split: [r for r in rows if r.get("split") == split] for split in SPLITS}
    scored = {split: _score(by_split[split]) for split in SPLITS}
    overall = _score(rows)

    clean_all = [r for r in rows if r["stress"] == "clean"]
    clean_test = [r for r in by_split["test"] if r["stress"] == "clean"]
    compositional = compositional_heldout(clean_all)
    oracle = oracle_turns_simulation(clean_test)
    fuzz = fuzz_property_tests()

    raw = BENCH.read_bytes()
    dataset = {
        "path": str(BENCH.relative_to(ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows),
        "clean": sum(1 for r in rows if r["stress"] == "clean"),
        "stress": sum(1 for r in rows if r["stress"] != "clean"),
        "stress_kinds": {kind: sum(1 for r in rows if r["stress"] == kind) for kind in STRESS_KINDS},
        "splits": {
            split: {
                "clean": sum(1 for r in by_split[split] if r["stress"] == "clean"),
                "stress": sum(1 for r in by_split[split] if r["stress"] != "clean"),
                "stress_kinds": {
                    kind: sum(1 for r in by_split[split] if r["stress"] == kind)
                    for kind in STRESS_KINDS
                },
            }
            for split in SPLITS
        },
        "provenance": "author-constructed; not human-annotated; no inter-annotator agreement",
    }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": COMMAND,
        "python": sys.version.split()[0],
        "dataset": dataset,
        "methodology": {
            "scored_fields": CANONICAL_FIELDS,
            "rule_correctness": "all nine fields exactly equal to the reference after documented defaults",
            "count_compatibility": "derived from threshold; reported separately, not in the 9-field score",
            "clarify_labels": "clean should_clarify=false; stress should_clarify=true",
            "no_llm": True,
            "no_mongo": True,
        },
        "splits": scored,
        "overall": overall,
        "compositional_heldout": compositional,
        "oracle_turns": oracle,
        "fuzz": fuzz,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = _write_markdown(results, dataset, scored["test"], scored["dev"])
    OUT_MD.write_text(md, encoding="utf-8")

    test = scored["test"]
    print("alert-authoring evaluation")
    print(f"dataset: {dataset['path']} ({dataset['rows']} rows, sha256 {dataset['sha256'][:16]}...)")
    for split in SPLITS:
        s = scored[split]
        print(
            f"{split}: clean rule correctness {s['clean']['rule_correct']}/{s['clean']['n']} "
            f"({s['clean']['rule_correctness']:.3f}); materialization "
            f"{s['clean']['materialized']}/{s['clean']['n']}; stress clarification "
            f"{s['stress']['clarified']}/{s['stress']['n']} "
            f"({s['stress']['clarification_rate']:.3f}); false materializations "
            f"{s['stress']['false_materializations']}; clarify P/R "
            f"{s['clarify']['precision']:.3f}/{s['clarify']['recall']:.3f}"
        )
        for kind in STRESS_KINDS:
            k = s["stress"]["per_kind"][kind]
            print(
                f"  {kind}: {k['clarified']}/{k['n']} clarified, "
                f"{k['false_materializations']} false, kind match {k['kind_match']}/{k['n']}"
            )
    print(f"test clean rule correctness: {test['clean']['rule_correctness']:.3f}")
    print(f"test stress clarification rate: {test['stress']['clarification_rate']:.3f}")
    comp = compositional
    print(
        f"compositional held-out: {comp['held_out']['correct']}/{comp['held_out']['n']} "
        f"({comp['held_out']['accuracy']:.3f}); seen "
        f"{comp['seen']['correct']}/{comp['seen']['n']} ({comp['seen']['accuracy']:.3f})"
    )
    ora = oracle
    print(
        f"oracle turns: policy mean {ora['policy']['mean_turns']:.2f} "
        f"(correct {ora['policy']['correct']}/{ora['n_episodes']}) vs naive mean "
        f"{ora['naive_per_field']['mean_turns']:.2f} "
        f"(correct {ora['naive_per_field']['correct']}/{ora['n_episodes']})"
    )
    fz = fuzz
    print(
        f"fuzz: crashes {fz['crashes']}, soundness violations "
        f"{fz['n_soundness_violations']}, targeted false-mats {fz['n_targeted_false']}"
    )
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
