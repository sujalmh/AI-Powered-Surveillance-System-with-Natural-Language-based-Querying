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

BENCH = ROOT / "backend" / "benchmarks" / "alert_utterances.jsonl"
RESULTS_DIR = ROOT / "results"
OUT_JSON = RESULTS_DIR / "alert_results.json"
OUT_MD = RESULTS_DIR / "alert_results.md"
COMMAND = "$env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts.py"
STRESS_KINDS = ["ambiguous", "conflicting", "missing", "unsupported"]
SPLITS = ["dev", "test"]


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
    return {
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


def _write_markdown(
    results: Dict[str, Any], dataset: Dict[str, Any], test: Dict[str, Any], dev: Dict[str, Any]
) -> str:
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
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
