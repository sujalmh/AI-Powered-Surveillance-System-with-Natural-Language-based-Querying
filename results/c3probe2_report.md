# C3 probe #2 — attack report (alert materializer / clarification policy)

**Command:** `$env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_probe2.py` (repository root, Python 3.12.4, UTF-8).

**Read-only probe.** No tracked file, no benchmark row and no pre-existing `results/` artifact was modified; the proposed fix is pasted below and proved with runtime monkeypatches only.

**Dataset:** `backend/benchmarks/alert_utterances.jsonl` — 120 rows, sha256 `c885f8b96025132f…` (unchanged).

## 0. Baseline replication (shipped numbers reproduced)

| Invariant | Result |
|---|---|
| `test_clean_rule_correct_30_of_30` | PASS |
| `test_stress_clarified_30_of_30` | PASS |
| `test_false_materializations_zero` | PASS |
| `held_out_9_of_9` | PASS |
| `seen_51_of_51` | PASS |
| `fuzz_crashes_zero` | PASS |
| `fuzz_soundness_violations_zero` | PASS |
| `fuzz_targeted_false_materializations_zero` | PASS |
| `oracle_policy_mean_1p0778` | PASS |
| `oracle_policy_max_turns_2` | PASS |
| `oracle_naive_mean_2p5778` | PASS |
| `oracle_naive_max_turns_4` | PASS |
| `oracle_both_correct_75_of_90` | PASS |
| `oracle_both_valid_90_of_90` | PASS |

## 1. Attack (a) — bug hunt (expanded grammar-aware fuzz)

Adversarial corpus: **20/38** cases violate an expectation. Randomized grammar-aware fuzz: N=20000 (seed 0xc3f022).

| Property | Value |
|---|---:|
| Crashes (materialize + validate) | 0 |
| Validator soundness violations (materialized ∧ ¬`validate_rule`) | 6341 |
| Semantic soundness violations (rule admits excluded counts) | 2517 |
| False materializations of incoherent-by-construction inputs | 172 |

### Corpus violations by family

| Family | Violations |
|---|---:|
| `between_inversion` | 3 |
| `cadence_out_of_range` | 3 |
| `contradiction` | 0 |
| `degenerate_window` | 4 |
| `stacked_lower_bounds` | 1 |
| `stacked_upper_bounds` | 1 |
| `two_sided_bounds` | 6 |
| `unicode` | 0 |
| `unit_confusion` | 0 |
| `zone_alias_object_leak` | 2 |

### Minimal seeded reproducers

- **semantic_soundness** (`two_sided_bounds`): `more than five fewer than ten gather Zone A`
  - threshold {'op': '>', 'value': 5, 'unit': 'persons'} admits counts outside [6, 9]
- **semantic_soundness** (`two_sided_bounds`): `at least five at most ten gather Zone A`
  - threshold {'op': '>=', 'value': 5, 'unit': 'persons'} admits counts outside [5, 10]
- **semantic_soundness** (`two_sided_bounds`): `more than five fewer than ten gather Zone A`
  - threshold {'op': '>', 'value': 5, 'unit': 'persons'} admits counts outside [6, 9]
- **semantic_soundness** (`two_sided_bounds`): `number of Zone A more than five less than ten`
  - threshold {'op': '>', 'value': 5, 'unit': 'persons'} admits counts outside [6, 9]
- **semantic_soundness** (`two_sided_bounds`): `more than two fewer than four are in Parking`
  - threshold {'op': '>', 'value': 2, 'unit': 'persons'} admits counts outside [3, 3]
- **semantic_soundness** (`two_sided_bounds`): `loiters Zone A for more than five minutes less than ten minutes`
  - threshold {'op': '>', 'value': 300, 'unit': 'seconds'} admits counts outside [301, 599]
- **semantic_soundness** (`stacked_lower_bounds`): `more than five more than three people gather Zone A`
  - threshold {'op': '>', 'value': 3, 'unit': 'persons'} admits counts outside [6, None]
- **semantic_soundness** (`stacked_upper_bounds`): `fewer than ten fewer than eight gather Zone A`
  - threshold {'op': '<', 'value': 10, 'unit': 'persons'} admits counts outside [None, 7]
- **validator_soundness** (`between_inversion`): `between ten and three people gather Zone A`
  - between threshold lower bound exceeds upper bound: [10, 3]
- **validator_soundness** (`between_inversion`): `number of Zone A between ten and three`
  - between threshold lower bound exceeds upper bound: [10, 3]
- **validator_soundness** (`between_inversion`): `count Parking between nine and two`
  - between threshold lower bound exceeds upper bound: [9, 2]
- **validator_soundness** (`degenerate_window`): `Zone A between 3 and 3`
  - time_of_day window is degenerate (start == end)
- **validator_soundness** (`degenerate_window`): `Zone A 9 PM to 9 PM`
  - time_of_day window is degenerate (start == end)
- **validator_soundness** (`degenerate_window`): `Zone A before 12 AM`
  - time_of_day window is degenerate (start == end)
- **validator_soundness** (`degenerate_window`): `Zone A between 12 and 12`
  - time_of_day window is degenerate (start == end)
- **validator_soundness** (`cadence_out_of_range`): `every 0.05 seconds more than five gather Zone A`
  - cadence_hz above the 10 Hz evaluation ceiling
- **validator_soundness** (`cadence_out_of_range`): `20 hz more than five gather Zone A`
  - cadence_hz above the 10 Hz evaluation ceiling
- **validator_soundness** (`cadence_out_of_range`): `99 hz more than five gather Zone A`
  - cadence_hz above the 10 Hz evaluation ceiling
- **object_leak** (`zone_alias_object_leak`): `during the day car park`
  - objects ['car'] (expected ['person']); the zone alias leaks an object
- **object_leak** (`zone_alias_object_leak`): `more than five gather car park`
  - objects ['car'] (expected ['person']); the zone alias leaks an object

### Residual findings NOT fixed by the proposed patch

- `two_zone_conjunction`: `alert me when more than five people gather in Zone A and more than three people gather in Zone B` — conjunction of two (zone, threshold) pairs; the grammar holds one zone, so the second clause is silently dropped (materialized: True, threshold: {'op': '>', 'value': 5, 'unit': 'persons'})
- `dropped_time_constraint`: `alert me when more than five people gather in Zone A until 3 AM` — 'until' is not in the grammar, so the time constraint is silently dropped (materialized: True, threshold: {'op': '>', 'value': 5, 'unit': 'persons'})
- `dropped_time_constraint`: `alert me when more than five people gather in Zone A between 10:70 and 11 PM` — invalid clock minute -> the whole time window is silently dropped (materialized: True, threshold: {'op': '>', 'value': 5, 'unit': 'persons'})

## 2. Attack (b) — clarification-policy frontier

**Turn counting (fixed for every variant):** One turn = one assistant message. A message may bundle several slot questions (this is the convention the shipped harness already uses: `minimal_questions` returns a LIST of questions and the simulated oracle answers every asked slot inside that one message). The dialogue ends when the partial rule is complete (`_partial_complete`) or the policy returns no question. `turns` = number of assistant messages; `slots_asked` = number of slot questions asked across the dialogue (reported separately because a bundled question costs one turn but one slot). The protocol is fixed across all variants and identical to the shipped harness (seed 0xC3A1, reps=3, k=rep+1 masked slots of {event, zone, threshold}).

**Selection rule (DEV only):** Pre-registered on the DEV split only: maximise rule-correctness; tie-break on the lower mean number of turns; final tie-break on the lower mean number of slot questions. The winner is frozen and evaluated ONCE on the TEST split.

### DEV split (selection)

| Variant | Correct | Mean turns | Max turns | Valid | Mean slot questions |
|---|---:|---:|---:|---:|---:|
| `v2_always_ask_threshold` | 90/90 | 1.00 | 1 | 90 | 2.00 |
| `v4_v2_plus_v3` | 90/90 | 1.00 | 1 | 90 | 2.00 |
| `v3_bundle_threshold_with_event` | 84/90 | 0.90 | 1 | 90 | 1.83 |
| `v0_baseline_minimal_questions` | 71/90 | 1.08 | 2 | 90 | 1.54 |
| `v1_never_ask_threshold` | 52/90 | 0.88 | 1 | 90 | 1.33 |

**Frozen variant: `v2_always_ask_threshold`**

### TEST split (single evaluation of the frozen variant)

| Policy | Mean turns | Max turns | Correct | Valid | Mean slot questions |
|---|---:|---:|---:|---:|---:|
| Baseline `minimal_questions` (shipped) | 1.08 | 2 | 75/90 | 90 | 1.54 |
| Frozen `v2_always_ask_threshold` | 1.00 | 1 | 90/90 | 90 | 2.00 |

Paired (n=90 episodes): baseline-correct-only 0, frozen-correct-only 15; exact McNemar p = 6.1e-05; row-clustered bootstrap 95% CI on Δcorrectness (frozen−baseline) [0.066667, 0.288889], on Δturns [-0.211111, 0.044444].

## 3. Attack (c) — paraphrase robustness (60 clean rows, no benchmark edit)

Clean rule-correctness: **60/60** (1.000).
Perturbed rule-correctness: **921/929** (0.991).

| Perturbation | n | Correct | Accuracy |
|---|---:|---:|---:|
| `case_punct` | 300 | 300 | 1.000 |
| `filler` | 240 | 240 | 1.000 |
| `synonym` | 360 | 352 | 0.978 |
| `word_order` | 29 | 29 | 1.000 |

Failures (first 40):

- `a_clean_04` (synonym, \bthe Parking lot\b -> the car park): `alert me when over twelve people congregate in the car park during the day` → mismatched ['objects']
- `a_clean_10` (synonym, \bthe Parking lot\b -> the car park): `alert me when five or more people gather in the car park every 30 seconds` → mismatched ['objects']
- `a_clean_22` (synonym, \bthe Parking lot\b -> the car park): `alert me if a person loiters in the car park for over two minutes, warning severity` → mismatched ['objects']
- `a_clean_28` (synonym, \bthe Parking lot\b -> the car park): `alert me when a fight occurs in the car park during the day, save a clip` → mismatched ['objects']
- `a_clean_34` (synonym, \bthe Parking lot\b -> the car park): `alert me if a truck enters the car park during the day` → mismatched ['objects']
- `a_clean_39` (synonym, \bthe Parking lot\b -> the car park): `notify me if a person enters the car park every 30 seconds` → mismatched ['objects', 'threshold']
- `a_clean_44` (synonym, \bthe Parking lot\b -> the car park): `alert me when the car park is occupied during the day` → mismatched ['objects', 'threshold']
- `a_clean_58` (synonym, \bthe Parking lot\b -> the car park): `notify me during the day in the car park` → mismatched ['objects', 'threshold']

## 4. Proposed fix (source patch — NOT applied)

```diff
--- a/backend/app/services/alert_materializer.py
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
     ("person", r"\b(?:persons?|people|someone|somebody|anyone|individuals?|intruders?|pedestrians?)\b"),
 ]
 
+# Zone aliases are masked before the object scan so that a zone phrase such
+# as "car park" does not contribute a spurious object ("car").
+_ZONE_SPAN_RE = re.compile(
+    r"\b(?:zone\s*[- ]?[abc]|gate|entrance|"
+    r"parking\s+lot|car\s+park|parking\s+area|parking)\b",
+    re.IGNORECASE,
+)
+
 _VEHICLE_OBJECTS = {"car", "truck", "bus", "van", "vehicle", "motorcycle"}
@@ -425,9 +434,10 @@ def _objects(text: str) -> List[str]:
     """Object mentions in the utterance, in order of appearance."""
     hits: List[Tuple[int, str]] = []
+    scan = _ZONE_SPAN_RE.sub(" ", text)
     for name, pattern in _OBJECT_PATTERNS:
-        for match in re.finditer(pattern, text, re.IGNORECASE):
+        for match in re.finditer(pattern, scan, re.IGNORECASE):
             hits.append((match.start(), name))
     hits.sort(key=lambda item: item[0])
@@ -188,6 +198,13 @@
     rf"backpacks?|bags?|objects?)\b",
     re.IGNORECASE,
 )
 
+# Duration bounds with their own unit, so that minutes and seconds are not
+# compared as raw numbers.
+_SECONDS_BOUND_RE = re.compile(
+    rf"\b(?P<cmp>{_COMPARATOR_RE})\s+(?P<num>{_NUM_RE})\s*(?P<unit>{_TIME_UNIT_RE})\b",
+    re.IGNORECASE,
+)
+
 _DURATION_RE = re.compile(
     rf"\bfor\s+(?:(?P<cmp>{_COMPARATOR_RE})\s+)?"
     rf"(?P<num>{_NUM_RE}|a|an)\s*(?P<unit>{_TIME_UNIT_RE})\b",
     re.IGNORECASE,
@@ -521,6 +529,44 @@ def _op_from_comparator(cmp_token: str) -> str:
     return ">="
 
 
+def _combine_bounds(bounds: List[Dict[str, Any]], unit: str) -> Optional[Dict[str, Any]]:
+    """Combine conjunctive bounds of one unit family into a single threshold.
+
+    A request may state several bounds of the same unit ("more than five but
+    fewer than ten people").  Keeping only one of them silently widens the
+    rule: the created alert then fires in configurations the user excluded.
+    This helper returns the conjunction of ``bounds`` as one threshold
+    (``between`` for a two-sided range, the strictest bound otherwise) and
+    ``None`` when the bounds cannot be combined.
+    """
+    lowers = [b for b in bounds if b["op"] in {">", ">="}]
+    uppers = [b for b in bounds if b["op"] in {"<", "<="}]
+    exact = [b for b in bounds if b["op"] == "=="]
+    if exact:
+        return {"op": "==", "value": exact[0]["value"], "unit": unit}
+    if lowers and uppers:
+        lo_b = max(lowers, key=lambda b: (b["value"], b["op"] == ">"))
+        hi_b = min(uppers, key=lambda b: (b["value"], b["op"] == "<"))
+        lo, hi = lo_b["value"], hi_b["value"]
+        # Counts and durations are integers: tighten strict bounds into
+        # inclusive integer bounds so the range is represented exactly.
+        if lo_b["op"] == ">" and isinstance(lo, int) and not isinstance(lo, bool):
+            lo += 1
+        if hi_b["op"] == "<" and isinstance(hi, int) and not isinstance(hi, bool):
+            hi -= 1
+        if lo > hi:
+            return None
+        return {"op": "between", "value": [lo, hi], "unit": unit}
+    if lowers:
+        b = max(lowers, key=lambda b: (b["value"], b["op"] == ">"))
+        return {"op": b["op"], "value": b["value"], "unit": unit}
+    if uppers:
+        b = min(uppers, key=lambda b: (b["value"], b["op"] == "<"))
+        return {"op": b["op"], "value": b["value"], "unit": unit}
+    return None
+
+
 def _quantifiers(text: str) -> List[Dict[str, Any]]:
     """All comparator+number mentions, for conflict detection and extraction."""
     out: List[Dict[str, Any]] = []
@@ -603,6 +653,16 @@ def _extract_count(text: str, objects: List[str]) -> Optional[Dict[str, Any]]:
     if unit == "seconds":
         # A duration used as a count is not meaningful for crowd/count events;
         # keep it as seconds and let the caller reject/clarify.
         return {"op": chosen["op"], "value": value, "unit": "seconds"}
+
+    # Keep every bound stated for the same unit family: a two-sided request
+    # ("more than five but fewer than ten people") must not lose one side.
+    family = [q for q in quantifiers if (q["unit"] or unit) == unit]
+    if len(family) > 1:
+        combined = _combine_bounds(family, unit)
+        if combined is not None:
+            return combined
     return {"op": chosen["op"], "value": value, "unit": unit}
 
 
@@ -625,7 +685,34 @@ def _extract_duration(text: str) -> Optional[Dict[str, Any]]:
     cmp_token = match.group("cmp")
     op = _op_from_comparator(cmp_token) if cmp_token else ">="
-    return {"op": op, "value": seconds, "unit": "seconds"}
+    bounds = [{"op": op, "value": seconds}]
+    for other in _SECONDS_BOUND_RE.finditer(text):
+        val = _num(other.group("num"))
+        if val is None:
+            continue
+        factor = _TIME_UNIT_SEC.get(other.group("unit").lower())
+        if not factor:
+            continue
+        secs = val * factor
+        if float(secs).is_integer():
+            secs = int(secs)
+        other_op = _op_from_comparator(other.group("cmp"))
+        if (other_op, secs) == (op, seconds):
+            continue
+        bounds.append({"op": other_op, "value": secs})
+    if len(bounds) > 1:
+        combined = _combine_bounds(bounds, "seconds")
+        if combined is not None:
+            return combined
+        lowers = [b for b in bounds if b["op"] in {">", ">="}]
+        if lowers:
+            strictest = max(lowers, key=lambda b: (b["value"], b["op"] == ">"))
+            return {"op": strictest["op"], "value": strictest["value"], "unit": "seconds"}
+    return {"op": op, "value": seconds, "unit": "seconds"}
 
 
 def _threshold_for_event(
@@ -760,6 +847,24 @@ def _conflict(text: str, objects: List[str], zones: List[str]) -> Optional[str]:
         if eq_values and (upper is not None and upper < next(iter(eq_values))):
             return "exact threshold above upper bound"
 
+    # An inverted "between X and Y <count-unit>" range contradicts itself.
+    match = _BETWEEN_COUNT_RE.search(text)
+    if match:
+        lo = _num(match.group("lo"))
+        hi = _num(match.group("hi"))
+        if lo is not None and hi is not None and lo > hi:
+            return "between bounds are inverted (lower exceeds upper)"
+    if re.search(r"\b(?:count|number\s+of|vehicle\s+count)\b", text):
+        match = re.search(
+            rf"\bbetween\s+(?P<lo>{_NUM_RE})\s+and\s+(?P<hi>{_NUM_RE})\b", text
+        )
+        if match and not re.search(
+            rf"\bbetween\s+{match.group('lo')}\s+and\s+{match.group('hi')}\s*(?:am|pm)\b",
+            text,
+        ):
+            lo = _num(match.group("lo"))
+            hi = _num(match.group("hi"))
+            if lo is not None and hi is not None and lo > hi:
+                return "between bounds are inverted (lower exceeds upper)"
+
     # Conflicting zones / object sets expressed as disjunctions.
     if re.search(r"\bor\b", text):
         if len(set(zones)) > 1:
@@ -952,6 +1057,20 @@ def materialize_alert_rule(
         "clarification_kind": kind,
     }
+
+    # Safety gate: a rule the validator rejects must never be reported as
+    # "clean" (the live chat path refuses to insert it and the dialogue would
+    # assume the rule is complete).  Such a request is contradictory for the
+    # typed grammar, so it is clarified instead of materialized.
+    if kind == "clean":
+        ok, errors, _warnings = validate_rule(rule)
+        if not ok:
+            kind = "conflicting"
+            detail = "the requested rule is not representable (" + "; ".join(errors) + ")"
+            rule["needs_clarification"] = True
+            rule["clarification"] = _clarification_reason(kind, detail)
+            rule["clarification_kind"] = kind
     return rule
```

### Runtime-monkeypatched proof (source tree untouched)

| Invariant | Result |
|---|---|
| `test_clean_rule_correct_30_of_30` | PASS |
| `test_stress_clarified_30_of_30` | PASS |
| `test_false_materializations_zero` | PASS |
| `held_out_9_of_9` | PASS |
| `seen_51_of_51` | PASS |
| `fuzz_crashes_zero` | PASS |
| `fuzz_soundness_violations_zero` | PASS |
| `fuzz_targeted_false_materializations_zero` | PASS |
| `oracle_policy_mean_1p0778` | PASS |
| `oracle_policy_max_turns_2` | PASS |
| `oracle_naive_mean_2p5778` | PASS |
| `oracle_naive_max_turns_4` | PASS |
| `oracle_both_correct_75_of_90` | PASS |
| `oracle_both_valid_90_of_90` | PASS |

Adversarial corpus violations after the fix: **0/38**; grammar-aware fuzz after the fix: validator violations 0, semantic violations 0, false materializations 0, crashes 0.

Paraphrase robustness after the fix: clean 60/60, perturbed 929/929 (1.000).

| Reproducer | Resolved after fix | New clarification kind |
|---|---|---|
| `more than five fewer than ten gather Zone A` | yes | clean |
| `at least five at most ten gather Zone A` | yes | clean |
| `more than five fewer than ten gather Zone A` | yes | clean |
| `number of Zone A more than five less than ten` | yes | clean |
| `more than two fewer than four are in Parking` | yes | clean |
| `loiters Zone A for more than five minutes less than ten minutes` | yes | clean |
| `more than five more than three people gather Zone A` | yes | clean |
| `fewer than ten fewer than eight gather Zone A` | yes | clean |
| `between ten and three people gather Zone A` | yes | conflicting |
| `number of Zone A between ten and three` | yes | conflicting |
| `count Parking between nine and two` | yes | conflicting |
| `Zone A between 3 and 3` | yes | conflicting |
| `Zone A 9 PM to 9 PM` | yes | conflicting |
| `Zone A before 12 AM` | yes | conflicting |
| `Zone A between 12 and 12` | yes | conflicting |
| `every 0.05 seconds more than five gather Zone A` | yes | conflicting |
| `20 hz more than five gather Zone A` | yes | conflicting |
| `99 hz more than five gather Zone A` | yes | conflicting |
| `during the day car park` | yes | clean |
| `more than five gather car park` | yes | clean |

