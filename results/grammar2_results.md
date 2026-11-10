# Grammar-2 alert materializer evaluation

Runtime monkeypatch of `backend/app/services/alert_materializer.py` (NOT applied to source).

## Silent-drop rate on seeded adversarial set

| Metric | Before | After |
|---|---:|---:|
| Inputs with a silent constraint drop | 173/215 | 0/215 |
| Silent-drop rate | 0.8047 | 0.0000 |
| Drop instances | 173 | 0 |

By category:

| Category | n | before drops | after drops |
|---|---:|---:|---:|
| two_zone | 70 | 53 | 0 |
| until | 67 | 54 | 0 |
| invalid_clock | 78 | 66 | 0 |

## Seeded case proofs

- `more than 5 people gather in Zone A and Zone B`
  - before: clarify=False kind=clean zone=Zone A time={} drops=['zone_conjunction']
  - after:  clarify=True kind=conflicting zone=Zone A time={} drops=[]
  - after clarification: Conflicting conditions (multiple zones specified (Zone A, Zone B)). Please give one consistent zone, object set, numeric threshold, and time window.
- `more than 5 people in Zone A until 10 PM`
  - before: clarify=True kind=missing zone=Zone A time={} drops=[]
  - after:  clarify=False kind=clean zone=Zone A time={'start': '00:00', 'end': '22:00'} drops=[]
- `more than 5 people in Zone A between 10:70 and 12:00`
  - before: clarify=True kind=missing zone=Zone A time={} drops=[]
  - after:  clarify=True kind=conflicting zone=Zone A time={} drops=[]
  - after clarification: Conflicting conditions (invalid clock time). Please give one consistent zone, object set, numeric threshold, and time window.

## Shipped 120-row benchmark (patched)

- mismatches: 0
- test clean rule correctness: 30/30
- test stress clarified: 30/30
- test clarify P/R/F1: 1.0/1.0/1.0

## Fuzz (patched)

- crashes: 0
- soundness violations: 0
- targeted false materializations: 0

## Exact source patch (unapplied unified diff)

```diff
--- a/backend/app/services/alert_materializer.py
+++ b/backend/app/services/alert_materializer.py
@@ -409,7 +409,29 @@
         if result:
             return result
 
+    # until/till H[:MM] [am/pm] -> 00:00 .. H (single lower bound).
+    m = re.search(
+        r"\b(?:until|till)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text,
+    )
+    if m:
+        h, minute, mer = m.groups()
+        pair = _parse_clock(h, minute, mer)
+        if pair:
+            hour = _apply_meridiem(pair[0], mer) if mer else pair[0]
+            mins = pair[1]
+            if 0 <= hour <= 23 and 0 <= mins <= 59:
+                result = _range((0, 0), (hour, mins))
+                if result:
+                    return result
+
     return None
+
+
+def _has_invalid_clock(text):
+    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", text):
+        if int(m.group(2)) > 59 or int(m.group(1)) > 23:
+            return True
+    return False
 
 
 _TIME_WINDOW_COUNT_RE = re.compile(
@@ -893,10 +915,15 @@
             if lo is not None and hi is not None and lo > hi:
                 return "between bounds are inverted (lower exceeds upper)"
 
+    # Multiple zones: schema holds a single zone_id, so never drop silently.
+    if len(set(zones)) > 1:
+        return f"multiple zones specified ({', '.join(zones)})"
+    # Invalid clock times must never be dropped silently.
+    if _has_invalid_clock(text):
+        return "invalid clock time"
+
     # Conflicting zones / object sets expressed as disjunctions.
     if re.search(r"\bor\b", text):
-        if len(set(zones)) > 1:
-            return f"ambiguous zone disjunction ({', '.join(zones)})"
         object_set = {name for name in objects}
         if len(object_set) > 1:
             return f"ambiguous object disjunction ({', '.join(sorted(object_set))})"
```

## New tests to add

- `test_two_zone_conjunction_clarifies`: assert `materialize_alert_rule('more than 5 people in Zone A and Zone B').needs_clarification is True` and clarification contains 'Zone A' and 'Zone B'.
- `test_until_single_materializes_window`: assert time_of_day == {'start':'00:00','end':'22:00'} for '... until 10 PM' and needs_clarification False.
- `test_invalid_clock_clarifies`: assert needs_clarification True for '... between 10:70 and 12:00'.
- `test_no_silent_drop_property`: for seeded adversarial set, after patch silent_drop_rate == 0.
- `test_benchmark_regression`: shipped 120 rows unchanged (30/30, 30/30).

## Exact commands

```
$env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_grammar2.py
```

