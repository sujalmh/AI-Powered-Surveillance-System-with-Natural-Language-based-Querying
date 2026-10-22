# Alert semantics evaluation (C3)

Read-only probe over the shipped alert materializer/validator/policy.
Numbers below come only from files written under `results/semantics_*`.

Exact command: `$env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts_semantics.py --stage all`

## Task 1 - Semantic-vs-syntactic gap probe

Seeded variants: 78 total, 78 resolved silently to a default, 0 clarified. Semantic-gap (silent) rate = 1.0000.

| Class | n | Silent-to-default | Clarified |
|---|---:|---:|---:|
| D1_default_object_person_vs_intended | 12 | 12 | 0 |
| D2_default_cooldown_60s_vs_intended_urgency | 55 | 55 | 0 |
| D3_night_window_20_06_vs_intended_daytime | 6 | 6 | 0 |
| D3_night_window_20_06_vs_intended_evening | 5 | 5 | 0 |

## Task 2 - Per-field accuracy with 95% bootstrap CIs

### dev (n=30 clean rows)

| Field | Point (quoted) | Correct | 95% CI |
|---|---:|---:|---:|
| `event` | 1.000 | 30/30 | [1.000, 1.000] |
| `objects` | 1.000 | 30/30 | [1.000, 1.000] |
| `area` | 1.000 | 30/30 | [1.000, 1.000] |
| `threshold` | 1.000 | 30/30 | [1.000, 1.000] |
| `time_of_day` | 1.000 | 30/30 | [1.000, 1.000] |
| `severity` | 1.000 | 30/30 | [1.000, 1.000] |
| `cooldown_sec` | 1.000 | 30/30 | [1.000, 1.000] |
| `side_effects` | 1.000 | 30/30 | [1.000, 1.000] |
| `cadence_hz` | 1.000 | 30/30 | [1.000, 1.000] |

Specified (non-default) fields:

| Field | n (spec) | Point (quoted) | 95% CI |
|---|---:|---:|---:|
| `severity` | 2 | 1.000 | [1.000, 1.000] |
| `cadence_hz` | 4 | 1.000 | [1.000, 1.000] |
| `cooldown_sec` | 4 | 1.000 | [1.000, 1.000] |

### test (n=30 clean rows)

| Field | Point (quoted) | Correct | 95% CI |
|---|---:|---:|---:|
| `event` | 1.000 | 30/30 | [1.000, 1.000] |
| `objects` | 1.000 | 30/30 | [1.000, 1.000] |
| `area` | 1.000 | 30/30 | [1.000, 1.000] |
| `threshold` | 1.000 | 30/30 | [1.000, 1.000] |
| `time_of_day` | 1.000 | 30/30 | [1.000, 1.000] |
| `severity` | 1.000 | 30/30 | [1.000, 1.000] |
| `cooldown_sec` | 1.000 | 30/30 | [1.000, 1.000] |
| `side_effects` | 1.000 | 30/30 | [1.000, 1.000] |
| `cadence_hz` | 1.000 | 30/30 | [1.000, 1.000] |

Specified (non-default) fields:

| Field | n (spec) | Point (quoted) | 95% CI |
|---|---:|---:|---:|
| `severity` | 7 | 1.000 | [1.000, 1.000] |
| `cadence_hz` | 2 | 1.000 | [1.000, 1.000] |
| `cooldown_sec` | 1 | 1.000 | [1.000, 1.000] |

## Task 3 - Inconsistent-oracle contradiction detection

Combined-utterance re-materialization: 68/225 detected = 0.3022 (95% CI [0.244, 0.364]).

| Contradiction slot | n | Detected | Rate (95% CI) |
|---|---:|---:|---:|
| cadence | 30 | 0 | 0.000 [0.000, 0.000] |
| cooldown | 30 | 0 | 0.000 [0.000, 0.000] |
| event | 30 | 2 | 0.067 [0.000, 0.167] |
| object | 30 | 3 | 0.100 [0.000, 0.233] |
| severity | 30 | 0 | 0.000 [0.000, 0.000] |
| threshold | 30 | 18 | 0.600 [0.433, 0.767] |
| time | 15 | 15 | 1.000 [1.000, 1.000] |
| zone | 30 | 30 | 1.000 [1.000, 1.000] |

Dialogue-level policies:

| Policy | Detection rate | Silent-overwrite rate |
|---|---:|---:|
| shipped_minimal_questions | 0.000 | 1.000 |
| consistency_check (upper bound) | 1.000 | 0.000 |

## Verdict

Syntactic validity (validator pass) is not semantic correctness: all 78 seeded variants with divergent intent materialize to clean, valid rules (silent-to-default rate 1.000, 0 clarified). Per-field accuracy on the released set is 1.000 on every field; the bootstrap percentile CIs are degenerate [1.000, 1.000] because no field has an observed error on 30 rows (one miss -> 0.967). Combined-utterance contradiction detection is perfect for zone (30/30) and time (15/15), partial for threshold (18/30), and near-zero for event (2/30), object (3/30), severity, cooldown, and cadence (0/30 each); the shipped dialogue policy detects 0/225 inconsistent oracle answers (silent overwrite).

