# Alert-authoring evaluation (C3)

**Provenance.** The 120 utterances are **author-constructed** for this revision; they are not human-annotated and no inter-annotator agreement is claimed. The clean rows are a closed-vocabulary reference set for the documented grammar, so the high clean accuracy should not be read as open-world robustness.

**Exact command:** `$env:PYTHONIOENCODING='utf-8'; python backend/scripts/eval_alerts.py` (repository root, Python 3.12, UTF-8).

**Dataset:** `backend/benchmarks/alert_utterances.jsonl` — 120 rows (60 clean + 60 stress), sha256 `c885f8b96025132f…`.

## Test split (primary, n=30 clean / 30 stress)

| Metric | Value |
|---|---:|
| Clean rule correctness (9/9 fields exact) | 30/30 = **1.000** |
| Clean materialization rate (no clarification) | 30/30 = 100.0% |
| Stress clarification rate | 30/30 = **100.0%** |
| Stress false materializations | 0 |
| Clarify precision / recall / F1 | 1.000 / 1.000 / 1.000 |
| Derived `count` compatibility | 30/30 = 100.0% |

### Per-field accuracy (test clean rows)

| Field | Correct | Accuracy |
|---|---:|---:|
| `event` | 30/30 | 100.0% |
| `objects` | 30/30 | 100.0% |
| `area` | 30/30 | 100.0% |
| `threshold` | 30/30 | 100.0% |
| `time_of_day` | 30/30 | 100.0% |
| `severity` | 30/30 | 100.0% |
| `cooldown_sec` | 30/30 | 100.0% |
| `side_effects` | 30/30 | 100.0% |
| `cadence_hz` | 30/30 | 100.0% |

### Specified-field accuracy (test clean rows, defaults excluded)

Severity `info`, cadence `1.0` Hz and cooldown `60` s score 1.00 by construction when the utterance does not state them. The rows below score only the utterances whose reference does NOT use the default.

| Field | Specified n | Correct | Accuracy |
|---|---:|---:|---:|
| `severity` | 7 | 7 | 100.0% |
| `cadence_hz` | 2 | 2 | 100.0% |
| `cooldown_sec` | 1 | 1 | 100.0% |

### Per-stress-kind breakdown (test)

| Kind | n | Clarified | Rate | False mat. | Kind match |
|---|---:|---:|---:|---:|---:|
| ambiguous | 7 | 7 | 100.0% | 0 | 7/7 |
| conflicting | 8 | 8 | 100.0% | 0 | 8/8 |
| missing | 7 | 7 | 100.0% | 0 | 7/7 |
| unsupported | 8 | 8 | 100.0% | 0 | 8/8 |

## Dev split (tuning/diagnostic, n=30 clean / 30 stress)

| Metric | Dev | Test |
|---|---:|---:|
| Clean rule correctness | 30/30 = 1.000 | 30/30 = 1.000 |
| Clean materialization rate | 100.0% | 100.0% |
| Stress clarification rate | 100.0% | 100.0% |
| Stress false materializations | 0 | 0 |
| Clarify precision / recall | 1.000 / 1.000 | 1.000 / 1.000 |

All 60 clean rows (dev+test, diagnostic): 60/60 = 1.000; all 60 stress rows: 60/60 clarified.

## Interpretation

The 0.91 figure reported in the manuscript comes from the deployment study (a separate 60-utterance human-authored set that is not part of this release), and it is *not* reproducible from this benchmark: the placeholder 120-row file that previously shipped repeated a handful of templates, and the earlier evaluator built rules from the reference fields rather than from the utterance. The release therefore states two things separately and does not merge them: the deployment result of 0.91 on the human-authored set, and the reproducible coverage measurement above (30/30 fully correct 9-field rules, 30/30 stress utterances clarified) on the author-constructed set shipped here. Because the clean rows are author-constructed and the vocabulary is closed, this is a regression/coverage measurement and an upper bound for the authored grammar, not an open-world accuracy estimate.

## Failing test cases

None on the test split. All 30 clean rules were materialized with all nine fields exact, and all 30 stress utterances requested clarification.

## Compositional held-out (one (event, zone) combo per event)

The grammar has no learned parameters, so this is a coverage-uniformity check across the product space, not a test of learned compositionality. Held-out combos: (crowd, Zone A), (loiter, Entrance), (fight, Parking), (enter, Zone A), (zone, Zone B), (count, Gate), (time, Entrance).

| Slice | n | Correct | Accuracy |
|---|---:|---:|---:|
| Held-out combos | 9 | 9 | 100.0% |
| Seen combos | 51 | 51 | 100.0% |

## Simulated-oracle-user dialogue (clean test rows, masked slots)

Episodes: 90 (30 clean test rows x 3 reps, k=rep+1 masked required slots from {event, zone, threshold}, seed 0xc3a1). A truthful oracle fills exactly the slots asked; unasked masked slots fall back to the documented event defaults. Final rules are checked with `validate_rule` and against the reference.

| Policy | Mean turns | Max turns | Correct | Valid |
|---|---:|---:|---:|---:|
| Minimal-question policy | 1.08 | 2 | 75/90 = 83.3% | 90/90 |
| Naive per-field (9 fields, fixed order) | 2.58 | 4 | 75/90 = 83.3% | 90/90 |

## Fuzz property tests (stdlib `random`, seeded)

Noise: 2000 utterances (1-12 tokens from grammar/filler vocabulary, seed 0xf422); targeted: 200 utterances embedding one unsupported hint in filler.

| Property | Result |
|---|---:|
| Crashes (materialize + validate) | 0 |
| Soundness violations (materialized rule failing `validate_rule`) | 0 |
| Targeted false materializations (unsupported hint + filler) | 0 |
| Noise clarified / materialized (descriptive) | 1358 / 642 |

## Canonical fields and defaults

Scored fields: `event`, `objects`, `area`, `threshold`, `time_of_day`, `severity`, `cooldown_sec`, `side_effects`, `cadence_hz`. `count` is a derived compatibility view for `AlertRuleSpec`, not a tenth field.

Defaults when unspecified: objects `[person]`, severity `info`, cooldown `60` s, side effects `['snapshot', 'push_ws']`, cadence `1.0` Hz, time window all-day `{}`.

Event-specific defaults: fight `>= 2` persons; loiter `>= 60` seconds; zone/enter/time `>= 1` matching object. Crowd and count events require an explicit numeric threshold; a missing zone, crowd/count threshold, ambiguous quantifier/zone/time, contradictory conditions, or unsupported request requests clarification instead of materializing a rule.

The materializer is deterministic regex/keyword code; the benchmark loop makes no LLM calls and requires no MongoDB.

## Split construction

Every row carries `split`. Clean rows alternate dev/test (30/30). Each stress kind has 15 rows split 7/8 or 8/7, giving 30 dev and 30 test stress rows. The split is fixed in `backend/benchmarks/alert_utterances.jsonl`; no test row was moved after scoring.

