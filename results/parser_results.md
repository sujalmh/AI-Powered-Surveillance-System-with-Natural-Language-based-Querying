# NL parser evaluation (agreement with the metadata-derived reference)

- Model requested: `opencode-go/beam-free` (provider `opencode`)
- Models that answered: `opencode-go/beam-free` x150
- Generated at: 2026-10-21T20:56:42.294815+00:00
- Repo commit: `430bede` (working tree dirty: True)
- Queries: 150 of 150 in `backend/benchmarks/queries_control.jsonl` (splits: test 100, tune 50)
- Reference: the 'parsed' field of the query file used for the run

> **Read this before quoting a number.** The reference is an approximation of the true query intent; a disagreement may be a wrong reference rather than a wrong parse. The rates below are agreement rates, not accuracy.

## Headline

- Parse failures (LLM error / unusable JSON): **0** of 150 queries (0.0%).
- Parse latency: mean **2.96 s**, median 2.46 s, p95 **4.98 s**, max 26.65 s.
- Agreement `camera_id`: **16.0%** (24/150 queries whose reference states the field).
- Agreement `time_window`: **78.0%** (117/150 queries whose reference states the field).
- Agreement `object_name`: **81.1%** (73/90 queries whose reference states the field).
- Agreement `color`: **100.0%** (90/90 queries whose reference states the field).
- Agreement `zone`: **100.0%** (14/14 queries whose reference states the field).
- Agreement `action`: **33.3%** (10/30 queries whose reference states the field).
- Agreement `count_constraint`: **100.0%** (30/30 queries whose reference states the field).
- Agreement `intent`: **56.0%** (84/150 queries whose reference states the field).

## Per-field agreement (all queries)

Agreement is computed over the queries whose *reference* states the field; a field the parser does not state counts as a disagreement. `rate among both` restricts the denominator to the queries where both sides state the field.

| field | reference states | parser states | both | agree | agreement % | rate among both % | missing in parse | spurious in parse |
|---|---|---|---|---|---|---|---|---|
| `camera_id` | 150 | 24 | 24 | 24 | 16.0% | 100.0% | 126 | 0 |
| `time_window` | 150 | 120 | 120 | 117 | 78.0% | 97.5% | 30 | 0 |
| `object_name` | 90 | 128 | 73 | 73 | 81.1% | 100.0% | 17 | 55 |
| `color` | 90 | 90 | 90 | 90 | 100.0% | 100.0% | 0 | 0 |
| `zone` | 14 | 14 | 14 | 14 | 100.0% | 100.0% | 0 | 0 |
| `action` | 30 | 19 | 10 | 10 | 33.3% | 100.0% | 20 | 9 |
| `count_constraint` | 30 | 30 | 30 | 30 | 100.0% | 100.0% | 0 | 0 |
| `intent` | 150 | 150 | 150 | 84 | 56.0% | 56.0% | 0 | 0 |

`time_window` agreement means *interval overlap* with the reference window, not exact equality of the two bounds.

## Per-field agreement by category

| category | `camera_id` | `time_window` | `object_name` | `color` | `zone` | `action` | `count_constraint` | `intent` |
|---|---|---|---|---|---|---|---|---|
| action | 20.0% | 0.0% | n/a | n/a | n/a | 33.3% | n/a | 0.0% |
| count | 0.0% | 100.0% | n/a | n/a | n/a | n/a | 100.0% | 100.0% |
| filter | 20.0% | 100.0% | 60.0% | 100.0% | 100.0% | n/a | n/a | 0.0% |
| find | 20.0% | 100.0% | 100.0% | 100.0% | n/a | n/a | n/a | 100.0% |
| track | 20.0% | 90.0% | 83.3% | 100.0% | n/a | n/a | n/a | 80.0% |

A cell is `n/a` when the reference never states that field for the category (e.g. there is no reference zone for the count queries).

## Time window

| metric | value |
|---|---:|
| queries evaluated | 120 |
| queries without a parsed absolute window | 30 |
| median absolute error, start bound (s) | 22.90 |
| median absolute error, end bound (s) | 25.15 |
| median absolute error, both bounds (s) | 23.35 |
| mean absolute error, both bounds (s) | 24.24 |
| window overlap rate | 97.5% |
| window overlap rate (+/- 60 s tolerance) | 100.0% |
| median IoU | 0.7649 |

The reference window is a metadata *context window* around the anchor of the query, while the parser emits the window written in the text (often a single instant for "around HH:MM:SS"). The endpoint errors therefore mix parser error with a difference in the definition of the window; the overlap rate with a tolerance is given as the more forgiving diagnostic.

## Camera id and intent (restricted views)

| metric | value |
|---|---:|
| queries whose text names a camera | 24 |
| camera-id agreement on those queries | 100.0% (24/24) |
| queries whose reference states an intent | 30 |
| intent agreement on those queries | 80.0% (24/30) |

## Latency

| scope | n | mean (s) | median (s) | p95 (s) | max (s) |
|---|---|---|---|---|---|
| all queries | 150 | 2.96 | 2.46 | 4.98 | 26.65 |
| action | 30 | 2.09 | 1.91 | 3.17 | 3.57 |
| count | 30 | 2.05 | 1.92 | 3.06 | 3.18 |
| filter | 30 | 3.21 | 2.76 | 5.59 | 6.24 |
| find | 30 | 3.84 | 2.79 | 4.91 | 26.65 |
| track | 30 | 3.59 | 3.04 | 7.00 | 7.88 |

## Parse failures

No parse failures: all 150 queries produced a usable filter.

## Queries with a value conflict

Both sides state the field but with different values. Each of these is a case where the parse and the metadata-derived reference disagree; it may be a wrong reference rather than a wrong parse.

| qid | category | split | conflicting fields |
|---|---|---|---|
| q_track_02 | track | test | intent |
| q_track_06 | track | test | time_window |
| q_track_07 | track | test | intent |
| q_track_12 | track | tune | intent |
| q_track_13 | track | test | time_window |
| q_track_15 | track | tune | time_window |
| q_track_17 | track | test | intent |
| q_track_22 | track | tune | intent |
| q_track_27 | track | test | intent |
| q_filter_01 | filter | test | intent |
| q_filter_02 | filter | test | intent |
| q_filter_03 | filter | test | intent |
| q_filter_04 | filter | test | intent |
| q_filter_05 | filter | tune | intent |
| q_filter_06 | filter | tune | intent |
| q_filter_07 | filter | test | intent |
| q_filter_08 | filter | test | intent |
| q_filter_09 | filter | test | intent |
| q_filter_10 | filter | test | intent |
| q_filter_11 | filter | test | intent |
| q_filter_12 | filter | test | intent |
| q_filter_13 | filter | tune | intent |
| q_filter_14 | filter | tune | intent |
| q_filter_15 | filter | test | intent |
| q_filter_16 | filter | tune | intent |
| q_filter_17 | filter | test | intent |
| q_filter_18 | filter | tune | intent |
| q_filter_19 | filter | test | intent |
| q_filter_20 | filter | tune | intent |
| q_filter_21 | filter | test | intent |
| q_filter_22 | filter | tune | intent |
| q_filter_23 | filter | test | intent |
| q_filter_24 | filter | test | intent |
| q_filter_25 | filter | test | intent |
| q_filter_26 | filter | tune | intent |
| q_filter_27 | filter | test | intent |
| q_filter_28 | filter | test | intent |
| q_filter_29 | filter | tune | intent |
| q_filter_30 | filter | test | intent |
| qc_action_01 | action | tune | intent |
| qc_action_02 | action | tune | intent |
| qc_action_03 | action | tune | intent |
| qc_action_04 | action | tune | intent |
| qc_action_05 | action | tune | intent |
| qc_action_06 | action | tune | intent |
| qc_action_07 | action | tune | intent |
| qc_action_08 | action | test | intent |
| qc_action_09 | action | test | intent |
| qc_action_10 | action | test | intent |
| qc_action_11 | action | test | intent |
| qc_action_12 | action | test | intent |
| qc_action_13 | action | test | intent |
| qc_action_14 | action | test | intent |
| qc_action_15 | action | test | intent |
| qc_action_16 | action | test | intent |
| qc_action_17 | action | test | intent |
| qc_action_18 | action | test | intent |
| qc_action_19 | action | test | intent |
| qc_action_20 | action | test | intent |
| qc_action_21 | action | tune | intent |
| qc_action_22 | action | tune | intent |
| qc_action_23 | action | tune | intent |
| qc_action_24 | action | test | intent |
| qc_action_25 | action | test | intent |
| qc_action_26 | action | test | intent |
| qc_action_27 | action | test | intent |
| qc_action_28 | action | test | intent |
| qc_action_29 | action | test | intent |
| qc_action_30 | action | test | intent |

## Queries where the parser dropped a field

The reference states the field, the parser does not. The breakdown shows how often each field was dropped; a dropped `camera_id` is expected for the queries whose text does not name a camera (the camera is an archive property, not something the query can state).

| dropped field | queries |
|---|---|
| `camera_id` | 126 |
| `time_window` | 30 |
| `object_name` | 17 |
| `color` | 0 |
| `zone` | 0 |
| `action` | 20 |
| `count_constraint` | 0 |
| `intent` | 0 |

Queries with at least one disagreement of any kind: 144.

## Reading the disagreements

The counters below separate the cases where the *reference* is the odd one from the cases where the *parse* is the odd one. They are computed from the same records as the tables above.

- `camera_id`: 126 queries state no camera because their text does not name one; on the queries that do name a camera the agreement is 100.0%.
- `object_name`: 17 queries state no object (the text names clothing or a track, not a person), while 55 queries state an object the reference does not (the parser says 'person' for the count queries, whose reference states no object at all) - the parse is more specific there, not wrong.
- `action`: 30 reference action values are natural-language labels and 0 are the internal token 'motion_energy'; the parser dropped the field on 20 queries and produced a conflicting value on 0.
- `intent`: 60 mismatches come from the action and filter categories, whose reference falls back to the category name (a value the parser's intent vocabulary does not contain); 6 mismatches remain for the queries whose reference states an explicit intent.

## Known limitations

- The reference parse is metadata-derived, not human-annotated: it was reconstructed from the archive metadata that generated each query (camera, colour/zone/action label, anchor timestamps). It approximates the true query intent, so a disagreement may be a wrong reference rather than a wrong parse.
- The reported numbers are agreement rates with a metadata-derived reference; they are NOT accuracy against a human-annotated gold standard.
- The released benchmark subset is single-camera (camera 0) and does not represent the 4-camera deployment used in the paper.
- Free-tier quotas differ between providers: the opencode transport falls back to a secondary free model when the primary model's daily quota is exhausted, so a re-run on another day may be answered by a different model (the model that answered each query is recorded in the JSON).
- The count-constraint agreement rate is undefined for this benchmark: 30 of the 150 reference parses state a numeric count target (count_target_in_text is null everywhere), because the count queries carry no number in their text, so there is nothing to agree on for most of them.
- The action reference of the 'highest-motion' queries is the internal token 'motion_energy' (0 of the 30 reference action values), which no natural-language parser can emit; those queries are structurally unmatchable.
- The intent reference falls back to the query category (count / find / track / action / filter), a wider vocabulary than the parser's __intent (find / count / track); the action and filter categories are therefore structurally unmatched in the headline intent rate.
- The reference camera id is 0 for all 150 queries although only 24 of the texts names a camera; the strict camera-id rate is therefore dominated by queries where the camera is not expressible. The camera_id_explicit_text block reports the restricted rate.
- The reference zone is stated for only 14 queries, and the reference window is a metadata context window around the anchor rather than the window written in the query text; endpoint errors therefore mix parser error with reference/window-definition mismatch.

## Reproduce

```powershell
$env:PYTHONIOENCODING='utf-8'
python backend/scripts/eval_parser.py --limit 150 --model opencode-go/beam-free
```

Outputs: `results/parser_results.json` and `results/parser_results.md`. The parses are cached in `backend/benchmarks/parsed_cache/opencode-go_beam-free.json` (150 from cache, 0 model calls in this run). Use `--refresh` to ignore the cache and `--model <id>` to evaluate another model.
