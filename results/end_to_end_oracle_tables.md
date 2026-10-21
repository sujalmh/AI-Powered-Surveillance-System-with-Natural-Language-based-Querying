# End-to-end released-subset evaluation (real LLM parser)

Model: `opencode-go/beam-free` via provider `opencode`.
Queries: `queries_control.jsonl` -- protocol-matched with eval_retrieval_control.py: same queries, split, labels and per-category wording.
Query window used by the structured branch: parsed by the LLM.

Parse: 0/150 answered by the LLM (0%), mean 0.0s, p95 0.0s per query.
Count queries whose parsed filter carries an explicit numeric target: 30/30.

## Test macro P@5

| Method | Macro P@5 | Count | Find | Track | Action | Filter |
|---|---:|---:|---:|---:|---:|---:|
| fixed_0.0 | 0.048 | 0.010 | 0.070 | 0.020 | 0.090 | 0.050 |
| fixed_0.3 | 0.380 | 0.270 | 0.630 | 0.550 | 0.090 | 0.360 |
| fixed_0.5 | 0.534 | 0.390 | 0.910 | 0.810 | 0.090 | 0.470 |
| fixed_0.7 | 0.538 | 0.390 | 0.920 | 0.810 | 0.090 | 0.480 |
| fixed_1.0 | 0.520 | 0.390 | 0.920 | 0.810 | 0.000 | 0.480 |
| adaptive | 0.538 | 0.390 | 0.920 | 0.810 | 0.090 | 0.480 |
| per_intent_tuned | 0.536 | 0.390 | 0.920 | 0.810 | 0.090 | 0.470 |
| paper_lookup | 0.428 | 0.390 | 0.630 | 0.550 | 0.090 | 0.480 |

Adaptive vs fixed_0.7 (tune-selected): delta P@5 = +0.000 [+0.000, +0.000], p = 1.0000; grouped [+0.000, +0.000], p = 1.0000 (3000 resamples, seed 11).

## Per category (test split, vs the best tune-selected fixed alpha)

| Category | n | Adaptive | Best fixed | Delta | 95% CI | p | p (Holm) |
|---|---:|---:|---:|---:|---|---:|---:|
| count | 20 | 0.390 | 0.390 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |
| find | 20 | 0.920 | 0.920 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |
| track | 20 | 0.810 | 0.810 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |
| action | 20 | 0.090 | 0.090 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |
| filter | 20 | 0.480 | 0.480 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |

## What this track does not establish

- Single camera, 123 clips, ~7.2 min of selected footage; the deployment numbers are not reproduced here.
- Query relevance labels are metadata-derived, not human-annotated.
- The MongoDB structured branch is replaced by an in-memory metadata store built from the released sidecar; the scoring code is the production code.
- The fusion is the plain alpha blend, not the live stack (no recency boost, no MMR, no track merging); n_semantic is fixed at 20 rather than the live top-k.
- The tune and test splits share relevant clips (the split is stratified by category, not by clip or time bucket).
- The count EM/MAE helper reads the caption count at the benchmark's anchor timestamp, so it measures whether the anchor clip ranked first rather than counting ability.

## Comparison with the oracle-parse control

The control track replays the metadata-derived reference filter; this track uses the parser's own answer.  The difference is the price of not having an oracle parse: the reference filter carries the anchor person's attribute list and a time window that follows the subject past the window the query states, and neither is recoverable from the query text.

| Category | Control (oracle parse) | End-to-end (LLM parse) | Delta |
|---|---:|---:|---:|
| count | 0.390 | 0.390 | +0.000 |
| find | 0.920 | 0.920 | +0.000 |
| track | 0.810 | 0.810 | +0.000 |
| action | 0.090 | 0.090 | +0.000 |
| filter | 0.480 | 0.480 | +0.000 |
| **macro** | **0.538** | **0.538** | **+0.000** |

Parse fidelity against the reference parse:

| Category | n | Colour agreement | Count target | Reference has label-side attributes | Reference window differs |
|---|---:|---|---|---:|---:|
| count | 30 | 0/- | 30/30 | 0 | 0 |
| find | 30 | 30/30 | 0/- | 20 | 0 |
| track | 30 | 30/30 | 0/- | 0 | 0 |
| action | 30 | 0/- | 0/- | 0 | 0 |
| filter | 30 | 30/30 | 0/- | 0 | 0 |
