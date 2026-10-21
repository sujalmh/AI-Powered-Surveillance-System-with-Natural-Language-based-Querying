# End-to-end released-subset evaluation (real LLM parser)

Model: `opencode-go/beam-free` via provider `opencode`.
Queries: `queries_control.jsonl` -- protocol-matched with eval_retrieval_control.py: same queries, split, labels and per-category wording.
Query window used by the structured branch: parsed by the LLM.

Parse: 150/150 answered by the LLM (100%), mean 3.0s, p95 5.0s per query.
Count queries whose parsed filter carries an explicit numeric target: 30/30.

## Test macro P@5

| Method | Macro P@5 | Count | Find | Track | Action | Filter |
|---|---:|---:|---:|---:|---:|---:|
| fixed_0.0 | 0.048 | 0.010 | 0.070 | 0.020 | 0.090 | 0.050 |
| fixed_0.3 | 0.314 | 0.240 | 0.570 | 0.310 | 0.090 | 0.360 |
| fixed_0.5 | 0.448 | 0.360 | 0.820 | 0.530 | 0.090 | 0.440 |
| fixed_0.7 | 0.450 | 0.360 | 0.820 | 0.530 | 0.090 | 0.450 |
| fixed_1.0 | 0.428 | 0.360 | 0.790 | 0.540 | 0.000 | 0.450 |
| adaptive | 0.450 | 0.360 | 0.820 | 0.530 | 0.090 | 0.450 |
| per_intent_tuned | 0.448 | 0.360 | 0.820 | 0.530 | 0.090 | 0.440 |
| paper_lookup | 0.356 | 0.360 | 0.570 | 0.310 | 0.090 | 0.450 |

Adaptive vs fixed_0.5 (tune-selected): delta P@5 = +0.002 [-0.004, +0.010], p = 0.7973; grouped [-0.004, +0.009], p = 0.7927 (3000 resamples, seed 11).

## Per category (test split, vs the best tune-selected fixed alpha)

| Category | n | Adaptive | Best fixed | Delta | 95% CI | p | p (Holm) |
|---|---:|---:|---:|---:|---|---:|---:|
| count | 20 | 0.360 | 0.360 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |
| find | 20 | 0.820 | 0.820 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |
| track | 20 | 0.530 | 0.530 | -0.000 | [-0.030, +0.030] | 0.9420 | 1.0000 |
| action | 20 | 0.090 | 0.090 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 |
| filter | 20 | 0.450 | 0.440 | +0.010 | [+0.000, +0.030] | 0.7033 | 1.0000 |

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
| count | 0.390 | 0.360 | -0.030 |
| find | 0.920 | 0.820 | -0.100 |
| track | 0.810 | 0.530 | -0.280 |
| action | 0.090 | 0.090 | +0.000 |
| filter | 0.480 | 0.450 | -0.030 |
| **macro** | **0.538** | **0.450** | **-0.088** |

Parse fidelity against the reference parse:

| Category | n | Colour agreement | Count target | Reference has label-side attributes | Reference window differs |
|---|---:|---|---|---:|---:|
| count | 30 | 0/- | 30/30 | 0 | 3 |
| find | 30 | 30/30 | 0/- | 20 | 2 |
| track | 30 | 30/30 | 0/- | 0 | 30 |
| action | 30 | 0/- | 0/- | 0 | 0 |
| filter | 30 | 30/30 | 0/- | 0 | 4 |
