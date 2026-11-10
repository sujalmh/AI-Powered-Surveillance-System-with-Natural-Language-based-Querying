# MMR / recency ablation (released control, 100-query test split)

**Scope.** Protocol-matched control (`backend/benchmarks/queries_control.jsonl`), adaptive-alpha fused scores from `eval_retrieval_control.py`, post-ranking ablated 2x2. Production defaults are recency ON + MMR ON.

| Config | Macro P@5 | Count | Find | Track | Action | Filter |
|---|---:|---:|---:|---:|---:|---:|
| base (no recency, no MMR) | 0.538 | 0.390 | 0.920 | 0.810 | 0.090 | 0.480 |
| recency only | 0.538 | 0.390 | 0.920 | 0.810 | 0.090 | 0.480 |
| MMR only | 0.400 | 0.370 | 0.580 | 0.550 | 0.090 | 0.410 |
| both (production default) | 0.400 | 0.370 | 0.580 | 0.550 | 0.090 | 0.410 |
| both (archive-anchored now, sensitivity) | 0.400 | 0.370 | 0.580 | 0.550 | 0.090 | 0.410 |

## Deltas vs base (paired bootstrap, 3000 resamples)

| Config | ΔP@5 [95% CI] | p |
|---|---:|---:|
| recency only | +0.000 [+0.000, +0.000] | 1.0000 |
| MMR only | -0.138 [-0.172, -0.100] | < 0.0007 |
| both (production default) | -0.138 [-0.172, -0.100] | < 0.0007 |

## Recency diagnostics

Wall-clock now: `2026-10-22T07:07:01.253768`; sample boosts: min 1.0000, max 1.0000 (identically ~1.0: every archived clip is ~5,000 h old against a 24 h half-life). Archive-anchored now (`2026-03-01T19:56:17.989377`): min 1.460, max 1.460.

## What cannot be reproduced

- Single camera (camera_0), 100 test queries; cross-camera and deployment-scale effects are untestable.
- Recency uses production wall-clock semantics; on the archived subset every boost is ~1.0 by construction (age saturation), so recency is ranking-inert here, not necessarily in deployment.
- MMR (lambda=0.3) penalises same-(camera, hour-bucket) candidates; on this single-camera subset the relevant clips share the query window bucket, so MMR demotes them (-0.138 macro P@5). The 0.538 control is therefore pre-MMR; the production default scores 0.400 on this subset. Multi-camera deployment effects are untestable here.
