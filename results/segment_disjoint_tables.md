# Segment-disjoint (leak-free) retrieval results

**Scope.** Released single-camera subset; temporal-segment proxy for camera-disjointness. Not the 4-camera / 312k deployment.

## Leakage audit of the shipped tune/test split

- tune-relevant clips: 98; test-relevant clips: 104; shared: 98 (94.2% of test-relevant).

- test queries touching a tune clip: 98/100.

- tune queries touching a test clip: 50/50.


## Segments

| Segment | Clips | Single-segment queries |
|---|---:|---:|
| S1 (17:01-17:08) | 28 | 21 |
| S2 (18:04-18:28) | 90 | 99 |
| S3 (18:55-18:56) | 3 | 10 |

Cross-segment (multi-burst) queries: 20.


## Disjointness of this split

- held_out_segment: 2
- held_out_segment_name: S3 (18:55-18:56)
- train_segments: [0, 1]
- train_segment_names: ['S1 (17:01-17:08)', 'S2 (18:04-18:28)']
- n_train_clips: 118
- n_test_clips: 3
- n_shared_clips_train_test: 0
- n_tune_queries: 140
- n_test_queries: 10
- n_cross_segment_queries_total: 20
- n_queries_dropped_for_disjointness: 0
- n_tune_relevant_clips: 101
- n_test_relevant_clips: 3
- n_shared_relevant_clips: 0
- shared_relevant_pct_of_test: 0.0
- clip_disjoint: True


Held-out segment: S3 (18:55-18:56) (index 2). Train queries: 140, test queries: 10.


## Macro P@5 on held-out segment


### minmax

| Method | Macro P@5 | R@10 | F1@10 | MRR |
|---|---:|---:|---:|---:|
| adaptive | 0.3500 | 1.0000 | 0.5119 | 1.0000 |
| combMAX | 0.3500 | 1.0000 | 0.5119 | 1.0000 |
| fixed_0.7 | 0.3500 | 1.0000 | 0.5119 | 1.0000 |
| per_intent_tuned | 0.3500 | 1.0000 | 0.5119 | 0.9167 |
| fixed_0.0 | 0.0000 | 0.0417 | 0.0000 | 0.0354 |
| fixed_0.3 | 0.1458 | 0.4271 | 0.2143 | 0.6159 |
| fixed_0.5 | 0.3500 | 1.0000 | 0.5119 | 0.9167 |
| fixed_1.0 | 0.3500 | 1.0000 | 0.5119 | 1.0000 |

Best fixed alpha selected on tune: `fixed_0.7` (tune macro P@5 0.5657).


### zscore

| Method | Macro P@5 | R@10 | F1@10 | MRR |
|---|---:|---:|---:|---:|
| adaptive | 0.3500 | 1.0000 | 0.5119 | 0.9688 |
| combMAX | 0.3500 | 1.0000 | 0.5119 | 1.0000 |
| fixed_0.9 | 0.3500 | 1.0000 | 0.5119 | 0.9375 |
| per_intent_tuned | 0.3500 | 1.0000 | 0.5119 | 0.9167 |
| fixed_0.0 | 0.0000 | 0.0417 | 0.0000 | 0.0354 |
| fixed_0.3 | 0.0958 | 0.3021 | 0.1429 | 0.4365 |
| fixed_0.5 | 0.2000 | 0.6042 | 0.2956 | 0.6045 |
| fixed_0.7 | 0.3500 | 1.0000 | 0.5119 | 0.8854 |
| fixed_1.0 | 0.3500 | 1.0000 | 0.5119 | 1.0000 |

Best fixed alpha selected on tune: `fixed_0.9` (tune macro P@5 0.5657).


## Adaptive vs best-fixed

- **minmax**: ΔP@5 (pooled) = +0.0000 [+0.0000, +0.0000], p=1; grouped Δ = +0.0000 [+0.0000, +0.0000], p=1; macro-boot Δ = +0.0000 [+0.0000, +0.0000], p=1.
- **zscore**: ΔP@5 (pooled) = +0.0000 [+0.0000, +0.0000], p=1; grouped Δ = +0.0000 [+0.0000, +0.0000], p=1; macro-boot Δ = +0.0000 [+0.0000, +0.0000], p=1.

## combMAX vs prior (additive)

- **minmax**: ΔP@5 (pooled) = +0.0000 [+0.0000, +0.0000], p=1; grouped Δ = +0.0000 [+0.0000, +0.0000], p=1; macro-boot Δ = +0.0000 [+0.0000, +0.0000], p=1.
- **zscore**: ΔP@5 (pooled) = +0.0000 [+0.0000, +0.0000], p=1; grouped Δ = +0.0000 [+0.0000, +0.0000], p=1; macro-boot Δ = +0.0000 [+0.0000, +0.0000], p=1.

## Verdict
adaptive_vs_best_fixed=TIE-HOLDS; combMAX_zscore_survives=False

- adaptive vs best-fixed: **TIE-HOLDS**.

- combMAX z-score win survives: **False**.
