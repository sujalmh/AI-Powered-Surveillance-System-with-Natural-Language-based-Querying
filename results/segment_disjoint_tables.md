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

- held_out_segment: 1
- held_out_segment_name: S2 (18:04-18:28)
- train_segments: [0, 2]
- train_segment_names: ['S1 (17:01-17:08)', 'S3 (18:55-18:56)']
- n_train_clips: 31
- n_test_clips: 90
- n_shared_clips_train_test: 0
- n_tune_queries: 31
- n_test_queries: 99
- n_cross_segment_queries_total: 20
- n_queries_dropped_for_disjointness: 20
- n_tune_relevant_clips: 19
- n_test_relevant_clips: 81
- n_shared_relevant_clips: 0
- shared_relevant_pct_of_test: 0.0
- clip_disjoint: True


Held-out segment: S2 (18:04-18:28) (index 1). Train queries: 31, test queries: 99.


## Macro P@5 on held-out segment


### minmax

| Method | Macro P@5 | R@10 | F1@10 | MRR |
|---|---:|---:|---:|---:|
| adaptive | 0.5831 | 0.8100 | 0.6204 | 0.8714 |
| combMAX | 0.5831 | 0.8108 | 0.6208 | 0.8714 |
| fixed_0.5 | 0.5780 | 0.8068 | 0.6163 | 0.8666 |
| per_intent_tuned | 0.5780 | 0.8068 | 0.6163 | 0.8666 |
| fixed_0.0 | 0.0688 | 0.1388 | 0.0724 | 0.1855 |
| fixed_0.3 | 0.4960 | 0.7144 | 0.5219 | 0.7314 |
| fixed_0.7 | 0.5831 | 0.8108 | 0.6208 | 0.8714 |
| fixed_1.0 | 0.5511 | 0.7628 | 0.5888 | 0.8065 |

Best fixed alpha selected on tune: `fixed_0.5` (tune macro P@5 0.5226).


### zscore

| Method | Macro P@5 | R@10 | F1@10 | MRR |
|---|---:|---:|---:|---:|
| adaptive | 0.5633 | 0.8029 | 0.5977 | 0.8125 |
| combMAX | 0.5831 | 0.8108 | 0.6208 | 0.8714 |
| fixed_0.9 | 0.5831 | 0.8100 | 0.6204 | 0.8714 |
| per_intent_tuned | 0.5708 | 0.8033 | 0.6056 | 0.8109 |
| fixed_0.0 | 0.0688 | 0.1388 | 0.0724 | 0.1855 |
| fixed_0.3 | 0.3668 | 0.4891 | 0.3724 | 0.6101 |
| fixed_0.5 | 0.5201 | 0.7253 | 0.5454 | 0.7275 |
| fixed_0.7 | 0.5612 | 0.8039 | 0.5946 | 0.7824 |
| fixed_1.0 | 0.5511 | 0.7628 | 0.5888 | 0.8065 |

Best fixed alpha selected on tune: `fixed_0.9` (tune macro P@5 0.5226).


## Adaptive vs best-fixed

- **minmax**: ΔP@5 (pooled) = +0.0061 [+0.0000, +0.0141], p=0.1093; grouped Δ = +0.0061 [+0.0000, +0.0196], p=0.6767; macro-boot Δ = +0.0051 [+0.0000, +0.0115], p=0.09467.
- **zscore**: ΔP@5 (pooled) = -0.0222 [-0.0364, -0.0101], p=0; grouped Δ = -0.0222 [-0.0454, -0.0053], p=0.006667; macro-boot Δ = -0.0198 [-0.0316, -0.0092], p=0.

## combMAX vs prior (additive)

- **minmax**: ΔP@5 (pooled) = +0.0000 [+0.0000, +0.0000], p=1; grouped Δ = +0.0000 [+0.0000, +0.0000], p=1; macro-boot Δ = +0.0000 [+0.0000, +0.0000], p=1.
- **zscore**: ΔP@5 (pooled) = +0.0222 [+0.0101, +0.0364], p=0; grouped Δ = +0.0222 [+0.0053, +0.0447], p=0.006; macro-boot Δ = +0.0198 [+0.0093, +0.0315], p=0.

## Verdict
adaptive_vs_best_fixed=TIE-HOLDS; combMAX_zscore_survives=True

- adaptive vs best-fixed: **TIE-HOLDS**.

- combMAX z-score win survives: **True**.
