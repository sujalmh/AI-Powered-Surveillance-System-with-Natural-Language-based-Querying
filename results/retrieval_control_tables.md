# Released-subset retrieval results (protocol-matched control)

**Scope.** Single camera (`camera_0`), 121 indexed clips / 242 scene frames, 1,778 person descriptors, 2026-03-01 17:01--18:56. These are **not** the paper's full 4-camera / 312k-descriptor results.

**What changed vs. the raw released benchmark.** Count queries carry an explicit target count in the text (`show clips with exactly N people`), and action queries are event-only with no time cue. Find/track/filter rows are unchanged. This removes the two protocol mismatches that make the raw benchmark unfavourable to the paper's override logic. It is a post-hoc control, not a human-annotated benchmark.

Adaptive alpha source: `backend.app.services.result_merger.compute_adaptive_alpha`. Bootstrap resamples: 3000. Tune/test: 50/100 queries.


## Macro results (min-max branch normalisation)

| Method | P@5 | R@10 | F1@10 (P@10,R@10) | F1@10 paper (P@5,R@10) | MRR | P@5 grouped 95% CI |
|---|---:|---:|---:|---:|---:|---|
| fixed_0.0 | 0.048 | 0.093 | 0.053 | 0.046 | 0.144 | [0.016, 0.092] |
| fixed_0.3 | 0.380 | 0.597 | 0.337 | 0.417 | 0.621 | [0.202, 0.569] |
| fixed_0.5 | 0.534 | 0.793 | 0.446 | 0.584 | 0.841 | [0.345, 0.689] |
| fixed_0.7 | 0.538 | 0.798 | 0.448 | 0.588 | 0.853 | [0.349, 0.692] |
| fixed_1.0 | 0.520 | 0.787 | 0.443 | 0.572 | 0.824 | [0.309, 0.689] |
| adaptive | 0.538 | 0.797 | 0.447 | 0.588 | 0.853 | [0.349, 0.692] |
| per_intent_tuned | 0.536 | 0.793 | 0.446 | 0.585 | 0.841 | [0.346, 0.691] |
| paper_lookup | 0.428 | 0.696 | 0.365 | 0.483 | 0.763 | [0.251, 0.606] |

Best fixed alpha selected on the tune split: `fixed_0.7` (tune macro P@5 0.580). Oracle fixed on test (descriptive only): `fixed_0.7`.

## Per-category P@5 (min-max)

| Method | Count | Find | Track | Action | Filter | Macro |
|---|---:|---:|---:|---:|---:|---:|
| fixed_0.0 | 0.010 | 0.070 | 0.020 | 0.090 | 0.050 | 0.048 |
| fixed_0.3 | 0.270 | 0.630 | 0.550 | 0.090 | 0.360 | 0.380 |
| fixed_0.5 | 0.390 | 0.910 | 0.810 | 0.090 | 0.470 | 0.534 |
| fixed_0.7 | 0.390 | 0.920 | 0.810 | 0.090 | 0.480 | 0.538 |
| fixed_1.0 | 0.390 | 0.920 | 0.810 | 0.000 | 0.480 | 0.520 |
| adaptive | 0.390 | 0.920 | 0.810 | 0.090 | 0.480 | 0.538 |
| per_intent_tuned | 0.390 | 0.920 | 0.810 | 0.090 | 0.470 | 0.536 |
| paper_lookup | 0.390 | 0.630 | 0.550 | 0.090 | 0.480 | 0.428 |

## Count queries

The control's count queries are count-constrained retrieval questions (`exactly N people`), so the reported metric is P@5 over the clips whose caption count equals N; EM/MAE of a *free* count answer is reported by the raw released benchmark instead.


## Min-max vs z-score branch normalisation (macro P@5)

| Method | min-max | z-score | Δ (z - mm) |
|---|---:|---:|---:|
| fixed_0.0 | 0.048 | 0.048 | +0.000 |
| fixed_0.3 | 0.380 | 0.268 | -0.112 |
| fixed_0.5 | 0.534 | 0.450 | -0.084 |
| fixed_0.7 | 0.538 | 0.522 | -0.016 |
| fixed_1.0 | 0.520 | 0.520 | +0.000 |
| adaptive | 0.538 | 0.524 | -0.014 |
| per_intent_tuned | 0.536 | 0.534 | -0.002 |
| paper_lookup | 0.428 | 0.366 | -0.062 |

Adaptive z-score vs min-max paired ΔP@5 = +0.014 [+0.004, +0.026], p=0.0000.


## Adaptive vs tune-selected best fixed-alpha

- **minmax**: `adaptive vs fixed_0.7 (fixed alpha selected on tune split)`. ΔP@5 = +0.000 [+0.000, +0.000], p=1.0000; grouped Δ = +0.000 [+0.000, +0.000], p=1.0000.
  - count: Δ=+0.000 [+0.000, +0.000], p=1.0000, Holm-reject=False
  - find: Δ=+0.000 [+0.000, +0.000], p=1.0000, Holm-reject=False
  - track: Δ=+0.000 [+0.000, +0.000], p=1.0000, Holm-reject=False
  - action: Δ=+0.000 [+0.000, +0.000], p=1.0000, Holm-reject=False
  - filter: Δ=+0.000 [+0.000, +0.000], p=1.0000, Holm-reject=False
  - per-intent tuned vs adaptive: Δ=-0.002 [-0.006, +0.000], p=0.7253
- **zscore**: `adaptive vs fixed_0.7 (fixed alpha selected on tune split)`. ΔP@5 = +0.002 [+0.000, +0.006], p=0.7187; grouped Δ = +0.002 [+0.000, +0.007], p=0.7113.
  - count: Δ=+0.010 [+0.000, +0.030], p=0.7353, Holm-reject=False
  - find: Δ=+0.000 [+0.000, +0.000], p=1.0000, Holm-reject=False
  - track: Δ=+0.000 [+0.000, +0.000], p=1.0000, Holm-reject=False
  - action: Δ=+0.000 [+0.000, +0.000], p=1.0000, Holm-reject=False
  - filter: Δ=+0.000 [+0.000, +0.000], p=1.0000, Holm-reject=False
  - per-intent tuned vs adaptive: Δ=+0.010 [-0.000, +0.020], p=0.0773

## Per-intent alpha selected on the tune split

| Normalisation | Count | Find | Track | Action | Filter |
|---|---:|---:|---:|---:|---:|
| minmax | 0.5 | 0.6 | 0.5 | 0.5 | 0.5 |
| zscore | 0.7 | 0.9 | 0.8 | 0.5 | 0.9 |

## Realized adaptive alpha

| Category | Mean | Min | Max |
|---|---:|---:|---:|
| count | 1.000 | 1.000 | 1.000 |
| find | 0.651 | 0.596 | 0.659 |
| track | 0.622 | 0.622 | 0.622 |
| action | 0.100 | 0.100 | 0.100 |
| filter | 0.721 | 0.646 | 0.742 |

## Semantic aggregation sensitivity

| Method | max over frames | mean over frames |
|---|---:|---:|
| fixed_0.5 | 0.534 | 0.526 |
| adaptive | 0.538 | 0.536 |

## What cannot be reproduced

- Post-hoc control: the count/action query text was rewritten to remove the two protocol mismatches of the raw released benchmark; find/track/filter rows are unchanged.
- Single camera (camera_0) and a 2-hour released window; cross-camera retrieval and identity tracking are untestable.
- Labels are derived from released VLM captions, detector metadata and snapshots, not human annotations; no IAA is available.
- Two clips on disk are not in the scene FAISS index (121 indexed of 123); evaluation uses the indexed clips.
- Action relevance uses the released VLM caption events (fighting/smoke); no running/fighting labels are invented and the motion-energy proxy rows are excluded.
- Track queries are single-camera colour-continuity proxies, not ground-truth identities.
- The full-deployment 0.666->0.828 / 0.785->0.807 values are not reproduced and must not be replaced by these subset numbers.
