# Released-subset retrieval results

**Scope.** Single camera (`camera_0`), 121 indexed clips / 242 scene frames, 1,778 person descriptors, 2026-03-01 17:01--18:56. These are **not** the paper's full 4-camera / 312k-descriptor results.

Adaptive alpha source: `backend.app.services.result_merger.compute_adaptive_alpha`. Bootstrap resamples: 3000. Tune/test: 50/100 queries.


## Macro results (min-max branch normalisation)

| Method | P@5 | R@10 | F1@10 (P@10,R@10) | F1@10 paper (P@5,R@10) | MRR | P@5 grouped 95% CI |
|---|---:|---:|---:|---:|---:|---|
| fixed_0.0 | 0.036 | 0.108 | 0.045 | 0.039 | 0.127 | [0.008, 0.079] |
| fixed_0.3 | 0.378 | 0.604 | 0.336 | 0.419 | 0.636 | [0.208, 0.546] |
| fixed_0.5 | 0.560 | 0.831 | 0.467 | 0.612 | 0.819 | [0.459, 0.654] |
| fixed_0.7 | 0.568 | 0.906 | 0.493 | 0.623 | 0.832 | [0.464, 0.664] |
| fixed_1.0 | 0.578 | 0.954 | 0.505 | 0.639 | 0.787 | [0.481, 0.665] |
| adaptive | 0.486 | 0.825 | 0.440 | 0.531 | 0.702 | [0.371, 0.584] |
| per_intent_tuned | 0.564 | 0.915 | 0.491 | 0.617 | 0.832 | [0.464, 0.657] |
| paper_lookup | 0.370 | 0.685 | 0.347 | 0.415 | 0.593 | [0.222, 0.517] |

Best fixed alpha selected on the tune split: `fixed_0.7` (tune macro P@5 0.572). Oracle fixed on test (descriptive only): `fixed_1.0`.

## Per-category P@5 (min-max)

| Method | Count | Find | Track | Action | Filter | Macro |
|---|---:|---:|---:|---:|---:|---:|
| fixed_0.0 | 0.010 | 0.070 | 0.020 | 0.030 | 0.050 | 0.036 |
| fixed_0.3 | 0.080 | 0.630 | 0.550 | 0.270 | 0.360 | 0.378 |
| fixed_0.5 | 0.120 | 0.910 | 0.810 | 0.490 | 0.470 | 0.560 |
| fixed_0.7 | 0.120 | 0.920 | 0.810 | 0.510 | 0.480 | 0.568 |
| fixed_1.0 | 0.160 | 0.920 | 0.810 | 0.520 | 0.480 | 0.578 |
| adaptive | 0.160 | 0.920 | 0.810 | 0.060 | 0.480 | 0.486 |
| per_intent_tuned | 0.130 | 0.920 | 0.810 | 0.490 | 0.470 | 0.564 |
| paper_lookup | 0.160 | 0.630 | 0.550 | 0.030 | 0.480 | 0.370 |

## Count EM / MAE (test count queries)

| Method | n | EM | MAE |
|---|---:|---:|---:|
| fixed_0.0 | 20 | 0.150 | 6.050 |
| fixed_0.3 | 20 | 0.050 | 4.200 |
| fixed_0.5 | 20 | 0.100 | 2.200 |
| fixed_0.7 | 20 | 0.100 | 2.100 |
| fixed_1.0 | 20 | 0.100 | 2.050 |
| adaptive | 20 | 0.100 | 2.050 |
| per_intent_tuned | 20 | 0.100 | 2.100 |
| paper_lookup | 20 | 0.100 | 2.050 |

Structured window-count estimate (mode of caption counts) vs anchor-frame truth: EM 0.100, MAE 2.200 (n=30).


## Min-max vs z-score branch normalisation (macro P@5)

| Method | min-max | z-score | Δ (z - mm) |
|---|---:|---:|---:|
| fixed_0.0 | 0.036 | 0.036 | +0.000 |
| fixed_0.3 | 0.378 | 0.274 | -0.104 |
| fixed_0.5 | 0.560 | 0.470 | -0.090 |
| fixed_0.7 | 0.568 | 0.558 | -0.010 |
| fixed_1.0 | 0.578 | 0.578 | +0.000 |
| adaptive | 0.486 | 0.470 | -0.016 |
| per_intent_tuned | 0.564 | 0.566 | +0.002 |
| paper_lookup | 0.370 | 0.308 | -0.062 |

Adaptive z-score vs min-max paired ΔP@5 = +0.016 [+0.006, +0.028], p=< 0.0007.


## Adaptive vs tune-selected best fixed-alpha

P-values are two-sided paired bootstrap with 3000 resamples. `< 0.0007` is the resolution floor (2/3000): no resample fell on the null side, so the p-value is below the floor and not exactly zero. The raw p-values are in `retrieval_results.json`.

- **minmax**: `adaptive vs fixed_0.7 (fixed alpha selected on tune split)`. ΔP@5 = -0.082 [-0.134, -0.038], p=< 0.0007; grouped Δ = -0.082 [-0.190, -0.011], p=0.0067.
  - count: Δ=+0.040 [-0.020, +0.100], p=0.2633 (Holm p=1.0000), Holm-reject=False
  - find: Δ=+0.000 [+0.000, +0.000], p=1.0000 (Holm p=1.0000), Holm-reject=False
  - track: Δ=+0.000 [+0.000, +0.000], p=1.0000 (Holm p=1.0000), Holm-reject=False
  - action: Δ=-0.450 [-0.600, -0.300], p=< 0.0007 (Holm p=< 0.0007), Holm-reject=True
  - filter: Δ=+0.000 [+0.000, +0.000], p=1.0000 (Holm p=1.0000), Holm-reject=False
  - per-intent tuned vs adaptive: Δ=+0.078 [+0.032, +0.128], p=0.0007
- **zscore**: `adaptive vs fixed_1.0 (fixed alpha selected on tune split)`. ΔP@5 = -0.108 [-0.158, -0.064], p=< 0.0007; grouped Δ = -0.108 [-0.213, -0.039], p=< 0.0007.
  - count: Δ=+0.000 [+0.000, +0.000], p=1.0000 (Holm p=1.0000), Holm-reject=False
  - find: Δ=-0.010 [-0.030, +0.000], p=0.7267 (Holm p=1.0000), Holm-reject=False
  - track: Δ=-0.010 [-0.030, +0.000], p=0.7193 (Holm p=1.0000), Holm-reject=False
  - action: Δ=-0.470 [-0.620, -0.330], p=< 0.0007 (Holm p=< 0.0007), Holm-reject=True
  - filter: Δ=-0.050 [-0.090, -0.010], p=0.0100 (Holm p=0.0400), Holm-reject=True
  - per-intent tuned vs adaptive: Δ=+0.096 [+0.048, +0.146], p=< 0.0007

## Per-intent alpha selected on the tune split

| Normalisation | Count | Find | Track | Action | Filter |
|---|---:|---:|---:|---:|---:|
| minmax | 0.9 | 0.6 | 0.5 | 0.5 | 0.5 |
| zscore | 0.9 | 0.9 | 0.8 | 0.6 | 0.9 |

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
| fixed_0.5 | 0.560 | 0.556 |
| adaptive | 0.486 | 0.478 |

## What cannot be reproduced

- PROTOCOL NOTE: this raw benchmark is a code-path smoke test, not the benchmark quoted in the paper. It is not protocol-matched to the deployment C1 study: count questions are free 'how many' questions with no target count (so the structured branch has no numeric target to rank on) and action queries carry explicit time windows (so the structured branch has a time cue that the action override suppresses). The number quoted in the paper comes from the protocol-matched control, backend/scripts/eval_retrieval_control.py (results/retrieval_control_tables.md).
- Single camera (camera_0) and a 2-hour released window; cross-camera retrieval and identity tracking are untestable.
- Labels are derived from released VLM captions, detector metadata and snapshots, not human annotations; no IAA is available.
- Two clips on disk are not in the scene FAISS index (121 indexed of 123); evaluation uses the indexed clips.
- Action queries use 15 weak VLM caption events (fighting/smoke) plus 15 motion-energy proxy queries; no running/fighting labels are invented.
- Track queries are single-camera colour-continuity proxies, not ground-truth identities.
- The full-deployment 0.666->0.828 / 0.785->0.807 values are not reproduced and must not be replaced by these subset numbers.
