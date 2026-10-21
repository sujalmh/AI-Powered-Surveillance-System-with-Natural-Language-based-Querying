# Released-subset person fusion results

**Scope.** `camera_0` only, 1,778 shipped 1152-d descriptors, 1292 in the test track split. Same-track labels are deterministic pseudo-labels from visual+temporal matching, not human identities.

- Person descriptors: 1778
- Pseudo-tracks: 174 (130 test / 44 calibration)
- Test retrieval queries: 1255; mean gallery: 1291
- Same-track positives in the retrieval pool: 20180
- Different-track negatives in the retrieval pool: 1600023
- Pair file: 2016 pairs (test 744+ / 744-; calibration 264+ / 264-)
- Track threshold: 0.90 (grid in `fusion_results.json`); max gap 3 s, max duration 60 s, max track size 20.


## Person retrieval (same-track pseudo-labels)

| Method | P@5 | R@10 | F1@10 (P@10,R@10) | F1@10 paper (P@5,R@10) | P@5 95% CI |
|---|---:|---:|---:|---:|---|
| visual | 0.484 | 0.286 | 0.315 | 0.328 | [0.466, 0.502] |
| text | 0.090 | 0.061 | 0.068 | 0.059 | [0.078, 0.102] |
| concat | 0.451 | 0.258 | 0.288 | 0.300 | [0.432, 0.469] |
| zero_pad | 0.450 | 0.258 | 0.288 | 0.300 | [0.431, 0.469] |
| norm_matched | 0.450 | 0.258 | 0.288 | 0.300 | [0.431, 0.469] |
| rrf | 0.395 | 0.233 | 0.261 | 0.264 | [0.376, 0.413] |
| learned_proj | 0.473 | 0.274 | 0.306 | 0.319 | [0.455, 0.491] |
| randproj | 0.436 ± 0.000 | 0.241 ± 0.000 | 0.271 ± 0.000 | 0.286 ± 0.000 | mean±SD over seeds |

## Person-pair classification (threshold tuned on calibration pairs)

| Method | Threshold | AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| visual | 0.923 | 0.744 | 0.667 | 0.612 | 0.911 | 0.733 |
| text | 0.907 | 0.620 | 0.593 | 0.577 | 0.699 | 0.632 |
| concat | 0.940 | 0.693 | 0.628 | 0.616 | 0.680 | 0.646 |
| zero_pad | 0.939 | 0.692 | 0.630 | 0.618 | 0.681 | 0.648 |
| norm_matched | 0.939 | 0.692 | 0.630 | 0.618 | 0.681 | 0.648 |
| rrf | 0.018 | 0.749 | 0.647 | 0.717 | 0.487 | 0.580 |
| learned_proj | 0.965 | 0.718 | 0.636 | 0.584 | 0.944 | 0.721 |
| randproj | 0.934 | 0.664 ± 0.001 | 0.603 ± 0.003 | 0.589 ± 0.001 | 0.678 ± 0.014 | 0.630 ± 0.006 |

## Paired bootstrap vs concat (Holm over competitors)

| Task | Method | Δ [95% CI] | raw p | Holm p | reject |
|---|---|---:|---:|---:|---|
| retrieval P@5 | visual | +0.034 [+0.022, +0.046] | 0.0000 | 0.0000 | True |
| retrieval P@5 | text | -0.361 [-0.379, -0.341] | 0.0000 | 0.0000 | True |
| retrieval P@5 | zero_pad | -0.000 [-0.002, +0.001] | 0.6200 | 1.0000 | False |
| retrieval P@5 | norm_matched | -0.000 [-0.002, +0.001] | 0.6200 | 1.0000 | False |
| retrieval P@5 | learned_proj | +0.023 [+0.015, +0.030] | 0.0000 | 0.0000 | True |
| retrieval P@5 | rrf | -0.056 [-0.069, -0.044] | 0.0000 | 0.0000 | True |
| retrieval P@5 | randproj | -0.015 [-0.020, -0.010] | 0.0000 | 0.0000 | True |
| pair AUC | visual | +0.051 [+0.033, +0.068] | 0.0000 | 0.0000 | True |
| pair AUC | text | -0.073 [-0.089, -0.057] | 0.0000 | 0.0000 | True |
| pair AUC | zero_pad | -0.001 [-0.001, +0.000] | 0.2353 | 0.4707 | False |
| pair AUC | norm_matched | -0.001 [-0.001, +0.000] | 0.2353 | 0.4707 | False |
| pair AUC | learned_proj | +0.025 [+0.013, +0.038] | 0.0000 | 0.0000 | True |
| pair AUC | rrf | +0.056 [+0.039, +0.073] | 0.0000 | 0.0000 | True |
| pair AUC | randproj | -0.028 [-0.035, -0.021] | 0.0000 | 0.0000 | True |

## Grouped bootstrap by pseudo-track (P@5)

| Method | Mean | 95% CI |
|---|---:|---|
| concat | 0.451 | [0.400, 0.497] |
| visual | 0.484 | [0.437, 0.529] |
| text | 0.090 | [0.053, 0.135] |
| rrf | 0.395 | [0.342, 0.446] |
| learned_proj | 0.473 | [0.423, 0.519] |

## What cannot be reproduced

- Single camera (camera_0); cross-camera identity and generalisation are untestable.
- Pseudo-track labels, not human identity annotations; no IAA.
- Tracks were formed with visual-only cosine, so visual-only retrieval is optimistically biased.
- Track threshold is selected on a small calibration block (169 detections) with a self-supervised objective, not identity labels.
- Pair thresholds are tuned on calibration pairs; test pairs are track-disjoint but come from the same two-hour window.
- The paper's 312k-descriptor / 4-camera 0.785->0.807 fusion result is not reproduced and is not replaced by these numbers.
- The learned projection is a linear ridge baseline, not a jointly trained Re-ID model.
