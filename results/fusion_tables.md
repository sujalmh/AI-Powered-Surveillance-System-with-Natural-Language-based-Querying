# Released-subset person fusion results

**Scope.** `camera_0` only, 1,778 shipped 1152-d descriptors, 1270 in the test track split. Same-track labels are deterministic pseudo-labels from detector/attribute metadata (no visual cosine), not human identities. Threshold-free pair AUC is the primary metric.

- Person descriptors: 1778
- Pseudo-tracks (metadata-only formation): 280 (210 test / 70 calibration)
- Test retrieval queries: 1190; mean gallery: 1269
- Same-track positives in the retrieval pool: 16956
- Different-track negatives in the retrieval pool: 1493154
- Pairs (metadata-formed): 2016 pairs (test 504+ / 504-; calibration 504+ / 504-)
- Track rule: same camera + same colour + attribute-phrase Jaccard>=0.5, max gap 3 s, max duration 60 s, max track size 20 (fixed rule, no embedding threshold).
- Quality estimators: 1/sqrt(max(1, nearest_scene_frame_person_count)); min(1, n_attribute_phrases/4) (mean qv 0.424, mean qt 0.666).


## Person retrieval (same-track pseudo-labels)

| Method | P@5 | R@10 | F1@10 (P@10,R@10) | F1@10 paper (P@5,R@10) | P@5 95% CI |
|---|---:|---:|---:|---:|---|
| visual | 0.407 | 0.285 | 0.276 | 0.290 | [0.390, 0.424] |
| text | 0.102 | 0.095 | 0.084 | 0.077 | [0.089, 0.116] |
| concat | 0.427 | 0.311 | 0.289 | 0.309 | [0.410, 0.445] |
| gated | 0.444 | 0.307 | 0.289 | 0.315 | [0.425, 0.462] |
| zero_pad | 0.428 | 0.312 | 0.289 | 0.309 | [0.410, 0.446] |
| norm_matched | 0.428 | 0.312 | 0.289 | 0.309 | [0.410, 0.446] |
| rrf | 0.361 | 0.285 | 0.262 | 0.266 | [0.343, 0.378] |
| learned_proj | 0.432 | 0.311 | 0.290 | 0.312 | [0.415, 0.450] |
| randproj | 0.417 ± < 0.001 | 0.297 ± 0.001 | 0.278 ± < 0.001 | 0.298 ± < 0.001 | mean±SD over seeds |

`±` is the standard deviation over the 5 random-projection seeds; a spread below 0.001 is shown as `< 0.001` (it is not zero). The raw standard deviations are in `fusion_results.json`.

## Person-pair classification (threshold tuned on calibration pairs)

| Method | Threshold | AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| visual | 0.937 | 0.686 | 0.605 | 0.572 | 0.835 | 0.679 |
| text | 0.853 | 0.681 | 0.637 | 0.581 | 0.980 | 0.730 |
| concat | 0.930 | 0.726 | 0.669 | 0.612 | 0.919 | 0.735 |
| gated | 0.868 | 0.711 | 0.651 | 0.596 | 0.935 | 0.728 |
| zero_pad | 0.926 | 0.727 | 0.663 | 0.608 | 0.919 | 0.731 |
| norm_matched | 0.926 | 0.727 | 0.663 | 0.608 | 0.919 | 0.731 |
| rrf | 0.014 | 0.720 | 0.641 | 0.625 | 0.702 | 0.662 |
| learned_proj | 0.980 | 0.709 | 0.638 | 0.596 | 0.857 | 0.703 |
| randproj | 0.881 | 0.711 ± 0.001 | 0.642 ± 0.001 | 0.585 ± 0.001 | 0.979 ± 0.001 | 0.732 ± 0.001 |

## Paired bootstrap vs concat (Holm over competitors)

P-values are two-sided paired bootstrap with 3000 resamples. `< 0.0007` is the resolution floor (2/3000): no resample fell on the null side, so the p-value is below the floor and not exactly zero. The raw p-values are in `fusion_results.json`.

| Task | Method | Δ [95% CI] | raw p | Holm p | reject |
|---|---|---:|---:|---:|---|
| retrieval P@5 | visual | -0.021 [-0.030, -0.011] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | text | -0.325 [-0.345, -0.306] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | gated | +0.017 [+0.006, +0.027] | 0.0013 | 0.0053 | True |
| retrieval P@5 | zero_pad | +0.001 [-0.001, +0.003] | 0.3473 | 0.6947 | False |
| retrieval P@5 | norm_matched | +0.001 [-0.001, +0.003] | 0.3473 | 0.6947 | False |
| retrieval P@5 | learned_proj | +0.005 [-0.000, +0.010] | 0.0713 | 0.2140 | False |
| retrieval P@5 | rrf | -0.067 [-0.079, -0.054] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | randproj | -0.011 [-0.015, -0.007] | < 0.0007 | < 0.0007 | True |
| pair AUC | visual | -0.041 [-0.061, -0.022] | < 0.0007 | < 0.0007 | True |
| pair AUC | text | -0.045 [-0.075, -0.016] | 0.0040 | 0.0280 | True |
| pair AUC | zero_pad | +0.000 [-0.001, +0.002] | 0.4020 | 1.0000 | False |
| pair AUC | norm_matched | +0.000 [-0.001, +0.002] | 0.4020 | 1.0000 | False |
| pair AUC | learned_proj | -0.017 [-0.030, -0.006] | 0.0067 | 0.0400 | True |
| pair AUC | gated | -0.015 [-0.039, +0.010] | 0.2453 | 0.9813 | False |
| pair AUC | rrf | -0.006 [-0.026, +0.014] | 0.5333 | 1.0000 | False |
| pair AUC | randproj | -0.015 [-0.026, -0.004] | 0.0140 | 0.0700 | False |

## Grouped bootstrap by pseudo-track (P@5)

| Method | Mean | 95% CI |
|---|---:|---|
| concat | 0.427 | [0.379, 0.476] |
| gated | 0.444 | [0.396, 0.493] |
| visual | 0.407 | [0.358, 0.453] |
| text | 0.102 | [0.058, 0.153] |
| rrf | 0.361 | [0.303, 0.417] |
| learned_proj | 0.432 | [0.382, 0.481] |

## K-fold over pseudo-tracks (P@5, k=5)

Whole-track 5-fold CV (round-robin over tracks sorted by start time; gallery restricted to the held-out fold). Randproj uses seed 0 only; the 5-seed spread is in the main table above.

| Method | Mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| visual | 0.675 | 0.021 | 0.637 | 0.696 |
| text | 0.298 | 0.031 | 0.270 | 0.359 |
| concat | 0.699 | 0.033 | 0.643 | 0.729 |
| gated | 0.664 | 0.034 | 0.613 | 0.703 |
| zero_pad | 0.699 | 0.033 | 0.641 | 0.728 |
| norm_matched | 0.699 | 0.033 | 0.641 | 0.728 |
| learned_proj | 0.697 | 0.030 | 0.642 | 0.726 |
| randproj_s0 | 0.667 | 0.038 | 0.608 | 0.719 |
| rrf | 0.624 | 0.036 | 0.574 | 0.678 |

## Gated vs static vs RRF head-to-head (Holm over 2 per task)

| Task | Comparison | Δ [95% CI] | raw p | Holm p | reject |
|---|---|---:|---:|---:|---|
| retrieval P@5 | gated vs concat | +0.017 [+0.007, +0.027] | 0.0007 | 0.0007 | True |
| retrieval P@5 | gated vs rrf | +0.084 [+0.069, +0.098] | < 0.0007 | < 0.0007 | True |
| pair AUC | gated vs concat | -0.015 [-0.038, +0.008] | 0.2127 | 0.4253 | False |
| pair AUC | gated vs rrf | -0.009 [-0.037, +0.019] | 0.5540 | 0.5540 | False |

## Held-out temporal split (pair AUC by early/late test half)

Test pairs halved by gallery-timestamp median; same calibration thresholds as the main table. Point estimates (CIs are in the main pair-AUC table).

| Half | visual | text | concat | gated | zero_pad | norm_matched | rrf | learned_proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early (n=506) | 0.670 | 0.579 | 0.656 | 0.632 | 0.658 | 0.658 | 0.651 | 0.672 |
| late (n=502) | 0.703 | 0.762 | 0.795 | 0.793 | 0.795 | 0.795 | 0.783 | 0.746 |

randproj seed-mean test AUC: 0.711 (± 0.001).

## Visual-formation sensitivity (point estimates, biased labelling)

Same methods evaluated on visual-cosine pseudo-tracks (174 tracks, 1255 queries). The visual branch is favoured by construction here; these numbers exist only to show the bias direction versus the metadata-formed primary table.

| Method | Retrieval P@5 | Pair AUC |
|---|---:|---:|
| visual | 0.484 | 0.744 |
| text | 0.090 | 0.620 |
| concat | 0.451 | 0.693 |
| gated | 0.460 | 0.652 |
| zero_pad | 0.450 | 0.692 |
| norm_matched | 0.450 | 0.692 |
| rrf | n/a | 0.749 |
| learned_proj | 0.479 | 0.721 |
| randproj | n/a | n/a |

## Provenance (public Re-ID probe)

- Probe resolved: `False`.
  - `S:\college\ai-surveillance\data\reid\Market-1501-v15.09.15`: exists=False, market_layout=False
  - `S:\college\ai-surveillance\data\reid\MSMT17`: exists=False, market_layout=False
  - `S:\college\ai-surveillance\data\reid`: exists=False, market_layout=False
  - `S:\college\ai-surveillance\..\Market-1501-v15.09.15`: exists=False, market_layout=False
- No verifiable public Re-ID source on local disk; download not attempted (offline, CPU-only). Fell back to a held-out temporal split (early/late halves by test-timestamp median) of the released data, reported as temporal_halves.

## What cannot be reproduced

- Single camera (camera_0); cross-camera identity and generalisation are untestable.
- Pseudo-track labels, not human identity annotations; no IAA.
- Metadata-only formation removes the visual-cosine bias by construction, but colour/attribute matching has its own failure modes (colour changes, sparse attributes); reported as pseudo-labels, not identities.
- Pair thresholds are tuned on calibration pairs; test pairs are track-disjoint but come from the same two-hour window.
- The held-out temporal split is early/late halves of the same two-hour window, not an independent deployment; no public Re-ID source resolved offline.
- The paper's 312k-descriptor / 4-camera 0.785->0.807 fusion result is not reproduced and is not replaced by these numbers.
- The learned projection is a linear ridge baseline, not a jointly trained Re-ID model.
- Quality estimators (occlusion proxy, attribute richness) are unvalidated proxies from released metadata, not measured image quality.
