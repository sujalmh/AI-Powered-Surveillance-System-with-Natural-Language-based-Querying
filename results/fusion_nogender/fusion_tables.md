# Released-subset person fusion results

**Scope.** `camera_0` only, 1,778 shipped 1152-d descriptors, 1390 in the test track split. Same-track labels are deterministic pseudo-labels from detector/attribute metadata (no visual cosine), not human identities. Threshold-free pair AUC is the primary metric.

- Person descriptors: 1778
- Pseudo-tracks (metadata-only formation): 274 (205 test / 69 calibration)
- Test retrieval queries: 1325; mean gallery: 1389
- Same-track positives in the retrieval pool: 18868
- Different-track negatives in the retrieval pool: 1821555
- Pairs (metadata-formed): 2016 pairs (test 504+ / 504-; calibration 504+ / 504-)
- Track rule: same camera + same colour + attribute-phrase Jaccard>=0.5, max gap 3 s, max duration 60 s, max track size 20 (fixed rule, no embedding threshold).
- Quality estimators: 1/sqrt(max(1, nearest_scene_frame_person_count)); min(1, n_attribute_phrases/4) (mean qv 0.424, mean qt 0.656).


## Person retrieval (same-track pseudo-labels)

| Method | P@5 | R@10 | F1@10 (P@10,R@10) | F1@10 paper (P@5,R@10) | P@5 95% CI |
|---|---:|---:|---:|---:|---|
| visual | 0.387 | 0.264 | 0.260 | 0.274 | [0.370, 0.403] |
| text | 0.087 | 0.082 | 0.074 | 0.067 | [0.074, 0.101] |
| concat | 0.415 | 0.297 | 0.278 | 0.297 | [0.398, 0.433] |
| gated | 0.437 | 0.296 | 0.283 | 0.308 | [0.420, 0.455] |
| zero_pad | 0.416 | 0.297 | 0.278 | 0.297 | [0.399, 0.434] |
| norm_matched | 0.416 | 0.297 | 0.278 | 0.297 | [0.399, 0.434] |
| rrf | 0.338 | 0.258 | 0.243 | 0.248 | [0.321, 0.355] |
| learned_proj | 0.406 | 0.285 | 0.273 | 0.290 | [0.389, 0.424] |
| randproj | 0.405 ± 0.001 | 0.287 ± 0.001 | 0.270 ± < 0.001 | 0.288 ± < 0.001 | mean±SD over seeds |

`±` is the standard deviation over the 5 random-projection seeds; a spread below 0.001 is shown as `< 0.001` (it is not zero). The raw standard deviations are in `fusion_results.json`.

## Person-pair classification (threshold tuned on calibration pairs)

| Method | Threshold | AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| visual | 0.907 | 0.654 | 0.570 | 0.543 | 0.891 | 0.675 |
| text | 0.930 | 0.668 | 0.667 | 0.606 | 0.954 | 0.741 |
| concat | 0.919 | 0.715 | 0.650 | 0.597 | 0.925 | 0.725 |
| gated | 0.889 | 0.721 | 0.665 | 0.608 | 0.931 | 0.735 |
| zero_pad | 0.918 | 0.715 | 0.654 | 0.600 | 0.923 | 0.727 |
| norm_matched | 0.918 | 0.715 | 0.654 | 0.600 | 0.923 | 0.727 |
| rrf | 0.017 | 0.702 | 0.624 | 0.670 | 0.488 | 0.565 |
| learned_proj | 0.973 | 0.667 | 0.588 | 0.555 | 0.887 | 0.683 |
| randproj | 0.941 | 0.715 ± 0.001 | 0.662 ± 0.003 | 0.606 ± 0.001 | 0.925 ± 0.007 | 0.732 ± 0.003 |

## Paired bootstrap vs concat (Holm over competitors)

P-values are two-sided paired bootstrap with 3000 resamples. `< 0.0007` is the resolution floor (2/3000): no resample fell on the null side, so the p-value is below the floor and not exactly zero. The raw p-values are in `fusion_results.json`.

| Task | Method | Δ [95% CI] | raw p | Holm p | reject |
|---|---|---:|---:|---:|---|
| retrieval P@5 | visual | -0.028 [-0.037, -0.020] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | text | -0.328 [-0.346, -0.308] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | gated | +0.022 [+0.011, +0.032] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | zero_pad | +0.001 [-0.000, +0.002] | 0.1860 | 0.3720 | False |
| retrieval P@5 | norm_matched | +0.001 [-0.000, +0.002] | 0.1860 | 0.3720 | False |
| retrieval P@5 | learned_proj | -0.009 [-0.014, -0.003] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | rrf | -0.077 [-0.089, -0.065] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | randproj | -0.010 [-0.014, -0.005] | < 0.0007 | < 0.0007 | True |
| pair AUC | visual | -0.061 [-0.079, -0.044] | < 0.0007 | < 0.0007 | True |
| pair AUC | text | -0.047 [-0.075, -0.017] | 0.0027 | 0.0160 | True |
| pair AUC | zero_pad | +0.000 [-0.001, +0.001] | 0.4240 | 1.0000 | False |
| pair AUC | norm_matched | +0.000 [-0.001, +0.001] | 0.4240 | 1.0000 | False |
| pair AUC | learned_proj | -0.048 [-0.062, -0.034] | < 0.0007 | < 0.0007 | True |
| pair AUC | gated | +0.006 [-0.015, +0.026] | 0.6013 | 1.0000 | False |
| pair AUC | rrf | -0.013 [-0.034, +0.008] | 0.2367 | 1.0000 | False |
| pair AUC | randproj | -0.000 [-0.010, +0.010] | 0.9680 | 1.0000 | False |

## Grouped bootstrap by pseudo-track (P@5)

| Method | Mean | 95% CI |
|---|---:|---|
| concat | 0.415 | [0.370, 0.457] |
| gated | 0.437 | [0.391, 0.479] |
| visual | 0.387 | [0.342, 0.427] |
| text | 0.087 | [0.043, 0.139] |
| rrf | 0.338 | [0.287, 0.387] |
| learned_proj | 0.406 | [0.363, 0.447] |

## K-fold over pseudo-tracks (P@5, k=5)

Whole-track 5-fold CV (round-robin over tracks sorted by start time; gallery restricted to the held-out fold). Randproj uses seed 0 only; the 5-seed spread is in the main table above.

| Method | Mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| visual | 0.671 | 0.029 | 0.638 | 0.711 |
| text | 0.278 | 0.034 | 0.228 | 0.324 |
| concat | 0.695 | 0.028 | 0.660 | 0.736 |
| gated | 0.677 | 0.034 | 0.625 | 0.722 |
| zero_pad | 0.694 | 0.029 | 0.655 | 0.737 |
| norm_matched | 0.694 | 0.029 | 0.655 | 0.737 |
| learned_proj | 0.688 | 0.032 | 0.643 | 0.731 |
| randproj_s0 | 0.675 | 0.028 | 0.646 | 0.721 |
| rrf | 0.608 | 0.015 | 0.590 | 0.630 |

## Gated vs static vs RRF head-to-head (Holm over 2 per task)

| Task | Comparison | Δ [95% CI] | raw p | Holm p | reject |
|---|---|---:|---:|---:|---|
| retrieval P@5 | gated vs concat | +0.022 [+0.011, +0.032] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | gated vs rrf | +0.099 [+0.084, +0.114] | < 0.0007 | < 0.0007 | True |
| pair AUC | gated vs concat | +0.006 [-0.015, +0.026] | 0.5293 | 0.5293 | False |
| pair AUC | gated vs rrf | +0.019 [-0.009, +0.046] | 0.1873 | 0.3747 | False |

## Held-out temporal split (pair AUC by early/late test half)

Test pairs halved by gallery-timestamp median; same calibration thresholds as the main table. Point estimates (CIs are in the main pair-AUC table).

| Half | visual | text | concat | gated | zero_pad | norm_matched | rrf | learned_proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early (n=504) | 0.626 | 0.584 | 0.627 | 0.615 | 0.627 | 0.627 | 0.638 | 0.628 |
| late (n=504) | 0.681 | 0.767 | 0.807 | 0.817 | 0.806 | 0.806 | 0.764 | 0.708 |

randproj seed-mean test AUC: 0.715 (± 0.001).

## Visual-formation sensitivity (point estimates, biased labelling)

Same methods evaluated on visual-cosine pseudo-tracks (174 tracks, 1255 queries). The visual branch is favoured by construction here; these numbers exist only to show the bias direction versus the metadata-formed primary table.

| Method | Retrieval P@5 | Pair AUC |
|---|---:|---:|
| visual | 0.484 | 0.744 |
| text | 0.080 | 0.615 |
| concat | 0.455 | 0.693 |
| gated | 0.473 | 0.651 |
| zero_pad | 0.456 | 0.693 |
| norm_matched | 0.456 | 0.693 |
| rrf | n/a | 0.749 |
| learned_proj | 0.485 | 0.720 |
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
