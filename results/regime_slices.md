# Regime slices — person fusion (released subset)

**Scope.** released single-camera subset; not the 4-camera/312k deployment.

## Pre-registration

Frozen strata definitions are embedded in `results/regime_slices.json` under `pre_registration`.

## Retrieval P@5 by stratum

### illumination

| stratum | n | visual | text | concat | gated | zero_pad | norm_matched | learned_proj | rrf | randproj |
|---|---|---|---|---|---|---|---|---|---|---|
| early | 100 | 0.350 | 0.210 | 0.352 | 0.414 | 0.356 | 0.356 | 0.358 | 0.290 | 0.353 |
| late | 1090 | 0.412 | 0.092 | 0.434 | 0.447 | 0.435 | 0.435 | 0.439 | 0.367 | 0.422 |

### scene_density

| stratum | n | visual | text | concat | gated | zero_pad | norm_matched | learned_proj | rrf | randproj |
|---|---|---|---|---|---|---|---|---|---|---|
| high | 500 | 0.410 | 0.070 | 0.439 | 0.467 | 0.440 | 0.440 | 0.444 | 0.345 | 0.434 |
| low | 690 | 0.405 | 0.126 | 0.419 | 0.428 | 0.420 | 0.420 | 0.424 | 0.372 | 0.404 |

### motion_energy

| stratum | n | visual | text | concat | gated | zero_pad | norm_matched | learned_proj | rrf | randproj |
|---|---|---|---|---|---|---|---|---|---|---|
| high | 580 | 0.400 | 0.126 | 0.438 | 0.441 | 0.439 | 0.439 | 0.439 | 0.361 | 0.420 |
| low | 610 | 0.413 | 0.079 | 0.418 | 0.447 | 0.418 | 0.418 | 0.427 | 0.360 | 0.413 |

## Pair AUC by stratum

### illumination

| stratum | n | visual | text | concat | gated | zero_pad | norm_matched | learned_proj | rrf | randproj |
|---|---|---|---|---|---|---|---|---|---|---|
| early | 90 | 0.584 | 0.518 | 0.551 | 0.590 | 0.549 | 0.549 | 0.578 | 0.539 | 0.516 |
| late | 918 | 0.697 | 0.697 | 0.750 | 0.734 | 0.750 | 0.750 | 0.726 | 0.742 | 0.736 |

### scene_density

| stratum | n | visual | text | concat | gated | zero_pad | norm_matched | learned_proj | rrf | randproj |
|---|---|---|---|---|---|---|---|---|---|---|
| high | 380 | 0.758 | 0.777 | 0.835 | 0.831 | 0.834 | 0.834 | 0.802 | 0.831 | 0.828 |
| low | 628 | 0.639 | 0.610 | 0.654 | 0.653 | 0.656 | 0.656 | 0.649 | 0.653 | 0.637 |

### motion_energy

| stratum | n | visual | text | concat | gated | zero_pad | norm_matched | learned_proj | rrf | randproj |
|---|---|---|---|---|---|---|---|---|---|---|
| high | 492 | 0.710 | 0.696 | 0.735 | 0.717 | 0.735 | 0.735 | 0.727 | 0.738 | 0.721 |
| low | 516 | 0.661 | 0.668 | 0.718 | 0.708 | 0.719 | 0.719 | 0.692 | 0.702 | 0.702 |
