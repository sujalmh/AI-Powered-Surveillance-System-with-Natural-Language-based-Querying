# Calibrated-fusion baselines + candidate-set probe -- TUNE split (n=50)

**This file is the pre-registration record.**  The comparison set below was fixed *before* the test split was evaluated and no selection is performed on this split: the entire fixed set is evaluated once on TEST.  The tune numbers are reported for transparency only.


## Pre-registered protocol

- Comparison set is fixed a priori; there is no selection stage, so the whole set is evaluated once on TEST (n=100).
- TUNE (n=50) is computed for transparency only and is not used to choose anything.
- Arms: minmax (released control protocol, primary) and zscore (repository's alternative calibration, secondary = score-distribution shift).
- control protocol: dense branches over the whole 121-clip gallery, calibrated over the whole gallery. Systems: prior_only (source-count-ratio alpha) and fixed_0.5 (equal-weight calibrated fusion).
- prod protocol (production candidate model): structured shortlist = clips with a structured result (threshold 0.05, the control's own n_structured rule), semantic shortlist = top-20 semantic hits (the control's own n_semantic), union = the gallery, each branch calibrated over its own shortlist, missing candidates zero-filled. Systems: prior_only_prod, comb_sum_cal, comb_anz_cal, comb_mnz_cal.
- Statistic: macro P@5 (mean of five category means). Primary: paired bootstrap with within-category resampling (n_boot=3000). Secondary: grouped 5-minute-cluster bootstrap on the macro statistic, pooled per-query paired bootstrap, pooled per-query grouped bootstrap, Wilcoxon signed-rank, paired sign-flip randomisation.
- Holm: one family per calibration arm (4 comparisons) plus a conservative global family over all 8 arm-level comparisons.
- The paper's metadata prior alpha is reused unchanged; these are competing baselines, not replacements.

## Literature basis

- **comb_sum** -- CombSUM (normalise-then-sum) of Fox & Shaw, 'Combination of Multiple Searches', TREC-2 (NIST, 1994), https://trec.nist.gov/pubs/trec2/papers/txt/23.txt
- **comb_anz** -- CombANZ (normalise-then-sum, multiplied by the number of systems that returned the document) of Fox & Shaw, TREC-2 (1994); the UMass CIIR 'Analysis of Multiple Evidence Combination' report https://ciir-publications.cs.umass.edu/getpdf.php?id=63
- **comb_mnz** -- CombMNZ (CombANZ with the count taken over the systems that assigned a non-zero score) of Fox & Shaw, TREC-2 (1994); Lee (1997), 'Hybrid test collections and the fusion of retrieval systems'
- **empty_branch** -- Empty-branch rebalancing: a branch that returns nothing must not deflate the other branch (the canonical fix for an empty result set); see also Raza et al. (2019) 'A survey on deep learning for hybrid information retrieval', and the UMass CIIR evidence-combination report

## Candidate-set sizes (production candidate model)

| Category | mean structured shortlist | mean semantic shortlist | mean union gallery | mean alpha |
|---|---:|---:|---:|---:|
| count | 10.8 | 20.0 | 27.6 | 1.000 |
| find | 14.6 | 20.0 | 32.5 | 0.654 |
| track | 121.0 | 20.0 | 121.0 | 0.622 |
| filter | 66.5 | 20.0 | 76.1 | 0.716 |
| action | 120.0 | 20.0 | 120.0 | 0.100 |

Alpha consistency check (production pool sizes vs. the control's published prior alpha): max |difference| = 0.00e+00 (ok = True).


## TUNE results -- minmax arm

| Protocol | System | macro P@5 | R@10 | F1@10 (paper) | MRR | count EM | count MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| control | prior_only | 0.5800 | 0.7762 | 0.6035 | 0.8812 | 1.000 | 0.000 |
| control | fixed_0.5 | 0.5760 | 0.7762 | 0.6013 | 0.8812 | 1.000 | 0.000 |
| prod | prior_only_prod | 0.5800 | 0.7737 | 0.6022 | 0.8809 | 1.000 | 0.000 |
| prod | comb_sum_cal | 0.5520 | 0.7647 | 0.5811 | 0.7326 | 0.600 | 1.000 |
| prod | comb_anz_cal | 0.4320 | 0.6887 | 0.4587 | 0.7104 | 0.700 | 0.500 |
| prod | comb_mnz_cal | 0.5240 | 0.7609 | 0.5552 | 0.7051 | 0.600 | 0.700 |

| Comparison | delta macro P@5 | 95% CI | p | grouped delta | grouped p | Holm p | Holm reject |
|---|---:|---|---:|---:|---:|---:|---|
| fixed_0.5_vs_prior_only | -0.0040 | [-0.0120, +0.0000] | 0.7033 | -0.0040 | 0.7093 | 0.7033 | False |
| comb_sum_cal_vs_prior_only_prod | -0.0280 | [-0.0480, -0.0120] | 0.0007 | -0.0280 | 0.0087 | 0.0013 | True |
| comb_anz_cal_vs_prior_only_prod | -0.1480 | [-0.2080, -0.0920] | < 0.0007 | -0.1480 | 0.0040 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod | -0.0560 | [-0.0880, -0.0280] | < 0.0007 | -0.0560 | 0.0087 | < 0.0007 | True |

## TUNE results -- zscore arm

| Protocol | System | macro P@5 | R@10 | F1@10 (paper) | MRR | count EM | count MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| control | prior_only | 0.5560 | 0.7667 | 0.5681 | 0.8020 | 1.000 | 0.000 |
| control | fixed_0.5 | 0.5040 | 0.6246 | 0.5033 | 0.7358 | 0.700 | 0.700 |
| prod | prior_only_prod | 0.5240 | 0.7574 | 0.5551 | 0.8150 | 1.000 | 0.000 |
| prod | comb_sum_cal | 0.2600 | 0.5701 | 0.3049 | 0.5696 | 0.500 | 0.800 |
| prod | comb_anz_cal | 0.2480 | 0.4993 | 0.2728 | 0.5682 | 0.500 | 0.800 |
| prod | comb_mnz_cal | 0.2480 | 0.4993 | 0.2728 | 0.5682 | 0.500 | 0.800 |

| Comparison | delta macro P@5 | 95% CI | p | grouped delta | grouped p | Holm p | Holm reject |
|---|---:|---|---:|---:|---:|---:|---|
| fixed_0.5_vs_prior_only | -0.0520 | [-0.0920, -0.0240] | < 0.0007 | -0.0520 | 0.0027 | < 0.0007 | True |
| comb_sum_cal_vs_prior_only_prod | -0.2640 | [-0.3320, -0.2040] | < 0.0007 | -0.2640 | < 0.0007 | < 0.0007 | True |
| comb_anz_cal_vs_prior_only_prod | -0.2760 | [-0.3440, -0.2160] | < 0.0007 | -0.2760 | < 0.0007 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod | -0.2760 | [-0.3440, -0.2160] | < 0.0007 | -0.2760 | < 0.0007 | < 0.0007 | True |

## Candidate-set-size robustness probe (tune split)


### minmax arm

| System | full | half_both | quarter_both | struct_half | struct_quarter | sem_half | warp_struct | warp_struct_half_both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior_only_prod | 0.5800 [0.476, 0.684] | 0.4400 [0.340, 0.544] | 0.3880 [0.300, 0.484] | 0.4400 [0.340, 0.544] | 0.3440 [0.248, 0.448] | 0.5800 [0.476, 0.684] | 0.5800 [0.476, 0.684] | 0.4400 [0.340, 0.544] |
| comb_sum_cal | 0.5520 [0.456, 0.652] | 0.4120 [0.316, 0.512] | 0.3480 [0.268, 0.436] | 0.4120 [0.316, 0.516] | 0.3040 [0.216, 0.404] | 0.5520 [0.456, 0.652] | 0.5560 [0.460, 0.656] | 0.4120 [0.316, 0.512] |
| comb_anz_cal | 0.4320 [0.340, 0.524] | 0.3520 [0.272, 0.440] | 0.3360 [0.260, 0.420] | 0.3440 [0.260, 0.436] | 0.2560 [0.180, 0.336] | 0.4280 [0.340, 0.520] | 0.4440 [0.352, 0.536] | 0.3720 [0.288, 0.460] |
| comb_mnz_cal | 0.5240 [0.428, 0.620] | 0.4040 [0.308, 0.504] | 0.3480 [0.268, 0.436] | 0.3960 [0.304, 0.496] | 0.2880 [0.204, 0.380] | 0.5440 [0.448, 0.644] | 0.5480 [0.448, 0.648] | 0.4120 [0.316, 0.512] |
| random gallery (reference floor) | 0.4360 | 0.2880 | 0.1640 | 0.3320 | 0.2720 | 0.3720 | 0.3920 | 0.2800 |

**Stability of the conclusion** -- paired delta macro P@5 of each calibrated baseline against `prior_only_prod` inside each gallery subsample (negative = the calibrated baseline is worse):

| System | full | half_both | quarter_both | struct_half | struct_quarter | sem_half | warp_struct | warp_struct_half_both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| comb_sum_cal - prior_only_prod | -0.0280 [-0.048, -0.012] | -0.0280 [-0.056, -0.008] | -0.0400 [-0.064, -0.016] | -0.0280 [-0.060, -0.008] | -0.0400 [-0.076, -0.012] | -0.0280 [-0.048, -0.012] | -0.0240 [-0.044, -0.008] | -0.0280 [-0.056, -0.008] |
| comb_anz_cal - prior_only_prod | -0.1480 [-0.208, -0.092] | -0.0880 [-0.132, -0.048] | -0.0520 [-0.080, -0.028] | -0.0960 [-0.152, -0.044] | -0.0880 [-0.140, -0.040] | -0.1520 [-0.204, -0.104] | -0.1360 [-0.196, -0.084] | -0.0680 [-0.108, -0.032] |
| comb_mnz_cal - prior_only_prod | -0.0560 [-0.088, -0.028] | -0.0360 [-0.068, -0.012] | -0.0400 [-0.064, -0.016] | -0.0440 [-0.076, -0.016] | -0.0560 [-0.096, -0.020] | -0.0360 [-0.060, -0.016] | -0.0320 [-0.052, -0.012] | -0.0280 [-0.056, -0.008] |

### zscore arm

| System | full | half_both | quarter_both | struct_half | struct_quarter | sem_half | warp_struct | warp_struct_half_both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior_only_prod | 0.5240 [0.428, 0.620] | 0.5600 [0.456, 0.660] | 0.5120 [0.424, 0.608] | 0.5000 [0.412, 0.592] | 0.4200 [0.348, 0.492] | 0.5720 [0.468, 0.676] | 0.4960 [0.396, 0.604] | 0.4440 [0.356, 0.544] |
| comb_sum_cal | 0.2600 [0.208, 0.316] | 0.3200 [0.264, 0.376] | 0.4200 [0.344, 0.496] | 0.2560 [0.196, 0.320] | 0.2280 [0.168, 0.296] | 0.3240 [0.260, 0.388] | 0.2680 [0.212, 0.328] | 0.2720 [0.212, 0.332] |
| comb_anz_cal | 0.2480 [0.196, 0.308] | 0.2240 [0.180, 0.272] | 0.4040 [0.336, 0.476] | 0.2200 [0.164, 0.280] | 0.1920 [0.136, 0.256] | 0.2560 [0.196, 0.316] | 0.2360 [0.176, 0.300] | 0.1760 [0.136, 0.216] |
| comb_mnz_cal | 0.2480 [0.196, 0.308] | 0.2240 [0.180, 0.272] | 0.4040 [0.336, 0.476] | 0.2200 [0.164, 0.280] | 0.1920 [0.136, 0.256] | 0.2560 [0.196, 0.316] | 0.2360 [0.176, 0.300] | 0.1760 [0.136, 0.216] |
| random gallery (reference floor) | 0.3920 | 0.2720 | 0.1320 | 0.3440 | 0.2480 | 0.3560 | 0.3720 | 0.3040 |

**Stability of the conclusion** -- paired delta macro P@5 of each calibrated baseline against `prior_only_prod` inside each gallery subsample (negative = the calibrated baseline is worse):

| System | full | half_both | quarter_both | struct_half | struct_quarter | sem_half | warp_struct | warp_struct_half_both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| comb_sum_cal - prior_only_prod | -0.2640 [-0.332, -0.200] | -0.2400 [-0.288, -0.192] | -0.0920 [-0.124, -0.060] | -0.2440 [-0.312, -0.184] | -0.1920 [-0.244, -0.144] | -0.2480 [-0.300, -0.200] | -0.2280 [-0.292, -0.164] | -0.1720 [-0.232, -0.116] |
| comb_anz_cal - prior_only_prod | -0.2760 [-0.344, -0.212] | -0.3360 [-0.396, -0.276] | -0.1080 [-0.144, -0.076] | -0.2800 [-0.348, -0.216] | -0.2280 [-0.288, -0.176] | -0.3160 [-0.384, -0.252] | -0.2600 [-0.332, -0.192] | -0.2680 [-0.336, -0.204] |
| comb_mnz_cal - prior_only_prod | -0.2760 [-0.344, -0.212] | -0.3360 [-0.396, -0.276] | -0.1080 [-0.144, -0.076] | -0.2800 [-0.348, -0.216] | -0.2280 [-0.288, -0.176] | -0.3160 [-0.384, -0.252] | -0.2600 [-0.332, -0.192] | -0.2680 [-0.336, -0.204] |

## Descriptive weight sweep (CombSUM weight, tune only -- NOT used for selection)

| weight on structured branch | CombSUM macro P@5 | prior_only_prod macro P@5 | delta |
|---:|---:|---:|---:|
| 0.1 | 0.0760 | 0.5800 | -0.5040 |
| 0.2 | 0.0960 | 0.5800 | -0.4840 |
| 0.3 | 0.1760 | 0.5800 | -0.4040 |
| 0.4 | 0.3520 | 0.5800 | -0.2280 |
| 0.5 | 0.5520 | 0.5800 | -0.0280 |
| 0.6 | 0.5760 | 0.5800 | -0.0040 |
| 0.7 | 0.5800 | 0.5800 | +0.0000 |
| 0.8 | 0.5800 | 0.5800 | +0.0000 |
| 0.9 | 0.5800 | 0.5800 | +0.0000 |

## Empty-branch sanity check (synthetic, not a benchmark result)

```json
{
  "qc_count_01": {
    "category": "count",
    "cases": {
      "empty_structured": {
        "alpha": 1.0,
        "systems": {
          "prior_only_prod": {
            "all_finite": true,
            "max_fused_score": 0.0,
            "top1_from_surviving_branch": false,
            "fused_is_degenerate": true
          },
          "comb_sum_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_anz_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_mnz_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          }
        }
      },
      "empty_semantic": {
        "alpha": 1.0,
        "systems": {
          "prior_only_prod": {
            "all_finite": true,
            "max_fused_score": 1.0,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_sum_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_anz_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_mnz_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          }
        }
      }
    },
    "note": "the paper's alpha keeps the empty branch's weight, so the surviving branch is multiplied by (1 - alpha) (by 0 when alpha = 1): its scores are deflated and are not comparable across queries.  The calibrated rules keep the surviving branch at full scale."
  },
  "q_find_01": {
    "category": "find",
    "cases": {
      "empty_structured": {
        "alpha": 0.5700000000000001,
        "systems": {
          "prior_only_prod": {
            "all_finite": true,
            "max_fused_score": 0.42999999999999994,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_sum_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_anz_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_mnz_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          }
        }
      },
      "empty_semantic": {
        "alpha": 0.77,
        "systems": {
          "prior_only_prod": {
            "all_finite": true,
            "max_fused_score": 0.77,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_sum_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_anz_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          },
          "comb_mnz_cal": {
            "all_finite": true,
            "max_fused_score": 0.5,
            "top1_from_surviving_branch": true,
            "fused_is_degenerate": false
          }
        }
      }
    },
    "note": "the paper's alpha keeps the empty branch's weight, so the surviving branch is multiplied by (1 - alpha) (by 0 when alpha = 1): its scores are deflated and are not comparable across queries.  The calibrated rules keep the surviving branch at full scale."
  }
}
```
