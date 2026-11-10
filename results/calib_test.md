# Calibrated-fusion baselines + candidate-set probe -- TEST split (n=100)

**One test evaluation of the frozen comparison set.**  The set was fixed on the TUNE split (see `calib_tune.md`); nothing was selected on test.  Bootstrap resamples: 3000.  `< 0.0007` is the resolution floor.


## minmax arm

| Protocol | System | macro P@5 | R@10 | F1@10 (paper) | MRR | count EM | count MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| control | prior_only | 0.5380 | 0.7967 | 0.5878 | 0.8527 | 0.950 | 0.050 |
| control | fixed_0.5 | 0.5340 | 0.7934 | 0.5843 | 0.8410 | 0.947 | 0.105 |
| prod | prior_only_prod | 0.5380 | 0.7977 | 0.5883 | 0.8476 | 0.950 | 0.050 |
| prod | comb_sum_cal | 0.5180 | 0.7900 | 0.5740 | 0.7263 | 0.450 | 1.200 |
| prod | comb_anz_cal | 0.3720 | 0.7401 | 0.4249 | 0.6105 | 0.500 | 1.200 |
| prod | comb_mnz_cal | 0.5020 | 0.7882 | 0.5588 | 0.7097 | 0.450 | 1.550 |

| Comparison | delta macro P@5 | 95% CI | p | grouped delta | grouped 95% CI | grouped p | Holm p | reject |
|---|---:|---|---:|---:|---|---:|---:|---|
| fixed_0.5_vs_prior_only | -0.0040 | [-0.0100, +0.0000] | 0.2733 | -0.0040 | [-0.0147, +0.0000] | 0.7047 | 0.2733 | False |
| comb_sum_cal_vs_prior_only_prod | -0.0200 | [-0.0360, -0.0080] | 0.0007 | -0.0200 | [-0.0365, -0.0050] | 0.0153 | 0.0013 | True |
| comb_anz_cal_vs_prior_only_prod | -0.1660 | [-0.2020, -0.1300] | < 0.0007 | -0.1660 | [-0.2240, -0.1289] | < 0.0007 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod | -0.0360 | [-0.0600, -0.0160] | < 0.0007 | -0.0360 | [-0.0672, -0.0121] | < 0.0007 | < 0.0007 | True |

| Comparison | pooled per-query delta | p | pooled grouped delta | p | Wilcoxon p | sign-flip p | better/worse queries |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed_0.5_vs_prior_only | -0.0040 [-0.0100, +0.0000] | 0.2680 | -0.0040 | 0.7053 | 1.0000 | 0.5077 | 0/2 |
| comb_sum_cal_vs_prior_only_prod | -0.0200 [-0.0360, -0.0060] | 0.0007 | -0.0200 | 0.0187 | 0.0117 | 0.0077 | 0/8 |
| comb_anz_cal_vs_prior_only_prod | -0.1660 [-0.2220, -0.1140] | < 0.0007 | -0.1660 | < 0.0007 | < 0.0007 | < 0.0007 | 0/34 |
| comb_mnz_cal_vs_prior_only_prod | -0.0360 [-0.0620, -0.0160] | < 0.0007 | -0.0360 | 0.0007 | 0.0022 | 0.0007 | 0/12 |

| Per-category | delta P@5 | 95% CI | p | Holm p | reject |
|---|---:|---|---:|---:|---|
| fixed_0.5_vs_prior_only / count | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| fixed_0.5_vs_prior_only / find | -0.0100 | [-0.0300, +0.0000] | 0.7020 | 1.0000 | False |
| fixed_0.5_vs_prior_only / track | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| fixed_0.5_vs_prior_only / action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| fixed_0.5_vs_prior_only / filter | -0.0100 | [-0.0300, +0.0000] | 0.7313 | 1.0000 | False |
| comb_sum_cal_vs_prior_only_prod / count | -0.0100 | [-0.0300, +0.0000] | 0.7247 | 1.0000 | False |
| comb_sum_cal_vs_prior_only_prod / find | -0.0500 | [-0.1000, -0.0100] | 0.0200 | 0.1000 | False |
| comb_sum_cal_vs_prior_only_prod / track | -0.0200 | [-0.0500, +0.0000] | 0.2387 | 0.9547 | False |
| comb_sum_cal_vs_prior_only_prod / action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| comb_sum_cal_vs_prior_only_prod / filter | -0.0200 | [-0.0600, +0.0000] | 0.7307 | 1.0000 | False |
| comb_anz_cal_vs_prior_only_prod / count | -0.0100 | [-0.0300, +0.0000] | 0.7247 | 1.0000 | False |
| comb_anz_cal_vs_prior_only_prod / find | -0.0500 | [-0.1000, -0.0100] | 0.0200 | 0.0600 | False |
| comb_anz_cal_vs_prior_only_prod / track | -0.5800 | [-0.7200, -0.4400] | < 0.0007 | < 0.0007 | True |
| comb_anz_cal_vs_prior_only_prod / action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| comb_anz_cal_vs_prior_only_prod / filter | -0.1900 | [-0.2900, -0.1000] | < 0.0007 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod / count | -0.0100 | [-0.0300, +0.0000] | 0.7247 | 1.0000 | False |
| comb_mnz_cal_vs_prior_only_prod / find | -0.0500 | [-0.1000, -0.0100] | 0.0200 | 0.0800 | False |
| comb_mnz_cal_vs_prior_only_prod / track | -0.0300 | [-0.0800, +0.0000] | 0.2387 | 0.7160 | False |
| comb_mnz_cal_vs_prior_only_prod / action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| comb_mnz_cal_vs_prior_only_prod / filter | -0.0900 | [-0.1800, -0.0200] | 0.0060 | 0.0300 | True |

## zscore arm

| Protocol | System | macro P@5 | R@10 | F1@10 (paper) | MRR | count EM | count MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| control | prior_only | 0.5240 | 0.7834 | 0.5740 | 0.8003 | 0.950 | 0.050 |
| control | fixed_0.5 | 0.4500 | 0.6822 | 0.4848 | 0.6967 | 0.500 | 1.600 |
| prod | prior_only_prod | 0.5000 | 0.7939 | 0.5634 | 0.7758 | 0.950 | 0.050 |
| prod | comb_sum_cal | 0.2180 | 0.6433 | 0.2682 | 0.4750 | 0.400 | 1.400 |
| prod | comb_anz_cal | 0.2080 | 0.4859 | 0.2456 | 0.4489 | 0.300 | 1.700 |
| prod | comb_mnz_cal | 0.2080 | 0.4859 | 0.2456 | 0.4489 | 0.300 | 1.700 |

| Comparison | delta macro P@5 | 95% CI | p | grouped delta | grouped 95% CI | grouped p | Holm p | reject |
|---|---:|---|---:|---:|---|---:|---:|---|
| fixed_0.5_vs_prior_only | -0.0740 | [-0.1000, -0.0500] | < 0.0007 | -0.0740 | [-0.1280, -0.0377] | < 0.0007 | < 0.0007 | True |
| comb_sum_cal_vs_prior_only_prod | -0.2820 | [-0.3240, -0.2400] | < 0.0007 | -0.2820 | [-0.3720, -0.2362] | < 0.0007 | < 0.0007 | True |
| comb_anz_cal_vs_prior_only_prod | -0.2920 | [-0.3360, -0.2480] | < 0.0007 | -0.2920 | [-0.3847, -0.2440] | < 0.0007 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod | -0.2920 | [-0.3360, -0.2480] | < 0.0007 | -0.2920 | [-0.3847, -0.2440] | < 0.0007 | < 0.0007 | True |

| Comparison | pooled per-query delta | p | pooled grouped delta | p | Wilcoxon p | sign-flip p | better/worse queries |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed_0.5_vs_prior_only | -0.0740 [-0.1020, -0.0480] | < 0.0007 | -0.0740 | < 0.0007 | < 0.0007 | < 0.0007 | 0/27 |
| comb_sum_cal_vs_prior_only_prod | -0.2820 [-0.3480, -0.2180] | < 0.0007 | -0.2820 | < 0.0007 | < 0.0007 | < 0.0007 | 0/53 |
| comb_anz_cal_vs_prior_only_prod | -0.2920 [-0.3580, -0.2280] | < 0.0007 | -0.2920 | < 0.0007 | < 0.0007 | < 0.0007 | 0/54 |
| comb_mnz_cal_vs_prior_only_prod | -0.2920 [-0.3580, -0.2280] | < 0.0007 | -0.2920 | < 0.0007 | < 0.0007 | < 0.0007 | 0/54 |

| Per-category | delta P@5 | 95% CI | p | Holm p | reject |
|---|---:|---|---:|---:|---|
| fixed_0.5_vs_prior_only / count | -0.0800 | [-0.1200, -0.0400] | < 0.0007 | < 0.0007 | True |
| fixed_0.5_vs_prior_only / find | -0.1200 | [-0.2100, -0.0400] | 0.0007 | 0.0020 | True |
| fixed_0.5_vs_prior_only / track | -0.1000 | [-0.1700, -0.0400] | < 0.0007 | < 0.0007 | True |
| fixed_0.5_vs_prior_only / action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| fixed_0.5_vs_prior_only / filter | -0.0700 | [-0.1200, -0.0200] | 0.0013 | 0.0027 | True |
| comb_sum_cal_vs_prior_only_prod / count | -0.0900 | [-0.1800, -0.0100] | 0.0267 | 0.0533 | False |
| comb_sum_cal_vs_prior_only_prod / find | -0.5700 | [-0.7000, -0.4400] | < 0.0007 | < 0.0007 | True |
| comb_sum_cal_vs_prior_only_prod / track | -0.5600 | [-0.6700, -0.4500] | < 0.0007 | < 0.0007 | True |
| comb_sum_cal_vs_prior_only_prod / action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| comb_sum_cal_vs_prior_only_prod / filter | -0.1900 | [-0.2800, -0.1100] | < 0.0007 | < 0.0007 | True |
| comb_anz_cal_vs_prior_only_prod / count | -0.0900 | [-0.1800, -0.0100] | 0.0267 | 0.0533 | False |
| comb_anz_cal_vs_prior_only_prod / find | -0.6000 | [-0.7200, -0.4700] | < 0.0007 | < 0.0007 | True |
| comb_anz_cal_vs_prior_only_prod / track | -0.5600 | [-0.6700, -0.4500] | < 0.0007 | < 0.0007 | True |
| comb_anz_cal_vs_prior_only_prod / action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| comb_anz_cal_vs_prior_only_prod / filter | -0.2100 | [-0.3100, -0.1200] | < 0.0007 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod / count | -0.0900 | [-0.1800, -0.0100] | 0.0267 | 0.0533 | False |
| comb_mnz_cal_vs_prior_only_prod / find | -0.6000 | [-0.7200, -0.4700] | < 0.0007 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod / track | -0.5600 | [-0.6700, -0.4500] | < 0.0007 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod / action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| comb_mnz_cal_vs_prior_only_prod / filter | -0.2100 | [-0.3100, -0.1200] | < 0.0007 | < 0.0007 | True |

## Candidate-set-size robustness probe (test split)

Gallery subsample = the union of the two branches' shortlists after each shortlist is cut to its largest-scoring full / half / quarter candidates (seeded tie-break, seed 20260401).  Every frozen method is re-calibrated and re-ranked inside the subsampled gallery.  `mean alpha` shows how far the source-count-ratio weighting moves when the candidate-set size changes.


### minmax arm

| System | full (alpha 0.619) | half_both (alpha 0.619) | quarter_both (alpha 0.620) | struct_half (alpha 0.604) | struct_quarter (alpha 0.589) | sem_half (alpha 0.632) | warp_struct (alpha 0.619) | warp_struct_half_both (alpha 0.619) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior_only_prod | 0.5380 [0.466, 0.614] | 0.4040 [0.334, 0.478] | 0.3680 [0.308, 0.434] | 0.4040 [0.334, 0.478] | 0.3320 [0.270, 0.402] | 0.5380 [0.466, 0.614] | 0.5380 [0.466, 0.614] | 0.4040 [0.334, 0.478] |
| comb_sum_cal | 0.5180 [0.450, 0.592] | 0.3940 [0.328, 0.464] | 0.3460 [0.290, 0.412] | 0.3920 [0.326, 0.464] | 0.3060 [0.246, 0.374] | 0.5220 [0.454, 0.596] | 0.5200 [0.452, 0.594] | 0.3960 [0.328, 0.468] |
| comb_anz_cal | 0.3720 [0.306, 0.438] | 0.3520 [0.296, 0.410] | 0.3400 [0.286, 0.402] | 0.3340 [0.280, 0.394] | 0.2720 [0.222, 0.328] | 0.4160 [0.352, 0.482] | 0.3720 [0.308, 0.438] | 0.3500 [0.292, 0.410] |
| comb_mnz_cal | 0.5020 [0.434, 0.574] | 0.3860 [0.320, 0.454] | 0.3440 [0.288, 0.408] | 0.3780 [0.314, 0.446] | 0.2920 [0.236, 0.358] | 0.5120 [0.444, 0.584] | 0.5020 [0.434, 0.576] | 0.3940 [0.328, 0.464] |
| random gallery (reference floor) | 0.3640 | 0.2240 | 0.1400 | 0.2840 | 0.2180 | 0.3380 | 0.3660 | 0.2240 |

| Cell | mean structured pool | mean semantic pool | mean union gallery | mean alpha | gallery relevant coverage |
|---|---:|---:|---:|---:|---:|
| full | 67.6 | 20.0 | 76.5 | 0.619 | 1.000 |
| half_both | 34.0 | 10.0 | 41.4 | 0.619 | 0.869 |
| quarter_both | 17.2 | 5.0 | 21.5 | 0.620 | 0.750 |
| struct_half | 34.0 | 20.0 | 48.5 | 0.604 | 0.888 |
| struct_quarter | 17.2 | 20.0 | 34.3 | 0.589 | 0.778 |
| sem_half | 67.6 | 10.0 | 72.0 | 0.632 | 1.000 |
| warp_struct | 67.6 | 20.0 | 76.5 | 0.619 | 1.000 |
| warp_struct_half_both | 34.0 | 10.0 | 41.2 | 0.619 | 0.872 |

### zscore arm

| System | full (alpha 0.619) | half_both (alpha 0.619) | quarter_both (alpha 0.620) | struct_half (alpha 0.604) | struct_quarter (alpha 0.589) | sem_half (alpha 0.632) | warp_struct (alpha 0.619) | warp_struct_half_both (alpha 0.619) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior_only_prod | 0.5000 [0.432, 0.570] | 0.5340 [0.462, 0.610] | 0.4860 [0.424, 0.556] | 0.4660 [0.406, 0.530] | 0.4180 [0.362, 0.476] | 0.5320 [0.460, 0.608] | 0.4660 [0.400, 0.538] | 0.4120 [0.348, 0.482] |
| comb_sum_cal | 0.2180 [0.176, 0.262] | 0.3520 [0.302, 0.406] | 0.4440 [0.388, 0.502] | 0.2480 [0.210, 0.286] | 0.2400 [0.196, 0.288] | 0.3120 [0.266, 0.358] | 0.2540 [0.214, 0.298] | 0.2920 [0.244, 0.344] |
| comb_anz_cal | 0.2080 [0.164, 0.252] | 0.2420 [0.204, 0.280] | 0.4220 [0.372, 0.478] | 0.1860 [0.150, 0.224] | 0.1740 [0.140, 0.212] | 0.2340 [0.190, 0.280] | 0.1860 [0.146, 0.230] | 0.1840 [0.152, 0.216] |
| comb_mnz_cal | 0.2080 [0.164, 0.252] | 0.2420 [0.204, 0.280] | 0.4220 [0.372, 0.478] | 0.1860 [0.150, 0.224] | 0.1740 [0.140, 0.212] | 0.2340 [0.190, 0.280] | 0.1860 [0.146, 0.230] | 0.1840 [0.152, 0.216] |
| random gallery (reference floor) | 0.3720 | 0.2360 | 0.1520 | 0.2620 | 0.2320 | 0.3540 | 0.3500 | 0.2340 |

| Cell | mean structured pool | mean semantic pool | mean union gallery | mean alpha | gallery relevant coverage |
|---|---:|---:|---:|---:|---:|
| full | 67.6 | 20.0 | 76.5 | 0.619 | 1.000 |
| half_both | 34.0 | 10.0 | 41.3 | 0.619 | 0.855 |
| quarter_both | 17.2 | 5.0 | 21.5 | 0.620 | 0.751 |
| struct_half | 34.0 | 20.0 | 48.7 | 0.604 | 0.891 |
| struct_quarter | 17.2 | 20.0 | 34.3 | 0.589 | 0.781 |
| sem_half | 67.6 | 10.0 | 72.0 | 0.632 | 1.000 |
| warp_struct | 67.6 | 20.0 | 76.5 | 0.619 | 1.000 |
| warp_struct_half_both | 34.0 | 10.0 | 41.0 | 0.619 | 0.871 |

## Verdict

- calibrated baseline beats prior-only (Holm-significant): **False**
- robustness direction: **ROBUST (the sign and the size of the calibrated-vs-prior difference are preserved under every candidate-set-size and score-distribution perturbation)**
- overall verdict: **ROBUST**
- winner: no calibrated baseline is Holm-significant against the prior-only baseline in either arm
- probe_drop: absolute macro P@5 drop from the full gallery to the worst candidate-set-size cell: prior_only_prod -0.2060, the three calibrated baselines -0.1740 on average (the drop is driven by the gallery losing relevant candidates, and is shared by all methods)
- probe_stability: comb_sum_cal: delta range [-0.0260, -0.0080], sign stable=True; comb_anz_cal: delta range [-0.1660, -0.0280], sign stable=True; comb_mnz_cal: delta range [-0.0400, -0.0100], sign stable=True
- distribution_shift: score-distribution shift (calibrated structured branch x^2): prior_only_prod@warp_struct=0.5380 (alpha 0.619), comb_sum_cal@warp_struct=0.5200 (alpha 0.619), comb_anz_cal@warp_struct=0.3720 (alpha 0.619), comb_mnz_cal@warp_struct=0.5020 (alpha 0.619), prior_only_prod@warp_struct_half_both=0.4040 (alpha 0.619), comb_sum_cal@warp_struct_half_both=0.3960 (alpha 0.619), comb_anz_cal@warp_struct_half_both=0.3500 (alpha 0.619), comb_mnz_cal@warp_struct_half_both=0.3940 (alpha 0.619)
- empty_branch: with an empty branch the source-count-ratio prior collapses to an all-zero fused score (degenerate ranking) in 1 case(s): prior_only_prod@qc_count_01/empty_structured; the calibrated rules keep the surviving branch at full scale
- probe_detail: prior_only_prod: 0.5380 -> 0.3320 (struct_quarter); comb_sum_cal: 0.5180 -> 0.3060 (struct_quarter); comb_anz_cal: 0.3720 -> 0.2720 (struct_quarter); comb_mnz_cal: 0.5020 -> 0.2920 (struct_quarter)
- comparison_outcome: no calibrated baseline beats the source-count-ratio prior-only baseline Holm-significantly on test; the simple calibrated fusion is at best a tie under the released control protocol
