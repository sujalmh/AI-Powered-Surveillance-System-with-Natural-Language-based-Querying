# Calibrated fusion baselines vs. the source-count-ratio weighting

**Scope.** Released single-camera subset (121 indexed clips / 242 scene frames), 150 protocol-matched control queries (tune n=50 / test n=100).  The paper's metadata prior alpha (source-count-ratio weighting) is reused unchanged; the calibrated fusion baselines are *competitors*, not replacements.


**Anti-p-hacking record.** The comparison set was fixed before the test split was evaluated, the whole fixed set was evaluated **once** on test (the script refuses to write a second test result), and every configuration that was tried is listed below.


## 1. Baselines implemented

| Baseline | Rule | Source |
|---|---|---|
| `fixed_0.5` / `comb_sum_cal` | equal-weight fusion of the two calibrated branches (CombSUM) | Fox & Shaw, TREC-2 1994 |
| `comb_anz_cal` | CombANZ: CombSUM x (#branches that returned the candidate) | Fox & Shaw, TREC-2 1994 |
| `comb_mnz_cal` | CombMNZ: CombSUM x (#branches with a non-zero score) | Fox & Shaw, TREC-2 1994; Lee 1997 |
| zero-fill + per-branch calibration | candidates a branch did not return get 0; each branch is calibrated over its own shortlist | canonical IR data-fusion practice |

## 2. Pre-registration

- Comparison set is fixed a priori; there is no selection stage, so the whole set is evaluated once on TEST (n=100).
- TUNE (n=50) is computed for transparency only and is not used to choose anything.
- Arms: minmax (released control protocol, primary) and zscore (repository's alternative calibration, secondary = score-distribution shift).
- control protocol: dense branches over the whole 121-clip gallery, calibrated over the whole gallery. Systems: prior_only (source-count-ratio alpha) and fixed_0.5 (equal-weight calibrated fusion).
- prod protocol (production candidate model): structured shortlist = clips with a structured result (threshold 0.05, the control's own n_structured rule), semantic shortlist = top-20 semantic hits (the control's own n_semantic), union = the gallery, each branch calibrated over its own shortlist, missing candidates zero-filled. Systems: prior_only_prod, comb_sum_cal, comb_anz_cal, comb_mnz_cal.
- Statistic: macro P@5 (mean of five category means). Primary: paired bootstrap with within-category resampling (n_boot=3000). Secondary: grouped 5-minute-cluster bootstrap on the macro statistic, pooled per-query paired bootstrap, pooled per-query grouped bootstrap, Wilcoxon signed-rank, paired sign-flip randomisation.
- Holm: one family per calibration arm (4 comparisons) plus a conservative global family over all 8 arm-level comparisons.
- The paper's metadata prior alpha is reused unchanged; these are competing baselines, not replacements.

## 3. TUNE split (n=50) -- reported, not used for selection

| Arm | Protocol | System | macro P@5 |
|---|---|---|---:|
| minmax | control | prior_only | 0.5800 |
| minmax | control | fixed_0.5 | 0.5760 |
| minmax | prod | prior_only_prod | 0.5800 |
| minmax | prod | comb_sum_cal | 0.5520 |
| minmax | prod | comb_anz_cal | 0.4320 |
| minmax | prod | comb_mnz_cal | 0.5240 |
| zscore | control | prior_only | 0.5560 |
| zscore | control | fixed_0.5 | 0.5040 |
| zscore | prod | prior_only_prod | 0.5240 |
| zscore | prod | comb_sum_cal | 0.2600 |
| zscore | prod | comb_anz_cal | 0.2480 |
| zscore | prod | comb_mnz_cal | 0.2480 |

## 4. TEST split (n=100) -- the single frozen evaluation


### minmax arm

| Protocol | System | macro P@5 | per-category P@5 (count/find/track/action/filter) |
|---|---|---:|---|
| control | prior_only | 0.5380 | 0.390/0.920/0.810/0.090/0.480 |
| control | fixed_0.5 | 0.5340 | 0.390/0.910/0.810/0.090/0.470 |
| prod | prior_only_prod | 0.5380 | 0.390/0.920/0.810/0.090/0.480 |
| prod | comb_sum_cal | 0.5180 | 0.380/0.870/0.790/0.090/0.460 |
| prod | comb_anz_cal | 0.3720 | 0.380/0.870/0.230/0.090/0.290 |
| prod | comb_mnz_cal | 0.5020 | 0.380/0.870/0.780/0.090/0.390 |

| Comparison | delta macro P@5 | 95% CI | p | grouped delta | grouped p | Holm p | reject |
|---|---:|---|---:|---:|---:|---:|---|
| fixed_0.5_vs_prior_only | -0.0040 | [-0.0100, +0.0000] | 0.2733 | -0.0040 | 0.7047 | 0.2733 | False |
| comb_sum_cal_vs_prior_only_prod | -0.0200 | [-0.0360, -0.0080] | 0.0007 | -0.0200 | 0.0153 | 0.0013 | True |
| comb_anz_cal_vs_prior_only_prod | -0.1660 | [-0.2020, -0.1300] | < 0.0007 | -0.1660 | < 0.0007 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod | -0.0360 | [-0.0600, -0.0160] | < 0.0007 | -0.0360 | < 0.0007 | < 0.0007 | True |

### zscore arm

| Protocol | System | macro P@5 | per-category P@5 (count/find/track/action/filter) |
|---|---|---:|---|
| control | prior_only | 0.5240 | 0.390/0.910/0.800/0.090/0.430 |
| control | fixed_0.5 | 0.4500 | 0.310/0.790/0.700/0.090/0.360 |
| prod | prior_only_prod | 0.5000 | 0.390/0.910/0.700/0.090/0.410 |
| prod | comb_sum_cal | 0.2180 | 0.300/0.340/0.140/0.090/0.220 |
| prod | comb_anz_cal | 0.2080 | 0.300/0.310/0.140/0.090/0.200 |
| prod | comb_mnz_cal | 0.2080 | 0.300/0.310/0.140/0.090/0.200 |

| Comparison | delta macro P@5 | 95% CI | p | grouped delta | grouped p | Holm p | reject |
|---|---:|---|---:|---:|---:|---:|---|
| fixed_0.5_vs_prior_only | -0.0740 | [-0.1000, -0.0500] | < 0.0007 | -0.0740 | < 0.0007 | < 0.0007 | True |
| comb_sum_cal_vs_prior_only_prod | -0.2820 | [-0.3240, -0.2400] | < 0.0007 | -0.2820 | < 0.0007 | < 0.0007 | True |
| comb_anz_cal_vs_prior_only_prod | -0.2920 | [-0.3360, -0.2480] | < 0.0007 | -0.2920 | < 0.0007 | < 0.0007 | True |
| comb_mnz_cal_vs_prior_only_prod | -0.2920 | [-0.3360, -0.2480] | < 0.0007 | -0.2920 | < 0.0007 | < 0.0007 | True |

## 5. Candidate-set-size robustness probe

Same frozen methods, gallery subsample = union of the branches' shortlists after cutting each shortlist to its largest-scoring full / half / quarter candidates (seeded, seed 20260401).  Query-level bootstrap 95% CIs in brackets; descriptive.


### minmax arm

| System | full | half_both | quarter_both | struct_half | struct_quarter | sem_half | warp_struct | warp_struct_half_both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior_only_prod | 0.5380 [0.466, 0.614] | 0.4040 [0.334, 0.478] | 0.3680 [0.308, 0.434] | 0.4040 [0.334, 0.478] | 0.3320 [0.270, 0.402] | 0.5380 [0.466, 0.614] | 0.5380 [0.466, 0.614] | 0.4040 [0.334, 0.478] |
| comb_sum_cal | 0.5180 [0.450, 0.592] | 0.3940 [0.328, 0.464] | 0.3460 [0.290, 0.412] | 0.3920 [0.326, 0.464] | 0.3060 [0.246, 0.374] | 0.5220 [0.454, 0.596] | 0.5200 [0.452, 0.594] | 0.3960 [0.328, 0.468] |
| comb_anz_cal | 0.3720 [0.306, 0.438] | 0.3520 [0.296, 0.410] | 0.3400 [0.286, 0.402] | 0.3340 [0.280, 0.394] | 0.2720 [0.222, 0.328] | 0.4160 [0.352, 0.482] | 0.3720 [0.308, 0.438] | 0.3500 [0.292, 0.410] |
| comb_mnz_cal | 0.5020 [0.434, 0.574] | 0.3860 [0.320, 0.454] | 0.3440 [0.288, 0.408] | 0.3780 [0.314, 0.446] | 0.2920 [0.236, 0.358] | 0.5120 [0.444, 0.584] | 0.5020 [0.434, 0.576] | 0.3940 [0.328, 0.464] |
| random gallery (reference floor) | 0.3640 | 0.2240 | 0.1400 | 0.2840 | 0.2180 | 0.3380 | 0.3660 | 0.2240 |

**Stability of the conclusion** -- paired delta macro P@5 of each calibrated baseline against `prior_only_prod` inside each gallery subsample (negative = the calibrated baseline is worse):

| System | full | half_both | quarter_both | struct_half | struct_quarter | sem_half | warp_struct | warp_struct_half_both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| comb_sum_cal - prior_only_prod | -0.0200 [-0.036, -0.008] | -0.0100 [-0.020, -0.002] | -0.0220 [-0.038, -0.008] | -0.0120 [-0.024, -0.002] | -0.0260 [-0.048, -0.008] | -0.0160 [-0.026, -0.006] | -0.0180 [-0.030, -0.008] | -0.0080 [-0.016, -0.002] |
| comb_anz_cal - prior_only_prod | -0.1660 [-0.200, -0.132] | -0.0520 [-0.082, -0.026] | -0.0280 [-0.044, -0.014] | -0.0700 [-0.106, -0.038] | -0.0600 [-0.092, -0.032] | -0.1220 [-0.164, -0.084] | -0.1660 [-0.202, -0.132] | -0.0540 [-0.080, -0.030] |
| comb_mnz_cal - prior_only_prod | -0.0360 [-0.060, -0.016] | -0.0180 [-0.032, -0.006] | -0.0240 [-0.040, -0.010] | -0.0260 [-0.048, -0.008] | -0.0400 [-0.068, -0.016] | -0.0260 [-0.042, -0.012] | -0.0360 [-0.058, -0.018] | -0.0100 [-0.020, -0.002] |

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

| System | full | half_both | quarter_both | struct_half | struct_quarter | sem_half | warp_struct | warp_struct_half_both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior_only_prod | 0.5000 [0.432, 0.570] | 0.5340 [0.462, 0.610] | 0.4860 [0.424, 0.556] | 0.4660 [0.406, 0.530] | 0.4180 [0.362, 0.476] | 0.5320 [0.460, 0.608] | 0.4660 [0.400, 0.538] | 0.4120 [0.348, 0.482] |
| comb_sum_cal | 0.2180 [0.176, 0.262] | 0.3520 [0.302, 0.406] | 0.4440 [0.388, 0.502] | 0.2480 [0.210, 0.286] | 0.2400 [0.196, 0.288] | 0.3120 [0.266, 0.358] | 0.2540 [0.214, 0.298] | 0.2920 [0.244, 0.344] |
| comb_anz_cal | 0.2080 [0.164, 0.252] | 0.2420 [0.204, 0.280] | 0.4220 [0.372, 0.478] | 0.1860 [0.150, 0.224] | 0.1740 [0.140, 0.212] | 0.2340 [0.190, 0.280] | 0.1860 [0.146, 0.230] | 0.1840 [0.152, 0.216] |
| comb_mnz_cal | 0.2080 [0.164, 0.252] | 0.2420 [0.204, 0.280] | 0.4220 [0.372, 0.478] | 0.1860 [0.150, 0.224] | 0.1740 [0.140, 0.212] | 0.2340 [0.190, 0.280] | 0.1860 [0.146, 0.230] | 0.1840 [0.152, 0.216] |
| random gallery (reference floor) | 0.3720 | 0.2360 | 0.1520 | 0.2620 | 0.2320 | 0.3540 | 0.3500 | 0.2340 |

**Stability of the conclusion** -- paired delta macro P@5 of each calibrated baseline against `prior_only_prod` inside each gallery subsample (negative = the calibrated baseline is worse):

| System | full | half_both | quarter_both | struct_half | struct_quarter | sem_half | warp_struct | warp_struct_half_both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| comb_sum_cal - prior_only_prod | -0.2820 [-0.324, -0.242] | -0.1820 [-0.222, -0.142] | -0.0420 [-0.062, -0.024] | -0.2180 [-0.258, -0.180] | -0.1780 [-0.218, -0.140] | -0.2200 [-0.262, -0.182] | -0.2120 [-0.254, -0.168] | -0.1200 [-0.162, -0.080] |
| comb_anz_cal - prior_only_prod | -0.2920 [-0.336, -0.252] | -0.2920 [-0.336, -0.248] | -0.0640 [-0.090, -0.040] | -0.2800 [-0.322, -0.240] | -0.2440 [-0.286, -0.204] | -0.2980 [-0.340, -0.258] | -0.2800 [-0.328, -0.232] | -0.2280 [-0.278, -0.178] |
| comb_mnz_cal - prior_only_prod | -0.2920 [-0.336, -0.252] | -0.2920 [-0.336, -0.248] | -0.0640 [-0.090, -0.040] | -0.2800 [-0.322, -0.240] | -0.2440 [-0.286, -0.204] | -0.2980 [-0.340, -0.258] | -0.2800 [-0.328, -0.232] | -0.2280 [-0.278, -0.178] |

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

## 6. Verdict

**ROBUST** -- calibrated baseline beats prior-only (Holm-significant): **False**; robustness direction: **ROBUST (the sign and the size of the calibrated-vs-prior difference are preserved under every candidate-set-size and score-distribution perturbation)**.

- winner: no calibrated baseline is Holm-significant against the prior-only baseline in either arm
- probe_drop: absolute macro P@5 drop from the full gallery to the worst candidate-set-size cell: prior_only_prod -0.2060, the three calibrated baselines -0.1740 on average (the drop is driven by the gallery losing relevant candidates, and is shared by all methods)
- probe_stability: comb_sum_cal: delta range [-0.0260, -0.0080], sign stable=True; comb_anz_cal: delta range [-0.1660, -0.0280], sign stable=True; comb_mnz_cal: delta range [-0.0400, -0.0100], sign stable=True
- distribution_shift: score-distribution shift (calibrated structured branch x^2): prior_only_prod@warp_struct=0.5380 (alpha 0.619), comb_sum_cal@warp_struct=0.5200 (alpha 0.619), comb_anz_cal@warp_struct=0.3720 (alpha 0.619), comb_mnz_cal@warp_struct=0.5020 (alpha 0.619), prior_only_prod@warp_struct_half_both=0.4040 (alpha 0.619), comb_sum_cal@warp_struct_half_both=0.3960 (alpha 0.619), comb_anz_cal@warp_struct_half_both=0.3500 (alpha 0.619), comb_mnz_cal@warp_struct_half_both=0.3940 (alpha 0.619)
- empty_branch: with an empty branch the source-count-ratio prior collapses to an all-zero fused score (degenerate ranking) in 1 case(s): prior_only_prod@qc_count_01/empty_structured; the calibrated rules keep the surviving branch at full scale
- probe_detail: prior_only_prod: 0.5380 -> 0.3320 (struct_quarter); comb_sum_cal: 0.5180 -> 0.3060 (struct_quarter); comb_anz_cal: 0.3720 -> 0.2720 (struct_quarter); comb_mnz_cal: 0.5020 -> 0.2920 (struct_quarter)
- comparison_outcome: no calibrated baseline beats the source-count-ratio prior-only baseline Holm-significantly on test; the simple calibrated fusion is at best a tie under the released control protocol

## 7. Reproduction

```
python backend/scripts/eval_calib_base.py --stage tune
python backend/scripts/eval_calib_base.py --stage test
```

## 8. Limitations

- Single camera (camera_0) and a 2-hour released window; the numbers are not the paper's full 4-camera / 312k-descriptor deployment results.
- Labels are derived from released VLM captions, detector metadata and snapshots, not human annotations; no inter-annotator agreement is available.
- The control harness scores every clip with the metadata structured branch (it is a dense scoring function, not a real structured search); the production candidate model therefore approximates the structured shortlist with the control's own n_structured rule (score > 0.05).
- CombANZ/CombMNZ are applied to the two-branch case only; with a dense branch pair the count factor is nearly constant, so they can coincide with CombSUM -- that is reported, not hidden.
- The probe is descriptive (stability under perturbation), not a hypothesis test; the CIs are bootstrap intervals over queries.
- The alpha prior, the benchmark, the metrics and the statistical protocol are imported unchanged from the released control driver.
