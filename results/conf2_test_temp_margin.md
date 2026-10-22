# C1'' -- frozen winner evaluated on the TEST split (n=100)

**Frozen variant:** `temp_margin` with hyper-parameters `{"minmax": {"temp": 0.05, "blend": 0.75}, "zscore": {"temp": 1.0, "blend": 0.25}}` (selected on the tune split only; see `results/conf2_tune_sweep.md`).

**Protocol.** Paired bootstrap on per-query P@5 (n_boot=3000) and the same grouped (cluster) bootstrap by query group, Holm-corrected over the three pre-registered comparisons.  `< 0.0007` is the resolution floor (2/n_boot).


## Test results -- minmax normalisation

| System | Macro P@5 | Macro R@10 | Macro F1@10 (paper) | Macro MRR |
|---|---:|---:|---:|---:|
| prior_only (adaptive) | 0.5380 | 0.7967 | 0.5878 | 0.8527 |
| winner | 0.5360 | 0.7934 | 0.5853 | 0.8527 |
| best fixed alpha (tune-selected) | 0.5360 | 0.7934 | 0.5853 | 0.8527 |

| System | Count | Find | Track | Action | Filter |
|---|---:|---:|---:|---:|---:|
| prior_only | 0.390 | 0.920 | 0.810 | 0.090 | 0.480 |
| winner | 0.390 | 0.920 | 0.810 | 0.090 | 0.470 |

### Paired comparisons vs. the test split

| Comparison | ΔP@5 | 95% CI | p (query-level) | ΔP@5 (grouped) | grouped 95% CI | grouped p | Holm p | Holm reject |
|---|---:|---|---:|---:|---|---:|---:|---|
| winner_vs_prior | -0.0020 | [-0.0060, +0.0000] | 0.6953 | -0.0020 | [-0.0070, +0.0000] | 0.7113 | 1.0000 | False |
| winner_vs_best_fixed | +0.0000 | [+0.0000, +0.0000] | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| winner_vs_conf_only | +0.0000 | [+0.0000, +0.0000] | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |

### Per-category vs. prior-only (Holm over the five categories)

| Category | ΔP@5 | 95% CI | p | Holm p | Holm reject |
|---|---:|---|---:|---:|---|
| count | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| find | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| track | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| filter | -0.0100 | [-0.0300, +0.0000] | 0.7300 | 1.0000 | False |

## Test results -- zscore normalisation

| System | Macro P@5 | Macro R@10 | Macro F1@10 (paper) | Macro MRR |
|---|---:|---:|---:|---:|
| prior_only (adaptive) | 0.5240 | 0.7834 | 0.5740 | 0.8003 |
| winner | 0.5280 | 0.7834 | 0.5765 | 0.8223 |
| best fixed alpha (tune-selected) | 0.5380 | 0.7967 | 0.5878 | 0.8477 |

| System | Count | Find | Track | Action | Filter |
|---|---:|---:|---:|---:|---:|
| prior_only | 0.390 | 0.910 | 0.800 | 0.090 | 0.430 |
| winner | 0.390 | 0.910 | 0.800 | 0.090 | 0.450 |

### Paired comparisons vs. the test split

| Comparison | ΔP@5 | 95% CI | p (query-level) | ΔP@5 (grouped) | grouped 95% CI | grouped p | Holm p | Holm reject |
|---|---:|---|---:|---:|---|---:|---:|---|
| winner_vs_prior | +0.0040 | [+0.0000, +0.0100] | 0.2833 | +0.0040 | [+0.0000, +0.0090] | 0.2073 | 0.5520 | False |
| winner_vs_best_fixed | -0.0100 | [-0.0200, -0.0020] | 0.0127 | -0.0100 | [-0.0203, -0.0022] | 0.0160 | 0.0380 | True |
| winner_vs_conf_only | -0.0040 | [-0.0100, +0.0000] | 0.2760 | -0.0040 | [-0.0102, +0.0000] | 0.2447 | 0.5520 | False |

### Per-category vs. prior-only (Holm over the five categories)

| Category | ΔP@5 | 95% CI | p | Holm p | Holm reject |
|---|---:|---|---:|---:|---|
| count | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| find | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| track | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| filter | +0.0200 | [+0.0000, +0.0500] | 0.2413 | 1.0000 | False |

## Verdict

**NULL** -- Holm-significant min-max gain: False (Holm-adjusted p = 1.0000); Δmacro-P@5 min-max = -0.0020, z-score = +0.0040.

Success criteria: IMPROVED requires (a) Holm-significant test macro-P@5 gain over prior-only under min-max normalisation and (b) a non-negative test macro-P@5 delta under z-score normalisation.


## Ceiling diagnostic -- label-dependent oracle over the whole alpha family

For every query the *best* alpha in [0.00, 1.00] (step 0.01) is chosen with the ground truth, i.e. an upper bound on what **any** alpha rule (prior, confidence, rank-based, temperature-scaled or otherwise) can reach with these two branch score vectors.  It is descriptive only and was never used to select a variant.

| Normalisation | Split | Oracle macro-P@5 | Prior-only macro-P@5 | Head-room |
|---|---|---:|---:|---:|
| minmax | tune | 0.5800 | 0.5800 | +0.0000 |
| minmax | test | 0.5380 | 0.5380 | +0.0000 |
| zscore | tune | 0.5800 | 0.5560 | +0.0240 |
| zscore | test | 0.5380 | 0.5240 | +0.0140 |

Under min-max normalisation the oracle equals the prior-only baseline exactly on both splits: the alpha-blend family is **exhausted**, so no confidence rule of any form can produce the min-max gain that the success criterion requires.


## Pipeline integrity check

The script's own pipeline must reproduce the control's published test numbers before its own numbers can be trusted.

| Normalisation | Method | Recomputed test macro-P@5 | Control reference | Match |
|---|---|---:|---:|---|
| minmax | adaptive | 0.5380 | 0.538 | True |
| minmax | conf_only | 0.5360 | 0.5359999999999999 | True |
| minmax | combined | 0.5380 | 0.538 | True |
| zscore | adaptive | 0.5240 | 0.524 | True |
| zscore | conf_only | 0.5320 | 0.5319999999999999 | True |
| zscore | combined | 0.5260 | 0.526 | True |

All checks match: **True**
