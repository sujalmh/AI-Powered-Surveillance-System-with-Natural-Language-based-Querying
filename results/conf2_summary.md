# C1'' -- confidence-variant round 2: consolidated result

**Verdict: NULL.**  A new confidence/blend rule (`temp_margin`) was pre-registered on the tune split (n=50) and evaluated **once** on the test split (n=100) against the metadata prior-only baseline, with the same paired and grouped bootstrap and Holm correction as the control script.


## 1. Variant search on the TUNE split (n=50)

Three variants were explored; nothing else was tried.  The winner is the variant with the highest mean tune macro-P@5 over the two branch normalisations (min-max tune macro-P@5 is the first tie-break).

| Variant | Description | min-max tune macro-P@5 | z-score tune macro-P@5 | mean | mean tune macro F1@10 |
|---|---|---:|---:|---:|---:|
| prior_only (baseline) | metadata adaptive alpha | 0.5800 | 0.5560 | 0.5680 | 0.5858 |
| rank_conf | rank-based branch agreement, log-odds adjustment of the prior | 0.5800 | 0.5600 | 0.5700 | 0.5897 |
| percat_blend | per-category blend between prior and margin confidence (fitted on tune) | 0.5800 | 0.5680 | 0.5740 | 0.5952 |
| temp_margin **(winner)** | temperature-scaled (softmax) top-1 margins | 0.5800 | 0.5680 | 0.5740 | 0.5969 |

Frozen hyper-parameters (best tune trial per normalisation): `{"minmax": {"temp": 0.05, "blend": 0.75}, "zscore": {"temp": 1.0, "blend": 0.25}}`.


## 2. Frozen winner on the TEST split (n=100)


### minmax normalisation

| System | Macro P@5 | Count | Find | Track | Action | Filter |
|---|---:|---:|---:|---:|---:|---:|
| prior_only (adaptive) | 0.5380 | 0.390 | 0.920 | 0.810 | 0.090 | 0.480 |
| winner | 0.5360 | 0.390 | 0.920 | 0.810 | 0.090 | 0.470 |

| Comparison | ΔP@5 | 95% CI | p (query) | ΔP@5 (grouped) | grouped 95% CI | grouped p | Holm p | Holm reject |
|---|---:|---|---:|---:|---|---:|---:|---|
| winner_vs_prior | -0.0020 | [-0.0060, +0.0000] | 0.6953 | -0.0020 | [-0.0070, +0.0000] | 0.7113 | 1.0000 | False |
| winner_vs_best_fixed | +0.0000 | [+0.0000, +0.0000] | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| winner_vs_conf_only | +0.0000 | [+0.0000, +0.0000] | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |

### zscore normalisation

| System | Macro P@5 | Count | Find | Track | Action | Filter |
|---|---:|---:|---:|---:|---:|---:|
| prior_only (adaptive) | 0.5240 | 0.390 | 0.910 | 0.800 | 0.090 | 0.430 |
| winner | 0.5280 | 0.390 | 0.910 | 0.800 | 0.090 | 0.450 |

| Comparison | ΔP@5 | 95% CI | p (query) | ΔP@5 (grouped) | grouped 95% CI | grouped p | Holm p | Holm reject |
|---|---:|---|---:|---:|---|---:|---:|---|
| winner_vs_prior | +0.0040 | [+0.0000, +0.0100] | 0.2833 | +0.0040 | [+0.0000, +0.0090] | 0.2073 | 0.5520 | False |
| winner_vs_best_fixed | -0.0100 | [-0.0200, -0.0020] | 0.0127 | -0.0100 | [-0.0203, -0.0022] | 0.0160 | 0.0380 | True |
| winner_vs_conf_only | -0.0040 | [-0.0100, +0.0000] | 0.2760 | -0.0040 | [-0.0102, +0.0000] | 0.2447 | 0.5520 | False |

## 3. Why the null is structural, not a search failure

Upper bound of the *entire* alpha family: for every query the best alpha in [0.00, 1.00] is picked with the ground truth.  Any alpha rule (prior, confidence, rank-based, temperature-scaled, ...) is bounded by it.

| Normalisation | Split | Oracle macro-P@5 | Prior-only macro-P@5 | Head-room |
|---|---|---:|---:|---:|
| minmax | tune | 0.5800 | 0.5800 | +0.0000 |
| minmax | test | 0.5380 | 0.5380 | +0.0000 |
| zscore | tune | 0.5800 | 0.5560 | +0.0240 |
| zscore | test | 0.5380 | 0.5240 | +0.0140 |

Under min-max normalisation the oracle equals the prior exactly, so the required min-max gain is unreachable by construction.  Under z-score normalisation the family has head-room, and the winner recovers part of it without reaching significance.


## 4. Verdict against the pre-registered success criteria

| Criterion | Required | Observed | Met |
|---|---|---|---|
| (a) min-max test macro-P@5 | Holm-significant gain over prior-only | Δ = -0.0020, Holm-adjusted p = 1.0000 | NO |
| (b) z-score test macro-P@5 | same direction (Δ >= 0) | Δ = +0.0040 | yes |

**Verdict: NULL.**  IMPROVED requires (a) Holm-significant test macro-P@5 gain over prior-only under min-max normalisation and (b) a non-negative test macro-P@5 delta under z-score normalisation.


## 5. Commands

```
python backend/scripts/eval_retrieval_conf2.py --mode tune
python backend/scripts/eval_retrieval_conf2.py --mode test --winner temp_margin
python backend/scripts/eval_retrieval_conf2.py --mode diagnostics
python backend/scripts/eval_retrieval_conf2.py --mode report
```

New files: `backend/scripts/eval_retrieval_conf2.py`, `results/conf2_tune_sweep.json`, `results/conf2_tune_sweep.md`, `results/conf2_test_temp_margin.json`, `results/conf2_test_temp_margin.md`, `results/conf2_diagnostics.json`, `results/conf2_summary.json`, `results/conf2_summary.md`.
