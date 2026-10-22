# C1''' round 3 -- frozen variant on the TEST split (n=100)

**Frozen variant:** `max_fusion` (params {}).  Selected on the tune split only; evaluated **once** on the test split.  Bootstrap resamples: 3000.


## minmax branch normalisation

| System | macro P@5 | R@10 | F1@10 (paper) | MRR |
|---|---:|---:|---:|---:|
| prior-only (additive blend) | 0.5380 | 0.7967 | 0.5878 | 0.8527 |
| frozen variant | 0.5380 | 0.7977 | 0.5883 | 0.8527 |
| delta | +0.0000 | +0.0010 | +0.0005 | +0.0000 |

| Category | prior P@5 | variant P@5 | delta |
|---|---:|---:|---:|
| count | 0.3900 | 0.3900 | +0.0000 |
| find | 0.9200 | 0.9200 | +0.0000 |
| track | 0.8100 | 0.8100 | +0.0000 |
| action | 0.0900 | 0.0900 | +0.0000 |
| filter | 0.4800 | 0.4800 | +0.0000 |

- primary paired bootstrap (macro P@5, within-category resampling): delta = +0.0000 [+0.0000, +0.0000], p = 1.0000
- grouped cluster bootstrap (5-minute groups): delta = +0.0000 [+0.0000, +0.0000], p = 1.0000
- sign-flip randomisation test (secondary): delta = +0.0000, p = 1.0000
- Wilcoxon signed-rank on per-query P@5 (secondary): p = 1.0000
- pooled per-query paired bootstrap (control-script convention): delta = +0.0000 [+0.0000, +0.0000], p = 1.0000; grouped: delta = +0.0000, p = 1.0000
- queries changed: 0 (better 0, worse 0)

| Per-category | delta P@5 | 95% CI | p | Holm p | Holm reject |
|---|---:|---|---:|---:|---|
| count | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| find | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| track | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| filter | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |

## zscore branch normalisation

| System | macro P@5 | R@10 | F1@10 (paper) | MRR |
|---|---:|---:|---:|---:|
| prior-only (additive blend) | 0.5240 | 0.7834 | 0.5740 | 0.8003 |
| frozen variant | 0.5380 | 0.7977 | 0.5883 | 0.8527 |
| delta | +0.0140 | +0.0143 | +0.0143 | +0.0524 |

| Category | prior P@5 | variant P@5 | delta |
|---|---:|---:|---:|
| count | 0.3900 | 0.3900 | +0.0000 |
| find | 0.9100 | 0.9200 | +0.0100 |
| track | 0.8000 | 0.8100 | +0.0100 |
| action | 0.0900 | 0.0900 | +0.0000 |
| filter | 0.4300 | 0.4800 | +0.0500 |

- primary paired bootstrap (macro P@5, within-category resampling): delta = +0.0140 [+0.0060, +0.0240], p = 0.0007
- grouped cluster bootstrap (5-minute groups): delta = +0.0140 [+0.0064, +0.0244], p = 0.0020
- sign-flip randomisation test (secondary): delta = +0.0140, p = 0.0140
- Wilcoxon signed-rank on per-query P@5 (secondary): p = 0.0180
- pooled per-query paired bootstrap (control-script convention): delta = +0.0140 [+0.0040, +0.0240], p = 0.0027; grouped: delta = +0.0140, p = < 0.0007
- queries changed: 7 (better 7, worse 0)

| Per-category | delta P@5 | 95% CI | p | Holm p | Holm reject |
|---|---:|---|---:|---:|---|
| count | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| find | +0.0100 | [+0.0000, +0.0300] | 0.7167 | 1.0000 | False |
| track | +0.0100 | [+0.0000, +0.0300] | 0.7080 | 1.0000 | False |
| action | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 1.0000 | False |
| filter | +0.0500 | [+0.0200, +0.0900] | 0.0067 | 0.0333 | True |

## Holm family (arm-level, pre-registered)

| Arm | raw p | Holm-adjusted p | reject |
|---|---:|---:|---|
| zscore | 0.0007 | 0.0013 | True |
| minmax | 1.0000 | 1.0000 | False |

## Pre-registered success criteria

| Criterion | Required | Observed | Met |
|---|---|---|---|
| (a) z-score Holm-significant gain | Holm p < 0.05, delta > 0 | delta = +0.0140, Holm p = 0.0013 | YES |
| (b) min-max non-negative | delta >= 0 | delta = +0.0000 | YES |
| (c) grouped direction | delta > 0 | delta = +0.0140 | YES |

**Verdict: IMPROVED** (strict cluster-significant variant: IMPROVED).
