# C1''' round 3 -- tune-split sweep of combination operators

**Selection split: TUNE only (n=50).**  The test split was not read while these variants were designed or selected.  The baseline is the repository's own metadata prior-only alpha with the additive blend (`a*s + (1-a)*e`).


## Literature basis

- **max_fusion** -- combMAX (disjunctive / t-conorm) combination rule: Lee (1997) 'Hybrid test collections and the fusion of retrieval systems'; the combination-rule taxonomy of Fox & Shaw (1994) TREC-2 and of the UMass CIIR 'Analysis of Multiple Evidence Combination' report (CombMIN / CombMAX / CombSUM / CombMNZ).
- **pnorm** -- Hoelder / power-mean (generalised mean) combination: the interpolation between the additive blend (p=1) and the disjunctive max (p=inf), i.e. the family of compensatory fuzzy operators used for IR evidence combination (R-norm family).
- **rrf** -- Reciprocal Rank Fusion: Cormack, Clarke & Buettcher (2009), 'Reciprocal rank fusion outperforms Condorcet and individual rank learning methods', SIGIR 2009.

## All configurations, per branch normalisation

| Variant | Config | minmax tune P@5 | d(minmax) | z-score tune P@5 | d(z-score) | mean | changed q (mm/zs) |
|---|---|---:|---:|---:|---:|---:|---:|
| max_fusion | `max_fusion` | 0.5800 | +0.0000 | 0.5800 | +0.0240 | 0.5800 | 0/5 |
| pnorm | `pnorm:p=inf` | 0.5800 | +0.0000 | 0.5800 | +0.0240 | 0.5800 | 0/5 |
| pnorm | `pnorm:p=16` | 0.5800 | +0.0000 | 0.5800 | +0.0240 | 0.5800 | 0/5 |
| pnorm | `pnorm:p=8` | 0.5800 | +0.0000 | 0.5800 | +0.0240 | 0.5800 | 0/5 |
| pnorm | `pnorm:p=4` | 0.5800 | +0.0000 | 0.5760 | +0.0200 | 0.5780 | 0/5 |
| pnorm | `pnorm:p=6` | 0.5800 | +0.0000 | 0.5760 | +0.0200 | 0.5780 | 0/5 |
| pnorm | `pnorm:p=3` | 0.5800 | +0.0000 | 0.5720 | +0.0160 | 0.5760 | 0/4 |
| pnorm | `pnorm:p=2` | 0.5800 | +0.0000 | 0.5680 | +0.0120 | 0.5740 | 0/3 |
| rrf | `rrf:k=10` | 0.5720 | -0.0080 | 0.5720 | +0.0160 | 0.5720 | 2/6 |
| rrf | `rrf:k=20` | 0.5640 | -0.0160 | 0.5640 | +0.0080 | 0.5640 | 3/6 |
| rrf | `rrf:k=60` | 0.5400 | -0.0400 | 0.5400 | -0.0160 | 0.5400 | 9/10 |

Baseline (prior-only, additive blend): minmax tune macro-P@5 0.5800, z-score tune macro-P@5 0.5560.


## Pre-registered selection chain

- 1. mean tune macro-P@5 over {minmax, zscore}
- 2. minmax tune macro-P@5
- 3. mean tune macro F1@10 (paper)
- 4. mean tune macro R@10
- 5. mean tune macro MRR
- 6. variant name (alphabetical), then configuration key ascending

**Frozen winner: `max_fusion`** (mean tune macro-P@5 0.5800; minmax 0.5800, z-score 0.5800).
