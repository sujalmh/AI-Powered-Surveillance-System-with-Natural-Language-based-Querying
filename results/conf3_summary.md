# C1''' round 3 -- score-normalisation / combination-operator result

**Verdict: IMPROVED.**  One variant family was pre-registered on the TUNE split (n=50), one configuration was frozen, and the frozen variant was evaluated **once** on the TEST split (n=100) with paired and grouped bootstraps and Holm correction.


## 1. Literature ideas tried

| Family | Idea | Source |
|---|---|---|
| `max_fusion` | combMAX (disjunctive / t-conorm) combination: the branch with the stronger weighted opinion decides | Lee (1997) *Hybrid test collections and the fusion of retrieval systems*; combination-rule taxonomy of Fox & Shaw (1994) TREC-2 (https://trec.nist.gov/pubs/trec2/papers/txt/23.txt) and the UMass CIIR *Analysis of Multiple Evidence Combination* (https://ciir-publications.cs.umass.edu/getpdf.php?id=63) |
| `pnorm` | Hoelder / power-mean (generalised mean) family interpolating the additive blend (p=1) and the max (p=inf) | compensatory fuzzy combination operators in IR (R-norm family); score-normalisation survey *Data Fusion in Information Retrieval* (https://www.researchgate.net/publication/268724557_Data_Fusion_in_Information_Retrieval), which splits normalisation into linear (min-max, z-score) and non-linear (sigmoid/sum) families |
| `rrf` | Reciprocal Rank Fusion (rank-space, calibration-free) | Cormack, Clarke & Buettcher (2009), SIGIR (https://dl.acm.org/doi/10.1145/1571941.1572114, PDF https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) |

Considered but not implemented (they need a second index or a collection-level statistic that the released subset does not support): CORI-style collection-based belief (Cormack, Clarke & Buettcher, *Overview of document retrieval and clustering*, SIGIR 2001 -- https://dl.acm.org/doi/10.1145/502585.502600) and CSLS hubness correction (Conneau et al. 2018, https://arxiv.org/abs/1809.04760; Lample & Conneau 2019, https://arxiv.org/abs/1901.07291).  Score normalisation before fusion is the shared precondition of all of them: "each image must be mapped to a value that is normalized among all systems" (Zhou et al. 2010, https://www.fi.muni.cz/~xkohout7/Research/clanky_cizi/ranking/Zhou10.pdf); see also the RRF-vs-score-normalisation comparison at https://bigdataboutique.com/blog/hybrid-search-fusion-methods-compared.


## 2. Tune split (n=50) -- every variant, both calibrations

| Rank | Variant | Config | min-max tune P@5 | d | z-score tune P@5 | d | mean |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | max_fusion | `max_fusion` | 0.5800 | +0.0000 | 0.5800 | +0.0240 | 0.5800 |
| 2 | pnorm | `pnorm:p=inf` | 0.5800 | +0.0000 | 0.5800 | +0.0240 | 0.5800 |
| 3 | pnorm | `pnorm:p=16` | 0.5800 | +0.0000 | 0.5800 | +0.0240 | 0.5800 |
| 4 | pnorm | `pnorm:p=8` | 0.5800 | +0.0000 | 0.5800 | +0.0240 | 0.5800 |
| 5 | pnorm | `pnorm:p=4` | 0.5800 | +0.0000 | 0.5760 | +0.0200 | 0.5780 |
| 6 | pnorm | `pnorm:p=6` | 0.5800 | +0.0000 | 0.5760 | +0.0200 | 0.5780 |
| 7 | pnorm | `pnorm:p=3` | 0.5800 | +0.0000 | 0.5720 | +0.0160 | 0.5760 |
| 8 | pnorm | `pnorm:p=2` | 0.5800 | +0.0000 | 0.5680 | +0.0120 | 0.5740 |
| 9 | rrf | `rrf:k=10` | 0.5720 | -0.0080 | 0.5720 | +0.0160 | 0.5720 |
| 10 | rrf | `rrf:k=20` | 0.5640 | -0.0160 | 0.5640 | +0.0080 | 0.5640 |
| 11 | rrf | `rrf:k=60` | 0.5400 | -0.0400 | 0.5400 | -0.0160 | 0.5400 |

Baseline (metadata prior-only alpha, additive blend): min-max tune macro-P@5 0.5800, z-score tune macro-P@5 0.5560.  The `pnorm` grid is monotone in p, i.e. a dose-response from the additive blend (p=1) to combMAX (p=inf), and every p >= 2 is a no-op under min-max.


### Label oracle on the tune split (alpha x operator ceiling)

| Normalisation | prior (sum) | oracle alpha, sum | oracle alpha, combMAX | oracle alpha x operator | per-query ceiling |
|---|---:|---:|---:|---:|---:|
| minmax | 0.5800 | 0.5800 | 0.5800 | 0.5800 | 0.7520 |
| zscore | 0.5560 | 0.5800 | 0.5800 | 0.5800 | 0.7520 |

The alpha family is exhausted under min-max (the oracle equals the prior); under z-score the combMAX operator reaches the oracle of the whole (alpha x operator) family, so the residual gap to the per-query ceiling is unreachable by any weight or operator choice on this subset.

**Frozen after the pre-registered chain: `max_fusion`** (mean tune macro-P@5 0.5800).


## 3. Frozen variant on the TEST split (n=100)


### minmax branch normalisation

| System | macro P@5 | R@10 | F1@10 (paper) | MRR |
|---|---:|---:|---:|---:|
| prior-only (additive blend) | 0.5380 | 0.7967 | 0.5878 | 0.8527 |
| frozen combMAX variant | 0.5380 | 0.7977 | 0.5883 | 0.8527 |
| **delta** | **+0.0000** | +0.0010 | +0.0005 | +0.0000 |

- paired bootstrap, macro P@5 (within-category resampling): delta = +0.0000 [+0.0000, +0.0000], p = 1.0000, **Holm-adjusted p = 1.0000**, reject = False
- grouped cluster bootstrap (5-minute groups): delta = +0.0000 [+0.0000, +0.0000], p = 1.0000
- sign-flip randomisation (secondary): p = 1.0000; Wilcoxon signed-rank (secondary): p = 1.0000; pooled per-query paired bootstrap (control convention): delta = +0.0000, p = 1.0000
- per-category delta: count +0.0000, find +0.0000, track +0.0000, action +0.0000, filter +0.0000
- per-category Holm (family of 5): count 1.0000(keep), find 1.0000(keep), track 1.0000(keep), action 1.0000(keep), filter 1.0000(keep)
- per-query P@5 changed: 0 (better 0, worse 0)

### zscore branch normalisation

| System | macro P@5 | R@10 | F1@10 (paper) | MRR |
|---|---:|---:|---:|---:|
| prior-only (additive blend) | 0.5240 | 0.7834 | 0.5740 | 0.8003 |
| frozen combMAX variant | 0.5380 | 0.7977 | 0.5883 | 0.8527 |
| **delta** | **+0.0140** | +0.0143 | +0.0143 | +0.0524 |

- paired bootstrap, macro P@5 (within-category resampling): delta = +0.0140 [+0.0060, +0.0240], p = 0.0007, **Holm-adjusted p = 0.0013**, reject = True
- grouped cluster bootstrap (5-minute groups): delta = +0.0140 [+0.0064, +0.0244], p = 0.0020
- sign-flip randomisation (secondary): p = 0.0140; Wilcoxon signed-rank (secondary): p = 0.0180; pooled per-query paired bootstrap (control convention): delta = +0.0140, p = 0.0027
- per-category delta: count +0.0000, find +0.0100, track +0.0100, action +0.0000, filter +0.0500
- per-category Holm (family of 5): count 1.0000(keep), find 1.0000(keep), track 1.0000(keep), action 1.0000(keep), filter 0.0333(reject)
- per-query P@5 changed: 7 (better 7, worse 0)

### Changed test queries (z-score arm)

| qid | category | group | prior P@5 | variant P@5 |
|---|---|---|---:|---:|
| q_find_13 | find | 2026-03-01T18:10 | 0.80 | 1.00 |
| q_track_30 | track | 2026-03-01T18:20 | 0.80 | 1.00 |
| q_filter_01 | filter | 2026-03-01T17:00 | 0.00 | 0.20 |
| q_filter_03 | filter | 2026-03-01T17:05 | 0.00 | 0.20 |
| q_filter_09 | filter | 2026-03-01T18:05 | 0.60 | 0.80 |
| q_filter_15 | filter | 2026-03-01T18:10 | 0.40 | 0.60 |
| q_filter_19 | filter | 2026-03-01T18:15 | 0.80 | 1.00 |

The min-max arm changes **no** test query's P@5 (0 of 100) and its macro P@5 is bit-identical to the prior-only baseline; only R@10 moves by +0.001 (a re-ordering inside the top-10).  The gain is therefore not a re-tuning of the prior -- the prior alpha is never touched -- it is a repair of the z-score calibration.


**Mechanism.**  `zscore_norm` maps a branch through `sigmoid((x - mean) / sd)`, which saturates the *top* of the structured branch.  Measured on all 150 benchmark queries (label-free, `results/conf3_tune_sweep.json` -> `branch_contrast`):

| Calibration | structured contrast (top1 - median) | semantic contrast | structured range | semantic range |
|---|---:|---:|---:|---:|
| min-max | 0.800 | 0.441 | 1.000 | 1.000 |
| z-score | 0.441 | 0.385 | 0.546 | 0.809 |

so under z-score the structured branch loses most of its top-end contrast while the smooth semantic branch keeps its spread.  The additive blend therefore lets the semantic branch outvote the structured branch's own peak and pushes relevant clips out of the top-5.  A disjunctive (combMAX) combination keeps the structured branch's peak decisive and lets the semantic branch decide only when the structured branch has nothing to say (e.g. the action queries, whose structured score is flat because they carry no time cue).
  On the tune split the combMAX variant attains the per-query label oracle of the whole (alpha x operator) family in the z-score arm (oracle 0.5800 = variant 0.5800), so the remaining gap to the per-query ceiling (0.7520) is unreachable by any weight or operator choice on this subset.


## 4. Pre-registered success criteria

| Criterion | Required | Observed | Met |
|---|---|---|---|
| (a) z-score Holm-significant gain over prior-only | Holm p < 0.05, delta > 0 | delta = +0.0140, Holm p = 0.0013 | YES |
| (b) min-max non-negative | delta >= 0 | delta = +0.0000 | YES |
| (c) grouped cluster direction | delta > 0 | delta = +0.0140 | YES |

**Verdict: IMPROVED** (strict variant requiring cluster significance as well: IMPROVED).


## 5. Scope and limitations

- Exactly one variant was selected on the tune split and evaluated once on the test split; the script refuses a second test run of the same variant (`results/conf3_test_<variant>.json` already exists).  The `pnorm:p=8` / `pnorm:p=16` configurations tie `max_fusion` on the tune split but were **not** test-evaluated, so no claim is made about them.
- The prior alpha is reused unchanged from `backend.app.services.result_merger.compute_adaptive_alpha`; this round changes the *combination operator* only, not the weighting rule.
- The numbers are the released single-camera subset (camera_0, 121 indexed clips, 150 benchmark queries).  They are not the paper's 4-camera / 312k-descriptor deployment numbers and do not replace them.
- Labels come from released VLM captions, detector metadata and snapshots (no human annotation, no IAA available).
- The z-score branch is the repository's `sigmoid(z)` calibration (`backend/app/eval/baselines.zscore_norm`); the mechanism above is specific to a saturating, non-linear calibration and may not reproduce under a different z-score variant.


## 6. Commands

```
python backend/scripts/eval_retrieval_conf3.py --mode tune
python backend/scripts/eval_retrieval_conf3.py --mode test --winner max_fusion
python backend/scripts/eval_retrieval_conf3.py --mode report
```

## 7. New files

- `backend/scripts/eval_retrieval_conf3.py`
- `results/conf3_tune_sweep.json`, `results/conf3_tune_sweep.md`
- `results/conf3_test_max_fusion.json`, `results/conf3_test_max_fusion.md`
- `results/conf3_summary.json`, `results/conf3_summary.md`
