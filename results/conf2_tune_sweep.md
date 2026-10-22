# C1'' -- confidence variant sweep on the TUNE split (n=50)

**Pre-registration.** Every variant is selected on the tune split only (n=50).  The winner is the variant with the highest *mean tune macro-P@5* over the two branch normalisations (the success criterion spans both); the min-max tune macro-P@5 is the first tie-break, then mean tune macro F1@10 (paper definition) > mean tune macro R@10 > mean tune macro MRR > smallest mean |alpha - prior alpha| (least disruptive variant) > variant name (alphabetical).  The winner is frozen before the test split is touched and evaluated once.


**Amendment (tune-only, before any test-split evaluation of these variants).**  The first draft ranked the variants by the min-max tune macro-P@5 and broke ties with "smallest mean |alpha - prior alpha|".  All three variants score exactly the incumbent's 0.580 under min-max, so that chain froze a *no-op* variant (rank_conf with beta = 0, identical to the prior) and would have made the single permitted test evaluation vacuous.  The chain was therefore replaced by the tune-only chain stated above.  No test-split number of any of the three variants entered this decision.


## Tune results -- minmax normalisation

Prior-only (metadata adaptive alpha) tune macro-P@5 = 0.5800 (per-category: count 0.520, find 1.000, track 0.900, action 0.140, filter 0.340).

| Variant | Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | Tune macro R@10 | mean \|alpha-prior\| |
|---|---|---:|---:|---:|---:|
| prior_only (baseline) | metadata adaptive alpha | 0.5800 | 0.6035 | 0.7762 | 0.0000 |
| **rank_conf** (best of 5) | `{"beta": 0.0}` | 0.5800 | 0.6035 | 0.7762 | 0.0000 |
| **percat_blend** (best of 1) | `{"beta_by_cat": {"count": 0.5, "find": 0.5, "track": 0.5, "action": 0.5, "filter": 0.5}}` | 0.5800 | 0.6035 | 0.7762 | 0.0803 |
| **temp_margin** (best of 18) | `{"temp": 0.05, "blend": 0.75}` | 0.5800 | 0.6035 | 0.7762 | 0.0372 |

<details><summary>full trial grid</summary>


**rank_conf** -- rank-based branch agreement, log-odds adjustment of the prior

| Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | mean \|alpha-prior\| |
|---|---:|---:|---:|
| `{"beta": 0.0}` | 0.5800 | 0.6035 | 0.0000 |
| `{"beta": 0.25}` | 0.5800 | 0.6035 | 0.0328 |
| `{"beta": 0.5}` | 0.5800 | 0.6035 | 0.0612 |
| `{"beta": 1.0}` | 0.5760 | 0.5978 | 0.1058 |
| `{"beta": 2.0}` | 0.5400 | 0.5590 | 0.1619 |

**percat_blend** -- per-category blend between prior and margin confidence (fitted on tune)

| Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | mean \|alpha-prior\| |
|---|---:|---:|---:|
| `{"beta_by_cat": {"count": 0.5, "find": 0.5, "track": 0.5, "action": 0.5, "filter": 0.5}}` | 0.5800 | 0.6035 | 0.0803 |

**temp_margin** -- temperature-scaled (softmax) top-1 margins

| Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | mean \|alpha-prior\| |
|---|---:|---:|---:|
| `{"temp": 0.02, "blend": 0.25}` | 0.5760 | 0.5981 | 0.1556 |
| `{"temp": 0.02, "blend": 0.5}` | 0.5800 | 0.6035 | 0.1038 |
| `{"temp": 0.02, "blend": 0.75}` | 0.5800 | 0.6035 | 0.0519 |
| `{"temp": 0.05, "blend": 0.25}` | 0.5800 | 0.6035 | 0.1116 |
| `{"temp": 0.05, "blend": 0.5}` | 0.5800 | 0.6035 | 0.0744 |
| `{"temp": 0.05, "blend": 0.75}` | 0.5800 | 0.6035 | 0.0372 |
| `{"temp": 0.1, "blend": 0.25}` | 0.5800 | 0.6035 | 0.1212 |
| `{"temp": 0.1, "blend": 0.5}` | 0.5800 | 0.6035 | 0.0808 |
| `{"temp": 0.1, "blend": 0.75}` | 0.5800 | 0.6035 | 0.0404 |
| `{"temp": 0.25, "blend": 0.25}` | 0.5800 | 0.6035 | 0.1457 |
| `{"temp": 0.25, "blend": 0.5}` | 0.5800 | 0.6035 | 0.0972 |
| `{"temp": 0.25, "blend": 0.75}` | 0.5800 | 0.6035 | 0.0486 |
| `{"temp": 0.5, "blend": 0.25}` | 0.5800 | 0.6035 | 0.1501 |
| `{"temp": 0.5, "blend": 0.5}` | 0.5800 | 0.6035 | 0.1001 |
| `{"temp": 0.5, "blend": 0.75}` | 0.5800 | 0.6035 | 0.0500 |
| `{"temp": 1.0, "blend": 0.25}` | 0.5800 | 0.6035 | 0.1512 |
| `{"temp": 1.0, "blend": 0.5}` | 0.5800 | 0.6035 | 0.1008 |
| `{"temp": 1.0, "blend": 0.75}` | 0.5800 | 0.6035 | 0.0504 |

</details>


## Tune results -- zscore normalisation

Prior-only (metadata adaptive alpha) tune macro-P@5 = 0.5560 (per-category: count 0.520, find 0.960, track 0.900, action 0.140, filter 0.260).

| Variant | Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | Tune macro R@10 | mean \|alpha-prior\| |
|---|---|---:|---:|---:|---:|
| prior_only (baseline) | metadata adaptive alpha | 0.5560 | 0.5681 | 0.7667 | 0.0000 |
| **rank_conf** (best of 5) | `{"beta": 0.25}` | 0.5600 | 0.5758 | 0.7492 | 0.0328 |
| **percat_blend** (best of 1) | `{"beta_by_cat": {"count": 0.5, "find": 0.1, "track": 0.2, "action": 0.5, "filter": 0.0}}` | 0.5680 | 0.5869 | 0.7737 | 0.1266 |
| **temp_margin** (best of 18) | `{"temp": 1.0, "blend": 0.25}` | 0.5680 | 0.5902 | 0.7712 | 0.1470 |

<details><summary>full trial grid</summary>


**rank_conf** -- rank-based branch agreement, log-odds adjustment of the prior

| Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | mean \|alpha-prior\| |
|---|---:|---:|---:|
| `{"beta": 0.0}` | 0.5560 | 0.5681 | 0.0000 |
| `{"beta": 0.25}` | 0.5600 | 0.5758 | 0.0328 |
| `{"beta": 0.5}` | 0.5600 | 0.5748 | 0.0612 |
| `{"beta": 1.0}` | 0.5360 | 0.5511 | 0.1058 |
| `{"beta": 2.0}` | 0.5240 | 0.5440 | 0.1619 |

**percat_blend** -- per-category blend between prior and margin confidence (fitted on tune)

| Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | mean \|alpha-prior\| |
|---|---:|---:|---:|
| `{"beta_by_cat": {"count": 0.5, "find": 0.1, "track": 0.2, "action": 0.5, "filter": 0.0}}` | 0.5680 | 0.5869 | 0.1266 |

**temp_margin** -- temperature-scaled (softmax) top-1 margins

| Hyper-parameters | Tune macro-P@5 | Tune macro F1@10 | mean \|alpha-prior\| |
|---|---:|---:|---:|
| `{"temp": 0.02, "blend": 0.25}` | 0.5520 | 0.5604 | 0.1278 |
| `{"temp": 0.02, "blend": 0.5}` | 0.5600 | 0.5714 | 0.0852 |
| `{"temp": 0.02, "blend": 0.75}` | 0.5600 | 0.5758 | 0.0426 |
| `{"temp": 0.05, "blend": 0.25}` | 0.5600 | 0.5758 | 0.1113 |
| `{"temp": 0.05, "blend": 0.5}` | 0.5600 | 0.5758 | 0.0742 |
| `{"temp": 0.05, "blend": 0.75}` | 0.5600 | 0.5758 | 0.0371 |
| `{"temp": 0.1, "blend": 0.25}` | 0.5640 | 0.5835 | 0.1338 |
| `{"temp": 0.1, "blend": 0.5}` | 0.5640 | 0.5825 | 0.0892 |
| `{"temp": 0.1, "blend": 0.75}` | 0.5600 | 0.5758 | 0.0446 |
| `{"temp": 0.25, "blend": 0.25}` | 0.5640 | 0.5835 | 0.1449 |
| `{"temp": 0.25, "blend": 0.5}` | 0.5640 | 0.5825 | 0.0966 |
| `{"temp": 0.25, "blend": 0.75}` | 0.5600 | 0.5758 | 0.0483 |
| `{"temp": 0.5, "blend": 0.25}` | 0.5640 | 0.5835 | 0.1466 |
| `{"temp": 0.5, "blend": 0.5}` | 0.5640 | 0.5825 | 0.0977 |
| `{"temp": 0.5, "blend": 0.75}` | 0.5600 | 0.5758 | 0.0489 |
| `{"temp": 1.0, "blend": 0.25}` | 0.5680 | 0.5902 | 0.1470 |
| `{"temp": 1.0, "blend": 0.5}` | 0.5640 | 0.5825 | 0.0980 |
| `{"temp": 1.0, "blend": 0.75}` | 0.5600 | 0.5758 | 0.0490 |

</details>


## Selection (pre-registered, tune-only)

Selection metric: mean tune macro P@5 over both branch normalisations. Tie-break chain: mean tune macro P@5 over the two branch normalisations (min-max, z-score) > min-max tune macro P@5 > mean tune macro F1@10 (paper definition) > mean tune macro R@10 > mean tune macro MRR > smallest mean |alpha - prior alpha| (least disruptive variant) > variant name (alphabetical).

| Rank | Variant | Mean tune macro-P@5 | min-max tune macro-P@5 | Mean tune macro F1@10 |
|---:|---|---:|---:|---:|
| 1 | temp_margin | 0.5740 | 0.5800 | 0.5969 |
| 2 | percat_blend | 0.5740 | 0.5800 | 0.5952 |
| 3 | rank_conf | 0.5700 | 0.5800 | 0.5897 |

**Frozen winner: `temp_margin`**
