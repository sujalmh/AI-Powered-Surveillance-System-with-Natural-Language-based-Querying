# C1 robustness study on the released subset

Benchmark: `backend/benchmarks/queries_control.jsonl` (protocol-matched control, 50 tune / 100 test). These are **not** the paper's 4-camera / 312k-descriptor results.


## Reference methods (test macro P@5)

| Method | Macro P@5 |
|---|---:|
| fixed_0.0 | 0.048 |
| fixed_0.3 | 0.380 |
| fixed_0.5 | 0.534 |
| fixed_0.7 | 0.538 |
| fixed_1.0 | 0.520 |
| adaptive_default | 0.538 |
| paper_lookup | 0.428 |
| learned_alpha (logistic) | 0.534 |


## Soft-weight sensitivity

Grid of 16 weight triples ($w_\text{intent}+w_\text{complexity}+w_\text{source}=1$), searched on the tune split only.

| $w_I$ | $w_\text{cplx}$ | $w_\text{src}$ | tune P@5 | test P@5 |
|---:|---:|---:|---:|---:|
| 0.2 | 0.3 | 0.5 | 0.580 | 0.538 |
| 0.2 | 0.4 | 0.4 | 0.580 | 0.538 |
| 0.2 | 0.5 | 0.3 | 0.580 | 0.538 |
| 0.4 | 0.1 | 0.5 | 0.576 | 0.538 |
| 0.4 | 0.2 | 0.4 | 0.580 | 0.538 |
| 0.4 | 0.3 | 0.3 | 0.580 | 0.538 |
| 0.4 | 0.4 | 0.2 | 0.580 | 0.538 |
| 0.4 | 0.5 | 0.1 | 0.580 | 0.538 |
| 0.6 | 0.0 | 0.4 | 0.580 | 0.538 |
| 0.6 | 0.1 | 0.3 | 0.580 | 0.538 |
| 0.6 | 0.2 | 0.2 | 0.580 | 0.538 |
| 0.6 | 0.3 | 0.1 | 0.580 | 0.538 |
| 0.6 | 0.4 | 0.0 | 0.580 | 0.538 |
| 0.8 | 0.0 | 0.2 | 0.580 | 0.538 |
| 0.8 | 0.1 | 0.1 | 0.580 | 0.538 |
| 0.8 | 0.2 | 0.0 | 0.580 | 0.538 |

Test macro P@5 across the grid: min 0.538, median 0.538, p90 0.538, max 0.538. Default (0.6/0.2/0.2) = 0.538; best-on-tune = 0.538 (weights 0.6/0.0/0.4).

Soft-term isolation: overrides + intent prior only = 0.538; with the soft terms = 0.538 (delta +0.000).


## Learned-alpha competitor (R1-2)

- model: multinomial logistic regression on [intent one-hot, n_s, n_v, |F|, count, action]
- train / test: 50 / 100 queries
- test macro P@5: 0.534 (closed-form rule: 0.538; per-intent lookup: 0.428)
- the learned alpha equals the oracle-best alpha on 98% of test queries

## Count-branch accuracy and the label-noise ceiling (R1-5)

- n count queries: 30
- window-mode caption estimate vs anchor caption count: EM 0.100, MAE 2.200
- attainable ceiling for any window-based predictor: EM 1.000, MAE floor 0.000 (mean window count spread 4.47 persons)
- caption count spread inside one clip (67 clips): mean 0.63, max 2 persons

## What cannot be reproduced

- The released subset is single-camera with metadata-derived labels.
- The learned-alpha model is trained on 50 tune queries; the class distribution of the oracle-best alpha label is reported with it.
- The count ground truth is the released VLM caption count; the frame-level caption/detection disagreement bounds any count accuracy.
