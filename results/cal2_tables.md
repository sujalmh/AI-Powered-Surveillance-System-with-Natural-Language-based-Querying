# Round-2 literature-motivated quality-gate search (cal2)

Pre-registered two-stage protocol (`backend/scripts/eval_fusion_cal2.py`). Stage CAL scored every variant on **calibration data only**; exactly one winner was frozen and evaluated **once** on the test split. The round-1 winner `F2_lam1` appears as a reference row only.

## Stage CAL: candidate table (calibration data only)

| Variant | Family | lam | cal P@5 | d P@5 vs concat [95% CI] | cal pair AUC | cal 5-fold P@5 | d 5-fold | gate dev. | A1 | A2 | feasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| concat (reference) | - | - | 0.5927 | +0.0000 | 0.8552 | 0.8678 | +0.0000 | 0.0000 | - | - | - |
| gated (reference) | - | - | 0.5675 | -0.0252 | 0.8192 | 0.8461 | -0.0217 | 0.2489 | - | - | - |
| F2_lam1 (reference) | - | - | 0.6134 | +0.0207 | 0.8237 | 0.8606 | -0.0072 | 0.1485 | - | - | - |
| Grank_lam1 | Grank | 1 | 0.6102 | +0.0175 [+0.0053, +0.0293] | 0.8506 | 0.8761 | +0.0083 | 0.0480 | True | True | True |
| Grank_lam2 **(WINNER)** | Grank | 2 | 0.6268 | +0.0341 [+0.0146, +0.0528] | 0.8382 | 0.8757 | +0.0079 | 0.0931 | True | True | True |
| Grank_lam4 | Grank | 4 | 0.6138 | +0.0211 [-0.0028, +0.0447] | 0.8243 | 0.8627 | -0.0051 | 0.1693 | True | True | False |
| Grank_lam8 | Grank | 8 | 0.6081 | +0.0154 [-0.0085, +0.0390] | 0.7831 | 0.8500 | -0.0177 | 0.2680 | True | True | False |
| Ginv_lam0.5 | Ginv | 0.5 | 0.5846 | -0.0081 [-0.0199, +0.0028] | 0.8470 | 0.8700 | +0.0022 | 0.1086 | True | False | False |
| Ginv_lam1 | Ginv | 1 | 0.5870 | -0.0057 [-0.0236, +0.0110] | 0.8237 | 0.8484 | -0.0194 | 0.2032 | True | False | False |
| Ginv_lam2 | Ginv | 2 | 0.5585 | -0.0341 [-0.0557, -0.0134] | 0.7925 | 0.8250 | -0.0428 | 0.3313 | True | False | False |
| Ginv_lam4 | Ginv | 4 | 0.5333 | -0.0593 [-0.0793, -0.0402] | 0.7951 | 0.8218 | -0.0460 | 0.4088 | True | False | False |

Winner vs static concat on calibration, track-grouped paired bootstrap: dP@5 = +0.0341 [-0.0055, +0.0777], p = 0.0940.


Admissible candidates: 4 / 8; feasible: 2; rule applied: `A1+A2+S1+S2 (all filters satisfied)`.  
**Frozen winner: `Grank_lam2`** (spec hash `44cfdb542086`).


## Stage TEST: frozen winner

Variant `Grank_lam2` (family Grank, lam=2); calibration file `cal2_calibration.json`, spec hash `44cfdb542086`.

### Retrieval P@5 (test split)

| Method | P@5 | R@10 | F1@10 |
|---|---:|---:|---:|
| concat | 0.4274 | 0.3110 | 0.2887 |
| gated | 0.4442 | 0.3065 | 0.2889 |
| winner | 0.4721 | 0.3443 | 0.3186 |
| F2_lam1 | 0.4696 | 0.3347 | 0.3113 |

### Paired bootstrap (Holm over 2 per task)

| Task | Comparison | Delta [95% CI] | raw p | Holm p | reject |
|---|---|---:|---:|---:|:--:|
| retrieval P@5 | winner vs concat | +0.0447 [+0.0339, +0.0561] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | winner vs gated | +0.0279 [+0.0188, +0.0368] | < 0.0007 | < 0.0007 | True |
| pair AUC | winner vs concat | +0.0079 [-0.0085, +0.0241] | 0.3547 | 0.3547 | False |
| pair AUC | winner vs gated | +0.0227 [-0.0008, +0.0477] | 0.0580 | 0.1160 | False |

Secondary (not in the Holm family): winner vs the round-1 winner `F2_lam1`.

| Task | Delta [95% CI] | raw p |
|---|---:|---:|
| retrieval P@5 | +0.0025 [-0.0039, +0.0087] | 0.4413 |
| pair AUC | +0.0079 [-0.0075, +0.0228] | 0.3220 |

Secondary statistic (not part of the pre-registered decision): paired bootstrap resampling whole pseudo-tracks.

| Task | Comparison | Delta [95% CI] | raw p |
|---|---|---:|---:|
| retrieval P@5 (track-grouped) | winner vs concat | +0.0447 [+0.0235, +0.0675] | < 0.0007 |
| retrieval P@5 (track-grouped) | winner vs gated | +0.0279 [+0.0122, +0.0457] | < 0.0007 |

### Gate mechanism (mean effective branch weights over all detections)

| Gate | mean gw | visual weight | text weight |
|---|---:|---:|---:|
| static concat | 0.5000 | 0.6000 | 0.4000 |
| released gate (lambda=4, raw qualities) | 0.3247 | 0.3896 | 0.5403 |
| round-1 winner F2_lam1 (z-standardised, lambda=1) | 0.4986 | 0.5983 | 0.4011 |
| round-2 winner `Grank_lam2` (percentile-rank, lambda=2) | 0.5113 | 0.6135 | 0.3910 |

### Pair classification (threshold from calibration pairs)

| Method | Threshold | test AUC | test F1 | cal AUC |
|---|---:|---:|---:|---:|
| concat | 0.9297 | 0.7263 | 0.7349 | 0.8552 |
| gated | 0.8679 | 0.7115 | 0.7280 | 0.8192 |
| winner | 0.8896 | 0.7341 | 0.7232 | 0.8382 |
| F2_lam1 | 0.8594 | 0.7263 | 0.7225 | 0.8237 |

### 5-fold whole-track CV (P@5)

| Fold | winner | concat | released gated |
|---|---:|---:|---:|
| 0 | 0.6812 | 0.6430 | 0.6127 |
| 1 | 0.7043 | 0.6814 | 0.6681 |
| 2 | 0.7484 | 0.7239 | 0.7031 |
| 3 | 0.7045 | 0.7197 | 0.6408 |
| 4 | 0.7646 | 0.7292 | 0.6944 |
| **mean** | **0.7206** | 0.6994 | 0.6638 |

### Temporal halves (test pair AUC)

| Half | n pairs | concat | gated | winner | F2_lam1 |
|---|---:|---:|---:|---:|---:|
| early | 506 | 0.6564 | 0.6318 | 0.6585 | 0.6541 |
| late | 502 | 0.7954 | 0.7934 | 0.8009 | 0.7950 |

### Retrieval P@5 by query-timestamp half

| Half | n queries | concat | gated | winner | F2_lam1 |
|---|---:|---:|---:|---:|---:|
| early | 595 | 0.3818 | 0.3983 | 0.4020 | 0.4128 |
| late | 595 | 0.4729 | 0.4901 | 0.5422 | 0.5264 |

### Reproduction check against the released `fusion_results.json`

Maximum absolute difference over the recomputed reference quantities: **0.00e+00**.


### Verdict

- C1 retrieval Holm-significant win vs static concat: **True**
- C2 pair AUC not significantly worse than the released gate: **True**
- C2b pair AUC not significantly worse than static concat: **True**
- C3 5-fold CV mean within 0.005 of static concat: **True**
- **VERDICT: IMPROVED**


### Caveats

- The calibration split is small (70 pseudo-tracks, 492 retrieval queries); the track-grouped bootstrap on the winner's calibration delta is the honest uncertainty statement for the selection.
- F2_lam1 (the round-1 winner) is reported here as a reference row only; it was frozen and evaluated on test by round 1 (results/cal_gate_results.json) and was NOT re-selected here.
- The frozen normaliser constants are estimated on calibration detections only and applied unchanged to every detection; no test-split label, score or threshold enters the gate.
- Same-track labels are deterministic pseudo-labels formed from detector/attribute metadata, not human identities; the benchmark is the released single-camera subset, not the paper's 312k-descriptor / 4-camera identity benchmark.

