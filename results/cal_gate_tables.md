# Quality-gate variant search: calibration table and frozen test

Pre-registered two-stage protocol (`backend/scripts/eval_fusion_cal.py`). Stage CAL scored every variant on **calibration data only**; exactly one winner was frozen and evaluated **once** on the test split.

## Stage CAL: candidate table (calibration data only)

| Variant | Family | lam | beta | cal P@5 | d P@5 vs concat [95% CI] | cal pair AUC | cal 5-fold P@5 | gate dev. | A1 | A2 | feasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| concat (reference) | static | - | - | 0.5927 | - | 0.8552 | 0.8678 | 0.0000 | - | - | - |
| gated lam=4 (reference) | released | 4 | 1 | 0.5675 | -0.0252 | 0.8192 | 0.8461 | 0.2489 | - | - | - |
| F1_lam0.5 | F1 | 0.5 | 1 | 0.5825 | -0.0102 [-0.0167, -0.0037] | 0.8543 | 0.8684 | 0.0394 | True | False | False |
| F1_lam1 | F1 | 1 | 1 | 0.5752 | -0.0175 [-0.0285, -0.0073] | 0.8484 | 0.8671 | 0.0777 | True | False | False |
| F1_lam2 | F1 | 2 | 1 | 0.5650 | -0.0276 [-0.0447, -0.0114] | 0.8329 | 0.8532 | 0.1473 | True | False | False |
| F1_lam4 | F1 | 4 | 1 | 0.5675 | -0.0252 [-0.0467, -0.0045] | 0.8192 | 0.8461 | 0.2489 | True | False | False |
| F1_lam8 | F1 | 8 | 1 | 0.5541 | -0.0386 [-0.0602, -0.0187] | 0.8128 | 0.8436 | 0.3316 | True | False | False |
| F2_lam1 **(WINNER)** | F2 | 1 | 1 | 0.6134 | +0.0207 [+0.0012, +0.0394] | 0.8237 | 0.8606 | 0.1485 | True | True | False |
| F2_lam2 | F2 | 2 | 1 | 0.5980 | +0.0053 [-0.0187, +0.0280] | 0.7761 | 0.8354 | 0.2447 | True | True | False |
| F2_lam4 | F2 | 4 | 1 | 0.5533 | -0.0394 [-0.0602, -0.0203] | 0.7501 | 0.8178 | 0.3365 | True | False | False |
| F2_lam8 | F2 | 8 | 1 | 0.5350 | -0.0577 [-0.0772, -0.0390] | 0.7472 | 0.8086 | 0.3761 | True | False | False |
| F2_lam16 | F2 | 16 | 1 | 0.5268 | -0.0659 [-0.0854, -0.0484] | 0.7363 | 0.8109 | 0.3930 | True | False | False |
| F3_lam4_beta0 | F3 | 4 | 0 | 0.5927 | +0.0000 [+0.0000, +0.0000] | 0.8552 | 0.8678 | 0.0000 | False | False | False |
| F3_lam4_beta0.25 | F3 | 4 | 0.25 | 0.5805 | -0.0122 [-0.0211, -0.0037] | 0.8534 | 0.8674 | 0.0622 | True | False | False |
| F3_lam4_beta0.5 | F3 | 4 | 0.5 | 0.5707 | -0.0220 [-0.0358, -0.0093] | 0.8415 | 0.8627 | 0.1244 | True | False | False |
| F3_lam4_beta0.75 | F3 | 4 | 0.75 | 0.5553 | -0.0374 [-0.0549, -0.0207] | 0.8280 | 0.8496 | 0.1867 | True | False | False |
| F3_lam4_beta1 | F3 | 4 | 1 | 0.5675 | -0.0252 [-0.0467, -0.0045] | 0.8192 | 0.8461 | 0.2489 | True | False | False |

Winner vs static concat on calibration, track-grouped paired bootstrap: dP@5 = +0.0207 [-0.0208, +0.0649], p = 0.3400.


Selection rule: A1: a configuration with gate deviation 0 is the static baseline itself and is not admissible; A2: admissible candidates must beat the static baseline on calibration retrieval P@5; S1: feasible = (cal_kfold_p5 >= cal_kfold_p5(concat) - 0.005) and (cal_pair_auc >= cal_pair_auc(released gated) - 0.005); S2: argmax cal_retrieval_p5, ties -> cal_pair_auc -> smallest gate deviation; S3: if empty relax the CV filter, then the AUC filter, then take the global maximum cal_retrieval_p5  
Admissible candidates: 2 / 15; feasible: 0; rule applied: `A1+A2+S3 (CV filter relaxed)`.  
**Frozen winner: `F2_lam1`** (spec hash `4ef151e22575`).


## Stage TEST: frozen winner

Variant `F2_lam1` (family F2, lam=1, beta=1); calibration file `cal_gate_calibration.json`, spec hash `4ef151e22575`.

### Retrieval P@5 (test split)

| Method | P@5 | R@10 | F1@10 |
|---|---:|---:|---:|
| concat | 0.4274 | 0.3110 | 0.2887 |
| gated | 0.4442 | 0.3065 | 0.2889 |
| winner | 0.4696 | 0.3347 | 0.3113 |
| randproj | 0.4165 | 0.2973 | 0.2781 |

### Paired bootstrap (Holm over 2 per task)

| Task | Comparison | Delta [95% CI] | raw p | Holm p | reject |
|---|---|---:|---:|---:|:--:|
| retrieval P@5 | winner vs concat | +0.0422 [+0.0323, +0.0524] | < 0.0007 | < 0.0007 | True |
| retrieval P@5 | winner vs gated | +0.0254 [+0.0187, +0.0321] | < 0.0007 | < 0.0007 | True |
| pair AUC | winner vs concat | +0.0000 [-0.0235, +0.0236] | 0.9873 | 0.9873 | False |
| pair AUC | winner vs gated | +0.0148 [-0.0041, +0.0334] | 0.1300 | 0.2600 | False |

Secondary statistic (not part of the pre-registered decision): paired bootstrap resampling whole pseudo-tracks.

| Task | Comparison | Delta [95% CI] | raw p |
|---|---|---:|---:|
| retrieval P@5 (track-grouped) | winner vs concat | +0.0422 [+0.0236, +0.0616] | < 0.0007 |
| retrieval P@5 (track-grouped) | winner vs gated | +0.0254 [+0.0154, +0.0374] | < 0.0007 |

### Gate mechanism (mean effective branch weights over all detections)

| Gate | mean gw | visual weight | text weight |
|---|---:|---:|---:|
| static concat | 0.5000 | 0.6000 | 0.4000 |
| released gate (lambda=4, raw qualities) | 0.3247 | 0.3896 | 0.5403 |
| winner (z-normalised, lambda=1) | 0.4986 | 0.5983 | 0.4011 |

### Pair classification (threshold from calibration pairs)

| Method | Threshold | test AUC | test F1 | cal AUC |
|---|---:|---:|---:|---:|
| concat | 0.9297 | 0.7263 | 0.7349 | 0.8552 |
| gated | 0.8679 | 0.7115 | 0.7280 | 0.8192 |
| winner | 0.8594 | 0.7263 | 0.7225 | 0.8237 |

### 5-fold whole-track CV (P@5)

| Fold | winner | concat | released gated |
|---|---:|---:|---:|
| 0 | 0.6504 | 0.6430 | 0.6127 |
| 1 | 0.6957 | 0.6814 | 0.6681 |
| 2 | 0.7384 | 0.7239 | 0.7031 |
| 3 | 0.6824 | 0.7197 | 0.6408 |
| 4 | 0.7422 | 0.7292 | 0.6944 |
| **mean** | **0.7018** | 0.6994 | 0.6638 |

### Temporal halves (test pair AUC)

| Half | n pairs | concat | gated | winner |
|---|---:|---:|---:|---:|
| early | 506 | 0.6564 | 0.6318 | 0.6541 |
| late | 502 | 0.7954 | 0.7934 | 0.7950 |

### Retrieval P@5 by query-timestamp half

| Half | n queries | concat | gated | winner |
|---|---:|---:|---:|---:|
| early | 595 | 0.3818 | 0.3983 | 0.4128 |
| late | 595 | 0.4729 | 0.4901 | 0.5264 |

### Reproduction check against the released `fusion_results.json`

Maximum absolute difference over the recomputed reference quantities (concat / released-gated retrieval P@5, pair AUC, 5-fold means): **0.00e+00**.


### Verdict

- C1 retrieval Holm-significant win vs static concat: **True**
- C2 pair AUC not significantly worse than the released gate: **True**
- C3 5-fold CV mean within 0.005 of static concat: **True**
- **VERDICT: IMPROVED**


### Caveats

- The calibration split is small (70 pseudo-tracks, 492 retrieval queries): the track-grouped bootstrap on the winner's calibration delta includes zero, so the pre-registered selection was weakly resolved -- only 2 of 15 candidates were admissible under A1+A2 and the winner was the admissible one with the larger calibration retrieval delta.
- The calibration and test splits disagree in sign for the released raw-quality gate (calibration dP@5 = -0.025, test dP@5 = +0.017); the z-normalised winner is the only family that improved calibration retrieval over the static baseline, which is why the pre-registered screen selected it.
- The z-normalisation constants are estimated on calibration detections only and applied unchanged to every detection; no test-split label, score or threshold enters the gate.
- The winner is a re-weighting of the same two branches with the same released metadata quality proxies; it does not add a new encoder, embedding or label source.
- Same-track labels are deterministic pseudo-labels formed from detector/attribute metadata, not human identities; the benchmark is the released single-camera subset, not the paper's 312k-descriptor / 4-camera identity benchmark.
- The Holm family per task is the 2-member head-to-head family used by eval_fusion.py (winner vs concat, winner vs released gate); the track-grouped bootstrap is a secondary statistic and is not part of the decision rule.

