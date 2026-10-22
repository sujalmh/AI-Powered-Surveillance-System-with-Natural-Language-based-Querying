# Round-2 exploratory calibration sweep (calibration only)

Every gate configuration explored before the round-2 pre-registration was frozen.  **No test-split quantity is computed or read here**: the sweep uses calibration detections, calibration pairs, calibration-track retrieval and a calibration-track 5-fold CV only.  It is reported in full so that no variant is hidden.

| Variant | Family | cal P@5 | d P@5 vs concat | cal pair AUC | cal 5-fold P@5 | d 5-fold vs concat | gate dev. |
|---|---|---:|---:|---:|---:|---:|---:|
| concat_ref | static | 0.5927 | +0.0000 | 0.8552 | 0.8678 | +0.0000 | 0.0000 |
| released_F1_lam4 | F1-released | 0.5675 | -0.0252 | 0.8192 | 0.8461 | -0.0217 | 0.2489 |
| F2_lam1_round1 | round1 | 0.6134 | +0.0207 | 0.8237 | 0.8606 | -0.0072 | 0.1485 |
| F2z_lam0.25 | F2-z | 0.5911 | -0.0016 | 0.8543 | 0.8687 | +0.0010 | 0.0423 |
| F2z_lam0.5 | F2-z | 0.6114 | +0.0187 | 0.8447 | 0.8729 | +0.0051 | 0.0815 |
| F2z_lam0.75 | F2-z | 0.6130 | +0.0203 | 0.8328 | 0.8669 | -0.0008 | 0.1167 |
| F2z_lam1 | F2-z | 0.6134 | +0.0207 | 0.8237 | 0.8606 | -0.0072 | 0.1485 |
| F2z_lam1.5 | F2-z | 0.6061 | +0.0134 | 0.7974 | 0.8492 | -0.0186 | 0.2027 |
| F2z_lam2 | F2-z | 0.5980 | +0.0053 | 0.7761 | 0.8354 | -0.0323 | 0.2447 |
| F2z_lam3 | F2-z | 0.5622 | -0.0305 | 0.7535 | 0.8249 | -0.0429 | 0.3013 |
| F2z_lam1_clip0.15 | F2-z | 0.6159 | +0.0232 | 0.8228 | 0.8627 | -0.0051 | 0.1462 |
| F2z_lam1_clip0.25 | F2-z | 0.5947 | +0.0020 | 0.8224 | 0.8620 | -0.0058 | 0.1366 |
| F2z_lam1_clip0.35 | F2-z | 0.5866 | -0.0061 | 0.8294 | 0.8618 | -0.0060 | 0.1094 |
| F2w_lam0.5 | F2-winsorised-z | 0.6163 | +0.0236 | 0.8423 | 0.8757 | +0.0079 | 0.0770 |
| F2w_lam1 | F2-winsorised-z | 0.6175 | +0.0248 | 0.8222 | 0.8627 | -0.0051 | 0.1449 |
| F2w_lam2 | F2-winsorised-z | 0.5984 | +0.0057 | 0.7759 | 0.8354 | -0.0323 | 0.2436 |
| Grank_lam1 | Grank-PIT | 0.6102 | +0.0175 | 0.8506 | 0.8761 | +0.0083 | 0.0480 |
| Grank_lam2 | Grank-PIT | 0.6268 | +0.0341 | 0.8382 | 0.8757 | +0.0079 | 0.0931 |
| Grank_lam3 | Grank-PIT | 0.6195 | +0.0268 | 0.8333 | 0.8656 | -0.0021 | 0.1337 |
| Grank_lam4 | Grank-PIT | 0.6138 | +0.0211 | 0.8243 | 0.8627 | -0.0051 | 0.1693 |
| Grank_lam6 | Grank-PIT | 0.6089 | +0.0163 | 0.7969 | 0.8531 | -0.0147 | 0.2263 |
| Grank_lam8 | Grank-PIT | 0.6081 | +0.0154 | 0.7831 | 0.8500 | -0.0177 | 0.2680 |
| Grank_lam2_clip0.15 | Grank-PIT | 0.6268 | +0.0341 | 0.8382 | 0.8757 | +0.0079 | 0.0931 |
| Grank_lam3_clip0.15 | Grank-PIT | 0.6183 | +0.0256 | 0.8336 | 0.8656 | -0.0021 | 0.1328 |
| Ginv_lam0.5 | Ginv-log-ratio | 0.5846 | -0.0081 | 0.8470 | 0.8700 | +0.0022 | 0.1086 |
| Ginv_lam1 | Ginv-log-ratio | 0.5870 | -0.0057 | 0.8237 | 0.8484 | -0.0194 | 0.2032 |
| Ginv_lam2 | Ginv-log-ratio | 0.5585 | -0.0341 | 0.7925 | 0.8250 | -0.0428 | 0.3313 |
| Ginv_lam3 | Ginv-log-ratio | 0.5333 | -0.0593 | 0.7887 | 0.8235 | -0.0443 | 0.3902 |
