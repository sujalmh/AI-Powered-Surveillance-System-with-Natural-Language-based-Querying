# splits2: disjointness proof + multi-seed learned projection

## Disjointness verdict

- Track IDs: DISJOINT (0 overlap) for both visual and metadata pseudo-tracks.
- Unique detection keys (row_index, faiss_id): DISJOINT (0 overlap).
- detection_id strings (non-unique in release): overlap 187 (metadata) / 94 (person_pairs).
- Timestamps: cal/test co-occur; exact timestamp values overlap 187 (metadata) / 94 (person_pairs); test instances with timestamp in cal: 187 / 95.
- Pairs: disjoint tuples; fusion pair-tuple overlap 0; track-string overlap 0.
- Overall: CLEAN on canonical unique keys; no identity-level leakage. Temporal co-occurrence and detection_id aliasing quantified.

## Multi-seed learned projection (pre-registered: 80% cal-track subsample, 5 seeds, lambda=1.0)

| seed | cal tracks | cal dets | retrieval P@5 | retrieval R@10 | pair AUC |
|---:|---:|---:|---:|---:|---:|
| 0 | 56 | 395 | 0.4334 | 0.3106 | 0.7092 |
| 1 | 56 | 372 | 0.4336 | 0.3109 | 0.7068 |
| 2 | 56 | 385 | 0.4355 | 0.3110 | 0.7074 |
| 3 | 56 | 415 | 0.4361 | 0.3116 | 0.7091 |
| 4 | 56 | 399 | 0.4348 | 0.3119 | 0.7106 |

Spread (5 seeds): P@5 mean 0.4347 sd 0.0010 (range 0.4334-0.4361); pair AUC mean 0.7086 sd 0.0014 (range 0.7068-0.7106).

## Reference comparison

- learned_proj multiseed mean: P@5 0.4347, pair AUC 0.7086
- concat (recomputed): P@5 0.4274, pair AUC 0.7263
- Grank winner (cal2, frozen): P@5 0.4721, pair AUC 0.7341
- learned_proj full-data reference: P@5 0.4324, pair AUC 0.7091

## C2 implications

- learned_proj is below concat on pair AUC even full-data (0.709 vs 0.726) and below Grank (0.734); multi-seed spread is small relative to gaps, so single-run is not the cause.
- C2 (gated fusion) unaffected: gated pair AUC 0.711 vs concat 0.726; Grank winner is a gated variant (0.734) and is the strongest; learned linear projection is not competitive.