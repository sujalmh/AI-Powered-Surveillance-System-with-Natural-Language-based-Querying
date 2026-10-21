# Release manifest

Generated: `2026-10-21T18:02:15.720114+00:00`
Git: `fcf0f3701d4da1b5c86cc9deb740c08f74df79bb`

## Subset

- Clips: 123 (~7.2 min selected footage; 640x480 @ 5/1 fps)
- Snapshots: 2071
- Recordings: 11 (~34.0 min readable)
- Scene index: 242 frames, 512-d, 242 vectors
- Person index: 1778 descriptors, 1152-d, 1778 vectors
- Camera IDs: [0]
- Timestamp span: 2026-03-01T17:01:00.052839 to 2026-03-01T18:56:01.940526

## Benchmarks

- `backend\benchmarks\queries.jsonl`: 150 rows, 150 unique texts
- `backend\benchmarks\queries_control.jsonl`: 150 rows, 126 unique texts
- `backend\benchmarks\alert_utterances.jsonl`: 120 rows, 120 unique texts
- `backend\benchmarks\person_pairs.jsonl`: 2016 rows
- `backend\benchmarks\splits.json`: JSON object (keys: seed, tune, test, n_tune, n_test, stratified_by, note)
- `backend\benchmarks\splits_control.json`: JSON object (keys: n_tune, n_test, total, seed, source, tune, test, note, categories)

## Reproduce

```powershell
$env:PYTHONIOENCODING='utf-8'
python backend/scripts/check_release.py
python backend/benchmarks/build_subset.py
python backend/benchmarks/build_subset_control.py
python backend/scripts/eval_retrieval_control.py
python backend/scripts/eval_alpha_competitors.py
python backend/scripts/eval_fusion.py
python backend/scripts/eval_alerts.py
python backend/scripts/measure_latency.py
python backend/scripts/eval_retrieval.py   # raw smoke test, not protocol-matched
python backend/scripts/package_release.py  # build the supplementary archive
```

The C1 number quoted in the paper comes from `eval_retrieval_control.py`;
`eval_retrieval.py` is the raw released benchmark (a code-path smoke test
whose count/action queries are not protocol-matched to the deployment study).

## Notes

- The subset is single-camera and covers ~7.2 min of selected 3.6-s clips within a 2 h timestamp window; it is not a continuous 2 h recording.
- The person index stores 1,152-d fused descriptors (768-d visual + 384-d text).  All released person track IDs are -1; track construction for the person-pair benchmark is documented in the fusion results.
- FAISS indices are flat inner-product indices; no approximate index is released.
