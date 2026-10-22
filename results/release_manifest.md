# Release manifest

Generated: `2026-10-22T02:54:01.140860+00:00`
Git: `62d0aea62c264b6c971b69722bfb2f87aa35d64c`

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

## Delivery bundles

- **ok** -- release archive present: indicon2026_supplementary.zip (1.43 MB, sha256 124aa88a0f3cc905)
- **ok** -- release archive contents: 6 required entries present (llm_client.py, eval_end_to_end.py, test_llm_client.py, ...)
- **ok** -- no secrets in the release: no *.env/*.key/*.pem/credentials*/secrets* entries
- **ok** -- no parse caches in the release: no parsed_cache/parser_cache entries
- **ok** -- measured results shipped: all measured results present
- **ok** -- camera-ready PDF: 8 pages (0.44 MB, sha256 401aa7521ad1d7ac)
- **ok** -- source bundle: 4 entries (INDICON2026Conference.tex, References.bib, IEEEtran.cls, README.txt)
- **ok** -- source bundle carries the current paper: INDICON2026Conference.tex is byte-identical to INDICON2026Conference.tex
- **ok** -- LaTeX log clean: no errors, 8 pages reported
- **ok** -- folder package: INDICON2026_submit.zip carries the PDF, the source bundle and the sources (0.56 MB)
- **ok** -- source bundle compiles standalone: rebuilt from INDICON2026_850_source.zip into a clean 8-page PDF (scratch: C:\Users\sujal\AppData\Local\Temp\opencode\check_release)

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
python backend/scripts/eval_end_to_end.py
python backend/scripts/measure_latency.py
python backend/scripts/eval_retrieval.py   # raw smoke test, not protocol-matched
python backend/scripts/package_submission.py  # camera-ready PDF + source bundle
python backend/scripts/package_release.py    # build the supplementary archive
```

The C1 number quoted in the paper comes from `eval_retrieval_control.py`;
`eval_retrieval.py` is the raw released benchmark (a code-path smoke test
whose count/action queries are not protocol-matched to the deployment study).

## Notes

- The subset is single-camera and covers ~7.2 min of selected 3.6-s clips within a 2 h timestamp window; it is not a continuous 2 h recording.
- The person index stores 1,152-d fused descriptors (768-d visual + 384-d text).  All released person track IDs are -1; track construction for the person-pair benchmark is documented in the fusion results.
- FAISS indices are flat inner-product indices; no approximate index is released.
