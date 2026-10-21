# Benchmarks — construction, provenance, splits

Everything in this folder is regenerated from the **released single-camera
subset** by the scripts named below. No file here is a hand-written
placeholder, and no label is human-annotated.

## Files

| File | Rows | Builder | Provenance |
|---|---|---|---|
| `queries.jsonl` | 150 (30 per category: count / find / track / action / filter) | `build_subset.py` | metadata-derived labels from the released VLM captions, detector metadata and snapshot motion energy |
| `splits.json` | 50 tune / 100 test | `build_subset.py` | stratified by category (action additionally by event/motion subfamily), seed `20260301` |
| `person_pairs.jsonl` | 2,016 (528 calibration / 1,488 test; 1,008 same / 1,008 different) | `build_subset.py` | pseudo-tracks built from the 1,778 shipped 1,152-d descriptors; whole tracks are assigned to one split |
| `queries_control.jsonl`, `splits_control.json` | 150 (50/100) | `build_subset_control.py` | post-hoc protocol-matched control (see below) |
| `alert_utterances.jsonl` | 120 (60 clean + 60 stress) | authored for this revision | author-constructed, closed vocabulary, 30/30 dev/test |
| `alert_schema.json` | — | authored | canonical 9-field rule schema |

Regenerate everything with:

```powershell
$env:PYTHONIOENCODING='utf-8'
python backend/benchmarks/build_subset.py
python backend/benchmarks/build_subset_control.py
```

`build_subset.py` is deterministic (seed `20260301`); re-running it reproduces
the same query texts, relevance sets, splits and pairs.

## Label provenance and what it cannot support

1. **Query relevance** is derived from released metadata, not from human
   annotation: colour/attribute matches come from the detector sidecar,
   zone and event labels from the VLM captions, motion labels from snapshot
   motion energy, and count labels from the caption person counts. There is
   therefore **no inter-annotator agreement** for this subset and none is
   claimed.
2. **Count queries** in `queries.jsonl` are free questions (`how many people
   were visible around 17:01:00?`); their relevance set is the clip whose
   caption count equals the anchor count. The structured branch has no target
   count to rank on, so the paper's count *override* cannot be exercised by
   this form of question. The paper quotes the control below instead.
3. **Action queries** are weak proxies: 15 rows use VLM caption events
   (`smoke`, `fighting`) and 15 use a motion-energy proxy. The motion rows are
   not semantic actions.
4. **Track queries** are single-camera colour-continuity proxies, not
   ground-truth identities; the released person track ids are all `-1`.
5. **Person pairs** use pseudo-tracks formed by visual cosine similarity
   under a 3 s temporal gate, with the track threshold tuned on a
   calibration time block. Visual-only retrieval is therefore optimistically
   biased on this benchmark, which is stated in `results/fusion_tables.md`.
6. The subset is single camera (`camera_0`), 123 clips, ~7.2 min of selected
   footage inside a 2 h timestamp window. Cross-camera Re-ID, cross-site
   generalisation and full-deployment latency percentiles are untestable here.

## Protocol-matched control

`build_subset_control.py` rewrites only the two categories whose released
form is not protocol-matched to the paper's C1 study:

* count queries become count-constrained retrieval questions
  (`show clips with exactly N people between ...`) with the relevance set
  unchanged, so the structured count branch has the target it needs; and
* action queries become event-only queries with no explicit time cue
  (`show clips of smoke`), so the structured branch has no time evidence to
  compete with the semantic branch.

`find` / `track` / `filter` rows are copied unchanged. The control is
post-hoc, metadata-derived and clearly labelled as such in
`results/retrieval_control_tables.md`; it does not replace the deployment
measurements in the paper.

## Schemas

- `../app/eval/metrics.py` — P@5 / R@10 / F1@10 (paper and standard) /
  macro-mean / EM / MAE.
- `../app/eval/stats.py` — paired bootstrap, grouped bootstrap, Holm
  correction.
- `alert_schema.json` — the 9-tuple alert rule schema.
- `nlfilter_schema.json` — query filter schema pointer.

## Reproducing the numbers

```powershell
$env:PYTHONIOENCODING='utf-8'
python backend/scripts/eval_retrieval.py          # raw released benchmark
python backend/scripts/eval_retrieval_control.py  # protocol-matched control
python backend/scripts/eval_fusion.py             # person-pair fusion
python backend/scripts/eval_alerts.py             # alert authoring
python backend/scripts/measure_latency.py         # component latency
```

Outputs land in `results/` with a JSON, a markdown table and (where relevant)
a per-query CSV.
