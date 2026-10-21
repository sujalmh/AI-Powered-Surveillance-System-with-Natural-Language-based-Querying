# Benchmarks — construction, provenance, splits

Everything in this folder is regenerated from the **released single-camera
subset** by the scripts named below, except `parsed_cache/` and
`parser_cache/`, which are build artifacts of the end-to-end and the parser
track. No file here is a hand-written
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
| `parsed_cache/<model>.json` | one entry per parsed query | `backend/scripts/eval_end_to_end.py` (via `llm_client.ParseCache`) | build artifact: the LLM parse of each query id; not a benchmark input |
| `parser_cache/<model>.json` | one entry per parsed query | `backend/scripts/eval_parser.py` (via `llm_client.ParseCache`) | build artifact: the LLM parse of each query id; not a benchmark input |

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

## Parse cache

`parsed_cache/` and `parser_cache/` hold the on-disk parse caches of the
end-to-end track (`eval_end_to_end.py`) and of the parser track
(`eval_parser.py`).  They are **build artifacts**, not benchmark inputs:
`queries.jsonl` and `splits.json` stay the only inputs of the benchmark, and
`build_subset.py` / `build_subset_control.py` never read a cache.

*   One JSON file per model id; the file name is the model id with `/` and `:`
    replaced by `_` (for example `opencode-go_beam-free.json`).
*   The file is a JSON object keyed by query id (the `qid` used in
    `queries.jsonl`) plus a digest of the query text and the model
    (`<qid>:<sha1-12>`), so a stale parse is never reused after the benchmark
    is rebuilt or the model changes.  Each value holds the parsed filter in
    the backend filter shape (`timestamp`, `camera_id`,
    `objects.object_name`, `objects.color`, `zone`, `action`,
    `count_constraint`), the flag `ok` (`false` when the parse failed and the
    deterministic heuristic fallback was used) and the parse duration
    `seconds`; a failed parse also stores the truncated `error`.  The parser
    track (`parser_cache/`) uses the same value shape but keys it by the plain
    query id.
*   `backend/app/services/llm_client.ParseCache` reads and writes it, and
    both evaluation scripts save it after every query.  A run stopped by a
    provider quota can therefore be resumed, and a re-run reproduces the same
    filters without calling the model.
*   Regenerate it by deleting the file(s) and running the script again
    (`eval_end_to_end.py` / `eval_parser.py`); `--refresh` re-parses and
    overwrites the cached entries during the run.
*   The directories never ship: `package_release.py` excludes `parsed_cache`
    and `parser_cache` from the release archive and `check_release.py` fails
    when such a path reaches it, because the cached parses are regenerable.

See `parsed_cache/README.md` for the full description of the format.

## End-to-end track

`backend/scripts/eval_end_to_end.py` measures the whole query path on the
released subset:

1.  every query text is parsed by the deployed LLM backend
    (`nl_parser.parse_nl_with_llm` -> `llm_client`), and the parse is cached;
2.  the parsed filter is mapped onto the structured branch (colour, zone,
    action, count constraint and the time window when the parser produced
    one; the benchmark window is only the fallback);
3.  the semantic branch encodes the query text with the shipped OpenCLIP
    model and searches the released 242-frame scene index;
4.  the branches are merged with the production adaptive-alpha rule
    (`backend.app.services.result_merger.compute_adaptive_alpha`), the same
    rule the API uses;
5.  the merged ranking is scored with the metrics and bootstrap code of the
    control track (`eval_retrieval_control.py` is imported for the fusion,
    alpha and scoring helpers), on the standard 50 tune / 100 test split.

The default query file is `queries_control.jsonl` — the protocol-matched
control of `build_subset_control.py` — so the end-to-end track and the
control track use the same queries, the same split, the same relevance labels
and the same per-category wording, and differ only in where the filter comes
from.  The raw released benchmark (`queries.jsonl`) can be run with
`--queries`, but its numbers are not protocol-matched with the control track.

Outputs land in `results/end_to_end_results.json` and
`results/end_to_end_tables.md` (`--out-prefix` changes the names).

The track reuses the released scene index and the released metadata; it
replaces only the metadata-derived filter of the control track with the LLM
parse.  It is single-camera evidence for the deployed query path and does not
reproduce the 4-camera/312k deployment numbers.

## Parser track

`backend/scripts/eval_parser.py` measures the parser itself: every query in
`queries.jsonl` is parsed through the production path
(`nl_parser.parse_nl_with_llm`) and the resulting filter is compared field by
field with the reference parse stored next to the query in the same file.

The reference is **metadata-derived**, not human-annotated: it was
reconstructed from the archive metadata that generated the query, so it is an
approximation of the query intent and a disagreement can be a wrong reference
rather than a wrong parse.  The reported numbers are therefore **agreement
rates, not accuracy**.  Reported: the per-field agreement (overall and per
category), the time-window error, the camera-id and intent agreement on the
queries that state them, the parse latency, the parse failures and the
queries with at least one disagreement.

Outputs land in `results/parser_results.json` and `results/parser_results.md`;
the parses are cached in `parser_cache/`.

## Reproducing the numbers

```powershell
$env:PYTHONIOENCODING='utf-8'
python backend/scripts/eval_retrieval.py          # raw released benchmark
python backend/scripts/eval_retrieval_control.py  # protocol-matched control
python backend/scripts/eval_alpha_competitors.py  # C1 competitors + robustness
python backend/scripts/eval_fusion.py             # person-pair fusion
python backend/scripts/eval_alerts.py             # alert authoring
python backend/scripts/measure_latency.py         # component latency
python backend/scripts/eval_parser.py             # parser quality (real LLM parse)
python backend/scripts/eval_end_to_end.py         # end-to-end track (real LLM parse)
```

Outputs land in `results/` with a JSON, a markdown table and (where relevant)
a per-query CSV.
