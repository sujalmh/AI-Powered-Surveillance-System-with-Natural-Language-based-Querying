# Results provenance

This directory contains machine-readable outputs from the released-subset
evaluation.  It is deliberately separate from the paper's full-deployment
numbers.

## Two evidence tracks

- **Full deployment (paper headline):** 4 cameras, ~8 h, 312,544 person
  descriptors.  The numbers 0.666 -> 0.828, 0.785 -> 0.807, 91%, and 6 s/12 s
  are the authors' full-deployment measurements.  The original per-query CSV,
  annotator agreement sheet, and 312k index are not present in this
  repository, so those exact runs cannot be re-executed here.
- **Released subset (this directory):** 123 clips (640×480, 5 FPS, ~3.6 s
  each; ~7.2 min of selected footage spanning a 2 h timestamp window),
  `camera_0` only, 242 scene frames, 1,778 person descriptors.  Every number here is produced
  by a script in this repository from the shipped metadata/index.  It is a
  reproducibility and code-path check, not a reproduction of the 4-camera
  deployment.  Every file below belongs to this second track; none of them
  replaces a deployment number.

## Files

- `retrieval_results.json` / `retrieval_tables.md` / `retrieval_per_query.csv`:
  adaptive-alpha and fixed-alpha retrieval on the **raw** released benchmark.
  This is a code-path smoke test, not the benchmark quoted in the paper: its
  count questions carry no target count and its action queries carry explicit
  time windows, so it is not protocol-matched to the deployment C1 study.  The
  script prints the same note.  Use the control below for the quoted number.
- `retrieval_control_results.json` / `retrieval_control_tables.md` /
  `retrieval_control_per_query.csv`: the same evaluation on a
  **protocol-matched control** built by
  `backend/benchmarks/build_subset_control.py`.  The raw benchmark has two
  protocol mismatches with the paper's C1 study: count questions carry no
  target count (so the structured branch has nothing to rank on) and action
  queries carry explicit time windows (so the structured branch has a cue the
  action override suppresses).  The control removes both, so it is the number
  quoted in the paper as the released-subset check.  It is post-hoc and
  metadata-derived, not human-annotated.
- `alpha_competitors.json` / `alpha_competitors.md`: the C1 robustness study
  on the same control - a learned (logistic) per-query alpha competitor, a
  16-point soft-weight sensitivity sweep searched on the tune split, and the
  counting accuracy together with the label-noise ceiling that bounds it.
- `end_to_end_results.json` / `end_to_end_tables.md`: the **end-to-end**
  released-subset track, produced by `backend/scripts/eval_end_to_end.py`.
  The natural-language query is parsed by the deployed LLM backend (the local
  opencode runtime by default, `opencode-go/beam-free`; any OpenAI-compatible
  endpoint is supported), the parsed filter drives the structured branch
  (colour, zone, action, count constraint and the time window when the parser
  produced one), the semantic branch searches the released OpenCLIP/FAISS
  scene index, and the production adaptive-alpha rule
  (`result_merger.compute_adaptive_alpha`) merges the two branches.
  The default query file is `backend/benchmarks/queries_control.jsonl`, the
  protocol-matched control of `eval_retrieval_control.py` (same 150 queries,
  same 50 tune / 100 test split, same relevance labels and same per-category
  wording), so the two tracks differ only in where the filter comes from; the
  raw released benchmark (`queries.jsonl`) can be run with `--queries`, but its
  numbers are not protocol-matched with the control track.
  Reported: test macro P@5 and per-category P@5 for five fixed alphas,
  `adaptive`, `per_intent_tuned` and `paper_lookup`, plus count EM/MAE, the
  per-category paired bootstrap with Holm correction and the grouped
  bootstrap of `adaptive` against the tune-selected best fixed alpha.  The
  alphas are selected on the tune split and the numbers are reported on the
  test split.  Parses are cached under `backend/benchmarks/parsed_cache/` (see
  `backend/benchmarks/parsed_cache/README.md`), so the run is reproducible and
  resumable without the model.  A query the model cannot answer is scored
  with a deterministic heuristic filter and flagged with `parse_ok: false`.
  This track is released-subset evidence for the deployed query path; it is
  single-camera and does **not** reproduce the 4-camera/312k deployment
  numbers.
- `parser_results.json` / `parser_results.md`: the **parser-quality** track,
  produced by `backend/scripts/eval_parser.py`.  Every query of the released
  benchmark is parsed through the production path
  (`nl_parser.parse_nl_with_llm`) and the resulting filter is compared field
  by field with the reference parse stored next to the query.  The reported
  numbers are **agreement rates, not accuracy**: the reference is
  reconstructed from the archive metadata and not annotated by a human, so a
  disagreement can be a wrong reference rather than a wrong parse.  Reported:
  per-field agreement overall and per category, the time-window error
  (median absolute error, overlap rate with and without a tolerance), the
  camera-id and intent agreement on the queries that state them, the parse
  latency, the parse failures and the queries with at least one
  disagreement.  The parses are cached under
  `backend/benchmarks/parsed_cache/` (shared with the end-to-end track, so
  both measure the same parses on the same protocol) and are not shipped:
  they are regenerable build artefacts.
- `end_to_end_oracle_results.json` / `end_to_end_oracle_tables.md`: the
  **harness-equivalence control** for the end-to-end track, produced by
  `backend/scripts/eval_end_to_end.py --parse-source oracle`.  The benchmark's
  reference filter drives the structured branch instead of the parser's answer.
  It reproduces the control track's macro P@5 (0.538) exactly, which is what
  makes the two tracks comparable: the difference between 0.538 and the
  end-to-end 0.450 is the filter's origin, not the harness.
- `fusion_results.json` / `fusion_tables.md` / `fusion_queries.csv`:
  person-pair/track fusion baselines on the subset.
- `alert_results.json` / `alert_results.md`: alert-authoring test split and
  stress/clarify evaluation.
- `latency_results.json` / `latency_results.md`: real component timings,
  p50/p95, and concurrency on the subset.
- `privacy_notes.md`: implemented safeguards and their deployment switches.
- `release_manifest.json` / `release_manifest.md`: inventory, checksums and
  package versions of the shipped artifact.
- `subset_stats.json`: dataset statistics produced by the benchmark builder.

## Rules

1. No simulated or hard-coded paper numbers in these files.
2. If a metric cannot be computed from the released artifact, it is omitted
   and the limitation is stated in the corresponding markdown file.
3. The subset is single-camera and author/metadata-labelled; it does not
   provide human IAA, cross-camera Re-ID, cross-site generalization, or
   full-deployment latency percentiles.
4. Files that depend on an LLM parse record the provider and the model that
   produced them in their `scope` block.  A provider quota or a model change
   affects that track only; it never changes a deployment number.
