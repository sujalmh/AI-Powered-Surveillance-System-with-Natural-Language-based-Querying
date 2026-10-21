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
  deployment.

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
  14-point soft-weight sensitivity sweep searched on the tune split, and the
  counting accuracy together with the label-noise ceiling that bounds it.
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
