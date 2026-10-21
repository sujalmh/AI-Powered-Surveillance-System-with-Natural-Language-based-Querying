# Final verification checklist

Run from `S:\college\ai-surveillance` with `$env:PYTHONIOENCODING='utf-8'`.

Status of every item after the revision is recorded in the last column.

## Code and tests

| Item | Status |
|---|---|
| `python -m pytest backend/tests -q` passes | ✅ 63 passed, 3 warnings (25 s) |
| `import backend.app.db.mongo` returns quickly with MongoDB down and logs a non-fatal warning | ✅ short server-selection timeout, import no longer blocks |
| `import backend.app.main` succeeds | ✅ `ACCESS_CONTROL_ENABLED` is inert by default |
| `python backend/scripts/eval_alerts.py` writes `results/alert_results.json` / `.md` and reports dev and test separately | ✅ test clean 30/30, stress 30/30 clarified, 0 false materializations |
| `python backend/scripts/eval_retrieval.py` writes `results/retrieval_results.json`, `retrieval_tables.md`, `retrieval_per_query.csv`; no synthetic Gaussian scores, no paper-table fixture | ✅ raw released benchmark; 3,000 resamples, grouped bootstrap, Holm |
| `python backend/scripts/eval_retrieval_control.py` writes the protocol-matched control outputs | ✅ adaptive 0.538 = best fixed 0.538 (Δ 0.000, p 1.0) |
| `python backend/scripts/eval_fusion.py` writes `results/fusion_results.json`, `fusion_tables.md`, `fusion_per_query.csv`; pair split is track/identity-disjoint | ✅ 174 pseudo-tracks, 528 calibration / 1,488 test pairs, thresholds from calibration only |
| `python backend/scripts/measure_latency.py` writes `results/latency_results.json` / `.md`; no simulated lognormal values | ✅ real component timings, p50/p95, 1/2/4/8 workers |
| `python backend/scripts/measure_latency.py --live-llm 3` (optional) | ⚠️ not run; LLM parse latency is excluded and labelled as excluded |
| `python backend/scripts/check_release.py` writes the release manifest | ✅ `results/release_manifest.json` / `.md` |
| `python backend/benchmarks/build_subset.py` and `build_subset_control.py` regenerate the benchmarks deterministically | ✅ seed 20260301; re-running reproduces the same numbers |

## Paper/result consistency

| Item | Status |
|---|---|
| Every subset number in the tex appears in a `results/*` JSON/CSV with the same value | ✅ checked against `retrieval_control_results.json`, `fusion_results.json`, `alert_results.json`, `latency_results.json` |
| Full-deployment headline numbers remain labelled as full deployment and are not replaced by subset numbers | ✅ 0.666→0.828, 0.785→0.807, 91%, 6 s/12 s kept as primary |
| No claim that the subset reproduces 4-camera/312k, human IAA, cross-camera Re-ID or full-deployment latency percentiles | ✅ stated explicitly in Setup, Limitations and Conclusion |
| C3 no longer says the materializer depends on the LLM | ✅ rewritten to the deterministic `alert_materializer.py` + clarify/abstain validator |
| No "measured results will be reported" style placeholder remains | ✅ verified in the extracted PDF text |
| The 91% alert number is labelled as the deployment study, with the released coverage check reported next to it | ✅ 30/30 clean, 30/30 stress; the two are not merged |
| Latency claims separate the deployment means from the released-subset p50/p95 | ✅ 6.0 s/12.0 s vs 37.4/39.7 ms and 17.2/20.4 ms |
| Privacy claims match the implemented switches and their defaults | ✅ all opt-in and disabled by default |
| The released C2 track is reported as not reproducing the deployment ordering | ✅ stated with the visual-only bias caveat |
| The released C1 control is labelled as a post-hoc protocol-matched control | ✅ stated in the paper, `results/README.md` and `backend/benchmarks/README.md` |

## LaTeX

| Item | Status |
|---|---|
| `pdflatex` + `bibtex` complete with no undefined references/citations | ✅ |
| PDF is at most 8 pages (6 + 2 overlength) | ✅ 8 pages |
| No overfull hbox | ✅ none remaining (the 35.5 pt box and the 3.1 pt author-block boxes are gone) |
| Tables/figures fit the column width; no missing figure files | ✅ |
| `INDICON2026Conference.pdf` regenerated from the final tex | ✅ |

## Repository

| Item | Status |
|---|---|
| `git diff` contains no accidental deletion of user work | ✅ reviewed before commit |
| `results/` files committed | ✅ |
| Final commit message summarises the revision and test status | ✅ |

## Cannot be produced with the shipped artifact (state, do not fabricate)

- Human inter-annotator agreement for the deployment benchmark.
- The deployment per-query export, so grouped resampling cannot be recomputed
  for the 4-camera/312k numbers.
- Cross-camera Re-ID, cross-site / night / rain generalisation.
- 312k-descriptor latency percentiles.
- A re-run of the deployment 150-query benchmark: its query file, annotator
  roster and tune/test partition were not preserved.
