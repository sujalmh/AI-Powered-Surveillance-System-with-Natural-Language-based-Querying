# Response to the INDICON 2026 reviewers

This document maps every reviewer item to what changed in the revision.  All
numbers quoted here are produced by scripts in this repository and stored under
`results/`; the paper keeps the full-deployment measurements as its primary
evidence and reports the released single-camera subset as a separate
reproducibility track.

Summary of the revision in one line: **the paper's headline results are
unchanged** (`0.666 -> 0.828` macro P@5, `0.785 -> 0.807` F1@10, `91%` alert
rule-correctness, `6.0 s / 12.0 s` query latency), while every claim that
previously had no artefact behind it now has either a runnable artefact or an
explicit statement of what cannot be reproduced.

---

## Reviewer 1

| Item | Request | Status | Where / what changed |
|---|---|---|---|
| R1-1 | Reproducible benchmark (150 queries, 60 utterances) | **Fixed** | `backend/benchmarks/build_subset.py` rebuilds 150 metadata-derived queries + 2,016 track-disjoint person pairs deterministically (seed `20260301`); `build_subset_control.py` adds the protocol-matched control; `backend/benchmarks/alert_utterances.jsonl` now holds 120 unique author-constructed utterances (60 clean / 60 stress) with a 30/30 dev/test split. `backend/benchmarks/README.md` documents provenance per category. |
| R1-2 | Justify the soft weights; add a learned per-query-$\alpha$ competitor and a per-intent lookup | **Fixed on the released track** | `backend/scripts/eval_alpha_competitors.py`: a multinomial logistic model that predicts the best fixed $\alpha$ from the intent, branch sizes and filter size, fitted on the 50 tune queries, reaches `0.534` vs. `0.538` for the training-free rule; all 16 soft-weight triples give `0.538` on the test split; the per-intent lookup (`0.428`) is reported alongside. Paper §C1 "Competitors and robustness". |
| R1-3 | Remove the concavity-based Proposition | **Fixed as instructed** | The Proposition is replaced by an empirical *Observation* (no concavity assumed, no general lower bound claimed); the abstract and §C1 now say "we observe empirically". |
| R1-4 | Score normalisation (min-max vs. z-score) | **Fixed on the released track** | Both calibrations are run for every method in `eval_retrieval.py` and the paired min-max/z-score deltas are reported in `results/retrieval_control_tables.md`; the paper keeps per-query min-max as the primary normalisation and states it. |
| R1-5 | Counting: report EM/MAE | **Fixed with a stated ceiling** | `eval_alpha_competitors.py` measures the window-mode count estimate (EM `0.10`, MAE `2.20`) and the attainable ceiling for any window-based predictor, and shows the release contains no per-frame detector count to validate against. Paper §C1 (iii) and §Evaluation protocol. |
| R1-6 / R1-7 | Fairer fusion baselines, learned projection, 5-seed mean$\pm$SD | **Fixed on the released track** | `eval_fusion.py` adds zero-pad and norm-matched addition, a ridge text$\to$visual projection fitted on calibration tracks only, and a 5-seed random projection (`0.436$\pm$0.0004`); the deployment table keeps its original single-seed measurement and says so. |
| R1-8 | Calibration/test independence | **Fixed on the released track** | Whole pseudo-tracks are assigned to calibration or test (`528` / `1{,}488` pairs; `174` tracks, `130` in test); every pair threshold is selected on calibration pairs only; the deployment study's 50-pair split is now explicitly flagged as unlogged for disjointness. |
| R1-9 | Grouped resampling + multiplicity correction | **Fixed on the released track** | `backend/app/eval/stats.py` provides paired bootstrap, grouped bootstrap and Holm correction; every released comparison reports grouped CIs and Holm-adjusted p-values (3,000 resamples). The deployment per-query export was not preserved, so the deployment CIs remain query-level and this is now stated. |
| R1-10 | Alert reliability / stress | **Fixed** | 120 utterances (60 clean / 60 stress: ambiguous, conflicting, missing, unsupported) with a deterministic materializer and a clarify/abstain validator: 30/30 clean test utterances rule-correct, 30/30 stress test utterances clarified, **0 false materializations**; the chat router no longer inserts a rule when clarification is required. Paper §C3. |
| R1-11 + R2-3 | 2 FPS vs. 6 s/12 s; percentiles and concurrency | **Fixed** | `measure_latency.py` measures the real components: scene retrieval core `37.4/37.4/39.7 ms` (mean/p50/p95), person core `17.2/18.5/20.4 ms`, and `26.5-39.5 queries/s` at 1-8 workers with p95 `39.6-265.4 ms`; LLM parse, clip stitching, MongoDB and ingestion are explicitly excluded and labelled. The deployment means remain the end-to-end number. |
| R1-12 + R1-13 | Generalisation; privacy/consent | **Partially fixed** | Implemented and documented: opt-in role/camera/zone access control with audit log, 8 h retention prune, face-blur utility, gender-off output filter (`results/privacy_notes.md`), and the PAR bias discussion is restored with its per-attribute degradation detail. **Still open (needs data we do not have):** cross-site / night / rain evaluation and any demographic-bias measurement; the paper now states these as limitations. |
| R1-14 + R1-19 + R2-2 | Release, protocol definitions, repeatability | **Fixed** | `backend/scripts/check_release.py` writes a manifest with checksums, package versions and a copy-paste reproduce block; the release contains the benchmarks, the eval harness, the alert schema, the privacy switches and the measured outputs; macro-mean, F1@10 and every threshold are defined in §Evaluation protocol. |
| R1-15,16,17,20 + R1-18 + R2-1,4 | Figures, notation, storage, qualitative examples, consolidated table | **Fixed** | Qualitative retrieval table (2 wins, 2 failures including an intent-routing error), a consolidated two-axis comparison table, the dataflow figure, explicit storage figures (`1.44 GB` for 312k flat descriptors), and the storage/notation clarifications requested. |

## Reviewer 2

| Item | Request | Status | Where / what changed |
|---|---|---|---|
| R2-1 | Consolidated comparison | **Fixed** | Table VII compares the proposed system against the best fixed-$\alpha$ and the unimodal/fusion baselines on both axes, with the deployment numbers and the release cross-referenced. |
| R2-2 | Protocol / release details | **Fixed** | See R1-14 row. |
| R2-3 | Latency measurement | **Fixed** | See R1-11 row. |
| R2-4 | Notation and storage | **Fixed** | See R1-15 row. |

---

## Items that remain open, and why

These need artefacts that are **not in the released repository**, so they are
stated as limitations rather than estimated:

1. **Human inter-annotator agreement** for the deployment benchmark: the
   annotator roster and the per-query export were not preserved, and no human
   re-annotation was possible for this revision.
2. **Re-running the deployment 150-query benchmark**: the original query file,
   relevance labels and tune/test partition were not released, so the
   deployment CIs stay query-level and the released track is reported
   separately.
3. **Cross-camera / cross-site / night-rain generalisation**: the release is a
   single camera (`camera_0`, 123 clips, ~7.2 min inside a 2 h window).
4. **312k-descriptor latency percentiles**: the full index and its per-query
   export are not in the release; the subset percentiles are component
   figures and are labelled as such.
5. **Public-Re-ID / OSNet-only baseline** for C2: no such baseline was run on
   the deployment, and the paper says so instead of implying one.

## Notes on two numbers that are deliberately *not* replaced

* The deployment alert result stays **0.91** on the 60-utterance deployment
  study. The released 120-utterance benchmark is a separate author-constructed
  coverage set that reports `1.00` on its test split; the two are reported
  side by side and never merged.
* The deployment fusion result stays **0.807 vs. 0.785 (RRF) vs. 0.700
  (visual-only)**. The released subset carries pseudo-tracks rather than
  identity labels, so it cannot test that result: the paper states this scope
  explicitly, and the subset fusion evaluation is published with the release
  (`results/fusion_tables.md`, `backend/scripts/eval_fusion.py`) with its own
  limitation note, including the fact that its ordering differs from the
  deployment one because the pseudo-tracks were formed from the visual branch
  and therefore bias visual-only upwards.
