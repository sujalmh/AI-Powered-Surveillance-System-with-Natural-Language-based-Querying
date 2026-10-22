# Response to the INDICON 2026 reviewers

This document maps every reviewer item to what changed in the revision.  All
numbers quoted here are produced by scripts in this repository and stored under
`results/`; the paper keeps the full-deployment measurements as its primary
evidence and reports the released single-camera subset as a separate
reproducibility track.

---

## Major-revision round 2 (2026-10-22) — Weak Reject / Major Revision fixes

**C1 (intent-adaptive alpha).** The abstract and §C1 now state the full
honest picture: the per-intent fixed-alpha lookup reaches 0.822 vs. adaptive
0.828, so the soft terms contribute +0.006 on the deployment and +0.000 on
the released control (all 16 soft-weight triples give 0.538); the released
control ties the best tune-selected fixed alpha (0.538 vs. 0.538, Δ 0.000,
p = 1.0); the end-to-end track ties too (0.450 vs. 0.448, Δ +0.002,
p = 0.80); deployment per-category gains are +0.02 (p = 0.04, find) and +0.01
(p = 0.07, track), both marginal and not surviving Holm correction.
Observation 1 stays empirical ("not a theorem"). Deployment CIs/p-values are
now labelled in the Statistical protocol paragraph as recorded single-run
measurements, not re-computable significance; significance is claimed only on
the released track (grouped bootstrap + Holm).

**C2 (heterogeneous fusion).** (a) The deployment Add(proj.) baseline is
labelled illustrative single-seed and the concat-vs-RRF margin fragile (95%
CI [+0.4,+4.1]). (b) The released track already carried both F1 variants
(standard P@10-based first, paper P@5/R@10 second) plus 5-seed
random-projection and a calibration-only ridge learned projection; this round
adds 5-fold whole-track CV over the 174 pseudo-tracks (`kfold_by_track` in
`eval_fusion.py`, `results/fusion_tables.md`): visual-only beats concat on
every fold, so the inversion is systematic bias from visual-cosine track
formation, not fold noise. The deployment Table IV keeps its recorded
paper-definition F1@10 values (the per-query export needed for the standard
variant was not preserved, so those cells are not re-computable) and now
points to the released track, which leads with the standard P@10-based F1.
(c) Public Re-ID probe (CUHK-PEDES / Market-1501 frozen SigLIP+MiniLM):
time-boxed attempt stopped without downloading, for measured reasons — the
official CUHK-PEDES repo URL returns HTTP 404 and only unofficial mirrors
(≤9 likes, no verifiable file listing) remain, so dataset provenance is
unsuitable for a paper artefact; SigLIP weights are not in the offline cache
(+~1 GB download); CPU encode lower bound measured at 0.05 s/img
(OpenCLIP ViT-B/32 proxy, SigLIP-base heavier), putting even a 1,000-image
mini-split plus mirror format-debugging at 60–120 min all-in, over the
~30 min budget. Fallback-2 stands: Table IV remains deployment case-study
evidence and the released track a code-path check, with no new
cross-dataset claim. (d) The (0.6,0.4) weights are disclosed as a 9-point
unit-simplex grid on the 50-pair calibration split, no CV, gallery
disjointness unlogged.

**C3 (alert grammar).** The 91% stays attributed to the LLM-backed deployment
study with the export-loss caveat; the released 120-set is explicitly a
closed-vocabulary regression/coverage check. Specified-field accuracy
(excluding severity/cadence/cooldown defaults) is now measured:
`eval_alerts.py` reports test 7/7 severity, 2/2 cadence, 1/1 cooldown correct
— defaults dominate the released 1.00.

**Systems language.** 2 FPS is now "near-real-time" with keyframe-only VLM
captioning outside the budget; 6 s/12 s is "operator-in-the-loop, not
interactive"; 109 ms is the flat-index deployment median over N = 312,544.

**Post-ranking ablation (new).** `backend/scripts/eval_ablation.py` runs the
2x2 (recency on/off × MMR on/off) on the 100-query control test split
(`results/ablation_results.json`, `results/ablation_tables.md`): recency is
ranking-inert on the archived subset (Δ 0.000, all boosts ~1.0: ~5,000 h old
against a 24 h half-life); production MMR (λ = 0.3) costs −0.138 macro P@5
(p < 0.0007), so the quoted 0.538 control is pre-MMR and the production
default scores 0.400 on this single-camera subset. Limitations (ii) reports
this instead of "unablated".

**Related work (still 15 refs, all cited).** Added Bruch et al. 2022
convex-combination analysis (b33), DAT per-query alpha 2025 (b34, QDAP-style
family) and CLIP-ReID AAAI 2023 (b35); dropped the three single-cite
generics (Agentic-RAG survey b3, GPT-4o system card b20, BoT-SORT b22) with
prose rephrased to match. Kept CUHK-PEDES b32, HyST b30, RRF b27.
"Undemonstrated" is narrowed to override-routing + heterogeneous
single-index fusion scope.

**Counting.** EM/MAE (0.10/2.20 + label-noise ceiling) is now stated as the
primary counting-ability measure; count-P@5 stays as supporting-clip
retrieval with its clarifying clause.

No measured value was removed: a distinct-value diff of the pre- and
post-revision tex sources shows every headline number and caveat still
present. The paper builds to exactly 6 pages, 0 overfull, 0 undefined, 15/15
refs cited.

---

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
| R1-15,16,17,20 + R1-18 + R2-1,4 | Figures, notation, storage, qualitative examples, consolidated table | **Fixed** | Qualitative retrieval examples (2 wins, 2 failures including an intent-routing error), a consolidated two-axis position statement, the end-to-end dataflow, explicit storage figures (`1.44 GB` for 312k flat descriptors), and the storage/notation clarifications requested. To fit the 6-page limit the qualitative table, the positioning table and the dataflow figure are stated in prose; the numbers are unchanged and the per-category detail is in the supplementary artefact. |

## Reviewer 2

| Item | Request | Status | Where / what changed |
|---|---|---|---|
| R2-1 | Consolidated comparison | **Fixed** | §Related Work now compares the proposed system against the best fixed-$\alpha$ and the unimodal/fusion baselines on both axes in prose, with the deployment numbers and the release cross-referenced. |
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

---

## Independent evidence audit (2026-10-22)

An independent read-only trace of every numeric claim in the manuscript
against `results/`, `backend/` and the built PDF (`PAPER_EVIDENCE_AUDIT.md`,
5 blockers, 11 major items, 14 minor nits) was applied in full. **The
deployment numbers are unchanged**: `0.666 -> 0.828` macro P@5, `0.785 ->
0.807` F1@10, `91%` alert rule-correctness, `6.0 s / 12.0 s` query latency,
`312,544` person descriptors, 4 cameras, 2 FPS and the 109 ms person-index
median are all still the primary evidence, and the PDF still builds to exactly
6 pages with zero overfull box and zero undefined reference.

* **B1 — action-query mechanism vs. Table II.** §IV-B now states that the
  deployed pipeline routes action queries semantic-only, so `α` is inert for
  them in production, and that the `α`-sweep harness scores them by a
  time-overlap proxy; the `α=1.0` column of Table II therefore must not be
  read as structured-branch noise injection. The same caveat is repeated in
  the "Query resolution" paragraph of §VII.
* **B2 — override-gap wording.** The "Released-subset check (C1)" paragraph no
  longer claims that the released subset reproduces the override gap. It now
  says that the overrides are not harmful, that the best per-category fixed
  `α` (`0.7`) happens to match the adaptive rule on all five categories of the
  subset, and that the subset therefore does not reproduce the deployment gain.
* **B3 — the contradicting raw-benchmark ordering is now disclosed.** The
  paper states that the release also ships a *raw*, non-protocol-matched smoke
  benchmark (its count questions carry no target count, its action queries
  carry a time window the structured branch can exploit) which ranks adaptive
  below the best fixed `α` (`0.486` vs. `0.568`) and prefers `α=1.0` on action
  queries (`0.520`), and it calls the control a **post-hoc** protocol-matched
  control everywhere it is described. The abstract and the evaluation setup
  repeat that the deployment per-query export was not preserved, so the
  deployment CIs and `p`-values are not re-computable.
* **B4 — learned-`α` failure mode.** The learned competitor is now described
  correctly: it collapses to the constant `α=0.5`, matches the closed-form
  rule on both override categories (count `0.39`, action `0.09`) and loses
  only on the soft zone (find `0.91` vs. `0.92`, filter `0.47` vs. `0.48`).
* **B5 — release-link wording.** The cited repository URL returns HTTP 404 and
  has been removed. The footnote and the conclusion now refer to the
  supplementary artifact, and the abstract says the harness, benchmark
  definitions and reproducibility track are "included in the supplementary
  artifact" rather than "released"; the public URL is withheld from this
  preprint and will be given on acceptance.

Structural displacements were needed to keep the 6-page limit (the
conference maximum, so no overlength charge). The consolidated comparison
table (former Table V) was **deleted** because it
restated Tables II and IV and was never cited — the R1-15/R2-1 rows above are
therefore superseded by Tables II, IV and VI — and the "Generalization"
paragraph was folded into the conclusion, which now carries the
schema-decoupling statement and the single-premises caveat. To reach 6 pages
the off-the-shelf detection-and-indexing algorithm, the TikZ dataflow figure
and six supporting tables were folded into prose, and the display headings
became run-in headings; a distinct-value diff of the 8-page and 6-page sources
shows that **no** measured value was lost, and every caveat still appears at
least once. The remaining
major items (M1–M10) and the minor nits were applied in the manuscript
without changing any headline number; two audit items fall outside the
manuscript and were left to their owners: the README captioning stack and the
stale checkmark/test count in `VERIFICATION.md`.
