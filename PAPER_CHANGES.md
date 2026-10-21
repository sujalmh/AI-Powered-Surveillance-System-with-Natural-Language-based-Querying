# Paper revision plan (tex-level)

This file is a coordination note for the paper workstream.  It is not part of
the submission.  Every number added to the paper must come from a file under
`results/` produced by a runnable script.

## Central framing

The paper has two evidence tracks:

1. **Full deployment (primary):** 4 cameras, ~8 h, 312,544 person descriptors.
   The existing headline numbers (0.666 -> 0.828; 0.785 -> 0.807; 91%; 6 s /
   12 s) stay here.  Label them explicitly as full-deployment measurements.
2. **Released subset (reproducibility):** 123 clips (640×480, 5 FPS,
   ~3.6 s each; ~7.2 min of selected footage spanning a 2 h timestamp
   window), camera_0 only, 242 scene frames, 1,778 person descriptors.  Report real numbers from
   `results/retrieval_*`, `results/fusion_*`, `results/alert_*`,
   `results/latency_*`.

Never say the subset reproduces the 4-camera/312k numbers.  Never replace the
full-deployment headline with a subset number.  State what the subset cannot
test: cross-camera Re-ID, cross-site/night/rain generalization, full-deployment
latency percentiles, human IAA.

## Section-by-section edits

### Abstract
- Keep 0.67 -> 0.83, 10.7/2.2 pp, 91%, 2 FPS, 109 ms.
- Add "full-deployment" qualifier once; add one clause that the released
  single-camera subset reproduces the qualitative ordering / provides
  runnable evidence (only if true from results).
- Do not claim the subset reproduces 4-camera numbers.

### Introduction / Contributions
- Keep the three contribution claims.
- Item 4: replace "validating all three contributions on a 4-camera live
  deployment" with a precise statement: full-deployment evaluation plus a
  released, runnable single-camera reproducibility subset.
- Footnote: repository released; no "upon acceptance".

### C1 (adaptive alpha)
- Keep the full-deployment fixed-alpha table (`tab:alpha_baseline`) and CI
  table.  These are the headline results.
- Add a compact "released-subset" result: fixed vs adaptive, per-category
  P@5, macro-mean, and paired bootstrap.  Use `results/retrieval_results.json`.
- Add EM/MAE for count queries (full-deployment values if available; otherwise
  clearly label the subset values and say the full-deployment per-query counts
  were not preserved).
- Add alpha sensitivity: either a small figure or one sentence with the
  subset sweep.  Reviewer R1-2 wants sensitivity, not just five points.
- Add z-score normalization baseline (R1-4) and state whether it changes the
  ranking.  State that min-max is per-query (no cross-query leakage).
- Add per-intent lookup and learned/logistic alpha competitor if the subset
  results support it.  If the soft terms contribute ~0, say so explicitly.
- Observation: keep "empirical, not a theorem"; no lower-bound language.
- Remove "full sensitivity sweep is retained as future work" once the sweep
  exists.

### C2 (fusion)
- Keep the full-deployment ablation table and headline 0.807.
- Add released-subset fusion results: concat vs visual/text/zero-pad/
  norm-matched/random-proj (multi-seed mean±SD)/RRF/learned projection.
  Report real P@5/R@10/F1@10 and n (tracks/pairs).  Label single-camera
  track-derived identity proxy, not human cross-camera labels.
- State the subset's calibration/test split is identity/track-disjoint; the
  full-deployment 50-pair split's gallery overlap was not logged (honest
  limitation).
- Fix the "single-seed random Gaussian projection (illustrative)" wording:
  use multi-seed mean±SD, and include zero-pad/norm-matched/learned.
- Keep the 312k storage calculation.
- If the subset concat does not beat RRF, report it; do not hide it.  The
  full-deployment claim remains separate.

### C3 (alerts)
- Replace the 91% table with the real test-split results from
  `results/alert_results.json` (per-field accuracy + rule correctness).
- Add the stress-set result: clarification/abstain rate, false-materialization
  rate, per-kind breakdown.
- State the benchmark is **author-constructed** (not human-annotated), with
  dev/test split; no IAA claim.
- Fix the grammar paragraph: the live materializer is a deterministic
  keyword/regex module (`alert_materializer.py`), not an LLM.  Remove the
  Discussion sentence "Alert-grammar correctness depends on the LLM; we
  observed a 6-pp drop with local 7B models" — it contradicts the
  implementation.  If an LLM comparison exists, describe it separately;
  otherwise delete.
- Update the example to match the materializer's actual defaults
  (severity/side-effects/cadence) and validator behavior.
- Keep the anomaly-detector table; it is a separate component.

### Implementation / routing
- Algorithm 2: person text queries route to the 1152-d person index
  (`HybridSearchEngine`), scene queries to the 512-d OpenCLIP index.  This is
  now wired in `unified_retrieval.py`; cite the code path.
- Add one sentence on the fallback if the person index is unavailable.

### Evaluation setup
- Separate "Full deployment" and "Released subset" in one paragraph each.
- Full deployment: 4 cameras, ~8 h, 230k frames, 1.25M detections, 312k
  descriptors, 150 queries, 60 alert utterances, 50-pair calibration.
  State the original per-query CSV and annotator agreement were not preserved;
  the full-deployment numbers are the authors' measurements.
- Released subset: 123 clips, camera_0, 2 h, 242 scene frames, 1,778 person
  descriptors; 150-query (or actual count) metadata-derived benchmark with a
  disjoint 50/100 tune/test split; 120 alert utterances with a 30/30 dev/test
  split; track-derived person pairs.
- Remove "single annotator, no agreement reported" from the main claim and
  replace with the exact provenance: full-deployment labels are the authors';
  released-subset labels are deterministically derived from metadata, not
  human IAA.
- Define macro-mean and F1@10 exactly (already present; keep).
- State thresholds and how they were selected (detector 0.5, fight 0.03/24px,
  z 1.5/2.0/3.0).  If selection was manual, say so; do not claim a sweep.

### Statistics
- Full deployment: keep query-level bootstrap CIs; state grouped resampling
  was not possible because per-query group IDs were not preserved.
- Released subset: report grouped bootstrap by clip/time bucket and Holm
  correction across five categories.  If find/track lose significance, say so.
- Add multi-seed mean±SD for stochastic fusion baselines.

### Latency
- Keep the full-deployment mean table and 6.0 s/12.0 s.
- Add released-subset p50/p95 component table from
  `results/latency_results.json`; label CPU-only, single-camera, 242/1778
  vectors, and state these are not 312k percentiles.
- Add concurrency (1/2/4/8) results.
- Remove "measured results will be reported in the final version".
- Add the 2 FPS implication: query latency is interactive (seconds), ingest
  is 2 FPS/stream; a query spans ~12 frames at 2 FPS; FAISS is not the
  bottleneck; clip stitching/LLM parse dominate.  Do not call the query path
  real-time.

### Privacy / generalization / limitations
- Describe the new implementation: role/zone access control (deployment
  switch), audit JSONL, retention pruning (8 h default, disabled by default),
  face-blur switch, gender-off switch, gender ablation (if measured).
- Do not claim demographic-bias measurement without labels; state it as a
  limitation.
- Generalization: single site; released subset single camera/2 h; add a
  temporal holdout result if available; no cross-site/night/rain claim.
- Keep crowded/occlusion/domain-shift/ambiguous-action limitations.
- Add exact storage $312{,}544\times1152\times4$ B ~= 1.44 GB.

### Formatting and page budget
- Keep the PDF at 8 pages (6 + 2 overlength).  Replace verbose prose rather
  than adding pages.  The current draft already has 13 tables; do not add a
  14th without removing one.
- Suggested consolidation:
  * Drop the realized-alpha table (Table V) and fold its ranges into one
    sentence.
  * Replace the consolidated comparison (Table VII) with a two-track table:
    full-deployment rows plus released-subset rows.  This absorbs the new
    subset results without a second table.
  * Keep the full-deployment alpha-baseline table (III) and CI table (X); they
    are the C1 headline evidence.
  * Replace the alert table (XI) with test-split per-field + rule-correctness
    and add 4 stress rows (ambiguous/conflicting/missing/unsupported) for
    clarify/abstain.  One table, not two.
  * Report subset latency p50/p95 and concurrency in prose, not a new table,
    unless space remains.
  * If still over 8 pages, shrink the qualitative table (IV) to 2 rows or
    move the anomaly table (VIII) to prose.
- Fix the 35.5 pt overfull hbox in the Setup paragraph (line ~450) by splitting
  the long sentence and/or using `\sloppy` locally (not globally).
- Fix the 3.1 pt overfull lines in the abstract/related-work table if easy.
- Add at most one compact results figure (alpha sensitivity or latency
  distribution); avoid a second page.
- Ensure all `\ref`/`\cite` resolve; compile twice.

## Ready-to-integrate facts (from `results/`)

### Alert (C3)
- `results/alert_results.json`: 120 author-constructed utterances, 60 clean
  + 60 stress (15 ambiguous / 15 conflicting / 15 missing / 15 unsupported),
  60 dev / 60 test.
- Test clean: 30/30 rule-correct (1.00), all nine fields 1.00; test stress:
  30/30 clarified (1.00), 0 false materializations; per-kind 1.00.
- Dev clean/stress also 1.00.  This is a coverage check on an
  author-constructed set, not a human study.  Keep the full-deployment 0.91
  as the primary human-authored result and state the released test split
  separately.  Do not replace 0.91 in the abstract/conclusion with 1.00.

### Latency (R1-11 / R2-3)
- `results/latency_results.json`, CPU-only, released subset:
  scene retrieval core (OpenCLIP encode + FAISS + aggregation + merge):
  33.3 ms p50 / 40.6 ms p95; person retrieval core (MiniLM + person FAISS +
  timestamp mapping + merge): 16.9 ms p50 / 18.8 ms p95.
  Concurrency 1/2/4/8 workers: 26.3/29.3/37.0/38.2 queries/s; p95
  41.1/86.5/131.9/255.3 ms.
- Excluded: LLM parse/answer, clip stitching, MongoDB, ingestion, frontend
  network, full 312k deployment.  Do not present these as full-deployment
  percentiles.  The full-deployment 6.0 s mean / 12.0 s Find remains the
  end-to-end number; it is dominated by parse + clip stitching.

### Privacy (R1-13)
- `results/privacy_notes.md`: opt-in role/camera/zone access control
  (`ACCESS_CONTROL_ENABLED=false` default; header-based behind a trusted
  proxy), JSONL audit, 8 h retention prune (explicit, not automatic), Haar
  face blur (still-image utility, not wired into live ingestion), gender-off
  output filter (does not rewrite the released index).
- No demographic-bias labels; no IAA; no full-deployment privacy result.
  State these as limitations, not as measured outcomes.

### Retrieval (C1) -- important
- The full released-subset benchmark is **not protocol-matched** to the
  full-deployment C1 evaluation: count questions are "how many" without a
  target count (the structured branch has no target to rank), and action
  queries carry time windows (so the structured branch has a strong time
  cue, while the action override intentionally suppresses it).  The raw
  subset numbers therefore show adaptive (0.486) below fixed (0.568); do not
  replace the full-deployment table with these, and do not claim the subset
  reproduces C1.
- If a protocol-matched control is produced (target-count queries and
  action queries without time cues, or action omitted), label it explicitly
  as a post-hoc control, not the primary benchmark.  Otherwise report the
  subset as a code-path/smoke test with the protocol mismatch stated.
- Count EM/MAE on the raw subset: 0.10 / 2.05 for adaptive and fixed-1.0;
  full-deployment per-query counts were not preserved.

### Fusion (C2)
- Pending `results/fusion_results.json`; use only when it exists.  The
  subset person pairs are track-derived single-camera proxies, not human
  cross-camera identities.

## Evidence map (fill from results)

| Claim | Full deployment | Released subset |
|---|---|---|
| Fixed alpha 0.666 | `tab:alpha_baseline` | `results/retrieval_results.json` |
| Adaptive 0.828 | `tab:alpha_baseline_ci` | same |
| Fusion 0.807 / RRF 0.785 | `tab:fusion_ablation` | `results/fusion_results.json` |
| Alert 0.91 | `tab:alert_authoring` | `results/alert_results.json` |
| Latency 6.0 s / 12.0 s | `tab:query_performance` | `results/latency_results.json` |
| 312k descriptor / 1.44 GB | text | N/A (subset 1,778) |

---

## Final state (applied to `INDICON2026_submit/INDICON2026Conference.tex`)

The PDF is 8 pages, compiles with no overfull box, no undefined reference and
no undefined citation (`pdflatex` -> `bibtex` -> `pdflatex` x2).

### Applied edits

| Section | Change |
|---|---|
| Abstract | labelled the 150-query benchmark as the full four-camera deployment; 91% attributed to the deployment study; added one release sentence |
| Introduction | unchanged claims; release sentence in the contribution list |
| Background | dropped the nine-row off-the-shelf model table (information kept in prose) to stay within 8 pages |
| C1 | added "Released-subset check (C1)" paragraph with the protocol-matched control; removed the realized-alpha table (folded into text); updated the soft-weight follow-up wording |
| C2 | added the released-subset fusion check; multi-seed and learned-projection baselines now reported from the subset |
| C3 | removed the "materializer depends on the LLM" contradiction; the materializer is now the deterministic `alert_materializer.py`; added the released alert benchmark paragraph (30/30 clean, 30/30 stress) as a coverage check, keeping the 0.91 deployment result |
| Setup | split the overfull 35.5 pt paragraph; deployment benchmark vs released-subset track separated; annotator/tune-test provenance stated; 10k-resample CIs, grouped bootstrap and Holm now described per track |
| Latency | replaced "measured results will be reported in the final version" with the measured subset p50/p95 and 1-8 worker throughput, explicitly excluding parse/stitch/Mongo/ingestion |
| Limitations | (iii) and (iv) rewritten to match the implemented materializer, validator and privacy/access switches; no IAA, cross-camera or demographic-bias claim |
| Conclusion | release description updated; headline numbers unchanged |

### Numbers used (all from `results/`)

| Claim | Source | Value |
|---|---|---|
| Deployment C1 | paper table (primary, unchanged) | 0.666 -> 0.828 macro P@5 |
| Released C1 control | `results/retrieval_control_results.json` | adaptive 0.538 = best fixed 0.538 (Δ 0.000, p 1.0); paper lookup 0.428; α=0.0 0.048; α=0.3 0.380; count 0.390 vs 0.010; action 0.090 vs 0.000 |
| Raw released retrieval | `results/retrieval_results.json` | adaptive 0.486 vs best fixed 0.568 (protocol mismatch, reported in the release only) |
| Deployment C2 | paper table (primary, unchanged) | concat 0.807 vs RRF 0.785 vs visual 0.700 |
| Released C2 | `results/fusion_results.json` | visual 0.484, learned proj 0.473, concat 0.451, randproj 0.436±0.0004 (5 seeds), RRF 0.395 P@5; pair AUC 0.744/0.693/0.749 |
| Deployment C3 | paper table (primary, unchanged) | 0.91 rule-correctness, 60 held-out utterances |
| Released C3 | `results/alert_results.json` | test clean 30/30 nine fields exact; test stress 30/30 clarified, 0 false materializations |
| Deployment latency | paper table (primary, unchanged) | 6.0 s mean, 12.0 s Find |
| Released latency | `results/latency_results.json` | scene core 37.4/37.4/39.7 ms (mean/p50/p95); person core 17.2/18.5/20.4 ms; 26.5-39.5 q/s at 1-8 workers, p95 39.6-265.4 ms |
| Released privacy | `results/privacy_notes.md` | opt-in access control + audit, 8 h retention prune, face-blur utility, gender-off filter (all default off) |

### Deliberately not claimed

- The released subset does not reproduce the 4-camera/312k results.
- No human inter-annotator agreement, cross-camera Re-ID, cross-site or
  night/rain generalisation, or 312k-descriptor latency percentiles.
- The 0.91 alert figure is a deployment-study result; the released benchmark
  is a separate author-constructed coverage measurement and does not replace
  it.
- The released fusion track does not reproduce the deployment ordering; the
  pseudo-track construction biases visual-only and is stated in the paper.
