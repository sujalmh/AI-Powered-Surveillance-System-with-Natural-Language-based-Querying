# Paper 850 — improvement plan (v3)

**Status:** active plan for the panel revision. Supersedes v1/v2 of this file and
the older `REVISION_PLAN.md` / `PAPER_CHANGES.md`.

**Premise:** keep paper 850's **original title, structure and three-component
story**. Improve the evidence and the system. Do not rebuild the paper around
failures or errors. The remaining parser gap is a single evaluation finding and
a single future-scope sentence — nothing more.

**Scope:** INDICON 2026 paper 850,
`INDICON2026_submit/INDICON2026Conference.tex`. Arten is future work and is not
part of this plan.

---

## 0. What stays

- **Title (unchanged):**
  *Intent-Adaptive Hybrid Retrieval and Heterogeneous-Space Person Fusion for
  Conversational Video Surveillance.*
- **Structure (unchanged):** Introduction; Related Work and Position;
  Background and System Substrate; C1; C2; C3; Evaluation; Discussion and
  Conclusion.
- **The three components and the deployment:** C1 alert compiler, C2
  heterogeneous person descriptor, C3 adaptive routing, four-camera deployment,
  released single-camera track.
- **The 6-page INDICON format.**
- **The novelty:** an integrated conversational surveillance system that
  couples alert authoring, heterogeneous person retrieval and adaptive hybrid
  ranking, with a deployment study and a released reproducibility track.

No new sections. No new headline thesis. The paper reads as a stronger 850, not
a different paper.

---

## 1. What improves (the value increase)

| Area | Current paper | Improvement | Why it is stronger |
|---|---|---|---|
| C1 | 91% deployment headline; "verified"; long probe paragraph | Deterministic compiler becomes the primary result: 30/30 clean + 30/30 stress, zero false materializations; 91% moves to the deployment table with the LLM-materializer caveat; "schema-validated" replaces "verified" | Reproducible evidence; the deterministic compiler is a real artifact; no implementation mismatch |
| C2 | Deployment 0.807 table; "Re-ID"; single-seed projection | Released table with visual, text, concat, zero-pad/norm-matched, learned projection, 5-seed random projection, RRF, gated; two-view gain 0.427 vs. 0.407/0.361; deployment 0.807 in the deployment table; no identity claim | Fair baselines; the zero-pad/learned tie becomes a mechanism note (two-view combination, not the concat operator) |
| C3 | Deployment 0.828 tables; "by construction"; action α=0.10 conflict | Released control table: adaptive 0.538 = fixed 0.7 0.538; end-to-end 0.450 vs. 0.448 (p=0.80); deployment 0.828 in the deployment table; Eq. (3) action branch fixed to semantic-only | Training-free parity without a per-site α sweep; the specification matches the code |
| Evaluation | Mixed deployment/released numbers; deployment p-values | Two-track presentation: deployment study table + released tables; no deployment p-values; grouped/clip-cluster CIs; z-score robustness; MMR ablation | Trust and clarity; every claim has a source |
| System story | Three components, no figure | One system figure in Background; one claim → experiment → metric table in Evaluation; no new sections | The whole system is visible; the components are clearly coupled |
| Safety | Disclosed, not implemented | Gender off by default; approval gate for consequential alerts; MMR off by default | Responsible-deployment improvement, not just a caveat |
| Trust | No number checker | `check_paper_numbers.py` + `paper_claims.yaml`; one source per table | Prevents drift and overclaim |

**No value is removed.** The deployment numbers stay in the paper as a
deployment study. The three components stay. The title stays. The structure
stays.

---

## 2. Parser-gap rule (exactly two mentions)

The remaining end-to-end gap is real, but it is not the paper's thesis. Use it
**twice only**:

1. **Evaluation, after the end-to-end result:** one sentence:
   > The remaining end-to-end gap is dominated by query parsing rather than
   > ranking (oracle-filter 0.538 vs. parser-driven 0.450; track queries
   > −0.28), which identifies the next improvement.

2. **Discussion, future scope:** one sentence:
   > The highest-leverage next step is parser improvement for track time
   > windows, where the reference and parsed windows differ in 30/30 cases.

**Do not mention it in:** the title, abstract, contributions, C1/C2/C3
headings, the C3 result paragraph, the conclusion, or the response-letter
headline. No "error budget" section, no "bottleneck" repetition, no
"failure" language.

---

## 3. Claim ledger

| Claim | New role | Value added |
|---|---|---|
| 0.828 adaptive routing | **Deployment study row**; the released control is the reproducibility check | The re-computable result is training-free parity (0.538 = 0.538); the deployment shows the larger recorded gain. |
| 0.807 concat fusion | **Deployment study row**; the released table is the reproducibility check | The re-computable two-view gain is 0.427 vs. 0.407/0.361; the deployment shows the larger recorded gain. |
| 91% alert rule-correctness | **Deployment study row** with the LLM-materializer caveat | The released deterministic compiler is 30/30 + 30/30 with zero false materializations. |
| "verified alert compiler" | **"Schema-validated alert compiler"** | More precise, supported by the validator and tests. |
| "Person Re-ID" (keyword/abstract) | **"Person descriptor retrieval"** | No identity claim; the title's "Person Fusion" stays. |
| 10.7-pp / 2.2-pp fusion gains | **Released +2.0 pp over visual, +6.7 pp over RRF, plus the mechanism note**; deployment numbers in the deployment table | Fairer comparison; the zero-pad/learned tie explains why the two-view gain exists. |
| 0.666 → 0.828 routing gain | **Deployment study row**; the released control is the reproducibility check | The reproducible result is parity without per-site tuning (0.538 = 0.538). |
| 0.538 = 0.538 | **Primary C3 released result** | Training-free parity; no per-site α sweep. |
| 0.450 vs. 0.448, p=0.80 | **Primary end-to-end result** | The parser-driven system is measured. |
| 30/30 + 30/30 | **Primary C1 result** | Deterministic, reproducible compiler. |
| MMR 0.538 → 0.400 | **Deployment finding; MMR off by default** | A concrete system improvement. |
| Gender in descriptor/ranking | **Gender off by default; descriptors rebuilt** | Responsible-deployment improvement. |
| No approval gate | **`requires_approval` gate + test** | Responsible-deployment improvement. |
| 312k/109 ms/6.0/12.0 s | **Deployment study table** | Keeps deployment context without using it as the evidence. |
| "only this one provides all three" | **Scoped "to our knowledge" sentence** | Preserves the system novelty without overclaiming. |

---

## 4. Abstract

Replace the current abstract with:

> Conversational surveillance requires turning free text into executable alert
> rules, representing people across visual and attribute spaces, and ranking
> candidates from structured and semantic branches. We present an integrated
> system that couples these decisions in one pipeline and evaluate it on a
> four-camera deployment and a released single-camera reproducibility track.
> The system combines a deterministic, schema-validated 9-tuple alert compiler
> with clarify/abstain validation (30/30 clean and 30/30 stress on the released
> benchmark, no false materializations); a heterogeneous SigLIP+MiniLM person
> descriptor that improves P@5 over visual-only and RRF on the released track
> (0.427 vs. 0.407 and 0.361); and a training-free intent-adaptive routing rule
> that matches the best tune-selected fixed alpha (0.538 vs. 0.538) and ties it
> end-to-end (0.450 vs. 0.448), removing the need for a per-site alpha sweep.
> The deployment study provides the motivating measurements; the released track
> makes the pipeline re-computable. The system is released with its benchmark
> definitions and measured outputs.

No deployment numbers, no parser mention, no "verified," no Re-ID, no
"10.7/2.2," no "0.67→0.83."

---

## 5. Contributions

Keep four contributions, positive and scoped:

1. **An integrated conversational surveillance system.** One query schema
   couples alert authoring, heterogeneous person descriptor retrieval and
   adaptive hybrid retrieval; the paper reports a four-camera deployment study
   and releases a single-camera reproducibility track with code, benchmark
   definitions and measured outputs.
2. **A deterministic, schema-validated alert compiler.** Nine-field rules with
   clarify/abstain; released test split 30/30 clean and 30/30 stress with zero
   false materializations; fuzz and adversarial probes. The deployment study
   used an LLM-backed materializer and is reported in the deployment table.
3. **A heterogeneous person descriptor with a fair comparison.** SigLIP+MiniLM
   concatenation gives P@5 0.427 vs. 0.407 visual-only and 0.361 RRF on the
   released track; zero-pad/norm-matched and learned projection are
   statistically indistinguishable, so the gain comes from combining views
   rather than the specific concatenation operator. No identity claim.
4. **A training-free intent-adaptive routing rule.** Matches the best
   tune-selected fixed alpha (0.538 = 0.538) and ties end-to-end (0.450 vs.
   0.448), removing the need for a per-site alpha sweep. The four-camera
   deployment recorded the larger gains in the deployment table; the released
   track is the reproducibility check.

**No parser mention in the contributions.**

---

## 6. Section-by-section edits

### Title (line 46)

Keep exactly:

> Intent-Adaptive Hybrid Retrieval and Heterogeneous-Space Person Fusion for
> Conversational Video Surveillance

### Abstract (lines 63–67)

Replace with §4. Remove "verified," "91%," "10.7-pp," "2.2-pp," "0.822 vs.
0.828," "0.67 to 0.83," and "Lemma."

### Keywords (line 70)

Replace "Multimodal Person Re-ID" with "Person Descriptor Retrieval." Keep the
rest; optionally add "Reproducibility."

### Introduction (lines 73–85)

- Keep the opening problem: ranking candidates from structured and semantic
  branches.
- Keep the three gaps: alert authoring, heterogeneous descriptor fusion, hybrid
  ranking.
- Replace the "no fixed value exceeds 0.67" motivation with the released
  control: best fixed 0.538; adaptive 0.538; the deployment 0.666/0.828 is the
  motivating deployment observation.
- Replace the four contributions with §5.
- Remove "only this one provides all three"; keep the scoped "to our knowledge"
  sentence.

### Related Work (lines 87–95)

- Keep the three axes.
- Replace "only this one provides all three" with:
  > To our knowledge, no prior system couples conversational alert authoring,
  > heterogeneous frozen-encoder person descriptors and adaptive hybrid
  > retrieval, and reports a deployment study alongside a released
  > reproducibility track.
- Keep the rest; no new claims.

### Background and System Substrate (lines 97–101)

- Keep the nine-model pipeline description.
- Add **Figure 1**: capture → YOLO/OSNet/attributes → scene/person FAISS → NL
  parser → structured/semantic branches → adaptive merge → post-ranking →
  top-k; alert compiler + approval gate on the side; release boundary marked.
- No new section; the figure sits inside Background.

### C1 (lines 103–117)

- Keep grammar, authoring example, defaults, six rule families.
- Replace the 0.91 headline with: the released deterministic compiler is
  evaluated at 30/30 clean and 30/30 stress; the deployment study with the
  LLM-backed materializer recorded 91%, and the two paths are complementary,
  not interchangeable. Put the 0.91/per-field numbers in the deployment table.
- Keep validation safeguards; move the 68/225 contradiction-detection probe to
  a footnote/supplement.
- Condense release probes to 2–3 sentences; keep fuzz/adversarial.
- Replace "verified" with "schema-validated."

### C2 (lines 119–158)

- Keep the problem, construction, Eq. (1)–(2), and "why concatenation" note.
- State deployment gallery N=312,544 and released track N=1,778; deployment
  calibration 50 pairs; released track 1,008 calibration pairs + 5-fold track
  CV.
- Replace Table II with the released table:

  | Method | P@5 | R@10 | F1(P@5,R@10) | 95% CI |
  |---|---:|---:|---:|---|
  | Visual only | 0.407 | 0.285 | 0.290 | [0.390,0.424] |
  | Text only | 0.102 | 0.095 | 0.077 | [0.089,0.116] |
  | **Concat (ours)** | **0.427** | **0.311** | **0.309** | **[0.410,0.445]** |
  | Zero-pad / norm-matched | 0.428 | 0.312 | 0.309 | [0.410,0.446] |
  | Learned projection | 0.432 | 0.311 | 0.312 | [0.415,0.450] |
  | 5-seed random projection | 0.417 ± <0.001 | 0.297 | 0.298 | seeds |
  | RRF | 0.361 | 0.285 | 0.266 | [0.343,0.378] |
  | Gated rank (post-hoc) | 0.444 | 0.307 | 0.315 | [0.425,0.462] |

  Caption: "Released single-camera track; metadata-formed pseudo-tracks (280;
  1,190 queries; 1,778 descriptors), not identities. Paired bootstrap, 3,000
  resamples; Holm over competitors. Concat beats visual (Δ+0.021, p<0.0007) and
  RRF (Δ+0.067, p<0.0007); zero-pad and learned projection tie; gated is +0.017
  P@5 but pair-AUC parity. The tie shows the gain comes from combining views,
  not the concatenation operator."
- Keep the deployment 0.807/0.785/0.700 as one sentence and in the deployment
  table; do not put it in the primary table.
- Replace "the released subset reverses the deployment ordering" with: "the
  metadata-only track keeps concat first but puts visual above RRF
  (0.427/0.407/0.361); the earlier visual-cosine formation inverted the order
  by construction, which is why formation is fixed to metadata."
- If the calibrated gate 0.472/0.734 is kept, add it as a separate row with its
  own source (`eval_fusion_cal2.py`) and Holm p; do not mix it with the 0.444
  uncalibrated gate. Prefer dropping it if space is tight.

### C3 (lines 166–232)

- Keep Eq. (1) and the fixed-α baseline analysis, based on the released control:
  best fixed 0.538; extremes 0.048 (α=0.0) and 0.520 (α=1.0); adaptive 0.538.
- Eq. (3): change the action branch to `semantic-only (R_s skipped; α
  irrelevant)`; remove 0.10.
- Text: "The deployed path skips R_s for action queries; the α-sweep harness
  uses α=0.0 as the semantic-only proxy. The 0.10 value is a harness/config
  artifact, not the deployed behavior."
- Replace Tables III and IV with the released control table:

  | Method | Count | Find | Track | Action | Filter | Macro | 95% CI |
  |---|---:|---:|---:|---:|---:|---:|---|
  | fixed 0.0 | 0.010 | 0.070 | 0.020 | 0.090 | 0.050 | 0.048 | [0.016,0.092] |
  | fixed 0.3 | 0.270 | 0.630 | 0.550 | 0.090 | 0.360 | 0.380 | [0.202,0.569] |
  | fixed 0.5 | 0.390 | 0.910 | 0.810 | 0.090 | 0.470 | 0.534 | [0.345,0.689] |
  | fixed 0.7 (tune-best) | 0.390 | 0.920 | 0.810 | 0.090 | 0.480 | 0.538 | [0.349,0.692] |
  | fixed 1.0 | 0.390 | 0.920 | 0.810 | 0.000 | 0.480 | 0.520 | [0.309,0.689] |
  | **Adaptive** | **0.390** | **0.920** | **0.810** | **0.090** | **0.480** | **0.538** | **[0.349,0.692]** |
  | Per-intent tuned | 0.390 | 0.920 | 0.810 | 0.090 | 0.470 | 0.536 | [0.346,0.691] |
  | Paper lookup | 0.390 | 0.630 | 0.550 | 0.090 | 0.480 | 0.428 | [0.251,0.606] |

  Caption: "Released protocol-matched control (50 tune / 100 test; post-hoc;
  metadata-derived labels). Adaptive equals the tune-selected fixed α=0.7 on
  every category (Δ=0.000, p=1.0). The four-camera deployment recorded the
  larger gains in the deployment table; the released control is the
  reproducibility check."
- End-to-end: adaptive 0.450 vs. tune-selected fixed 0.5 0.448 (Δ+0.002,
  p=0.80; 95% CI [−0.004,+0.010]); oracle-filter control 0.538.
- **Parser gap:** one sentence, only here in the results:
  "The remaining end-to-end gap is dominated by query parsing rather than
  ranking (oracle-filter 0.538 vs. parser-driven 0.450; track queries −0.28),
  which identifies the next improvement."
- Lemma: downgrade to "Observation (conditional)." Keep A1–A2 and the scope;
  replace "attains both by construction" with "attains both on the deployment
  benchmark; the released control ties, so the gap is not observed there."
- Post-ranking: "Recency is inert on the released control; MMR at λ=0.3 costs
  0.138 P@5 (0.538→0.400), so it is off by default and reported as an
  ablation."
- Remove the duplicate segment-disjoint sentence; if mentioned, say "n=10,
  inconclusive" once.

### Evaluation (lines 234–284)

- Setup: two tracks, explicitly: deployment study (recorded) + released
  reproducibility track (primary comparison).
- Statistical protocol: remove the deployment macro p<0.0002; say "deployment
  CIs/p-values are recorded single-run measurements, not re-computable
  significance." Keep the released grouped bootstrap, paired CIs, Holm, and the
  n=30 resolution sentence.
- C1: report the released 30/30 first; put the 0.91/per-field deployment
  numbers in the deployment table.
- C2: the released table is the table; the deployment 0.807/0.785/0.700 is one
  observation sentence.
- C3: the released control table is the table; the deployment α-sweep is one
  observation sentence; the end-to-end tie is the result.
- Replace the "Supporting numbers" paragraph with the deployment study table:

  | Quantity | 4-camera deployment (recorded) | Released track (reproducibility check) |
  |---|---|---|
  | Alert compiler | 91% full-rule, 60 utterances, LLM materializer | 30/30 clean, 30/30 stress, deterministic |
  | Person descriptor | 0.807 concat / 0.785 RRF / 0.700 visual; 312,544 descriptors | 0.427 / 0.361 / 0.407 P@5; 1,778 descriptors |
  | Routing | 0.666→0.828 macro P@5; 150 queries | 0.538 = 0.538; 100 control queries |
  | End-to-end | 6.0 s / 12.0 s; 109 ms person index | 0.450 vs. 0.448; 3.0 s parse; 37.4/17.2 ms FAISS |
  | Post-ranking | production MMR 0.400 | MMR off 0.538; MMR on 0.400 |

- Add Table I (claim → experiment → metric → source) in Evaluation; no new
  section.
- Add the z-score robustness sentence: min-max 0.538 = 0.538; z-score 0.524 vs.
  0.522 (p=0.72); the parity holds, the absolute score shifts by 0.014.
- Cut the long C1 probe paragraph, the duplicate segment sentence, and the
  supporting-numbers prose to stay at 6 pages.

### Discussion and Conclusion (lines 286–290)

- Limitations: keep (i) conditional observation, (ii) descriptor cost + MMR
  ablation, (iii) alert compiler paths complementary, not interchangeable +
  author-constructed benchmark, (iv) deployment study recorded, not
  re-computable, (v) privacy/fairness: no identity/fairness claim, gender
  excluded from descriptor by default, approval gate, access/audit/retention
  supplied but off by default.
- Future scope: public identity-labelled benchmark; **one parser sentence**
  (track time windows); runtime alert quality (per-family P/R, false alerts per
  camera-hour, latency); cross-camera generalization.
- Conclusion: "We presented an integrated conversational surveillance system
  that couples a deterministic alert compiler, a heterogeneous person
  descriptor and a training-free adaptive routing rule, with a four-camera
  deployment study and a released single-camera reproducibility track. The
  released evidence supports the compiler, the two-view descriptor and routing
  parity without per-site tuning; the deployment study supplies the motivating
  measurements. The release makes the claims re-computable and provides the
  basis for the next iteration."
- No parser mention in the conclusion. No "verified," 91%, 10.7/2.2, 0.828.

---

## 7. Code and experiment tasks

### E1 — C2 evidence table

- Source: `results/fusion_tables.md`, `results/fusion_results.json`.
- Replace Table II with the released table; add the fair baselines; add the
  mechanism sentence.
- Keep the deployment 0.807/0.785/0.700 in the deployment table and one
  sentence.
- Reconcile the gated 0.472/0.734 (calibrated, `eval_fusion_cal2.py`) with the
  0.444/0.711 (uncalibrated, `fusion_tables.md`). Use one; if both, separate
  rows and sources.
- Gate: every table cell maps to `fusion_tables.md` or `fusion_results.json`.

### E2 — C3 released control + end-to-end

- Source: `results/retrieval_control_tables.md`,
  `results/end_to_end_results.json`.
- Replace Tables III/IV with the control table; add the end-to-end tie.
- Add the z-score robustness sentence: 0.524 vs. 0.522, p=0.72; min-max
  0.538 = 0.538.
- Add a clip-cluster bootstrap to `eval_retrieval_control.py` (resample by clip,
  not query) and report it as a sensitivity; the split is query-disjoint, not
  clip-disjoint (98/104 leakage).
- Gate: 0.538 = 0.538 on every category; end-to-end 0.450 vs. 0.448, p=0.80.

### E3 — Parser-gap analysis (one result sentence)

- New script: `backend/scripts/analyze_end_to_end_gap.py`.
- Read `end_to_end_results.json` (`oracle_delta.per_category`,
  `parse_fidelity_vs_reference`) and write `results/end_to_end_gap.md`.
- The paper uses **one sentence** in Evaluation and **one sentence** in future
  scope; do not add a section, table, or headline.
- State the oracle is an upper bound (label-side attributes and a reference
  window) so the sentence is precise.
- Gate: numbers match the JSON; the paper contains at most two parser mentions.

### E4 — MMR default off

- `backend/app/config.py`: set MMR off by default (`MMR_ENABLED=false` or
  `MMR_LAMBDA=1.0`); keep the option.
- Re-run `eval_ablation.py` and `eval_end_to_end.py`; confirm the default
  matches the 0.538 control and the 0.450 end-to-end.
- Paper: "MMR costs 0.138 P@5 on the released track; disabled by default;
  reported as an ablation." The deployment 0.400 stays in the deployment table.
- Gate: config default = off; ablation table unchanged; paper matches.

### E5 — Gender off by default + descriptor rebuild

- `backend/app/config.py`: `INCLUDE_GENDER_ATTRIBUTES` default `false`.
- `backend/scripts/rebuild_person_index.py` (new): load
  `data/index/faiss_person/{index.bin,meta.json}`; recover the visual/text
  blocks from the 1152-d fused vector; strip gender terms from
  `attribute_text`; re-embed the text with MiniLM; re-fuse with the same
  weights; L2-normalize; write `data/index/faiss_person_nogender/`; verify the
  reconstruction against the original vector before writing.
- Re-run `eval_fusion.py`, `eval_retrieval_control.py`, `eval_end_to_end.py`
  with the new index; update the C2/C3 tables and the paper's numbers.
- If the rebuild is not numerically faithful, re-index from the source ROIs;
  do not ship a descriptor that still contains gender.
- Paper: one limitations sentence: "Gender is excluded from the descriptor by
  default; no per-group error rate is claimed."
- Gate: no gender term in the released descriptor text; C2/C3 tables re-run.

### E6 — Approval gate

- Add `requires_approval` to the alert rule schema (9-tuple + flag).
- `alert_validator.py`: default `true` for consequential event families; reject
  a rule that sets it false without an explicit authorization flag.
- `alert_engine.py`: when `requires_approval` is true, insert a
  `pending_approval` alert log and do not execute side effects; add an
  approve/reject path (API or queue) and a test.
- Paper: one limitations/operations sentence: "Consequential alerts are staged
  for operator approval by default; the artifact exposes the flag and the
  pending queue."
- Gate: test proves no side effect before approval.

### E7 — Number checker

- `backend/scripts/check_paper_numbers.py` + `INDICON2026_submit/paper_claims.yaml`.
- Parse the `.tex` for numeric tokens; map each claimed number to a JSON path
  or a deployment-table row; fail on mismatch or on a deployment number outside
  the deployment table; whitelist section numbers, years, dataset sizes and
  units.
- Run before packaging.
- Gate: all headline numbers mapped; zero unmapped deployment claims.

### E8 — Figure 1 + Table I

- Figure 1 in Background: capture → perception → indices → parser →
  structured/semantic branches → adaptive merge → post-ranking → top-k; alert
  compiler + approval gate; release boundary.
- Table I in Evaluation: claim → experiment → metric → source (C1 30/30; C2
  0.427/0.407/0.361; C3 0.538=0.538; end-to-end 0.450 vs. 0.448).
- No new sections; cut prose to keep 6 pages.
- Gate: figure/table referenced; 6 pages; 0 overfull; 0 undefined.

### E9 — Optional public identity benchmark (stretch)

- Only if a public dataset with identity labels is available and the deadline
  allows: Market1501/MSMT17/RAP with SigLIP+MiniLM, same fusion, mAP/CMC-1/5/10;
  report visual vs. concat vs. zero-pad vs. RRF.
- If not available, keep the pseudo-track table and state "no public identity
  benchmark" in limitations. Do not call the pseudo-track evaluation Re-ID.
- Gate: dataset provenance + script + numbers; otherwise skip.

---

## 8. Gates

- **G0 — identity.** Title unchanged; structure unchanged; abstract,
  contributions and conclusion contain no `0.828`, `0.807`, `91%`, `verified`,
  `10.7`, `2.2`, `0.67 to 0.83`; "Person Re-ID" is gone from the abstract and
  keywords; the parser gap appears at most twice (Evaluation + future scope).
- **G1 — provenance.** Every number maps to a measured JSON or the deployment
  table; the checker passes.
- **G2 — C2 table.** Matches `fusion_tables.md`: visual 0.407, concat 0.427,
  RRF 0.361, zero-pad 0.428, learned 0.432, gated 0.444; paired/Holm values
  match; no 0.807 in the primary table; the mechanism sentence is present.
- **G3 — C3 table.** Matches `retrieval_control_tables.md`: adaptive 0.538 =
  fixed 0.7 0.538, per-category identical; end-to-end 0.450 vs. 0.448, p=0.80;
  0.828 only in the deployment table; the parity framing is present.
- **G4 — C1 table.** Matches `alert_results.md`: 30/30 clean, 30/30 stress, 0
  false materializations; 91% only in the deployment table; "schema-validated"
  not "verified."
- **G5 — equation/code agreement.** Eq. (3) action branch is semantic-only;
  text says α=0.0 proxy; no 0.10.
- **G6 — statistics.** No deployment p-values; released C3 tie with grouped CI
  and clip-cluster sensitivity; C2 paired bootstrap + Holm; z-score robustness;
  n=30 resolution stated.
- **G7 — story.** Figure 1 + Table I; scoped "to our knowledge" novelty
  sentence; no "only this one provides all three"; no new sections.
- **G8 — safety.** Gender off by default; descriptor rebuilt; approval gate test
  passes; MMR off by default; paper states the defaults; no fairness claim.
- **G9 — build.** 6 pages, 0 overfull, 0 undefined; every table/figure
  referenced; artifact URL 200 logged out; source zip compiles.
- **G10 — package.** PDF, source zip and response letter regenerated from the
  final `.tex`; number checker passes.

---

## 9. Timeline (10 working days)

**Days 1–2 — C1 and abstract/contributions.**
Deterministic 30/30 + 30/30 as the C1 result; 91% to the deployment table;
abstract/contributions/title unchanged; "verified" out; "schema-validated" in.

**Days 3–4 — C2 table and mechanism note.**
Released table swap; fair baselines; two-view gain; zero-pad/learned tie note;
deployment 0.807 in the deployment table; no Re-ID claim.

**Days 5–6 — C3 table, equation, parity.**
Eq. (3) action branch; released control table; end-to-end tie; MMR default off;
one parser sentence; deployment 0.828 in the deployment table.

**Days 7–8 — evaluation, safety, figure.**
Deployment table; stats; z-score/clip-cluster; gender off + rebuild; approval
gate; Figure 1 + Table I; cut prose to 6 pages.

**Days 9–10 — verify and package.**
Tests, number checker, build, page/overfull/undefined checks, artifact URL,
source zip, response letter.

**If the window is 3 days:** G0, G2, G3, G4, G5, G9, G10; keep the safety fixes
as code changes with a limitations sentence if the rebuild cannot finish.
**If the window is 7 days:** add G6–G8; skip E9.

---

## 10. Response letter (positive framing)

| Panel point | Response: what the revision improves |
|---|---|
| C3 0.828 not established | The deployment result is in the deployment table; the revision adds the released control (0.538 = 0.538) and the end-to-end tie (0.450 vs. 0.448, p=0.80), with the parity/no-per-site-sweep framing. |
| C2 Re-ID unsupported | C2 is now person descriptor retrieval; the release adds a fair two-view comparison (0.427 vs. 0.407/0.361) on pseudo-tracks, with no identity claim and a disclosed split audit. |
| C2 comparison unfair | The revision adds zero-pad/norm-matched, learned projection, 5-seed random projection, RRF and gated baselines, paired bootstrap + Holm, and 5-fold track CV. |
| Eq. 3 conflict | The action branch is now semantic-only; α=0.0 is the sweep proxy; 0.10 removed. |
| 91% different implementation | The deterministic compiler is now the C1 result (30/30 + 30/30); 91% is a deployment observation with the LLM-materializer caveat. |
| Statistics | The revision removes deployment p-values, adds grouped/clip-cluster CIs and Holm on the released track, and states the n=30 resolution. |
| No single story | The paper keeps its three-component structure and adds one system figure + one claim table; the components are clearly coupled. |
| Privacy/fairness/safety | Gender off by default (descriptors rebuilt), `requires_approval` gate, MMR off; governance defaults stated. |
| Normalization | Both min-max (0.538 = 0.538) and z-score (0.524 vs. 0.522, p=0.72) are reported; the parity is robust. |
| Runtime alert quality | Named as future scope with the metrics; no number claimed. |
| Metric consistency | One source per table; number checker; gated 0.472/0.734 separated or dropped; duplicate sentence removed. |
| Novelty | The system novelty is preserved: integrated query-to-alert pipeline + three components + deployment study + release; the "only this one" overclaim is replaced by a scoped "to our knowledge" sentence. |
| Diagram/eval map | Figure 1 + Table I added inside the existing sections; no new sections. |

---

## 11. Day-1 commands and first edits

```powershell
cd S:\college\ai-surveillance
git status --short --branch
python -m pytest backend/tests -q
$env:PYTHONIOENCODING='utf-8'
python backend/scripts/eval_alerts.py
python backend/scripts/eval_retrieval_control.py
python backend/scripts/eval_fusion.py
python backend/scripts/eval_end_to_end.py
```

First edits, in order:

1. `INDICON2026_submit/INDICON2026Conference.tex`: replace the abstract with §4;
   replace the contributions with §5; keep the title and structure.
2. §C1: deterministic 30/30 + 30/30 as the primary result; 91% to the
   deployment table; "verified" → "schema-validated."
3. §C2: released table + mechanism sentence; deployment 0.807 to the deployment
   table; no Re-ID.
4. §C3: Eq. (3) action branch; released control table; end-to-end tie; one
   parser sentence.
5. §Evaluation: deployment table; stats; z-score/clip-cluster; Table I.
6. `config.py`: gender default false; MMR default off.
7. `alert_validator.py` / `alert_engine.py`: `requires_approval` + test.
8. Figure 1; build; check 6 pages.

Do not rewrite the paper's thesis. The improvements are evidence-level; the
title, structure and three-component identity stay.

---

## 12. Why this is the right revision

- **The title and structure stay.** The paper remains 850, not a new system
  paper.
- **The components get stronger evidence.** C1 deterministic 30/30 + 30/30; C2
  fair two-view table; C3 released parity + end-to-end tie.
- **The deployment story stays.** 0.828/0.807/91% remain in a clearly labelled
  deployment study, with the released track as the reproducibility check.
- **The parser gap is a single finding, not the thesis.** One sentence in
  Evaluation, one in future scope.
- **The paper gains trust and safety.** Fair baselines, number checker, gender
  off, approval gate, MMR off, Figure 1, Table I.
- **Nothing is removed from the system.** The paper is stronger because its
  claims are precise, not because it claims less.
