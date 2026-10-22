# Evaluation-gap proposal: what can be closed on the released artifact, what cannot

**Status:** design proposal. No tracked file in this repository was modified; this file is the only
write. All numbers quoted as *measured* come from two read-only, offline probes over the **shipped**
artifacts (no model inference, no re-encoding, no LLM call, CPU only, ≤ 2 min each); all numbers
marked `<FILL>` must be produced by the proposed analyses before the corresponding sentence is used.

**Target of the proposal:** `INDICON2026_submit/INDICON2026Conference.tex` (the camera-ready, commit
`889a945`, 6 pages, verified `Output written on ... (6 pages)`). The older working copy
`research_paper_6pages.tex` is *not* the submission (different contribution order, April mtime) and is
not addressed here.

---

## 0. Executive summary

| Review sub-point | Verdict | Mechanism |
|---|---|---|
| (a) 150 queries / 30 per category is small | **text-only** (size) + **closable-with-analysis** (what the design can resolve) | Holm-adjusted p, resolution/MDE sentence, seed-stability distribution |
| (a) single-site / multi-premise representativeness | **unfixable** | one premises, one camera; no new premises can be created |
| (a) no camera-disjoint test | **closable-by-proxy** (temporal, not spatial) | segment-disjoint split + gallery-restricted fold (D1) |
| (b) no generalization across lighting / occlusion / tracking-failure | **closable-with-analysis** (proxies, single camera, small n) | regime slices (D3), photometric stress suite (D4) |
| (b) no stability across seeds / splits | **closable-with-analysis** (zero new compute) | 50 stratified re-splits (D2) |
| (c) uncorrected deployment *p* | **text-only** (arithmetic on the paper's own *p*-values) | declared families + Holm-adjusted *p* (N4) |
| (c) marginal find/track deltas | **text-only** | resolution statement + scope ("unresolvable, not improvements") (N4, N2) |
| (d) public repo | **closed** (already in the camera-ready footnote; verify the URL is live) | — |
| (d) deployment export | **unfixable** (not preserved) | text-only disclosure (U2) |
| (d) independent Re-ID identities | **unfixable** (no identities; track ids `-1`) | text-only disclosure (U3) |
| (d) 60-utterance C3 study | **unfixable** for the 0.91 figure | text-only scoping; **new generalization axis** from the already-computed perturbation study (D5) |

The 6-page budget is full: page 6's last column ends at y = 585.3 pt with the text block bottom at
708.9 pt, i.e. **123.6 pt ≈ 10.3 free lines**. The proposed prose adds ≈ 34 lines (2 138 chars), so
**8 cuts** (−1 550 chars ≈ 24 lines) are required to net out at ≈ +9 lines. The full edit set
(8 cuts + 5 additions) was compiled in a temp copy: **6 pages, 39.9 pt (≈ 3.3 lines) of slack**,
+286 characters net.

---

## 1. What the camera-ready already discloses (do not duplicate)

These were checked line by line; the proposal below only *adds* to them.

| Anchor (camera-ready line) | Existing text | Review sub-point already covered |
|---|---|---|
| Abstract, L65 | "the deployment per-query export was not preserved, so its CIs and $p$-values are not re-computable" | (d) deployment export |
| L244 `Setup` | "Its per-query export, annotator roster and tune/test partition were not preserved." | (a), (d) |
| L244 `Setup` | "in one notified indoor/outdoor premises" | (a) single-site, stated |
| L246 | "$p<0.05$ **without multiple-testing correction**" | (c) — this is the sentence N4 replaces |
| L254 | "this is a coverage/regression measurement, **not** the benchmark behind the $0.91$" | (d) C3 study |
| L260 | "per-category gains are nil on count/action/filter and **marginal on find/track, not surviving Holm**" | (c) — prose already says it; the *table* does not show adjusted *p* |
| L295 (i) | "crowded scenes, occlusion, camera-domain shifts and ambiguous action phrasing degrade both branches, **unisolated here**" | (b) — this is the sentence N2 replaces |
| L295 (iv) | "no inter-annotator agreement, cross-camera generalisation or demographic-bias measurement is claimed" | (a), (b) |

Consequence: the reviewer is *right* that generalization is unisolated, but the paper is already
honest about the absence. The fix is therefore **not** more disclaimers — it is (i) the five
analyses below, (ii) one statistical paragraph (N4), and (iii) deleting the now-false phrase
"unisolated here".

---

## 2. Research: how applied-vision/IR work establishes generalization *without* new sites

Seven precedent families, each with the concrete device it uses and how it maps onto this artifact.

| # | Precedent family | Canonical device | Source | Mapping here |
|---|---|---|---|---|
| P1 | **Leave-one-camera-out (LOCO) / camera-disjoint** | hold out one *capture unit*; all its data is absent from training and from the gallery | face anti-spoofing Protocol III (LOCO), [Sensors 2021, PMC8321193](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321193/); cross-camera driver classification on 100-Driver, [arXiv:2411.13181](https://arxiv.org/html/2411.13181v2); traffic-light detection "camera-disjoint protocol", [CMES 2026](https://www.sciopen.com/article/10.32604/cmes.2026.078601); open-set Re-ID with camera-disjoint gallery/query, [Algorithms 2026](https://www.mdpi.com/1999-4893/19/8/675) | **Only one camera exists** → LOCO is impossible. Substitute = hold out a *temporal segment* of that camera (D1). Must be labelled *temporal*, never *camera-disjoint*. |
| P2 | **Scene-disjoint splits + honest scoping** | train/test on disjoint scenes; if the split is *not* scene-disjoint, say what the numbers then mean | MuSTMM scene-disjoint (23 train / 8 test scenes), [Springer](https://www.researchgate.net/publication/415629918); StereoGenBench seen-map/unseen-map, [arXiv:2605.23237](https://arxiv.org/html/2605.23237v1); Sign-Language Recognition: a non-scene-disjoint split must be read as "comparisons under a common signer- and scene-dependent distribution", [MDPI Sensors 16(19):9640](https://www.mdpi.com/2076-3417/16/19/9640) | Same discipline: the released track is *segment-disjoint*, never *site-disjoint*. The SLR sentence is the template for the wording of U1. |
| P3 | **Temporal / blocked splits instead of random splits** | split by timepoint, forward-chaining, blocked *k*-fold | data-leakage study of IR papers (53.7 % random split, 12.3 % split-by-timepoint), [ACM TOIS 2023](https://dl.acm.org/doi/full/10.1145/3569930); DataSAIL, [PMC11978981](https://pmc.ncbi.nlm.nih.gov/articles/PMC11978981/); "forward-chaining, blocked k-fold" for temporal leakage, [arXiv:2512.06932](https://arxiv.org/html/2512.06932v1) | The shipped 50/100 split is *query*-stratified and leaks (measured: 94.2 % of test-relevant clips also occur in the tune split). D1 replaces it with a forward-chaining block protocol. |
| P4 | **Slice-based / worst-slice reporting** | report every slice, not only the aggregate; name the worst slice | Slice-based Learning, [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/351869bde8b9d6ad1e3090bd173f600d-Abstract.html); Language-Driven Slice Discovery, [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.1177.pdf); fairness in VL text-to-image retrieval under language shifts, [IPM 2026](https://www.sciencedirect.com/science/article/pii/S0045790626004556) | D3 reports per-stratum P@5 for every stratum, including the unfavourable ones. |
| P5 | **Corruption / perturbation suites** | fixed, pre-declared corruption grid; report the degradation curve; no new labels needed | ImageNet-C family, [IJCV 2019](https://www.researchgate.net/publication/332109774); MMMU perturbation suite, [arXiv:2608.29278](https://arxiv.org/pdf/2608.29278); robustness study of composed image retrieval, [arXiv:2311.14837](https://arxiv.org/html/2311.14837v1) | D4 = 8 frozen photometric degradations of the released frames; the *same* 150 queries and labels are reused, so the degradation is measured against fixed ground truth. |
| P6 | **Query-side paraphrase robustness** | perturb the *query wording*, keep the labels, measure the system's sensitivity | prompt/paraphrase robustness, [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1595.pdf); composed-retrieval robustness study (above) | D5(a): the repo already computes this for the alert grammar (0.991 over 929 perturbed rules) — it is only not in the paper. |
| P7 | **Seed/split stability + family-wise error control + resolution** | repeat the split; report the distribution of the effect; declare the test family; state the minimum effect the design can detect | "Don't let your analysis go to seed", [PMC12166761](https://pmc.ncbi.nlm.nih.gov/articles/PMC12166761/); effect of split seeds, [CEUR 3476](https://ceur-ws.org/Vol-3476/paper4.pdf); Holm uniformly dominates Bonferroni, [Epidemiology 1996](https://pubmed.ncbi.nlm.nih.gov/8629727/); sensitivity power analysis, [Lakens §8](https://lakens.github.io/statistical_inferences/08-samplesizejustification.html) | D2 (seed stability), N4 (declared families + Holm + resolution statement). |

**Take-away for the response letter:** the reviewer's request is standard practice; the only part that
cannot be met is *spatial* generalization (P1/P2), because the deployment is one premises and the
release is one camera. The temporal/perturbation/slice/stability surrogates (P3–P7) are all runnable
on the artifact that is already public.

---

## 3. Pre-registration block (anti-gaming rules, common to D1–D5)

To be committed as `results/generalization_protocol.md` **before** any of the analyses is run.

1. **Artifact lock.** Inputs are only: `backend/benchmarks/queries_control.jsonl`,
   `backend/benchmarks/splits_control.json`, `results/retrieval_control_per_query.csv`,
   `results/fusion_per_query.csv`, `backend/app/eval/subset.py` metadata, and the 123 released
   `data/clips/camera_0/*.mp4`. No new query, label or clip is created.
2. **Seed lock.** All resampling uses seed `20260301` (the seed already used by `build_subset.py`).
3. **Selection/evaluation disjointness.** Every α, threshold, stratum boundary or model choice used to
   produce a *number* is chosen on a set that is disjoint from the set the number is reported on:
   D1 → strictly earlier blocks; D2 → the tune half; D3 → the stratum boundary is a *fixed rule*
   (median), never a search; D4 → α frozen from the shipped tune split; D5 → no selection at all.
4. **Frozen grids.** D3 proxy set = {illumination, scene density, motion energy} (fixed, 3);
   D4 conditions = the 8 declared degradations + identity (fixed, 9). No condition may be added,
   removed or re-weighted after seeing results.
5. **Full reporting.** Every fold, stratum and condition is reported, including those where the
   adaptive rule loses. Null and negative results are reported with the same prominence.
6. **Family lock.** The set of hypothesis tests is declared per analysis *before* the run and the
   p-values are Holm-corrected inside that family only. A test that is not in the declared family is
   reported as exploratory and labelled as such.
7. **Pre-committed wording.** The sentence templates in §5 are fixed now; `<FILL>` slots are filled
   with whatever the run produces, including values that weaken a claim. If a claim does not survive,
   the claim is removed — the wording template is not edited to fit the outcome.
8. **Separation of tracks.** Every number produced here is a *released-subset* number. It never
   replaces or is blended with the deployment numbers (0.666 → 0.828, 0.785 → 0.807, 0.91). Each
   output file carries the same `scope` block the existing tracks carry.

---

## 4. The five designs

### D1 — Segment-disjoint (leak-free) protocol + leakage audit of the shipped split
*Closes:* (a) "no camera-disjoint test" **as a temporal proxy**; (b) temporal regime shift.
*Does not close:* cross-premise generalization (one premises).

*Design.*
- The released window is **not continuous**: gap analysis over the 121 indexed clips (gap > 120 s)
  yields **three segments** — S1 = 17:01:00–17:08:44 (28 clips), S2 = 18:04:23–18:27:56 (90 clips),
  S3 = 18:55:17–18:56:17 (3 clips) — with **21 / 89 / 40** queries anchored in them.
  (Equal-*duration* blocking is degenerate here: the 17:24–17:47 block contains **zero** clips and
  zero queries. The segment definition above is therefore the frozen rule.)
- A query is assigned to the segment of its **anchor clip** (`anchor_clip` when present, else the
  clip with maximal temporal overlap with the query window, else the nearest clip by start time).
  Queries whose relevance set spans >1 segment are kept in the anchor segment and flagged
  `cross_segment`; a sensitivity run drops them.
- **Forward chaining (primary):** for *k* = 1..|S|−1, select the best fixed α on the union of
  segments 1..*k* (macro P@5; ties → smaller |α−0.5|, then smaller α, matching
  `eval_retrieval_control.tune_per_intent`), then evaluate adaptive and that α on segment *k*+1.
- **Leave-one-segment-out (secondary):** select on the other segments, evaluate on the held-out one.
- **Gallery-restricted variant (the strongest available "unseen-scene" proxy):** in the held-out
  segment, restrict *both* branches' candidate set to the clips of that segment, so no test clip was
  visible during selection. Implementation: re-use `eval_retrieval_control.py`'s `load_scene_index` /
  `build_query_context` / `build_fused` helpers and filter the candidate array by
  `frames[faiss_id]["clip_id"] ∈ segment`; the α for the fold is taken from the forward-chaining
  selection step, never re-tuned.
- **Metrics:** macro P@5 (primary, five category means over the categories present in the fold);
  per-category P@5 (secondary); Δ = adaptive − selected fixed α with a paired bootstrap CI (3 000
  resamples, seed 11, grouped by segment); Holm over the folds.
- **Anti-gaming:** §3 rules 1–7, plus: the segment definition is fixed *before* the run; the
  selection set never contains the evaluation segment; α is never re-tuned per fold.

*Measured (probe, shipped per-query CSV, min-max normalization):*
| Protocol | Fold | n_eval | α* | adaptive | fixed | Δ |
|---|---|---|---|---|---|---|
| forward chaining | S1→S2 | 89 | 0.5 | 0.689 | 0.682 | **+0.006** |
| forward chaining | S1+S2→S3 | 40 | 0.7 | 0.301 | 0.301 | **+0.000** |
| leave-one-segment-out | hold out S1 | 21 | 0.7 | 0.696 | 0.696 | +0.000 |
| leave-one-segment-out | hold out S2 | 89 | 0.5 | 0.689 | 0.682 | +0.006 |
| leave-one-segment-out | hold out S3 | 40 | 0.7 | 0.301 | 0.301 | +0.000 |

*Leakage audit (also measured):* the shipped 50/100 split shares **98 of 104** test-relevant clips
with the tune split (**94.2 %**); **98 of 100** test queries have a relevance set that touches a tune
clip, and 50/50 tune queries touch a test clip. The clip-count variant of the blocked protocol
(5 contiguous blocks of 24–25 clips) gives Δ = +0.000 / +0.021 / +0.000 / +0.000 and the equal-duration
variant is degenerate — both are recorded as pre-registered sensitivities.

*Expected honest outcome:* on this subset the adaptive rule **ties** the tune-selected best fixed α
(Δ ≤ +0.006 everywhere). D1 therefore supports a *protocol* claim ("the C1 conclusion is stable under
a leak-free split"), **not** the deployment gain. Say so.

*Cost:* seconds for the query-split variant (pure re-analysis of `results/retrieval_control_per_query.csv`,
whose per-query P@5 does not depend on the split); ~1 min for the gallery-restricted variant
(150 OpenCLIP text encodes on CPU + 3 FAISS searches, no LLM — use `--parse-source oracle` or the
cached parses).

---

### D2 — Split-seed stability of the routing rule
*Closes:* (b) "stability across seeds/splits"; (c) "is the headline an artifact of one split?".

*Design.* 50 stratified re-splits (10 tune / 20 test per category, seeds 20260301–20260350) of the
shipped control; for each split select the best fixed α on the tune half (same rule as D1) and report
Δ = adaptive − selected fixed α on the test half. Report the **distribution** (min, quartiles,
median, max), the fraction of seeds with Δ ≥ 0, and the selection frequency of each α. Secondary:
the same distribution against the paper's deployment best fixed α (0.3).

*Anti-gaming:* the split family is fixed (stratified, 10/20, seed list); α selection uses only the
tune half; every seed is reported.

*Measured (probe):* Δ vs. tune-selected best fixed α = **+0.000 in 50/50 seeds** (min = max = 0.000);
the selected α is **fixed 0.7 in 50/50 seeds**; Δ vs. fixed α = 0.3 = **+0.132** (min +0.098, median
+0.132, max +0.168), adaptive ≥ fixed 0.3 in **50/50** seeds.

*Expected honest outcome:* the released track cannot discriminate adaptive from the best fixed α —
this is the "reported subset null" the paper already states, now quantified as a *distribution*
instead of a single split. It also means the deployment margin (+0.162) must **not** be attributed to
the released track.

*Cost:* seconds, from the shipped CSV.

---

### D3 — Regime slices: illumination, scene density, motion energy
*Closes:* (b) "no generalization testing across lighting / occlusion / tracking-failure regimes" —
as **proxies**, on one camera, with small *n*.

*Design (frozen).*
- Proxies (all label-independent, all computed from the release):
  1. **Illumination** — mean luma of every frame of the anchor clip, decoded from the released mp4
     (`cv2`). Distribution is **bimodal** (31+37 clips below luma 117.3; valley ≈ 120.7; max 183.6),
     i.e. the window spans a real dusk transition, and bright/dim clips occur in **all three
     segments** → illumination is *not* a proxy for time of day.
  2. **Scene density / occlusion proxy** — maximum number of persons simultaneously visible in the
     anchor clip (0.5 s gate) from the released detector sidecar; median = 1, range 1–3.
  3. **Motion energy** — the *already shipped* `subset.compute_motion_energy` value of the anchor
     clip (mean |Δ| of consecutive 96×96 snapshots); median 0.00253, range 0.00147–0.06276.
- **Split rule (primary): within-category median.** For each category separately, split its queries at
  the median of the proxy over that category's anchor clips → every category contributes ≈ half its
  queries to each stratum. This removes the category-mix confound.
- **Split rule (secondary, must be reported as confounded): marginal median over all 150 queries.**
  The measured confounded result is macro P@5 0.549 (dim) vs. 0.675 (bright) — but **all 30 action
  queries live in the dim stratum** (action P@5 = 0.107), so the gap is a category-mix artifact, not
  an illumination effect. Reporting only the marginal split would be misleading; it is therefore
  reported *with* the per-category composition.
- **Metrics:** per-category P@5 in each stratum (primary within-category); the adaptive − fixed-α
  contrast per stratum; a paired bootstrap CI per stratum (3 000 resamples, clustered by anchor
  clip). Holm over the (category × stratum) cells within each proxy family.
- **Anti-gaming:** the three proxies are frozen; the median rule is fixed; no proxy may be added
  after seeing results; a stratum with < 10 queries per category is reported as descriptive only;
  the confounded marginal split is always shown next to the balanced one.

*Measured (probe, within-category median split, adaptive P@5 low/high):*
| proxy | count | find | track | action | filter |
|---|---|---|---|---|---|
| illumination | 0.440 / 0.427 | 0.933 / 0.960 | 0.893 / 0.787 | 0.107 / – (all dim) | 0.376 / 0.508 |
| density | 0.435 / 0.429 | 0.947 / 0.940 | 0.787 / 0.893 | 0.107 / – | 0.422 / 0.486 |
| motion energy | 0.427 / 0.440 | 0.965 / 0.923 | 0.853 / 0.827 | 0.107 / – | 0.438 / 0.429 |

*Expected honest outcome:* the adaptive rule is **flat across regimes** (largest within-category
spread 0.11 P@5: track in bright vs. dim scenes); the **action** category cannot be split at all on
this artifact (all 30 action anchors sit in the dim, single-person, low-motion stratum) — that is a
property of the release and must be stated, not hidden.

*Cost:* ~60 s (mp4 decode) + seconds.

---

### D4 — Photometric stress suite on the released gallery
*Closes:* (b) "no generalization testing" on an **appearance-shift** axis; also produces the
degradation curve that supports the paper's own thesis (the structured branch carries the load when
the semantic branch degrades).

*Design (frozen).* Apply a fixed grid of 8 degradations + identity to every frame of every released
clip, re-encode with the **same** OpenCLIP ViT-B/32 already cached, rebuild the 242-frame scene index
per condition, and re-run the same 150 control queries with the **same** α (the tune-selected α from
the shipped split; no per-condition tuning). Conditions: brightness ×{0.6, 0.8, 1.3}; Gaussian blur
σ ∈ {1.0, 2.0}; additive Gaussian noise σ ∈ {0.03, 0.08} on the [0,1] scale; downscale ×0.5 then
upscale.

*Metrics:* macro P@5 per condition (adaptive and best fixed α); **retention** = P@5(perturbed) /
P@5(clean); **primary (pre-declared): the worst-condition retention**; full curve for both methods.

*Anti-gaming:* the grid and its severities are fixed in the pre-registration file; α is frozen from
the clean tune split; the same 150 queries and the same relevance labels are used, so degradation is
measured against fixed ground truth; no condition is dropped; the primary statistic is declared in
advance; the report states that the perturbations are synthetic and that the structured branch is
unaffected *by construction*, so the measured drop isolates the semantic branch.

*Expected honest outcome:* the semantic-only branch should degrade far more than the hybrid — that
is the paper's central mechanism, and this suite measures it on the public artifact. If the hybrid
degrades as much as the semantic branch, the claim in §V must be softened. (Numbers `<FILL>`.)

*Cost:* 242 frames × 9 conditions ≈ 2 200 OpenCLIP image encodes on CPU (≈ 2–4 min) + 9 FAISS
searches + re-scoring (seconds). **No LLM calls** — run with the reference filters
(`--parse-source oracle`) or the cached parses.

---

### D5 — Perturbation robustness for C1, and an occlusion-stratified C2 margin
*Closes:* (b) for the other two contributions, using artifacts that already exist.

*Design D5(a) — C1 query perturbation (already computed in the repo, zero new work).*
`backend/scripts/eval_alerts_probe2.py` already evaluates the alert grammar under four deterministic
query perturbations on the 60 released clean utterances: **clean 60/60 = 1.000; perturbed 921/929 =
0.991** (case/punctuation 300/300, filler 240/240, synonym 352/360, word-order 29/29). This is a
ready-made paraphrase-robustness measurement (precedent P6). It only has to be *surfaced* in the
paper, scoped as: sensitivity of the deterministic materializer to **wording**, on a closed
vocabulary — not open-world accuracy, and not the 0.91 figure.

*Design D5(b) — C2 margin under the occlusion proxy.* From `results/fusion_per_query.csv`
(method, detection id, P@5) joined with the occlusion proxy already defined in
`backend/scripts/eval_fusion.py` (`qv = 1/sqrt(max(1, persons in the nearest scene frame)`), split the
test retrieval queries at the median `qv` and report P@5 per stratum for `concat`, `rrf`, `visual`,
plus the paired Δ with a track-clustered bootstrap. This asks whether the paper's *narrowest* claim
(concat over RRF, +2.2 pp on the deployment) survives in the crowded regime. Anti-gaming: the proxy
and the median rule are frozen; the same gallery and the same thresholds as the shipped table are
used; no method is re-tuned per stratum.

*Cost:* D5(a) seconds (`eval_alerts_probe2.py` is shipped); D5(b) ~1 min.

---

## 5. Exact paper prose (LaTeX, with placement anchors)

Everything below was compiled in a temp copy of the camera-ready (§5.4): **6 pages, 39.9 pt slack**.
`<FILL>` slots are to be replaced by the corresponding measured value; their *length* in the mock-up
was chosen to match a realistic number, so the page budget survives a small length change.

### 5.1 N1 — new paragraph, §VI Evaluation
**Anchor:** insert immediately **before** `\paragraph*{Supporting numbers}` (camera-ready line 288),
i.e. after the `\paragraph*{Competitors and robustness (C3)}` paragraph (line 286).
**Measured values already known:** the Δ figures, the leakage figures and the segment structure. The
`<FILL>` slots are only the stress-suite numbers (D4), which appear in N2.

```latex
\paragraph*{Generalization protocol (no new site).} The deployment benchmark is one premises and its per-query export was not preserved, so it cannot be re-split; the released track can, and we stress the \emph{protocol} there. Four pre-registered analyses use the release only: (i)~a \emph{segment-disjoint} split---the released clips form three temporally separated bursts, so $\alpha$ is selected on the other bursts and scored on the held-out one; (ii)~50 stratified re-splits, reporting the \emph{distribution} of $\Delta$; (iii)~regime slices by label-independent proxies (luminance, persons per frame, motion energy) split \emph{within} category; (iv)~a frozen photometric stress suite of eight frame degradations with $\alpha$ frozen. Here the adaptive rule and the tune-selected best fixed $\alpha$ \emph{tie} ($\Delta=+0.000$ in 50/50 re-splits; $+0.006$ in the best segment-disjoint fold), so the release does not reproduce Table~\ref{tab:alpha_baseline}; the shipped split leaks (94\% of test-relevant clips recur in the tune split).
```

### 5.2 N2 — §VII Limitations (i)
**Anchor:** `INDICON2026Conference.tex` line 295, `\textbf{Limitations.}`, first item, replacing the
sentence that ends "...degrade both branches, unisolated here."

```latex
(i)~Lemma~\ref{lem:override_gap} assumes (A1)--(A2) and is not guaranteed in general. On the released track we isolate two regimes it ignores---illumination and scene density, split within category---and the adaptive rule is flat across strata (<FILL> P@5 at most), while a photometric stress suite costs the semantic branch <FILL> P@5 and the hybrid <FILL> P@5; camera-domain shift stays unisolated (one camera, one premises) and ambiguous action phrasing stays a labelling limit.
```

### 5.3 N4 — §VI `Statistical protocol` (replaces the "without multiple-testing correction" sentence)
**Anchor:** camera-ready line 246. **Arithmetic only** — the Holm values are computed from the
*p*-values the paper already prints:

- C3 category family (m = 5): find *p* = 0.04 is the smallest → Holm-adjusted **0.20**; track
  *p* = 0.07 → **≥ 0.20** (its adjusted value is ≥ 0.20 whatever the three unprinted "n.s." values
  are, since the second-smallest is ≥ 0.05). Neither reaches 0.05.
- C2 family (m = 4): sorted {< 0.0002, 0.001, 0.003, 0.018} → adjusted {< 0.0008, 0.003, 0.006,
  **0.018**}. The narrowest comparison (concat vs. RRF) is the largest *p* in its family and is
  therefore unchanged by Holm.
- The macro-mean C3 test is a *separate, pre-specified primary endpoint* (*p* < 0.0002, unaffected).
- Resolution: with n = 30 per category the deployment category means carry ±0.05–0.08 bootstrap
  half-widths, while the observed deltas are +0.02 and +0.01 — i.e. below the design's resolution.

```latex
\textbf{Statistical protocol.} Mean P@5 with 95\% bootstrap CI (10{,}000 query-level resamples); paired comparisons use a two-sided paired bootstrap on per-query P@5 deltas with a Wilcoxon cross-check. We declare two pre-specified families: the five category-level C3 tests (Holm-adjusted: find $+0.02$, $p_{\mathrm{Holm}}{=}0.20$; track $+0.01$, $p_{\mathrm{Holm}}{\ge}0.20$; count/action/filter nil) and the four C2 fusion comparisons (Holm-adjusted $p\le0.018$; the narrowest, concat vs.\ RRF at $p{=}0.018$, is the largest $p$ in its family and is unchanged). The macro-mean C3 gain is a separate pre-specified primary endpoint ($p<0.0002$). At $n{=}30$ per category the deployment category means carry $\pm0.05$--$0.08$ bootstrap half-widths, so the observed $+0.02$/$+0.01$ category deltas are below the design's resolution and are reported as unresolvable, not as improvements. Deployment CIs/$p$-values are recorded single-run measurements, not re-computable significance; significance is claimed only on the released track (grouped bootstrap, Holm-corrected per-category tests, 3{,}000 resamples).
```

### 5.4 N5 / N6 — one-line additions
* **N5** — abstract (line 65): after "…plus a fixed adversarial round." insert
  ` and 0.991 rule-correctness under four deterministic query perturbations` (D5(a); the number is
  already in `results/c3probe2_report.md`).
* **N6** — Contributions bullet 4 (line 84): replace
  "plus a released single-camera reproducibility track exercising the C1--C3 code paths (\S\ref{sec:eval})."
  with
  "plus a released single-camera track exercising the C1--C3 code paths and a pre-registered generalization protocol (segment-disjoint splits, seed stability, regime slices, photometric stress) on it (\S\ref{sec:eval})."

### 5.5 The honestly-unfixable items (exact sentences)

**U1 — multi-premise / cross-site (unfixable).** Anchor: §VI `Setup`, line 244, after
"…in one notified indoor/outdoor premises;":
```latex
We report no cross-premise result: the deployment covers one notified premises and the release one camera, so site-level generalization is untested here and any statement about it would be extrapolation, not measurement.
```
(≈ 2 lines; if the budget is tighter, fold it into N4's last sentence rather than adding a
sentence.)

**U2 — deployment export (unfixable).** Anchor: §VI `Setup`, line 244, replacing
"Its per-query export, annotator roster and tune/test partition were not preserved.":
```latex
The deployment corpus---the 312{,}544-descriptor index, the MongoDB collections and the per-query export of the 150-query benchmark---was not preserved, so the deployment numbers cannot be re-derived, re-split or re-tested by a third party; what the artifact ships is the single-camera subset, the benchmark definitions and the code path behind every number reported here.
```

**U3 — independent Re-ID identities (unfixable).** Anchor: §VI `Setup`, line 244, at the end of the
"Released track." sentence, or §VII (iii):
```latex
The person-pair labels are pseudo-labels formed from detector and attribute metadata, not identities (all released track ids are $-1$), so no cross-camera identity and hence no independent Re-ID evaluation can be computed from the release; a third party can reproduce the ablation, not the identity claim.
```

**U4 — the 60-utterance C3 study (unfixable for 0.91).** Anchor: §VII (iii) — the paper already
states it; tighten to:
```latex
The $0.91$ comes from a 60-utterance human-authored study whose per-utterance export was not preserved; the released 120 utterances are author-constructed over a closed vocabulary and the released materializer is deterministic, so neither the study nor the $0.91$ is reproducible from the artifact, and the two measurements are not interchangeable.
```

### 5.6 Paired cuts (verified budget)

The camera-ready has **zero** slack in the body text (every page's body ends at y = 708.9 pt) but
**123.6 pt (≈ 10.3 lines) of free space at the bottom of the last column** (the bibliography ends at
y = 585.3 pt on page 6). The additions above need ≈ 21 lines, so these cuts are required. Each cut is
verified to apply (the mock-up asserts the fragment) and each is compensated by a specific addition.

| # | Anchor (camera-ready line) | Cut | chars saved | Compensated by |
|---|---|---|---|---|
| CUT-1 | L122, whole `\paragraph*{Release probes} Compositional held-out …` paragraph | **delete** (content duplicated in the abstract and in §VI `Released alert benchmark`) | 572 | N1, N6 |
| CUT-2 | L124, `\paragraph*{Statistical anomaly complement} To surface unknown behaviors …` | compress to one sentence (keep the three F₁ numbers) | 163 | N1 |
| CUT-3 | L254, `\paragraph*{Released alert benchmark} …` | drop the always-ask-threshold turn/question detail and compress the adversarial-round sentence; **add** the 0.991 perturbation clause (D5(a)) | 215 | N1, N5 |
| CUT-4 | L257 (body of `\paragraph*{C2 -- fusion ablation}`) | drop "and reproduces the old inversion by construction" and compress the released-subset competitor clause | 56 | N2 |
| CUT-5 | L291, "The 4-camera deployment sustains …" | drop the released-subset latency worker/percentile detail (it is in `results/latency_results.md`) | 229 | N4 |
| CUT-6 | L95, `\textbf{Hybrid structured/semantic retrieval.}` | compress the IR query-dependent-weighting sentence | 71 | N4 |
| CUT-7 | L286, `\paragraph*{Competitors and robustness (C3)}` | compress the learned-competitor and counting sentences | 77 | N2 |
| CUT-8 | L252 (body of `\paragraph*{C1 -- alert compiler}`) | compress the "Hardest fields" enumeration (keep every number) | 167 | N4 |
| | **total** | | **1 550** | additions 2 138 → **net +286** (script-measured; the per-edit sum is +358 because of newline accounting) |

**Verification.** A temp copy of the camera-ready with all 8 cuts and all 5 additions
(N1, N2, N4, N5, N6) was compiled with `pdflatex` + `bibtex` + `pdflatex` ×2:
`Output written on paper.pdf (6 pages, 345135 bytes)`, no LaTeX errors, last-column free space
39.9 pt (≈ 3.3 lines). The mock-up scripts live in the temp dir only
(`…\opencode\evalgaps_sub\mockup_v2.py`).

---

## 6. Reproducibility sub-items (d) — item-by-item

| Sub-item | Status | Evidence in the repo | Action |
|---|---|---|---|
| Public repo | **closed** | camera-ready L84 footnote → `github.com/sujalmh/intent-adaptive-hybrid-surveillance`; `results/release_manifest.md` lists the archives and checksums; git log "Public repo single-commit push" | verify the URL resolves and the README states the subset-only scope; no paper change |
| Deployment export | **unfixable** | `results/README.md`: "The original per-query CSV, annotator agreement sheet, and 312k index are not present in this repository" | U2 (text only). Optionally ship a *schema-level* export script (no data) — a code change, out of scope here |
| Independent Re-ID identities | **unfixable** | `results/subset_stats.json` `cannot_reproduce: ["cross-camera identity tracking or generalisation"]`; `results/fusion_tables.md`: "no verifiable public Re-ID source on local disk … Fell back to a held-out temporal split"; all released track ids are `-1` | U3 (text only). A third-party protocol (identity annotation + IAA form) can be *specified* but not executed |
| 60-utterance C3 study | **unfixable for 0.91** | `results/alert_results.md`: "The 0.91 figure reported in the manuscript comes from the deployment study … and it is *not* reproducible from this benchmark" | U4 (text only) + **D5(a)**, which converts an existing probe into a *generalization* measurement for C1 |

---

## 7. Execution plan (total ≈ 15 min CPU, no LLM calls)

| Order | Step | Script (to be added, names suggested) | Produces | Needed for |
|---|---|---|---|---|
| 0 | Commit the pre-registration | `results/generalization_protocol.md` (text) | protocol | §3 |
| 1 | D2 seed stability | `backend/scripts/eval_generalization.py --mode seeds` | `results/generalization_seeds.json/.md` | N1 |
| 2 | D1 segment protocol + leakage audit | same script `--mode segments` (query-split; gallery-restricted variant re-uses `eval_retrieval_control` helpers) | `results/generalization_segments.json/.md` | N1 |
| 3 | D3 regime slices | same script `--mode regimes` (decodes the 123 mp4s, joins `subset` metadata) | `results/generalization_regimes.json/.md` | N2, N1 |
| 4 | D4 photometric stress suite | same script `--mode stress` (OpenCLIP re-encode of 242 frames × 9 conditions) | `results/generalization_stress.json/.md` | N2 |
| 5 | D5(a) C1 perturbation | `backend/scripts/eval_alerts_probe2.py` (already shipped) | `results/c3probe2_report.md` (already present) | N1/N5/CUT-3 |
| 6 | D5(b) C2 occlusion slice | `backend/scripts/eval_fusion.py --slice occlusion` or a thin variant | `results/generalization_fusion_slices.md` | optional (not in the 6-page text) |
| 7 | Apply the paper edits | `INDICON2026_submit/INDICON2026Conference.tex` | camera-ready | §5 |

**Risks / stop conditions.**
- If D4 shows the hybrid degrades as much as the semantic branch, N2's sentence must be rewritten to
  report that (the template is pre-committed; only the numbers move).
- If D3 shows a regime with a large, significant drop, the Limitations sentence changes from
  "flat across strata" to the measured value; the *claim structure* does not change.
- If D1/D2 reproduce Δ > 0.02 consistently, the sentence "the adaptive rule and the tune-selected
  best fixed α tie" must become the measured value (still released-subset scoped).
- Every output file must carry a `scope` block stating "released single-camera subset; not the
  4-camera / 312k deployment", as the existing tracks do.

---

## 8. Appendix — measured values, provenance, and how to reproduce

**Probe runs (read-only, offline, CPU):**
- P1: `…\opencode\evalgaps_sub\probe_evalgaps.py` — re-analysis of
  `results/retrieval_control_per_query.csv`; sanity check reproduced the shipped macro P@5 of
  `adaptive` (0.538), `paper_lookup` (0.428), `per_intent_tuned` (0.536) and the five fixed α exactly.
- P2/P3/P4: `…\opencode\evalgaps_sub\probe_evalgaps2.py`, `probe_evalgaps3.py`,
  `probe_evalgaps4.py` — metadata joins + mp4 luma decoding + the blocked/segment protocols.
- Mock-up: `…\opencode\evalgaps_sub\mockup_v2.py` — compiles the proposed edits in a temp copy.

**Key measured numbers (all released-subset; not deployment numbers):**
| Quantity | Value |
|---|---|
| Shipped split: test-relevant clips that also occur in the tune split | 98 / 104 = **94.2 %** |
| Shipped split: test queries whose relevance set touches a tune clip | **98 / 100** |
| Seed stability: Δ(adaptive − tune-selected best fixed α), 50 seeds | **+0.000** (min = max), adaptive ≥ fixed in 50/50 |
| Seed stability: selected α across seeds | **fixed 0.7 in 50/50** |
| Seed stability: Δ(adaptive − fixed α = 0.3), 50 seeds | **+0.132** [+0.098, +0.168], ≥ 0 in 50/50 |
| Temporal segments (gap > 120 s) | S1 17:01–17:08 (28 clips / 21 q), S2 18:04–18:28 (90 / 89), S3 18:55–18:56 (3 / 40) |
| Forward chaining Δ | S1→S2 **+0.006**, S1+S2→S3 **+0.000** |
| Leave-one-segment-out Δ | +0.000 / +0.006 / +0.000 |
| Clip-count blocked variant Δ (sensitivity) | +0.000 / +0.021 / +0.000 / +0.000 |
| Equal-duration blocks | degenerate: 17:24–17:47 contains 0 clips, 0 queries |
| Illumination | bimodal; luma 113.8–183.6, valley ≈ 120.7; bright clips in all 3 segments |
| Regime slice (within-category median) | adaptive flat; largest spread 0.11 P@5 (track: dim 0.893 → bright 0.787) |
| Regime slice (marginal, confounded) | dim 0.549 vs bright 0.675 — entirely explained by all 30 action queries being dim (action P@5 0.107) |
| C1 perturbation (already in repo) | clean 60/60 = 1.000; perturbed **921/929 = 0.991** (synonym 352/360 is the only loss) |
| Page budget | last column free 123.6 pt (10.3 lines); mock-up with 8 cuts + 5 additions → 6 pages, 39.9 pt free |

**Reproduction of the analysis artefacts (once implemented):**
```powershell
$env:PYTHONIOENCODING='utf-8'
python backend/scripts/eval_generalization.py --mode seeds
python backend/scripts/eval_generalization.py --mode segments
python backend/scripts/eval_generalization.py --mode regimes
python backend/scripts/eval_generalization.py --mode stress
python backend/scripts/eval_alerts_probe2.py          # D5(a), already shipped
```

**Do not do (integrity rules carried over from `results/README.md`):** do not print a p-value as
`0.0000` (resolution floor 2/n_boot) or a dispersion as `± 0.000`; keep every released-subset number
in a file whose `scope` block says it is not the deployment measurement; never merge a released-subset
number into the deployment tables.
