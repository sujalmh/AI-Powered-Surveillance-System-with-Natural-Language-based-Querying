# Text-fix proposal — INDICON 2026 submission (read-only review, no edits made)

Scope: `INDICON2026_submit/INDICON2026Conference.tex` (302 lines, read fully), `References.bib`,
`backend/app/services/fusion.py`, `attribute_encoder.py`, `hybrid_search.py`, `alert_validator.py`,
`alert_materializer.py`, `backend/app/eval/metrics.py`, `backend/app/config.py`, `backend/app/main.py`,
`backend/app/core/privacy.py`, `backend/app/core/security.py`, `backend/scripts/eval_retrieval_control.py`,
`backend/benchmarks/queries_control.jsonl`, `results/{fusion_results,alert_results,retrieval_control_results,alpha_competitors}.json`.
Web search used only for items 6–7. No tracked file edited. Page budget: every item carries a paired cut; totals at end.

## Arithmetic verification (item 1)

`fusion.py:64-66` defaults `(visual_weight, text_weight) = (0.6, 0.4)`; `fuse` (`fusion.py:95-101`)
scales each branch, concatenates, then L2-normalizes. With unit-norm branches
$\mathbf{v}\in\mathbb{R}^{768}$, $\mathbf{t}\in\mathbb{R}^{384}$:

- $\Vert[0.6\mathbf{v}\,\Vert\,0.4\mathbf{t}]\Vert_2^2 = 0.36 + 0.16 = 0.52$; norm $=\sqrt{0.52}\approx0.7211$.
- Cosine of two fused vectors $= (0.36\cos_v + 0.16\cos_t)/0.52$.
- Effective block weights: $0.36/0.52 = 9/13 \approx 0.692$ visual, $0.16/0.52 = 4/13 \approx 0.308$ text.

Verified by execution (`py -3`: `norm2= 0.52`, `vis= 0.6923076923076923`, `txt= 0.3076923076923077`).
Shipped operating point confirmed in `results/fusion_results.json:62`
(`"concat": "MultimodalFusion(0.6,0.4), 1152-d, reproduces the shipped index"`).

---

## 1. WEIGHT MATH (anchor tex:139-142, Eq.~\eqref{eq:fusion} paragraph)

OLD (tex:142):
```
with $(\alpha_v,\beta_t)=(0.6, 0.4)$ from a coarse grid search on the unit simplex (step $0.1$, nine pairs) maximizing F$_1$@10 on the held-out calibration set, where F$_1$@10 $\triangleq$ H-mean(P@5,R@10); the standard P@10-based F$_1$ is lower.
```

NEW:
```
with nominal $(\alpha_v,\beta_t)=(0.6, 0.4)$ from a coarse grid search on the unit simplex (step $0.1$, nine pairs) maximizing F$_{5,10}$ on the held-out calibration set. Both branches enter unit-norm, so $\Vert[0.6\mathbf{v}\,\Vert\,0.4\mathbf{t}]\Vert_2^2{=}0.52$ and the fused cosine is $(0.36\cos_v{+}0.16\cos_t)/0.52$: the effective block weights are $9/13{\approx}0.69$ visual and $4/13{\approx}0.31$ text, not $0.6/0.4$.
```
(F$_{5,10}$ rename + standard-F$_1$ numbers are item 2; keep its clause out of here.)

PAIRED CUT (tex:165, same paragraph family): delete
`, The coarse $(\alpha_v,\beta_t)$ sweep is unimodal around $(0.6,0.4)$ (the $(0.5,0.5){=}0.785$ tie with RRF needs multi-seed re-verification).`
 folding its content into the sentence above (grid selected nominal $(0.6,0.4)$; weight-specific RRF tie already stated in the preceding sentence of tex:257 as `the RRF tie is weight-specific, not family-wide` — keep that one, cut this one).
NET: +2 lines / −2 lines = 0.

VERIFICATION: math above (executed). Nominal point: `fusion.py:64-66`, `fusion_results.json:62`.
NOT VERIFIABLE: the full nine-pair deployment grid — only three points appear anywhere
(tex:257: $0.785/0.799/0.774$ at $(0.5,0.5)/(0.7,0.3)/(0.4,0.6)$); no deployment sweep exists in
`results/` (deployment export not preserved, tex:244). The sentence must not quote grid points beyond these three.

## 2. F1 RENAME (anchors tex:142, tex:155 Table I header, tex:249)

OLD (tex:249):
```
Precision@5 $=|\mathrm{top\!-\!5}\cap\mathrm{relevant}|/5$; Recall@10 $=|\mathrm{top\!-\!10}\cap\mathrm{relevant}|/|\mathrm{relevant}|$; F$_1$@10 is the harmonic mean of P@5 and R@10 (not the standard P@10-based F$_1$).
```
NEW:
```
Precision@5 $=|\mathrm{top\!-\!5}\cap\mathrm{relevant}|/5$; Recall@10 $=|\mathrm{top\!-\!10}\cap\mathrm{relevant}|/|\mathrm{relevant}|$; F$_{5,10}$ is the harmonic mean of P@5 and R@10 (subscripts name the cutoffs; not the standard same-cutoff F$_1$). On the released track the concat descriptor scores F$_{5,10}{=}0.309$ alongside standard F$_1{=}0.289$.
```
OLD (tex:155 header): `Person index & P@5 & R@10 & F$_1$@10 \\`
NEW: `Person index & P@5 & R@10 & F$_{5,10}$ \\`
OLD (tex:142 tail): `where F$_1$@10 $\triangleq$ H-mean(P@5,R@10); the standard P@10-based F$_1$ is lower.`
NEW: `where F$_{5,10}$ $\triangleq$ H-mean(P@5,R@10).`
Replace every other `F$_1$@10` token with `F$_{5,10}$` (abstract tex:65, contributions tex:82, C2 tex:157-165, conclusion tex:297) — pure renames, zero line impact.

PAIRED CUT: tex:142 tail deletion above absorbs the rename; plus delete tex:257 clause
`Five-seed random-projection means (P@5 $0.417\pm{<}0.001$) and the learned-projection competitor (P@5 $0.432$) come from the released subset (\texttt{eval\_fusion.py}).`
(these two numbers live in `fusion_results.json:168-179` and the fusion tables artifact; the sentence adds no deployment evidence).
NET: +2 / −2 = 0.

VERIFICATION (`results/fusion_results.json`, `test.retrieval.concat`): `f1` = 0.28866860857606186
(line 100) → 0.289; `f1_paper` = 0.3087860381311794 (line 101) → 0.309. Definitions:
`backend/app/eval/metrics.py:36-42` (`f1_at_10_paper` = H-mean(P@5,R@10)),
`metrics.py:45-55` (`f1_at_10_standard` = H-mean(P@10,R@10)); both reported by
`eval_fusion.py:514-515`. The `F1@10 (P@10,R@10) | F1@10 paper (P@5,R@10)` header convention already
exists in `eval_retrieval_control.py:970`. Scope note: 0.289/0.309 are released-track values, not
deployment values — the sentence says so.

## 3. FUZZ WORDING (anchor tex:122; abstract tex:65)

OLD (tex:122):
```
seeded fuzz (2{,}000 noise utterances plus 200 unsupported-hint cases, stdlib \texttt{random}) gives zero crashes, zero soundness violations and zero false materializations (\texttt{eval\_\allowbreak alerts.py}).
```
NEW:
```
seeded fuzz (stdlib \texttt{random}) over 2{,}000 noise utterances---642 materialized and all passed the validator, so zero crashes and zero soundness violations---plus 200 targeted unsupported-hint cases with zero false materializations (\texttt{eval\_\allowbreak alerts.py}).
```
Abstract tex:65: replace `and seeded fuzz with zero crashes, zero soundness violations and zero false materializations, plus a fixed adversarial round`
with `and seeded fuzz (642/2{,}000 noise inputs materialized, all valid; 0/200 targeted unsupported-hint cases materialized), plus a fixed adversarial round`.
PAIRED CUT: tex:124 anomaly-complement sentence trim — delete ` (crowd-spike)`, ` (off-hours)`, ` (unusual-object)`
(parenthetical labels; the three F$_1$ values stay ordered as introduced). Saves ~1 line; fuzz split costs ~1 line.
NET: +1 / −1 = 0 (body); abstract rephrase is character-neutral.

VERIFICATION (`results/alert_results.json:1647-1658`): `n_noise` 2000, `materialized_noise` 642,
`clarified_noise` 1358, `crashes` 0, `n_soundness_violations` 0 (`soundness_violations: []`),
`n_targeted` 200, `n_targeted_false` 0 (`targeted_false_materializations: []`), `seed` 62498.
"All passed the validator" = 642 materialized with zero entries in `soundness_violations`. The 200
targeted cases are a separate set from the 2000 noise inputs — the current tex:122 merges them.

## 4. GROUPED PRIMARY (anchors tex:246, tex:257, tex:284)

Grouping units (verified):
- Retrieval control: `group` = 5-minute wall-clock bucket per query, e.g. `"group": "2026-03-01T17:00"`
  (`backend/benchmarks/queries_control.jsonl:1`; buckets 17:00/17:05/18:00/18:05/… across lines 1–100).
  Consumed as cluster labels in `eval_retrieval_control.py:647` (`r["group"] = q.get("group")`),
  `eval_retrieval_control.py:712-716` (paired grouped bootstrap, `paired_grouped_bootstrap_ci`),
  `eval_retrieval_control.py:760-782` (ablation family, query-level + grouped reported side by side).
- Fusion: groups = pseudo-tracks (`eval_fusion.py:1259-1269`, `group_of = track`;
  `fusion_results.json:590` `grouped_bootstrap_by_track`).
- Statistics text already promises this order (`eval_retrieval_control.py:15-16`,
  `eval_retrieval_conf3.py:73`, `eval_calib_base.py:1283`); the paper inverts it.

OLD (tex:246):
```
\textbf{Statistical protocol.} Mean P@5 with 95\% bootstrap CI (10{,}000 query-level resamples); paired comparisons use a two-sided paired bootstrap on per-query P@5 deltas with a Wilcoxon cross-check, $p<0.05$ without multiple-testing correction. Deployment CIs/$p$-values are recorded single-run measurements, not re-computable significance; significance is claimed only on the released track (grouped bootstrap, Holm-corrected per-category tests, 3{,}000 resamples).
```
NEW:
```
\textbf{Statistical protocol.} Primary uncertainty is grouped (cluster) bootstrap---5-minute wall-clock query buckets on the released retrieval track (3{,}000 resamples), pseudo-tracks on the person-descriptor track---with paired-difference CIs and Holm-corrected per-category tests; query-level bootstrap is secondary. Deployment CIs/$p$-values are recorded single-run measurements, not re-computable significance; significance is claimed only on the released track.
```

OLD (tex:257 lead): `The heterogeneous-space concat descriptor reaches $F_1@10=0.807\,(\pm0.024)$: $+10.7$~pp over the best unimodal index ($0.700\pm0.031$; paired bootstrap 95\% CI $[+7.9,+13.4]$, $p<0.0002$) and ...`
NEW lead: `The heterogeneous-space concat descriptor reaches $F_{5,10}=0.807$: $+10.7$~pp over the best unimodal index ($0.700$; paired $\Delta$ 95\% CI $[+7.9,+13.4]$, $p<0.0002$) and ...; on the released track the grouped (by-track) bootstrap gives concat P@5 $0.427$ $[0.379,0.476]$.`
(grouped numbers: `fusion_results.json:591-595`: concat p5 0.42739495798319327, lo 0.37850697292863006, hi 0.47584.)

Released-subset check (tex:284): append one clause after `($\Delta=0.000$, $p=1.0$)`:
`, grouped $\Delta{=}0.000$, $p{=}1.0$; adaptive grouped 95\% CI $[0.349,0.692]$`
(`retrieval_control_results.json:1977-1989` overall minmax: grouped_delta 0.0, grouped_p 1.0;
`retrieval_control_results.json:581-588` adaptive ci: p5_grouped_lo 0.34909090909090906, p5_grouped_hi 0.6917431192660551.)

PAIRED CUT: tex:257 delete `, Wilcoxon $p=0.022` (keep the paired-bootstrap $p$; Wilcoxon stays in code
and artifact) and delete `A single-seed random-Gaussian-projection additive baseline gains $+5.4$~pp over visual-only ($p=0.003$) yet stays $5.3$~pp below concat ($p=0.001$);` (single-seed, already
caveated as illustrative in Table I caption tex:149). Saves ~2 lines; grouped promotion costs ~2.
NET: +2 / −2 = 0.

VERIFICATION: all numbers above with JSON paths. `n_boot` 3000: `retrieval_control_results.json:257`
(`eval_retrieval_control.py:66`). Holm families: `eval_retrieval_control.py:728-729,783-784`.
NOT VERIFIABLE: which script wrote the 5-minute `group` labels into `queries_control.jsonl`
(builder not located; values are self-evidently 5-minute buckets). Deployment grouped CIs do not exist.

## 5. ACTION/FILTER SENTENCE (anchor tex:284, Released-subset check paragraph)

NEW (insert after `so overrides are not harmful;` in tex:284):
```
The $0.538$ macro-average obscures weak absolute levels where overrides fire: action P@5 is $0.090$ and filter P@5 is $0.480$ on the released control, equal to the best tune-selected fixed $\alpha$ on both.
```
PAIRED CUT (same paragraph): shorten
`The release ships a \emph{raw}, non-protocol-matched smoke benchmark (no target count in its count questions; an exploitable time window on its action queries) ranking adaptive below the best fixed $\alpha$ ($0.486$ vs.\ $0.568$; oracle best $0.578$) and preferring $\alpha=1.0$ on action queries ($0.520$);`
to
`The release ships a \emph{raw}, non-protocol-matched smoke benchmark ranking adaptive below the best fixed $\alpha$ ($0.486$ vs.\ $0.568$);`
(parenthetical diagnoses already stated in `retrieval_control_results.json:860-868` scope note and `eval_retrieval_control.py:956-963`).
NET: +2 / −2 = 0.

VERIFICATION (`results/retrieval_control_results.json`, `test.minmax.adaptive.per_category`):
action `p5` 0.09 (lines 563–568), filter `p5` 0.48 (lines 569–574); macro `p5` 0.538 (lines 537–538).
Tie check: `test.minmax.fixed_0.7.per_category` action 0.09 (lines 453–458), filter 0.48 (lines 459–464);
`significance.minmax.overall` Δ=0.0, p=1.0 (lines 1980–1984). Production calibration is minmax
(tex:284: `production uses minmax calibration`), so minmax rows are the right ones.

## 6. BRUCH CITE (anchor References.bib:111-116, citing tex:95)

STATUS: b33 is the arXiv preprint. The published version exists:
S. Bruch, S. Gai, A. Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval",
ACM Trans. Inf. Syst., vol. 42, no. 1, Article 20, pp. 1–35, Jan. 2024, DOI 10.1145/3596512.
(Verified: arXiv:2210.11934 page lists Related DOI 10.1145/3596512; WebIS anthology entry
`rec+journals+tois+BruchGI24` gives vol 42/no 1/pp 20:1–20:35/year 2024; ACM TOIS 42(1) Jan 2024 TOC
lists Article 20 pp 1–35. Full names: Sebastian Bruch, Siyu Gai, Amir Ingber.)

NEW .bib entry:
```
@article{b33,
  author  = {S. Bruch and S. Gai and A. Ingber},
  title   = {An analysis of fusion functions for hybrid retrieval},
  journal = {ACM Trans. Inf. Syst.},
  volume  = {42},
  number  = {1},
  pages   = {20:1--20:35},
  year    = {2024},
  doi     = {10.1145/3596512}
}
```

CLAIM CHECK (tex:95): `convex-combination analysis finds the learned blend sample-efficient and normalization-agnostic, beating RRF in- and out-of-domain~\cite{b33}.`
The published abstract (identical to arXiv:2210.11934 abstract, verified verbatim) states exactly:
CC learning "generally agnostic to the choice of score normalization", "outperforms RRF in in-domain
and out-of-domain settings", and "sample efficient, requiring only a small set of training examples".
NO CLAIM ADJUSTMENT REQUIRED — the sentence already matches the published version (note: the paper
finds RRF "sensitive to its parameters"; we do not claim otherwise, so nothing to add).

PAIRED CUT (same .bib, net control): trim b30 and b34 `journal = {arXiv preprint arXiv:...}` to
`note = {arXiv:...}` (2 lines saved across the two entries; IEEEtran prints note compactly).
Entry grows +3 lines (pages/doi/volume/number vs journal/year), cuts −2, plus the b33 arXiv line
removal −1. NET: 0.

## 7. SURVEY SENTENCE (anchor tex:93, person Re-ID related-work paragraph)

REFERENCE (verified via author page markwjones.github.io/publications/2025_CoRRReID.html + arXiv:2408.00096):
F. Jiang, S. Yang, M. W. Jones and L. Zhang, "From Attributes to Natural Language: A Survey and
Foresight on Text-based Person Re-identification", Information Fusion, vol. 118, art. 102879,
June 2025, DOI 10.1016/j.inffus.2024.102879 (arXiv:2408.00096, July 2024). The survey's taxonomy spans
Evaluation, Strategy, Architecture and Optimization over attribute/natural-language person Re-ID,
covering benchmark datasets, feature-extraction strategies, network architectures and
modality-alignment loss functions.
CAVEAT: that author page's human-readable citation line says "vol. 30" while its own BibTeX says
vol. 118 with the same article number; vol. 118 matches the journal's 2025 volume sequence — use 118.

NEW (append to tex:93 after `...calibrated on 50 pairs.`):
```
A 2025 survey of text-based person Re-ID taxonomizes evaluation, strategy, architecture and optimization over established attribute-to-language retrieval designs~\cite{b36}; our construction differs in fusing frozen, dimensionally-mismatched encoders with no joint training as a single FAISS index.
```
PAIRED CUT (same paragraph tex:93): delete `CLIP-ReID~\cite{b35} still needs Re-ID training data;`
(covered by `Prior NL-based person-search~\cite{b32} couples a dual encoder by joint training` +
the new survey sentence; drop `\cite{b35}` only if b35 is cited nowhere else — it is cited only here,
so remove the b35 entry too: −5 .bib lines for +6 survey lines ≈ net 0 with the sentence swap).
NET: +2 prose / −2 prose = 0; .bib +6/−5 ≈ 0.

## 8. GENDER/GOVERNANCE (anchor tex:295 Limitations (iv))

Verified code facts:
- Gender exists because the PAR detector emits `male_confidence` and the encoder verbalizes it as a
  soft cue: `attribute_encoder.py:61-67` (male ≥ 0.65 → "male"; ≤ 0.35 → "female"; gated by
  `INCLUDE_GENDER_ATTRIBUTES`, default true per `config.py:221-223`).
- It DOES affect ranking on both sides: gallery `attribute_text` carries gender tokens into the fused
  1152-d descriptors (`hybrid_search.py:236-239` embeds text queries the same way); the released
  benchmark contains gender-constrained queries (`queries_control.jsonl:39` q_find_09 attributes
  `["female"]`; line 48 q_find_18 attributes `["male"]`), matched by `_det_attr`
  (`eval_retrieval_control.py:308-311`).
- Gender-off is exposure-level, NOT ranking-level: `hybrid_search.py:458-461` strips gender only from
  output text post-hoc via `privacy.py:387-392`; `config.py:218-220` states explicitly
  "captions/queries then omit it (the index itself is unchanged)".
- False-positive handling for protected attributes: `UNSUPPORTED_HINTS` includes `"gender"`
  (`alert_validator.py:80-88`); protected-attribute inference routes to clarify
  (`alert_materializer.py:227`).
- Oversight that exists (all opt-in / default-off): role/zone guard + audit trail inactive unless
  `ACCESS_CONTROL_ENABLED=true` (`main.py:274-277`; `security.py:1,215-240,300-307` audit JSONL);
  `RETENTION_ENABLED` default false (`config.py:211-214`); `FACE_BLUR_ENABLED` default false
  (`config.py:215-217`); 8-h retention prune + face-blur utility exist (`privacy.py:1,63-76,149-169`).
- NOT FOUND in code: any human-approval gate before alert side-effects fire
  (`snapshot`/`push_ws` defaults, `alert_validator.py:223-227`), any demographic-bias measurement,
  any wiring of gender-off into live ingestion (paper tex:295 already admits the last).

NEW (replace tex:295 limitation (iv) sentences from `its binary \textit{gender} attribute...`):
```
(iv)~\emph{Demographic bias, privacy and consent.} Its binary \textit{gender} attribute is a coarse verbalization of the PAR detector's \texttt{male\_confidence} (male $\ge$0.65 / female $\le$0.35) and, with the default \texttt{INCLUDE\_GENDER\_ATTRIBUTES=true}, it enters both gallery descriptors and text queries, so it affects ranking; the gender-off switch strips output text only and leaves the index unchanged. No demographic-bias evaluation is possible without demographic labels, and no measurement here isolates gender's ranking contribution. Live capture was in notified premises with face imagery retained at most 8~h; access control with audit logging, the retention prune and the face-blur utility exist but are opt-in (all default off), and no human-approval gate stands between a compiled rule and its side-effects, so consequential actions require operator review and deployment authorization outside this artifact.
```
PAIRED CUT: delete tex:295 `Gender is a soft cue only; no inter-annotator agreement, cross-camera generalisation or demographic-bias measurement is claimed.`
(subsumed by the new sentences; IAA/cross-camera non-claims already appear in tex:295 (iii) tail and
`retrieval_control_results.json:2338-2346` limitations).
NET: +3 / −3 = 0.

## 9. DEPLOYMENT–SIGNIFICANCE RECONCILIATION (anchors tex:246, Table III tex:262-264)

One sentence (replace Table III caption tex:263 first clause AND tex:246's second sentence with it —
single-sourcing the caveat):
```
Table~III's intervals are recorded single-run deployment measurements whose per-query export was not preserved, so they are not re-computable; verifiable significance is claimed only on the released track ($\Delta{=}0.000$, $p{=}1.0$, grouped $p{=}1.0$).
```
Concretely: NEW caption tex:263:
`\caption{Recorded deployment measurements (per-query export not preserved; not re-computable); verifiable significance is claimed only on the released track. Per-category P@5 with 95\% bootstrap CIs (10{,}000 resamples, $n{=}30$); $\Delta$ = mean of (adaptive $-$ best fixed-$\alpha$), $p$ = two-sided paired bootstrap (floor $0.0002$).}`
and delete tex:246's `Deployment CIs/$p$-values are recorded single-run measurements, not re-computable significance; significance is claimed only on the released track (grouped bootstrap, Holm-corrected per-category tests, 3{,}000 resamples).`
(folded into item 4's new protocol sentence, which already carries the grouped/Holm/3000 content).
NET: −1 line overall (duplication removed).

VERIFICATION: caption text tex:263-264; protocol tex:246; released-track null
`retrieval_control_results.json:1981-1988` (Δ 0.0, p 1.0, grouped p 1.0); p-floor convention
`eval_retrieval_control.py:1027-1033` (`< 2/n_boot` resolution floor).

## 10. COUNTING DISTINCTION (anchor tex:182; related tex:286)

OLD (tex:182 tail):
```
Counting top-5 are supporting clips relevant iff they contain the true count; exact-count EM/MAE (0.10/2.20, \texttt{eval\_\allowbreak alpha\_\allowbreak competitors.py}) is the primary counting-ability measure, in the release.
```
NEW:
```
Counting top-5 are supporting clips relevant iff they contain the true count, so P@5 measures clip relevance, not counting ability; the primary counting-ability measure is exact-count EM/MAE, only $0.10$/$2.20$ for the window-mode caption estimate against the anchor-frame caption count (\texttt{eval\_\allowbreak alpha\_\allowbreak competitors.py}).
```
PAIRED CUT (tex:286): shorten
`(iii)~Counting is label-limited: the window-mode caption estimate reaches EM $0.10$/MAE $2.20$ against the anchor-frame caption count, while counts in one window span $4.5$ persons on average, so no window-based predictor can beat the labels themselves (\texttt{eval\_\allowbreak alpha\_\allowbreak competitors.py}).`
to
`(iii)~Counting is label-limited: counts in one window span $4.5$ persons on average, so no window-based predictor can beat the labels themselves (\texttt{eval\_\allowbreak alpha\_\allowbreak competitors.py}).`
(EM/MAE now stated once, at tex:182.)
NET: +1 / −1 = 0.

VERIFICATION: `results/alpha_competitors.json:379-389`: `caption_estimate.em` 0.1, `.mae` 2.2 (lines
381–383); `label_noise_ceiling.mean_window_count_spread` 4.466666666666667 → 4.5 (line 388).
Cross-check: `results/retrieval_control_results.json:2305-2310` independently reports the same
estimator (`count_estimate`: n 30, em 0.1, mae 2.2, "window mode of released caption person counts vs
anchor-frame count"). Do NOT confuse with `test.minmax.fixed_0.0.count` (em 0.1, mae 1.8, lines
301–305) — that is top-1-clip prediction accuracy under retrieval, a different estimator.

---

## Page-budget tally (estimated typeset lines, two-column)

| Item | Added | Cut | Net |
|---|---|---|---|
| 1 weight math | +2 | −2 (tex:165 tail) | 0 |
| 2 F1 rename | +2 | −2 (tex:142 tail + tex:257 release-provenance sentence) | 0 |
| 3 fuzz split | +1 | −1 (tex:124 label parentheticals) | 0 |
| 4 grouped primary | +2 | −2 (Wilcoxon + single-seed clauses, tex:257) | 0 |
| 5 action/filter | +2 | −2 (raw-benchmark diagnosis clause, tex:284) | 0 |
| 6 Bruch .bib | +3/−3 (.bib only) | 0 (prose untouched — claim already correct) | 0 |
| 7 survey | +2 prose, +6/−5 .bib | −2 prose (CLIP-ReID clause + b35 entry) | 0 |
| 8 governance | +3 | −3 (tex:295 tail) | 0 |
| 9 reconciliation | 0 (merge) | −1 (dedupe) | −1 |
| 10 counting | +1 | −1 (tex:286 dedupe) | 0 |
| TOTAL | | | ≤ 0 (−1) |

## Could NOT verify (do not write these into the paper)

1. Full nine-pair deployment $(\alpha_v,\beta_t)$ grid values — only $(0.5,0.5)/(0.7,0.3)/(0.4,0.6) \to 0.785/0.799/0.774$ are attested (tex:257); nothing in `results/`.
2. Any deployment CI/$p$ beyond what is printed (export not preserved by design; tex:244).
3. The script that assigns 5-minute `group` labels in `queries_control.jsonl` (values self-evident; builder not found).
4. Author-page inconsistency for the survey citation line ("vol. 30" prose vs vol. 118 BibTeX on the same page) — used 118 + article number + DOI, all mutually consistent.
5. TOIS dating: issue is Jan 2024 (vol. 42 no. 1), online Aug 2023 — cite year 2024 per issue.
6. Whether gender helps or hurts ranking accuracy — no ablation exists; the draft states only that it affects ranking and is unmeasured.
7. Any human-approval gate on alert side-effects — none found; the draft states the absence as a limitation, not a feature.
