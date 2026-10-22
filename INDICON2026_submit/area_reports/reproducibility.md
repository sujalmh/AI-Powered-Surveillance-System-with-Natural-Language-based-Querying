# Area report — reproducibility (Figure 1, number checker, build/package)

Scope: plan §7 E7–E8 (Figure 1 only), §8 G1/G9–G10. No edit to the paper `.tex`.
Parser-gap sentences are not owned by this area (none added).

## 1. Exact LaTeX-ready replacement text

### 1a. Preamble (one line to re-add; `tikz` was dropped for the 6-page build)

```latex
\usepackage{tikz}\usetikzlibrary{arrows.meta,backgrounds,fit,positioning,shapes.geometric}
```

### 1b. Background / System Substrate — insert after the nine-model pipeline paragraph, no new section

```latex
\begin{figure}[!t]
\centering
\input{figures/pipeline}
\caption{System dataflow. Capture feeds perception (YOLOv11m-seg, OSNet, PAR
attributes, colour) into scene/person FAISS indices and MongoDB. The NL parser
(Beam) emits filter $F$, semantic query $q_{\mathrm{sem}}$ and intent $\iota$;
structured ($R_s$) and semantic ($R_v$) branches merge with the adaptive weight
$\alpha(F,\iota,n_s,n_v)$, then post-ranking (recency; MMR off by default)
returns top-$k$ clips. The schema-validated alert compiler plus the
\texttt{requires\_approval} gate stage consequential alerts in a pending queue.
Dashed nodes replay in the released single-camera reproducibility track.}
\label{fig:system}
\end{figure}
```

(Reference the figure as Fig.~\ref{fig:system}.)

### 1c. Figure source

`INDICON2026_submit/figures/pipeline.tex` (TikZ, `\resizebox` to
`\columnwidth`; contains only the `tikzpicture`, input inside the figure
environment above). Stands alone: compiles with `article` + the preamble
line; verified 0 errors, 0 overfull inside a `figure` environment
(standalone `wrap2.pdf`, 1 page).

## 2. Number checker + claims map

New files (stdlib only, no new dependencies):

- `backend/scripts/check_paper_numbers.py` — usage from repo root:
  `python backend/scripts/check_paper_numbers.py [--tex PATH] [--claims PATH]`.
  Exit 0 = all pass, 1 = fail with merge-actionable lines.
- `INDICON2026_submit/paper_claims.yaml` — every headline number mapped to a
  JSON path or a deployment-table row, plus whitelist and G0 rules.

What the checker enforces: (i) each released value present in the `.tex` and
equal to its JSON source within half a display digit; fractions (`30/30`) and
same-line CI endpoints checked; (ii) every deployment literal
(`0.828/0.807/0.785/0.700/91%/312,544/109 ms/6.0/12.0 s`) inside a table whose
caption mentions "deployment", else FAIL with line number; (iii) G0 —
`0.828/0.807/91%/verified/10.7/2.2/0.67–0.83` banned from abstract,
contributions, conclusion; `Person Re-ID` banned from abstract and keywords;
(iv) parser-gap sentences capped at two.

### Result on the CURRENT paper (expected FAIL — this is the merge fix-list)

`28 passed, 50 failed`, exit 1. Merge must fix, by group:

- Deployment numbers outside the deployment table (20 hits): `0.828` lines
  65 (abstract), 192, 253; `0.666` lines 175, 187, 253; `0.807` lines 154,
  250; `0.785` lines 153, 158, 250; `0.700` lines 150, 250; `91%` lines 65
  (abstract), 81 (contributions), 284, 290 (conclusion); `312,544` lines 130,
  237, 284; `109 ms`, `6.0 s`, `12.0 s` line 284. (`0.828/0.666` on line 272
  already sit in the recorded deployment table and pass.)
- G0 (15 hits): `verified` abstract + conclusion; `10.7/2.2` abstract,
  contributions, conclusion; `0.822` abstract; `0.67/0.83` abstract; `0.828`
  abstract; `91%` abstract, contributions, conclusion; `Person Re-ID` in
  keywords line 70.
- Released values missing from the current paper (merge must add with the new
  tables): C2 `0.311` (R@10), CI `[0.410, 0.445]`, `0.407`, `0.361`, `0.432`,
  `0.417`, `0.444`; C3 per-category `0.390`, z-score `0.524` vs `0.522`,
  end-to-end track gap `-0.28`.
- Caution: several PASSes on the current paper are positional coincidences
  (e.g. `0.80` inside the old deployment table). The gate applies to the
  REVISED paper, where each value must sit in its revised table.

### Checker self-tests (all green)

- Conforming synthetic paper (abstract + contributions + deployment table +
  all released values): `47 passed, 0 failed`, exit 0.
- Negative test (one value perturbed `0.427`→`0.429`): exit 1, exactly the
  `C2-concat-p5` MISMATCH line.

## 3. Number → evidence-source map

Values below are the REVISED-paper numbers (rounded display; checker
tolerates half a display digit).

| # | Paper value | Source file | JSON path / table row |
|---|---|---|---|
| C1 | 30/30 clean | `results/alert_results.json` | `splits.test.clean.rule_correct`=30 / `.n`=30 |
| C1 | 30/30 stress | `results/alert_results.json` | `splits.test.stress.clarified`=30 / `.n`=30 |
| C1 | 0 false materializations | `results/alert_results.json` | `splits.test.stress.false_materializations`=0 |
| C2 | 0.427 P@5 | `results/fusion_results.json` | `test.retrieval.concat.p5`=0.42739 |
| C2 | 0.311 R@10 | same | `test.retrieval.concat.r10`=0.31096 |
| C2 | 0.309 F | same | `test.retrieval.concat.f1_paper`=0.30879 |
| C2 | [0.410, 0.445] | same | `test.retrieval.concat.p5_ci` |
| C2 | 0.407 visual | same | `test.retrieval.visual.p5`=0.40689 |
| C2 | 0.361 RRF | same | `test.retrieval.rrf.p5`=0.36050 |
| C2 | 0.428 zero-pad/norm | same | `test.retrieval.zero_pad.p5`=0.42840 |
| C2 | 0.432 learned proj | same | `test.retrieval.learned_proj.p5`=0.43244 |
| C2 | 0.417 ±<0.001 randproj | same | `test.retrieval.randproj.p5`=0.41654, sd=0.00017 |
| C2 | 0.444 gated | same | `test.retrieval.gated.p5`=0.44420 |
| C2 | Holm stats | same | `significance_vs_concat.retrieval`: visual/rrf p=0.0 (floor <0.0007); zero_pad p=0.3473 (Holm 0.6947, n.s.); learned p=0.0713 (Holm 0.2140, n.s.); gated Δ+0.0168 p=0.0013 (Holm 0.0053) |
| C3 | 0.538 = 0.538 | `results/retrieval_control_results.json` | `test.minmax.adaptive.macro.p5`=0.538; tune-best `best_fixed_on_tune.minmax.method`=`fixed_0.7` (tune 0.580) |
| C3 | 0.390/0.920/0.810/0.090/0.480 | same | `test.minmax.adaptive.per_category.{count,find,track,action,filter}.p5` |
| C3 | fixed 0.048/0.380/0.534/0.520 | same | `test.minmax.fixed_{0.0,0.3,0.5,1.0}.macro.p5` |
| C3 | per-intent 0.536 | same | `test.minmax.per_intent_tuned.macro.p5`=0.53599 |
| C3 | paper lookup 0.428 | same | `test.minmax.paper_lookup.macro.p5`=0.42800 |
| C3 | conf-only 0.536 | same | `test.minmax.conf_only.macro.p5`=0.53599 |
| C3 | CI [0.349, 0.692], p=1.0 | same | `test.minmax.adaptive.ci.p5_grouped_{lo,hi}`; `significance.minmax.overall.p`=1.0 |
| C3 | z-score 0.524 vs 0.522, p=0.72 | same | `normalization_comparison.{adaptive,fixed_0.7}.zscore_macro_p5`; `significance.zscore.overall.p`=0.7187 |
| C3 | end-to-end 0.450 vs 0.448, p=0.80, CI [−0.004, +0.010] | `results/end_to_end_results.json` | `test.{adaptive,fixed_0.5}.macro`; `significance.{p,lo,hi}`=0.7973/−0.004/+0.010 |
| C3 | oracle 0.538; track −0.28 | same | `oracle_delta.control_macro_p5`; `oracle_delta.per_category.track.delta`=−0.28 |
| ABL | base 0.538; MMR 0.400 (Δ−0.138, p<0.0007) | `results/ablation_results.json` + `results/ablation_tables.md` | `summary["base (no recency, no MMR)"].macro`; `summary["MMR only"].macro`; Δ table row |
| DEP | 0.828 / 0.666 | recorded only (per-query export not preserved) | deployment study table; current `tab:alpha_baseline_ci` |
| DEP | 0.807 / 0.785 / 0.700 | recorded only | deployment study table |
| DEP | 91% | recorded (LLM materializer) | deployment study table; caveat in `results/alert_results.md` §Interpretation |
| DEP | 312,544 / 109 ms / 6.0 s / 12.0 s | recorded only | deployment study table |

## 4. Code changes made

No `backend/` logic touched (not permitted in this area). Files created:

- `INDICON2026_submit/figures/pipeline.tex` (new) — Figure 1 TikZ source.
- `backend/scripts/check_paper_numbers.py` (new) — number checker.
- `INDICON2026_submit/paper_claims.yaml` (new) — claims map.
- `INDICON2026_submit/area_reports/reproducibility.md` (new) — this report.

Test commands / results (repo root, PowerShell):

- `python backend/scripts/check_paper_numbers.py` → current paper: 28/50,
  exit 1 (fix-list above); synthetic conforming paper: 47/0, exit 0;
  perturbed synthetic: exit 1 with the single correct MISMATCH line.
- Figure: `pdflatex -interaction=nonstopmode -halt-on-error` on a wrapper
  with `article` + preamble line + `figure` env + `\input{pipeline}` →
  rc 0, 0 errors, 0 overfull. (One 15-pt overfull appears only with a bare
  `\input` in body text — paragraph indent; absent in the `figure`
  environment the parent will use.)
- Paper build (clean copy in `%TEMP%\paperbuild`, paper `.tex` untouched):
  `pdflatex` → `bibtex` → `pdflatex` → `pdflatex`, all rc 0 →
  `Output written on INDICON2026Conference.pdf (6 pages, 343362 bytes)`,
  0 overfull, 0 undefined control sequences, 0 LaTeX warnings; all 13
  `\label`s matched by `\ref`s, all `\cite` keys present in
  `References.bib`. Uncommitted build artefacts live only in `%TEMP%`.

## 5. Risks, blockers, page-budget impact

- **Figure bug found and fixed here:** `\\{` (newline + group-open) corrected
  to `\{` (literal brace) in the approval-gate node; plus the
  `label={{...}}` double-brace TikZ gotcha. Both verified by the clean
  compile above.
- **Re-run coupling (blocker for final gate):** other contributors have
  uncommitted changes (`config.py`, alert engine/validator, safety tests,
  `rebuild_person_index.py`, `results/safety_run/`). If the gender rebuild or
  MMR-default flip re-runs `eval_fusion.py` / `eval_retrieval_control.py` /
  `eval_end_to_end.py`, released numbers may shift — **re-run
  `check_paper_numbers.py` on the revised `.tex` after their re-runs** before
  the final build.
- **Table I ownership:** plan E8 pairs Figure 1 with a claim→experiment→map
  table in Evaluation. Table I prose is outside this area's task list; the
  checker already covers every number Table I will contain. Parent: assign
  explicitly or confirm the Evaluation owner takes it.
- **Parser-gap budget:** checker caps gap sentences at two (current paper
  passes with two pattern hits); my area adds none.
- **Page budget:** Figure 1 is `\resizebox`-scaled to `\columnwidth`
  (~3 rows, ≈1/3 column). Room must come from the planned Evaluation prose
  cuts (probe paragraph, duplicate segment sentence, supporting-numbers
  prose → deployment table) — other owners' edits. Preamble `tikz` line
  must be re-added (it was dropped for the 6-page build).
- **Parent final-build checklist:** (1) apply content edits; (2) run
  `python backend/scripts/check_paper_numbers.py` → expect exit 0;
  (3) `pdflatex → bibtex → pdflatex → pdflatex` in a clean dir;
  (4) confirm `6 pages`, `Overfull` count 0, no `undefined`, all
  tables/figures referenced; (5) regenerate PDF + source zip from the final
  `.tex`.
