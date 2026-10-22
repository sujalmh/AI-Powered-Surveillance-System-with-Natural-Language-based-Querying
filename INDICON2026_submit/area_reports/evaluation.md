# Area report — Evaluation section

## 1. Exact LaTeX-ready replacement text

Replace everything in `INDICON2026Conference.tex` from `\section{Evaluation}\label{sec:eval}` through the end of the Evaluation section (i.e. up to, but not including, `\section{Discussion and Conclusion}`). Delete the current `\label{tab:alpha_baseline_ci}` table and all deployment-$p$-value prose as part of this replacement (see merge note in §5).

```latex
\section{Evaluation}\label{sec:eval}

\paragraph*{Setup: two tracks}
We report two tracks. \textbf{(i)~Four-camera deployment study (recorded, not re-computable).} Four 1080p IP cameras at 2~FPS per stream for $\sim$8~h ($\sim$230k frames, 1.25M detections) in one notified indoor/outdoor premises; 312{,}544 person descriptors; a 150-query NL benchmark (30 per category: count/find/track/action/filter). Its per-query export, annotator roster and tune/test partition were not preserved, so its means are single-run estimates and its CIs/$p$-values are recorded single-run measurements, not re-computable significance. \textbf{(ii)~Released single-camera reproducibility track (primary comparison).} A single-camera subset (123 clips; 242 scene frames; 1{,}778 person descriptors) with metadata-derived labels and 2{,}016 track-disjoint person pairs (1{,}008 calibration); harness \texttt{eval\_*.py}, outputs \texttt{results/}. Significance is claimed only on the released track. Hardware: i7-13700K, RTX~4070, 32~GB; Python~3.12, PyTorch~2.3, CUDA~12.1, FAISS~1.8, MongoDB~7.

\paragraph*{Statistical protocol}
On the released track, primary uncertainty is grouped (cluster) bootstrap---5-minute wall-clock query buckets on the retrieval track (3{,}000 resamples), pseudo-tracks on the person-descriptor track---with paired-difference CIs and Holm-corrected per-category tests; query-level bootstrap is secondary. At $n{=}30$ per category the deployment category means carry $\pm0.05$--$0.08$ recorded half-widths, so the observed $+0.02$/$+0.01$ category deltas are below the design's resolution and are reported as unresolvable, not as improvements.

\paragraph*{Metric definitions}
Precision@5 $=|\mathrm{top\!-\!5}\cap\mathrm{relevant}|/5$; Recall@10 $=|\mathrm{top\!-\!10}\cap\mathrm{relevant}|/|\mathrm{relevant}|$; F$_{5,10}$ is the harmonic mean of P@5 and R@10 (subscripts name the cutoffs; not the standard same-cutoff F$_1$). Macro-mean is the unweighted mean of the five category means.

\paragraph*{C1 -- alert compiler (summary)}
The released deterministic compiler with clarify/abstain validation scores 30/30 fully-correct 9-field rules on the clean test split and 30/30 clarifications with zero false materializations on the stress test split (120 author-constructed utterances: 60 clean + 60 stress, 30/30 dev/test split; \texttt{eval\_alerts.py}). The deployment study with the LLM-backed materializer recorded 91\% full-rule correctness on 60 utterances; the two paths are complementary, not interchangeable (Table~\ref{tab:deployment_study}).

\paragraph*{C2 -- person descriptor (summary)}
On the released track (280 metadata-formed pseudo-tracks, not identities; 1{,}190 queries; 1{,}778 descriptors), the concat descriptor reaches P@5 0.427, ahead of visual-only (0.407, $\Delta{+}0.021$, Holm $p{<}0.0007$) and RRF (0.361, $\Delta{+}0.067$, Holm $p{<}0.0007$); zero-pad/norm-matched (0.428) and learned projection (0.432) tie concat, so the gain comes from combining views, not the concatenation operator (full table in \S\ref{sec:fusion}). The deployment study recorded 0.807 concat / 0.785 RRF / 0.700 visual (F$_{5,10}$) over 312{,}544 descriptors (Table~\ref{tab:deployment_study}).

\paragraph*{C3 -- routing rule (summary)}
On the released protocol-matched control (50 tune / 100 test queries), adaptive routing matches the best tune-selected fixed $\alpha{=}0.7$ on every category (macro 0.538 = 0.538, $\Delta{=}0.000$, $p{=}1.0$; grouped 95\% CI $[0.349,0.692]$), removing the need for a per-site $\alpha$ sweep (full table in \S\ref{sec:adaptive_alpha}). The deployment $\alpha$-sweep recorded a larger gain (0.666 $\to$ 0.828 macro P@5 on 150 queries; Table~\ref{tab:deployment_study}). Normalization does not drive the tie: under z-score normalization adaptive scores 0.524 vs.\ 0.522 for fixed $\alpha{=}0.7$ ($p{=}0.72$), while min-max gives 0.538 = 0.538, so parity holds while the absolute level shifts by 0.014. End-to-end with parsed queries, adaptive ties the tune-selected fixed $\alpha{=}0.5$ (0.450 vs.\ 0.448, $\Delta{+}0.002$, $p{=}0.80$, 95\% CI $[-0.004,+0.010]$), which is the result. The remaining end-to-end gap is dominated by query parsing rather than ranking (oracle-filter 0.538 vs.\ parser-driven 0.450; track queries -0.28), which identifies the next improvement.

\paragraph*{Post-ranking (ablation)}
Recency is inert on the released control; MMR at $\lambda{=}0.3$ costs 0.138 P@5 (0.538 $\to$ 0.400), so it is off by default and reported as an ablation (Table~\ref{tab:deployment_study}).

\paragraph*{Latency (summary)}
On the released track, mean LLM parse time is 3.0~s and production FAISS lookup is 37.4~ms mean (scene) / 17.2~ms mean (person). Deployment operator-in-the-loop totals and the 109~ms person-index median are recorded in Table~\ref{tab:deployment_study}.
```

## 2. Exact table LaTeX

### Table A — deployment study table (place after the C3 summary paragraph)

```latex
\begin{table}[!t]
\caption{Deployment study (recorded single-run measurements, not re-computable) vs.\ released reproducibility track (primary comparison). Deployment CIs/$p$-values are not claimed as significance.}
\label{tab:deployment_study}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcc@{}}
\toprule
Quantity & 4-camera deployment (recorded) & Released track (reproducibility check) \\
\midrule
Alert compiler & 91\% full-rule, 60 utterances, LLM materializer & 30/30 clean, 30/30 stress, deterministic \\
Person descriptor & 0.807 concat / 0.785 RRF / 0.700 visual (F$_{5,10}$); 312{,}544 descr.\ & 0.427 / 0.361 / 0.407 P@5; 1{,}778 descriptors \\
Routing & 0.666 $\to$ 0.828 macro P@5; 150 queries & 0.538 = 0.538; 100 control queries \\
End-to-end & 6.0~s / 12.0~s; 109~ms person index & 0.450 vs.\ 0.448; 3.0~s parse; 37.4/17.2~ms FAISS \\
Post-ranking & production MMR 0.400 & MMR off 0.538; MMR on 0.400 \\
\bottomrule
\end{tabular}
\end{table}
```

### Table B — Table I, claim $\to$ experiment $\to$ metric $\to$ source (place after the post-ranking paragraph)

```latex
\begin{table}[!t]
\caption{Claim $\to$ experiment $\to$ metric $\to$ source. Deployment rows are recorded single-run measurements; released rows are re-computable from the listed outputs.}
\label{tab:claim_map}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llll@{}}
\toprule
Claim & Experiment & Metric & Source \\
\midrule
Deterministic compiler & 30/30 clean + 30/30 stress test & 9/9-field correctness; 0 false mats & \texttt{results/alert\_results.md} \\
Two-view descriptor & Released pseudo-track retrieval & P@5 0.427 vs.\ 0.407 / 0.361 & \texttt{results/fusion\_tables.md} \\
Training-free routing & Released 50-tune/100-test control & 0.538 = 0.538 & \texttt{results/retrieval\_control\_tables.md} \\
End-to-end parity & Parsed-query run, 100 test & 0.450 vs.\ 0.448, $p{=}0.80$ & \texttt{results/end\_to\_end\_results.json} \\
Post-ranking & MMR ablation & 0.538 $\to$ 0.400 & \texttt{results/ablation\_tables.md} \\
\bottomrule
\end{tabular}
\end{table}
```

## 3. Number $\to$ evidence-source map

Every number in the proposed LaTeX above, with file + key/row:

| Number in LaTeX | Source file | Key / table row |
|---|---|---|
| 4 cameras, 2 FPS, $\sim$8 h, $\sim$230k frames, 1.25M detections, 312,544 descriptors | Recorded deployment study (not re-computable); previously stated `.tex` line 237 | Deployment-setup sentence; kept as recorded context, duplicated only in Table A deployment column |
| 150-query deployment benchmark, 30/category | Same as above | Recorded; Table A routing row "150 queries" |
| $\pm0.05$--$0.08$ recorded half-widths; $+0.02$/$+0.01$ unresolvable | Previously stated `.tex` line 239 (recorded deployment CIs) | Statistical-protocol paragraph; no significance claimed |
| 123 clips; 242 scene frames; 1,778 descriptors | `results/latency_results.md` header (subset line); `results/fusion_tables.md` scope bullets (1778 descriptors); `results/retrieval_control_tables.md` scope (121 indexed clips / 242 scene frames / 1,778 descriptors) | Setup track (ii) |
| 2,016 pairs, 1,008 calibration | `results/fusion_tables.md` scope bullets ("2016 pairs (test 504+/504-; calibration 504+/504-)") | Setup track (ii) |
| 3,000 resamples | `results/fusion_tables.md` ("Paired bootstrap vs concat", 3000 resamples); `results/retrieval_control_tables.md` header ("Bootstrap resamples: 3000"); `results/end_to_end_results.json` `significance.n_boot = 3000` | Statistical-protocol paragraph |
| C1: 30/30 clean test (9/9 exact), 30/30 stress clarified, 0 false mats | `results/alert_results.md` "Test split (primary, n=30 clean / 30 stress)" table | C1 summary paragraph; Table A released column; Table B row 1 |
| C1: 120 utterances, 60 clean + 60 stress, 30/30 dev/test | `results/alert_results.md` dataset line + dev/test table | C1 summary paragraph |
| C1 deployment: 91%, 60 utterances, LLM materializer | Recorded deployment; `results/alert_results.md` "Interpretation" ("The 0.91 figure ... comes from the deployment study (a separate 60-utterance human-authored set...)") | C1 summary; Table A alert row |
| C2 released: 280 pseudo-tracks (210 test / 70 cal), 1,190 queries | `results/fusion_tables.md` scope bullets | C2 summary paragraph |
| C2: concat 0.427 [0.410, 0.445]; visual 0.407; RRF 0.361 | `results/fusion_tables.md` "Person retrieval" table | C2 summary; Table A; Table B row 2 |
| C2: $\Delta{+}0.021$, Holm $p{<}0.0007$ (vs visual); $\Delta{+}0.067$, Holm $p{<}0.0007$ (vs RRF) | `results/fusion_tables.md` "Paired bootstrap vs concat" rows (visual $-0.021$, rrf $-0.067$, Holm $<0.0007$) | C2 summary paragraph |
| C2: zero-pad 0.428, norm-matched 0.428, learned 0.432 tie | `results/fusion_tables.md` retrieval table + paired rows ($\Delta{+}0.001$, $p{=}0.3473$, Holm $0.6947$; learned $\Delta{+}0.005$, $p{=}0.0713$, Holm $0.2140$, reject=False) | C2 summary paragraph |
| C2 deployment: 0.807 / 0.785 / 0.700 F$_{5,10}$, 312,544 | Recorded deployment (previously stated `.tex` Table II + line 250); audit-confirmed not in released files | Table A person-descriptor row only; one C2 observation sentence |
| C3: 50 tune / 100 test; adaptive 0.538 = fixed 0.7 0.538; $\Delta{=}0.000$, $p{=}1.0$; grouped CI [0.349, 0.692] | `results/retrieval_control_tables.md` "Adaptive vs tune-selected best fixed-alpha" (minmax $\Delta{+}0.000$, $p{=}1.0000$, grouped same) + "Macro results" table (grouped CI [0.349, 0.692]) | C3 summary; Table A routing row; Table B row 3 |
| C3 per-category identical (all Holm $p{=}1.0$); extremes 0.048 ($\alpha{=}0.0$), 0.380 ($\alpha{=}0.3$) | `results/retrieval_control_tables.md` per-category table + adaptive-vs-best section (Holm $p{=}1.0000$ all five) | "on every category" wording |
| C3 deployment: 0.666 $\to$ 0.828, 150 queries | Recorded deployment (previously stated `.tex` line 252 / Table III) | One observation sentence; Table A routing row |
| z-score 0.524 vs 0.522 ($p{=}0.72$); min-max 0.538 = 0.538; shift 0.014 | `results/retrieval_control_tables.md` "Min-max vs z-score" table (adaptive 0.538/0.524; fixed_0.7 0.538/0.522) + zscore section ($\Delta{+}0.002$, $p{=}0.7187$); 0.014 = 0.538 $-$ 0.524 | z-score sentence |
| End-to-end 0.450 vs 0.448, $\Delta{+}0.002$, $p{=}0.80$, CI $[-0.004,+0.010]$ | `results/end_to_end_results.json`: `test.adaptive.macro = 0.45`, `best_fixed_on_tune.method = fixed_0.5` (test 0.448), `significance` (delta 0.002, lo $-0.004$, hi $0.010$, p $0.7973 \to 0.80$) | C3 summary; Table A end-to-end row; Table B row 4 |
| Parser-driven 0.450 vs oracle-filter 0.538; track $-0.28$ | `results/end_to_end_results.json` `oracle_delta` (control 0.538, end-to-end 0.450; track delta $-0.28$) | The single parser sentence (only parser mention in this area) |
| MMR 0.538 $\to$ 0.400, $\Delta -0.138$, $p{<}0.0007$; recency inert | `results/ablation_tables.md` 2x2 table + deltas table (MMR only $-0.138$, $p{<}0.0007$; recency $+0.000$, $p{=}1.0000$) | Post-ranking paragraph; Table A; Table B row 5 |
| 3.0 s mean parse | `results/end_to_end_results.json` `parse.mean_seconds = 2.957` $\to$ 3.0 s | Latency paragraph; Table A |
| FAISS 37.4 ms scene / 17.2 ms person (mean) | `results/latency_results.md` "End-to-end component composition" (scene 37.46, person 17.17) | Latency paragraph; Table A end-to-end row |
| 6.0 s / 12.0 s deployment totals; 109 ms person-index median | Recorded deployment (previously stated `.tex` line 284) | Table A end-to-end row only |

Rounding: 0.7973 $\to$ 0.80, 0.7187 $\to$ 0.72, 2.957 $\to$ 3.0 s, 37.46 $\to$ 37.4, 17.17 $\to$ 17.2, 0.014 exact. No silent rounding beyond one decimal place shown.

## 4. Code changes made

None. Evaluation area performs no code edits (per constraints; the safety contributor owns re-runs). No test commands run; no released results files modified. Verified latency composition rows by reading `results/latency_results.md` (scene 37.46/39.68, person 17.17/20.35) against the current `.tex` line 284 figures.

## 5. Risks, blockers, and page-budget impact

- **Merge dependency (C3 refs):** this replacement deletes `\label{tab:alpha_baseline_ci}` and all deployment-$p$-value prose. The C3 section text (current lines 252--253) still cites `Table~\ref{tab:alpha_baseline_ci}`; the C3 contributor must retarget that reference (to their released control table) or the build will emit an undefined reference. Parent must enforce.
- **Label assumptions:** my LaTeX adds `\label{tab:deployment_study}` and `\label{tab:claim_map}` (new, no collision) and references `\S\ref{sec:fusion}` / `\S\ref{sec:adaptive_alpha}` (existing labels, unchanged). I deliberately do not `\ref` the C2/C3 released tables to avoid label collisions with those areas.
- **Parser-mention budget:** exactly one parser sentence is included (byte-identical to plan §2). No other parser-gap language in this area. The second allowed mention belongs to Discussion/future scope (not my area).
- **Deployment numbers contained:** 0.828, 0.807, 91%, 312,544, 109 ms, 6.0 s, 12.0 s appear only in Table A and at most one observation sentence per component; none in abstract/contributions/conclusion (untouched).
- **Cut list (page budget):** deleted the long C1 probe paragraph (compositional held-out / oracle dialogue / fuzz / adversarial detail), the deployment per-field C1 paragraph (moved to Table A), the C2 deployment-ablation paragraph, the C3 deployment-gain paragraph, the 8-row deployment CI table, the full released-subset check detail, the competitors-and-robustness paragraph, and the supporting-numbers prose. Added: two compact tables + ~6 short paragraphs. Net estimate **$-0.5$ to $-0.75$ columns** (roughly $-0.3$ pages), helping the 6-page budget. Table B can be dropped to `\small`/inline if space is tight.
- **Blockers:** none. No files outside `INDICON2026_submit/area_reports/evaluation.md` created or modified.
