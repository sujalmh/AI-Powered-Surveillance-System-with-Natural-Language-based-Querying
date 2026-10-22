# Area report — Framing (abstract, keywords, Introduction, Related Work novelty sentence, Discussion limitations / future scope, conclusion)

Scope: framing only. No edits to C1/C2/C3/Evaluation. No edits to `INDICON2026Conference.tex` (parent merges). No code changes (not allowed for this area). No long eval re-runs (read released files only).

Source of truth: `INDICON2026_submit/REVISION_PLAN_850.md` §§4–6. Title kept exactly. Structure unchanged.

---

## 1. Exact LaTeX-ready replacement text (with line anchors)

Line numbers refer to `INDICON2026_submit/INDICON2026Conference.tex` as read on 2026-10-23.

### 1a. Title — line 46 — NO CHANGE

```latex
\title{Intent-Adaptive Hybrid Retrieval and Heterogeneous-Space Person Fusion for Conversational Video Surveillance}
```

Keep byte-exact. No edit.

### 1b. Abstract — lines 63–67 — REPLACE whole block

Current (lines 63–67) contains forbidden tokens for the abstract: `verified`, `91\%`, `10.7-pp`, `2.2-pp`, `0.822 vs.\ 0.828`, `0.67 to 0.83`, `Lemma`, `Person Re-ID` (via keywords, not abstract body — abstract body itself avoids the phrase but the plan requires the positive rewrite anyway). Replace lines 63–67 inclusive with:

```latex
\begin{abstract}
\sloppy
Conversational surveillance requires turning free text into executable alert rules, representing people across visual and attribute spaces, and ranking candidates from structured and semantic branches. We present an integrated system that couples these decisions in one pipeline and evaluate it on a four-camera deployment and a released single-camera reproducibility track. The system combines a deterministic, schema-validated 9-tuple alert compiler with clarify/abstain validation (30/30 clean and 30/30 stress on the released benchmark, no false materializations); a heterogeneous SigLIP+MiniLM person descriptor that improves P@5 over visual-only and RRF on the released track (0.427 vs.\ 0.407 and 0.361); and a training-free intent-adaptive routing rule that matches the best tune-selected fixed alpha (0.538 vs.\ 0.538) and ties it end-to-end (0.450 vs.\ 0.448), removing the need for a per-site alpha sweep. The deployment study provides the motivating measurements; the released track makes the pipeline re-computable. The system is released with its benchmark definitions and measured outputs.
\fussy
\end{abstract}
```

Notes for parent merge:
- Verbatim plan §4 text, with only LaTeX escaping (`vs.\ `, `%`-free).
- Contains: no deployment numbers, no `verified`, no `Person Re-ID`, no parser-gap language, no `Lemma`, no `10.7/2.2`, no `0.67 to 0.83`.
- `30/30`, `0.427/0.407/0.361`, `0.538/0.538`, `0.450/0.448` are released-track numbers (see §3 map).

### 1c. Keywords — line 70 — REPLACE term

Current line 70:

```latex
    Conversational Alert Authoring, Multimodal Person Re-ID, Heterogeneous Embedding Fusion, Hybrid Retrieval, Intent-Adaptive Ranking, Vision--Language Surveillance.
```

Replace with:

```latex
    Conversational Alert Authoring, Person Descriptor Retrieval, Heterogeneous Embedding Fusion, Hybrid Retrieval, Intent-Adaptive Ranking, Vision--Language Surveillance.
```

One-term swap per plan §6. Rest unchanged. (`Reproducibility` addition is optional per plan; NOT proposed, to save space.)

### 1d. Introduction — lines 73–85 — two targeted edits

**Edit D1 — motivation sentence (end of line 75).** Current tail of line 75:

```latex
..., and no fixed value exceeds 0.67 macro-mean P@5.
```

Replace that clause with (keeps opening problem + three gaps, swaps headline to released control, demotes deployment to one observation sentence):

```latex
\IEEEPARstart{N}{atural}-language interfaces over surveillance video are feasible: instance segmentation, person descriptor retrieval and contrastive vision--language models are mature, and structured-output LLMs turn an utterance into a typed query schema. The unsolved problem is \emph{how to rank} candidate frames: a structured filter (\textit{``persons in Zone~A between 22:00 and 23:00''}) and a semantic vector lookup (\textit{``red jacket loitering near the entrance''}) give two incommensurable scales. Dominant fusion strategies: a fixed convex sum $\alpha S_{\text{struct}} + (1-\alpha) S_{\text{sem}}$ tuned once per system~\cite{b7,b12,b29}, parameter-free Reciprocal Rank Fusion~\cite{b27}, or sequential filter-then-rank pipelines~\cite{b30}. On the released protocol-matched control the best tune-selected fixed $\alpha$ reaches 0.538 macro-mean P@5 and the training-free adaptive rule matches it at 0.538, removing the need for a per-site $\alpha$ sweep; the four-camera deployment recorded larger absolute gains (0.666 best fixed to 0.828 adaptive) as a motivating deployment observation.
```

Also in line 77, replace `same-space-only Re-ID fusion` with `same-space-only descriptor fusion` (removes a `Re-ID` instance inside framing-owned Introduction; C2 owns the section but this sentence is in Introduction so framing owns it):

Current line 77:

```latex
Two gaps compound this: hand-translation of behavior into typed rules, and same-space-only Re-ID fusion needing matched backbones deployments rarely have.
```

Replace with:

```latex
Two gaps compound this: hand-translation of behavior into typed rules, and same-space-only descriptor fusion needing matched backbones deployments rarely have.
```

**Edit D2 — Contributions (lines 79–85).** Replace the whole `\subsection*{Contributions}` + `enumerate` block (lines 79–85) with the plan §5 positive, scoped four items. No parser mention, no deployment numbers:

```latex
\subsection*{Contributions}
\begin{enumerate}
\item \textbf{An integrated conversational surveillance system.} One query schema couples alert authoring, heterogeneous person descriptor retrieval and adaptive hybrid retrieval; the paper reports a four-camera deployment study and releases a single-camera reproducibility track with code, benchmark definitions and measured outputs.
\item \textbf{A deterministic, schema-validated alert compiler.} Nine-field rules with clarify/abstain; released test split 30/30 clean and 30/30 stress with zero false materializations; fuzz and adversarial probes. The deployment study used an LLM-backed materializer and is reported in the deployment table.
\item \textbf{A heterogeneous person descriptor with a fair comparison.} SigLIP+MiniLM concatenation gives P@5 0.427 vs.\ 0.407 visual-only and 0.361 RRF on the released track; zero-pad/norm-matched and learned projection are statistically indistinguishable, so the gain comes from combining views rather than the specific concatenation operator. No identity claim.
\item \textbf{A training-free intent-adaptive routing rule.} Matches the best tune-selected fixed alpha (0.538 = 0.538) and ties end-to-end (0.450 vs.\ 0.448), removing the need for a per-site alpha sweep. The four-camera deployment recorded the larger gains in the deployment table; the released track is the reproducibility check.
\end{enumerate}
```

Note: item 2 says ``reported in the deployment table'' and item 4 says ``recorded ... in the deployment table'' without printing any deployment number — compliant with the ``never in contributions'' rule. The footnote on item 4 in the current tex (supplementary-artifact URL) should be RETAINED by parent on the last item if space allows; it was omitted above only to keep the replacement block clean. Parent decision: keep existing `\footnote{...}` on the fourth item unchanged.

### 1e. Related Work — lines 87–95 — ONE sentence swap

Current line 89:

```latex
We compare with the closest published work along the three axes below; only this one provides all three, and none combines override-routing with heterogeneous-space Re-ID fusion.
```

Replace with (plan §6 verbatim):

```latex
We compare with the closest published work along the three axes below. To our knowledge, no prior system couples conversational alert authoring, heterogeneous frozen-encoder person descriptors and adaptive hybrid retrieval, and reports a deployment study alongside a released reproducibility track.
```

No other Related Work edit proposed by framing. The `Multimodal person Re-ID` bold heading (line 93) and body Re-ID phrasing are left to the C2 contributor; framing does not touch them to avoid cross-area conflict.

### 1f. Discussion — limitations (line 288) — NO EDIT proposed by framing

Keep the five-limitation structure (i)–(v) as-is. Framing proposes **no replacement text** for the limitations paragraph. Rationale:

- The substantive wording of (ii) MMR, (iii) alert-compiler paths, (v) privacy/fairness (gender default, approval gate) is owned by the Safety and Evaluation contributors and their code changes. A competing framing rewrite would create a merge conflict and risk a second parser mention.
- Current limitations contain **zero** parser-gap language — compliant. Parent must ensure the merged limitations still contain zero parser mentions (the single allowed future-scope sentence lives outside limitations).

### 1g. Discussion — future scope (end of §Discussion, before Conclusion) — REPLACE / INSERT one paragraph

Parent to place this as the future-scope paragraph immediately after the limitations paragraph (line 288) and before `\textbf{Conclusion.}` (line 290). It contains **exactly one** parser sentence — the area's single owned parser mention. No other parser-gap language anywhere else in framing text:

```latex
\textbf{Future scope.} A public identity-labelled benchmark for the descriptor; runtime alert quality (per-family precision/recall, false alerts per camera-hour, latency); and cross-camera generalization. The highest-leverage next step is parser improvement for track time windows, where the reference and parsed windows differ in 30/30 cases.
```

Word-for-word parser sentence per plan §2/§6. Do not rephrase (rephrasing risks breaking the ``exactly one sentence'' gate and the 30/30 evidence match).

### 1h. Conclusion — line 290 — REPLACE

Current line 290 (`A verified alert compiler (91\% ...) ... They are schema-decoupled.`) contains `verified`, `91\%`, `10.7-/2.2-pp` — all forbidden in the conclusion. Replace line 290 with (plan §6 verbatim):

```latex
\textbf{Conclusion.} We presented an integrated conversational surveillance system that couples a deterministic alert compiler, a heterogeneous person descriptor and a training-free adaptive routing rule, with a four-camera deployment study and a released single-camera reproducibility track. The released evidence supports the compiler, the two-view descriptor and routing parity without per-site tuning; the deployment study supplies the motivating measurements. The release makes the claims re-computable and provides the basis for the next iteration.
```

No parser mention, no deployment numbers, no `verified` in the replacement. (Parent may render ``deterministic, schema-validated alert compiler'' if the C1 merge prefers that adjective; framing accepts either since both avoid `verified`. Proposed text keeps plan wording to minimize drift.)

---

## 2. Table / figure LaTeX

None. Framing owns no table or figure. (Figure 1 and Tables I–IV + deployment-study table are owned by System-story / C1–C3 / Evaluation contributors.)

---

## 3. Number → evidence-source map

Every number printed in the §1 replacement text. No invented, silently rounded, or overclaimed values. Deployment numbers appear in framing text only in the single Introduction observation sentence (0.666/0.828), never in abstract / contributions / conclusion.

| Number in proposed text | Location | Evidence file | JSON key / table row | Notes |
|---|---|---|---|---|
| 30/30 clean, 30/30 stress, 0 false materializations | Abstract; Contributions item 2 | `results/alert_results.md` §Test split (primary) | Table rows: `Clean rule correctness 30/30`, `Stress clarification rate 30/30`, `Stress false materializations 0`; also `results/alert_results.json` test counters | Released deterministic compiler + validator. |
| 0.427 (concat P@5) | Abstract; Contributions item 3 | `results/fusion_tables.md` §Person retrieval table | Row `concat`: P@5 `0.427` | Released single-camera track, 280 pseudo-tracks / 1,190 queries / 1,778 descriptors (header of same file). |
| 0.407 (visual-only P@5) | Abstract; Contributions item 3 | `results/fusion_tables.md` §Person retrieval table | Row `visual`: P@5 `0.407` | Same table. |
| 0.361 (RRF P@5) | Abstract; Contributions item 3 | `results/fusion_tables.md` §Person retrieval table | Row `rrf`: P@5 `0.361` | Same table. |
| zero-pad/norm-matched + learned projection ``statistically indistinguishable'' (no number printed) | Contributions item 3 | `results/fusion_tables.md` §Paired bootstrap vs concat | Rows `zero_pad Δ+0.001 p=0.3473 Holm 0.6947 reject False`; `learned_proj Δ+0.005 p=0.0713 Holm 0.2140 reject False` | Supports ``gain comes from combining views'' mechanism note without printing extra numbers. |
| 0.538 adaptive = 0.538 best tune-selected fixed α=0.7 | Abstract; Introduction D1; Contributions item 4 | `results/retrieval_control_tables.md` §Macro results + §Per-category P@5 + §Adaptive vs tune-selected best fixed-alpha | Macro table rows `fixed_0.7 0.538`, `adaptive 0.538`; per-category table rows identical (0.390/0.920/0.810/0.090/0.480); text `ΔP@5 = +0.000, p=1.0000; grouped Δ = +0.000, p=1.0000` | Released protocol-matched control, 50 tune / 100 test. Tune-best is `fixed_0.7` (`Best fixed alpha selected on the tune split` line). |
| 0.450 vs. 0.448 end-to-end tie | Abstract; Contributions item 4 | `results/end_to_end_results.json` | `test.adaptive.macro = 0.45`; `test.fixed_0.5 = 0.44800000000000006`; `best_fixed_on_tune.method = fixed_0.5`; `significance.delta_p5 = 0.002, p = 0.79733, grouped_p = 0.79267` | Parser-driven (LLM parse) end-to-end; tune-selected fixed is α=0.5 here, not 0.7 — wording ``ties it end-to-end'' deliberately avoids naming α. p≈0.80 rounds the measured 0.7973. |
| 0.666 → 0.828 deployment observation | Introduction D1 only (one sentence) | Deployment study (non-released; Evaluation deployment table row per plan §6) — NOT in any `results/` JSON | Plan §3 ledger row ``0.666 → 0.828 routing gain → deployment study row''; current tex Table `tab:alpha_baseline` / `tab:alpha_baseline_ci` | Framing prints these two numbers exactly once, labelled ``motivating deployment observation'', not the headline. Parent/Evaluation own the authoritative deployment-table cell. |
| 30/30 track window differences | Future scope (single parser sentence) | `results/end_to_end_results.json` | `parse_fidelity_vs_reference.track.reference_window_differs_from_parsed = 30` with `track.n = 30` | The ``30/30 cases'' in the parser sentence. Oracle-gap numbers (0.538 vs. 0.450, −0.28) are Evaluation's sentence, NOT repeated here. |
| Four-camera deployment / single-camera track (counts, no numbers) | Abstract; Contributions; Conclusion | Plan §0–1 + `VERIFICATION.md` Setup rows | Qualitative scope statement, no numeric claim | Avoids 312,544 / 109 ms / 6.0 s / 12.0 s entirely in framing text. |
| 9-tuple | Abstract; Contributions | System substrate (`INDICON2026Conference.tex` §C1 grammar, plan §5) | Structural definition, not a measurement | Retained from original paper. |

Forbidden-token sweep on §1 text (verified by re-reading the replacement blocks): `0.828` appears only in §1d observation sentence (allowed once); `0.807`, `91%`, `312,544`, `109 ms`, `6.0 s`, `12.0 s` appear nowhere in §1; `verified` nowhere (replaced by `schema-validated` / `deterministic`); `Person Re-ID` nowhere (uses `person descriptor retrieval` / `person descriptor`); `Lemma` nowhere; parser-gap language (`parser`, `parsing`, `oracle-filter`, `error budget`, `bottleneck`, `failure`) appears exactly once in the whole §1 (the future-scope sentence).

---

## 4. Code changes

None. Framing area explicitly disallows code edits.

- Files created: `INDICON2026_submit/area_reports/framing.md` (this report).
- Files changed: none.
- Test commands run: none (no code touched; per constraints, long eval scripts were not re-run — released files were read only).

---

## 5. Risks, blockers, page-budget impact

**Risks / blockers:**
1. No blocker in framing scope. All framing numbers trace to already-released files (`alert_results.md`, `fusion_tables.md`, `retrieval_control_tables.md`, `end_to_end_results.json`).
2. Merge-collision risk (low, managed): Related Work body Re-ID heading and Discussion limitations (v) gender/approval wording belong to C2/Safety — this report deliberately leaves them untouched. Parent must take limitations (v) from the Safety contributor, not from framing.
3. Gate risk for parent checker (`check_paper_numbers.py` / G0): after merge, abstract + contributions + conclusion must contain zero instances of `0.828/0.807/91%/verified/10.7/2.2/Re-ID/Lemma`. The §1 blocks satisfy this; parent must ensure Evaluation/C3 merges do not leak deployment numbers or a second parser sentence into the conclusion.
4. Parser-count gate: final paper must contain at most two parser mentions — Evaluation's one sentence (Evaluation owner) + framing's future-scope sentence (§1g). Parent must reject any C1/C2/C3/abstract/conclusion edit that adds `parser/parsing/oracle` wording.
5. `paper_lookup 0.428` vs `per_intent_tuned 0.536` distinction in `retrieval_control_tables.md` is a known reader-confusion point; framing avoids naming either, using only ``best tune-selected fixed alpha'' (0.538). No action needed.

**Page-budget impact (6-page budget, net saver):**
- Abstract: current ~190 words → proposed ~155 words; saves ~4–6 column lines (removes 91% clause, three probe clauses, Lemma/score-confidence tail, artifact caveat sentences which move to Evaluation).
- Keywords: +0 lines (one-term swap).
- Introduction motivation: current deployment-α table walk (~5 lines) → proposed 3-line control + 1-line deployment observation; roughly neutral, −1 line from the `Re-ID` → `descriptor` swap (shorter).
- Contributions: four items ≈ current length; neutral (±2 lines depending on footnote retention).
- Related Work: one sentence → one sentence; neutral.
- Future scope: new ~3-line paragraph; Conclusion: current 2.5-line conclusion → proposed ~5-line conclusion (+2–3 lines).
- Net estimate: **−3 to −6 column lines** (small net saving). No new section, figure, or table from framing. Frees space for Figure 1 / Table I / deployment-study table owned by others.
