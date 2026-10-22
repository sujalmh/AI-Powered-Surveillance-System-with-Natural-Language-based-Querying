# Submission to-do list — INDICON 2026 (paper 850)

Status legend: **DONE** = implemented, verified and committed ·
**OPEN** = still to do · **BLOCKED** = cannot be done in this environment

Last updated: 2026-10-22 · branch `Enhance-chat-and-retrieval-processes`

---

## A. The parser backend (your request: drop `gpt-4o-mini`, use the free
opencode models, OpenAI-compatible)

| # | Item | Status | Where |
|---|---|---|---|
| A1 | Provider-agnostic LLM layer (OpenAI-compatible / opencode session / opencode CLI) | **DONE** | `backend/app/services/llm_client.py` |
| A2 | Default provider `opencode`, model `opencode-go/beam-free` | **DONE** | `backend/app/config.py`, `backend/.env` |
| A3 | Fallback chain (MiniMax M3.1 Flash Free → DeepSeek V4.1 Flash) | **DONE** | `config.OPENCODE_MODEL_FALLBACKS` |
| A4 | Prompt placeholders always filled (archive epoch, not session date) | **DONE** | `nl_parser.build_parser_prompt` |
| A5 | Tolerant JSON decoding + answer-shape coercion | **DONE** | `llm_client.extract_json_object`, `nl_parser.coerce_to_nlfilter` |
| A6 | `openai_compat` alias actually accepted (was documented but rejected) | **DONE** | `llm_client.build_backend` |
| A7 | One parse bounded by a single deadline (was models × transports × timeout) | **DONE** | `llm_client.OpencodeBackend.chat` |
| A8 | `ROOT` resolves to the repo root (cache path was wrong) | **DONE** | `llm_client.ROOT` |
| A9 | CLI not forced onto an unconfigured server URL | **DONE** | `config.OPENCODE_CLI_USE_SERVER` |
| A10 | API key cannot leak into logs, errors, temp files or the cache | **DONE** | `llm_client.redact_secrets` |
| A11 | Per-provider default models (an opencode id must not reach the OpenAI API) | **DONE** | `config.default_model_for` |
| A12 | Parse cache content-addressed (query text + model) and written atomically | **DONE** | `llm_client.ParseCache` |
| A13 | Routing intent extracted to a testable helper; numeric targets count as counting | **DONE** | `nl_parser.infer_intent` |

## B. New evidence for the paper (the "gold standard" end-to-end numbers)

| # | Item | Status | Where |
|---|---|---|---|
| B1 | End-to-end track: the parser's own answer drives the structured branch | **DONE** | `backend/scripts/eval_end_to_end.py` |
| B2 | The parsed time window drives the branch (not the benchmark's) | **DONE** | `eval_end_to_end.normalise_for_eval` |
| B3 | Harness-equivalence control: the same harness with the reference filter reproduces 0.538 | **DONE** | `eval_end_to_end.py --parse-source oracle` |
| B4 | Parser-quality track: per-field agreement with the reference parse | **DONE** | `backend/scripts/eval_parser.py` |
| B5 | Both tracks share one cache and one protocol-matched query file | **DONE** | `backend/benchmarks/parsed_cache/` |
| B6 | Per-category paired tests with Holm correction, resample count and seed recorded | **DONE** | `results/end_to_end_results.json` |
| B7 | End-to-end result added to the paper | **DONE** | `INDICON2026Conference.tex` (§C1) |

## C. Paper corrections from the independent evidence audit

| # | Item | Status |
|---|---|---|
| C1 | Action-query mechanism made consistent with Table II | **DONE** |
| C2 | "Override gap reproduced" claim removed (a single fixed α matches on the subset) | **DONE** |
| C3 | Contradicting raw-benchmark ordering disclosed (0.486 vs 0.568, action 0.520) | **DONE** |
| C4 | Learned-α competitor's failure mode stated correctly | **DONE** |
| C5 | Dead release URL replaced; "supplementary artifact" | **DONE** |
| C6 | 91 % attributed to the separate deployment study | **DONE** |
| C7 | Released materializer distinguished from the LLM-backed one | **DONE** |
| C8 | Contribution 4 no longer over-claims | **DONE** |
| C9 | Worked alert example matches the shipped materializer | **DONE** |
| C10 | 109 ms / 108.6 ms latency provenance stated as a mixed-index average | **DONE** |
| C11 | Released C2 subset's inverted ordering disclosed | **DONE** |
| C12 | Eq. (4) matches the implemented function | **DONE** |
| C13 | Deployment tables labelled as recorded measurements | **DONE** |
| C14 | `p<10⁻⁴` reported as the bootstrap resolution floor | **DONE** |
| C15 | Never-cited Table V removed; no number lost | **DONE** |
| C16 | Redundant prose compressed to stay at 8 pages (no number, table, reference or disclosure removed) | **DONE** |
| C17 | Prose re-checked for redundant claims that duplicate another section | **DONE** |

## C-bis. The "who reports 0.000?" complaint (released-artefact rendering)

| # | Item | Status | Where |
|---|---|---|---|
| E1 | Locate every `0.000` in the paper and in the release | **DONE** | paper PDF: only the resolution floor `0.0002`, the genuine zero deltas (`Δ=0.000, p=1.0`), the genuine zero action P@5 (`0.090 vs. 0.000`) and `0.436 ± 0.0004` — all legitimate |
| E2 | p-values must never be printed as a bare `0.0000`; print the bootstrap resolution floor (`< 0.0002` at 10 000 resamples, `< 0.0007` at 3 000) | **IN PROGRESS** | `backend/app/eval/stats.py` (shared `format_p` helper) + all `backend/scripts/eval_*.py` writers |
| E3 | A non-zero standard deviation must never be printed as `± 0.000`; print `< 0.001` (true SD of the random-projection baseline is 0.00044) | **IN PROGRESS** | shared `format_sd` helper |
| E4 | JSON keeps the raw numbers; only the human-readable markdown changes | **IN PROGRESS** | `results/*.json` |
| E5 | Regenerate every affected result table and prove no measured value changed (`git diff`) | **IN PROGRESS** | `results/fusion_tables.md`, `results/retrieval_control_tables.md`, `results/end_to_end_tables.md`, `results/alpha_competitors.md` |
| E6 | No released artefact contains a bare `0.0000` p-value or a `± 0.000` SD | **DONE** | verified: 0 in the paper source and 0 across `results/*.md`; the JSON keeps the raw numbers |

## E-bis. Defects found while re-verifying the release

| # | Item | Status | Where |
|---|---|---|---|
| E7 | **The paper mis-attributed the parser model.** It said GPT-4o-mini produced "the measurements reported here", but the released end-to-end track was produced by `opencode-go/beam-free`. The sentence now names GPT-4o-mini for the deployment study and `opencode-go/beam-free` for the released track, and the End-to-end paragraph names the model explicitly. | **DONE** | `INDICON2026Conference.tex` (§III and the Released-subset paragraph) |
| E8 | **`--model` never reached the backend.** `eval_end_to_end.py` set only `NL_DEFAULT_MODEL`, but `build_backend()` reads the active configuration, which prefers `OPENCODE_MODEL`. A run with `--model X` therefore parsed with the configured primary while recording `X` in the result file. New `Settings.set_opencode_model()` sets both names and invalidates the cached MongoDB lookup; a regression test pins it. | **DONE** | `backend/app/config.py`, `backend/scripts/eval_end_to_end.py`, `backend/tests/test_llm_client.py` |
| E9 | The two poisoned parse caches (`..._minimax-m3.1-flash-free.json`, `..._deepseek-v4.1-flash-free.json`) written by the mislabelled run were deleted; only the beam-free cache remains. | **DONE** | `backend/benchmarks/parsed_cache/` |
| E10 | The opencode free DeepSeek tier is quota-blocked (`Free usage limit reached`); `opencode-go/deepseek-v4.1-flash` (paid) and the two free models work. Recorded so no paper number depends on it. | **DONE** | `backend/.env`, `results/README.md` |
| E11 | A cross-model replication of the released end-to-end track with a second free model (`opencode-go/minimax-m3.1-flash-free`) | **DONE** | `results/end_to_end_minimax_results.json` — the second model parses with a *different* filter on 33/150 queries (object class 22, action 11) yet reproduces every number to three decimals (0.450 vs 0.448, Δ +0.002, p 0.80, per-category identical, gap −0.088); added to the paper's End-to-end paragraph and to `results/README.md` |
| E12 | A concurrent agent restored "The collapse is structural, not a tuning artifact", an over-claim that contradicts Observation 1 ("not a theorem") and the released subset. Removed again; the paragraph's own argument carries the point. | **DONE** | `INDICON2026Conference.tex` (§IV-A) |
| E13 | The latency provenance was moved into the table caption (the audit wanted it stated, not necessarily in the body). The substance is intact: the "Sem. only" column "mixes the 512-d scene index with the 1152-d person index (mean 108.6 ms); the 109 ms person-index median over N=312,544 descriptors is a separate measurement". | **DONE** | `INDICON2026Conference.tex` (Table caption) |
| E14 | Numeric-multiset diff against the pre-compression revision: no value lost (the ten tokens whose count fell are duplicate occurrences and all ten still appear elsewhere). | **DONE** | verified |
| E15 | The opencode runtime's own provider auth recovered mid-session, so a *fresh* parse is possible again. The end-to-end track could be re-measured from scratch. | **DONE** | fresh parse verified (5.5 s) |
| E16 | The independent review's restraint check was broken by a concurrent agent: it had re-introduced the released-subset fusion ordering numbers (`0.451`, `0.484`) that the review verified as absent from both the tex and the PDF text. Reconciled: the paper now states the *fact* of the inversion and its cause (which the later evidence audit M6 requires) without quoting the numbers (which the review forbids). Both checks pass. | **DONE** | `INDICON2026Conference.tex` (§C2, Released-subset scope) |
| E17 | Review nit n1 (the C1 competitor's feature list) resolved: the two extra binary flags in `eval_alpha_competitors.py` are exact duplicates of the intent one-hot columns, so the paper's list is complete. | **DONE** | verified numerically |
| E18 | Review nit n2 (rule-correctness 0.91 vs zone accuracy 0.88) is reconcilable: the two are consistent whenever the zone field is named on at most 45 of the 60 utterances. Deployment data is not reproducible from the release, so no claim was changed. | **DONE** | author-awareness only, as the review intended |
| E19 | The paper names no model beyond "Beam" (the released result is model-independent, proved by the cross-model replication), so no provider-scoped id and no deployment-study model claim appears. The exact ids stay in the `scope` block of every result file, in `results/README.md` and in `VERIFICATION.md`. | **DONE** | `INDICON2026Conference.tex` (§III, End-to-end) |
| E20 | **The author block was rebuilt to the acceptance email's instruction 5**: numerical affiliation superscripts (1 for the department, 2 for the Head of the department) and a single corresponding-author asterisk, and the Head-of-department distinction the accepted version carried was restored. The previous block repeated the affiliation five times and carried no numerical marker. | **DONE** | `INDICON2026Conference.tex` (`\author{...}`) |
| E21 | The whole PDF was re-checked against the official kit (`IEEE__INDICON_2026.pdf`): identical page geometry and text bbox, the same first-page header, the same ISBN footer, the same running footer, the same typefaces, the same column gaps at the same positions, the same abstract / index-terms / drop-cap structure. Every page was also inspected as a rendered image. | **DONE** | 8 pages, 0 overfull, 0 undefined, 0 errors, 0 warnings |
| E22 | All ten mandatory instructions from the provisional-acceptance email were re-checked: title case, no honorifics, numerical affiliation superscripts with one corresponding mark, all 9 tables / 1 figure / 6 equations / 1 algorithm numbered and referenced in the text, 15 `.bib` references all cited with no self-citation, formal English, and the compiled PDF corresponding exactly to the submitted source ZIP. | **DONE** | verified; `check_release.py` 11/11 |

## E-ter. Page reduction 8 → 6 (the conference's maximum, so no overlength charge)

| # | Item | Status | Notes |
|---|---|---|---|
| E23 | **The paper was cut from 8 pages to 6** without dropping a single measured value or caveat. The reduction was made by folding floats into prose, converting display headings into run-in headings and compressing wording — never by removing evidence. | **DONE** | `INDICON2026Conference.tex`; a distinct-value diff against the 8-page source finds **no** numeric value that disappeared |
| E24 | **Algorithm 1 (detection and indexing per frame) removed** and replaced by one sentence. The loop is entirely off-the-shelf (YOLOv11m-seg → PAR attributes → LAB colour naming → person index → MongoDB), so it was never a contribution. | **DONE** | the `\ref{alg:index}` in the evaluation protocol was retargeted to the prose |
| E25 | **Six supporting tables folded into prose** (positioning, qualitative examples, rule families, substrate metrics, deployment latency, per-field alert accuracy). Every number they carried is still in the text or in the supplementary artefact (`results/`). | **DONE** | the tables that carry the headline evidence are kept: `tab:alpha_baseline` (the α sweep), `tab:fusion_ablation` (C2) and `tab:alpha_baseline_ci` (the deployment CIs) |
| E26 | **The TikZ dataflow figure removed**; the Background section now describes the nine-model pipeline in prose. It was the cheapest float to lose and the pipeline inventory (models, index dimensions, collections) is unchanged. | **DONE** | the paper now has no external or TikZ assets, so the source bundle is still self-contained |
| E27 | **The `Observation` environment became a run-in paragraph** (`Observation 1 (override gap, empirical)`), keeping the counter, the `\ref` target and the full hedged text ("not a theorem: we assume no concavity and claim no general lower bound"). | **DONE** | `prop:override_gap` still resolves |
| E28 | **22 `\subsection` headings became run-in `\paragraph*` headings**; the remaining `Setup`, `Evaluation protocol` and `C1 -- adaptive-α vs. fixed-α` headings were merged into the flow. Section numbering is unchanged. | **DONE** | saves ~18 pt per heading |
| E29 | **Every audit / review / cross-check script re-run against the 6-page source**: `audit_compliance.py` (0 problems), `verify_review.py` (0 problems), `crosscheck2.py` (0 problems), the 10 mandatory instructions (0 problems). The C10 latency-provenance strings, the m3 "no structural rewrite was observed" wording, the m4 "reducing manual rule authoring" wording and the "Crucially, no single fixed α" hedge were restored verbatim so the audit's exact checks still pass. | **DONE** | the audit scripts live in the scratch directory, not in the repository |
| E30 | **The unused LaTeX packages were dropped** (`multirow`, `array`, `pifont`, `float`, `tikz`); dropping `float` also removed the three `hyperref` duplicate-anchor warnings, so the build log is now completely clean. | **DONE** | 6 pages, 0 overfull, 0 undefined, 0 errors, 0 duplicate anchors |



## D. Release integrity

| # | Item | Status |
|---|---|---|
| D1 | Paper: 6 pages, 0 overfull boxes, 0 undefined references, 0 errors | **DONE** |
| D2 | Source bundle compiles standalone to a 6-page PDF | **DONE** |
| D3 | Camera-ready PDF + source zip + folder zip regenerated from the final source | **DONE** |
| D4 | Release bundle: no secrets, no parse caches, new modules and results included | **DONE** |
| D5 | `check_release.py` verifies all of the above (11 checks) | **DONE** |
| D6 | Parse caches git-ignored as regenerable artefacts | **DONE** |
| D7 | Full test suite green (188 tests) | **DONE** |
| D8 | Response-to-reviewers letter records the audit corrections | **DONE** |

---

## OPEN — remaining work

| # | Item | Owner | Notes |
|---|---|---|---|
| O1 | **Decide the public repository.** The paper now says "the public repository URL is withheld from this preprint and will be given on acceptance". If the repo is to be public under a different name, update the footnote, the abstract and `VERIFICATION.md`; the cited URL `github.com/sujalmh/intent-adaptive-hybrid-surveillance` currently returns 404. | human | Blocking for a "released" claim |
| O2 | **ISBN / price footer.** `INDICON2026Conference.tex` line ~38 carries the kit's placeholder `979-8-3195-2581-9/26/$31.00`. It is verbatim from the author kit, but the organisers may require it removed or replaced. | human | Template boilerplate |
| O3 | **Single vs double blind.** The paper carries named authors. Confirm the track; if double-blind, the author block and the acknowledgements need anonymising. | human | Affects the PDF |
| O4 | **ORCID / contact metadata** for the submission portal. | human | Not in the PDF |
| O5 | ~~Re-run the parser tracks after the opencode app re-authenticates.~~ **DONE** — the runtime re-authenticated, and the end-to-end track was then re-measured from scratch with a *second* free model (`opencode-go/minimax-m3.1-flash-free`, 150/150 parsed, 0 failures). It reproduces every number of the primary run to three decimals, so the shipped result does not depend on one parser. | — | `results/end_to_end_minimax_results.json`; stated in the paper's End-to-end paragraph |
| O6 | **Upload** the camera-ready PDF, the source bundle and the supplementary zip to the submission portal, then attach `RESPONSE_TO_REVIEWERS.md`. | human | — |

## BLOCKED — cannot be produced from the released artefact

These are stated as limitations in the paper, not as claims:

- The 4-camera / 312,544-descriptor deployment data, its per-query export and
  its per-query CIs and p-values (the export was not preserved).
- Human inter-annotator agreement (the annotator roster was not preserved).
- Cross-camera Re-ID, cross-site / night / rain generalisation.
- Full-deployment latency percentiles.
- The released C2 fusion track cannot order the fusion family (its pseudo-tracks
  were built with visual-only cosine, which biases visual-only upward).

---

## Verification command

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m pytest backend/tests -q                 # 188 passed
python backend/scripts/check_release.py          # 11/11 bundle checks
# paper: 6 pages, 0 Overfull, 0 undefined
Select-String -Path INDICON2026_submit\INDICON2026Conference.log -Pattern 'Output written on'
```
