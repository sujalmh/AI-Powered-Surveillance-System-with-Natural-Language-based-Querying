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
| E6 | No released artefact contains a bare `0.0000` p-value or a `± 0.000` SD | **OPEN** | verified after E2–E5 land |

## E-bis. Defects found while re-verifying the release

| # | Item | Status | Where |
|---|---|---|---|
| E7 | **The paper mis-attributed the parser model.** It said GPT-4o-mini produced "the measurements reported here", but the released end-to-end track was produced by `opencode-go/beam-free`. The sentence now names GPT-4o-mini for the deployment study and `opencode-go/beam-free` for the released track, and the End-to-end paragraph names the model explicitly. | **DONE** | `INDICON2026Conference.tex` (§III and the Released-subset paragraph) |
| E8 | **`--model` never reached the backend.** `eval_end_to_end.py` set only `NL_DEFAULT_MODEL`, but `build_backend()` reads the active configuration, which prefers `OPENCODE_MODEL`. A run with `--model X` therefore parsed with the configured primary while recording `X` in the result file. New `Settings.set_opencode_model()` sets both names and invalidates the cached MongoDB lookup; a regression test pins it. | **DONE** | `backend/app/config.py`, `backend/scripts/eval_end_to_end.py`, `backend/tests/test_llm_client.py` |
| E9 | The two poisoned parse caches (`..._minimax-m3.1-flash-free.json`, `..._deepseek-v4.1-flash-free.json`) written by the mislabelled run were deleted; only the beam-free cache remains. | **DONE** | `backend/benchmarks/parsed_cache/` |
| E10 | The opencode free DeepSeek tier is quota-blocked (`Free usage limit reached`); `opencode-go/deepseek-v4.1-flash` (paid) and the two free models work. Recorded so no paper number depends on it. | **DONE** | `backend/.env`, `results/README.md` |
| E11 | A cross-model replication of the released end-to-end track with a second free model (`opencode-go/minimax-m3.1-flash-free`) | **IN PROGRESS** | `results/end_to_end_minimax_results.json` |


## D. Release integrity

| # | Item | Status |
|---|---|---|
| D1 | Paper: 8 pages, 0 overfull boxes, 0 undefined references, 0 errors | **DONE** |
| D2 | Source bundle compiles standalone to an 8-page PDF | **DONE** |
| D3 | Camera-ready PDF + source zip + folder zip regenerated from the final source | **DONE** |
| D4 | Release bundle: no secrets, no parse caches, new modules and results included | **DONE** |
| D5 | `check_release.py` verifies all of the above (11 checks) | **DONE** |
| D6 | Parse caches git-ignored as regenerable artefacts | **DONE** |
| D7 | Full test suite green (179 tests) | **DONE** |
| D8 | Response-to-reviewers letter records the audit corrections | **DONE** |

---

## OPEN — remaining work

| # | Item | Owner | Notes |
|---|---|---|---|
| O1 | **Decide the public repository.** The paper now says "the public repository URL is withheld from this preprint and will be given on acceptance". If the repo is to be public under a different name, update the footnote, the abstract and `VERIFICATION.md`; the cited URL `github.com/sujalmh/intent-adaptive-hybrid-surveillance` currently returns 404. | human | Blocking for a "released" claim |
| O2 | **ISBN / price footer.** `INDICON2026Conference.tex` line ~38 carries the kit's placeholder `979-8-3195-2581-9/26/$31.00`. It is verbatim from the author kit, but the organisers may require it removed or replaced. | human | Template boilerplate |
| O3 | **Single vs double blind.** The paper carries named authors. Confirm the track; if double-blind, the author block and the acknowledgements need anonymising. | human | Affects the PDF |
| O4 | **ORCID / contact metadata** for the submission portal. | human | Not in the PDF |
| O5 | **Re-run the parser tracks after the opencode app re-authenticates.** The app lost its provider session when the T3 server restarted mid-run; all shipped numbers were measured before that and are in the cache, but a fresh `eval_end_to_end.py --refresh` should be run once the runtime accepts requests again to confirm reproducibility end-to-end. | me (blocked on auth) | 150 parses ≈ 8 min |
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
python -m pytest backend/tests -q                 # 179 passed
python backend/scripts/check_release.py          # 11/11 bundle checks
# paper: 8 pages, 0 Overfull, 0 undefined
Select-String -Path INDICON2026_submit\INDICON2026Conference.log -Pattern 'Output written on'
```
