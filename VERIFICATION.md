# Final verification checklist

Run from `S:\college\ai-surveillance` with `$env:PYTHONIOENCODING='utf-8'`.

Status of every item after the revision is recorded in the last column.

## Code and tests

| Item | Status |
|---|---|
| `python -m pytest backend/tests -q` passes | ✅ all tests pass (3 warnings, ~26 s) |
| `import backend.app.db.mongo` returns quickly with MongoDB down and logs a non-fatal warning | ✅ short server-selection timeout, import no longer blocks |
| `import backend.app.main` succeeds | ✅ `ACCESS_CONTROL_ENABLED` is inert by default |
| `python backend/scripts/eval_alerts.py` writes `results/alert_results.json` / `.md` and reports dev and test separately | ✅ test clean 30/30, stress 30/30 clarified, 0 false materializations |
| `python backend/scripts/eval_retrieval.py` writes `results/retrieval_results.json`, `retrieval_tables.md`, `retrieval_per_query.csv`; no synthetic Gaussian scores, no paper-table fixture | ✅ raw released benchmark; 3,000 resamples, grouped bootstrap, Holm |
| `python backend/scripts/eval_retrieval_control.py` writes the protocol-matched control outputs | ✅ adaptive 0.538 = best fixed 0.538 (Δ 0.000, p 1.0) |
| `python backend/scripts/eval_alpha_competitors.py` writes `results/alpha_competitors.json` / `.md` | ✅ learned-alpha competitor 0.534 < 0.538; 16-point soft-weight sweep all 0.538; count EM 0.100 / MAE 2.20 with the label-noise ceiling |
| `python backend/scripts/eval_fusion.py` writes `results/fusion_results.json`, `fusion_tables.md`, `fusion_per_query.csv`; pair split is track/identity-disjoint | ✅ 174 pseudo-tracks, 528 calibration / 1,488 test pairs, thresholds from calibration only |
| `python backend/scripts/measure_latency.py` writes `results/latency_results.json` / `.md`; no simulated lognormal values | ✅ real component timings, p50/p95, 1/2/4/8 workers |
| `python backend/scripts/measure_latency.py --live-llm 3` (optional) | ⚠️ not run; LLM parse latency is excluded and labelled as excluded |
| `python backend/scripts/check_release.py` writes the release manifest | ✅ `results/release_manifest.json` / `.md` |
| `python backend/scripts/eval_end_to_end.py` writes `results/end_to_end_results.json` / `end_to_end_tables.md` from the released scene index and the deployed parser | ✅ real LLM parse of the 150 protocol-matched control queries (150/150, mean 3.0 s, p95 5.0 s); macro P@5 0.450 vs the control's 0.538; Δ vs the tune-selected best fixed-α is +0.002 (paired bootstrap p 0.80, n.s.) |
| `python backend/scripts/eval_end_to_end.py --parse-source oracle` writes `results/end_to_end_oracle_results.json` | ✅ the same harness driven by the reference filter reproduces the control's 0.538 exactly, which localises the 0.088 gap to the parse rather than to the harness |
| `python backend/benchmarks/build_subset.py` and `build_subset_control.py` regenerate the benchmarks deterministically | ✅ seed 20260301; re-running reproduces the same numbers |

## NL parser (provider-agnostic LLM layer)

| Item | Status |
|---|---|
| The parser default is the local opencode runtime and needs no hosted API key | ✅ `LLM_PROVIDER=opencode`, `OPENCODE_MODEL=opencode-go/beam-free` in `backend/app/config.py`; `llm_client.OpencodeBackend` talks to the local runtime, `OPENAI_API_KEY` is not read on that path |
| The opencode CLI transport hands the parser prompt over as an attached file | ✅ `OpencodeBackend._chat_cli` writes the prompt to a temporary `.md` file and passes `opencode run --file <path>`, because the prompt exceeds the Windows command-line length limit; the file is removed again, also on failure |
| The transports degrade instead of failing the parse | ✅ `http` -> `session` -> `cli`; a route the runtime does not expose is remembered as unavailable (`OPENCODE_TRANSPORT_RETRY_SECONDS`) and the next transport is used |
| The parser prompt never reaches the model with an unfilled placeholder | ✅ `nl_parser.build_parser_prompt` renders `{query}`, `{schema}`, `{format_instructions}` and `{reference_date}` and raises on a leftover placeholder, so relative dates resolve against `DATA_EPOCH_DATE` |
| The end-to-end released-subset run is reproducible from the parse cache | ✅ parses are cached in `backend/benchmarks/parsed_cache/<model>.json` and saved after every query; a re-run reuses them, `--refresh` re-parses |
| The free-tier quota limit is a known environment constraint, not a paper dependency | ✅ documented below; no paper number depends on a provider quota |
| The hosted path still works when it is asked for explicitly | ✅ `LLM_PROVIDER=openai` keeps the LangChain/OpenAI route and defaults to `gpt-4o-mini`; every provider has its own default model |

## Paper/result consistency

| Item | Status |
|---|---|
| Every subset number in the tex appears in a `results/*` JSON/CSV with the same value | ✅ checked against `retrieval_control_results.json`, `fusion_results.json`, `alert_results.json`, `latency_results.json` |
| Full-deployment headline numbers remain labelled as full deployment and are not replaced by subset numbers | ✅ 0.666→0.828, 0.785→0.807, 91%, 6 s/12 s kept as primary |
| No claim that the subset reproduces 4-camera/312k, human IAA, cross-camera Re-ID or full-deployment latency percentiles | ✅ stated explicitly in Setup, Limitations and Conclusion |
| C3 no longer says the materializer depends on the LLM | ✅ rewritten to the deterministic `alert_materializer.py` + clarify/abstain validator |
| No "measured results will be reported" style placeholder remains | ✅ verified in the extracted PDF text |
| The 91% alert number is labelled as the deployment study, with the released coverage check reported next to it | ✅ 30/30 clean, 30/30 stress; the two are not merged |
| Latency claims separate the deployment means from the released-subset p50/p95 | ✅ 6.0 s/12.0 s vs 37.4/39.7 ms and 17.2/20.4 ms |
| Privacy claims match the implemented switches and their defaults | ✅ all opt-in and disabled by default |
| The released C2 track is reported as not reproducing the deployment ordering | ✅ stated in the paper (subset scope sentence) with the visual-only bias caveat |
| The released C1 control is labelled as a post-hoc protocol-matched control | ✅ stated in the paper, `results/README.md` and `backend/benchmarks/README.md` |
| The contradicting raw-benchmark ordering is disclosed, not only the favourable control | ✅ the raw released benchmark (adaptive 0.486 vs best fixed 0.568, action best at alpha=1.0) is named in the paper next to the post-hoc control |
| The end-to-end track (the parser's own filter, not the metadata reference) is reported with the oracle gap attributed | ✅ `results/end_to_end_results.json`: macro P@5 0.450 vs the control's 0.538, gap attributed to the label-side attribute list and the reference time window |
| The NL parser runs without a hosted key | ✅ `LLM_PROVIDER=opencode` with `opencode-go/beam-free`; 150/150 released queries parsed (mean 3.0 s, p95 5.0 s) |

## LaTeX

| Item | Status |
|---|---|
| `pdflatex` + `bibtex` complete with no undefined references/citations | ✅ |
| PDF is at most 8 pages (6 + 2 overlength) | ✅ 8 pages |
| No overfull hbox | ✅ none remaining (the 35.5 pt box and the 3.1 pt author-block boxes are gone) |
| Tables/figures fit the column width; no missing figure files | ✅ |
| `INDICON2026Conference.pdf` regenerated from the final tex | ✅ |

## INDICON 2026 template conformance

| Item | Status |
|---|---|
| Class file | ✅ `IEEEtran` V1.8b — the kit's `IEEEtran.cls` is byte-identical (SHA256 `DA751920…`) to the one used for the build |
| Preamble | ✅ template preamble kept verbatim (`\documentclass[conference]{IEEEtran}`, `\IEEEoverridecommandlockouts`, `cite`, `amsmath`, `algorithmic`, `graphicx`, `textcomp`, `xcolor`, `fancyhdr`, `hyperref`, `\def\BibTeX{…}`) plus the packages the paper needs (`booktabs`, `multirow`, `array`, `makecell`, `pifont`, `amsthm`, `float`, `tikz`) |
| First-page header/footer | ✅ identical to the kit: `23rd Edition of IEEE India Council International Conference (INDICON 2026)`, `979-8-3195-2581-9/26/$31.00 ©2026 IEEE`, footer `INDICON 2026`, `\thispagestyle{firstpage}` |
| Author block | ✅ per-author `\IEEEauthorblockN` + `\IEEEauthorblockA` with `\and` separators, one affiliation per author, per-author e-mail, corresponding author marked `\textsuperscript{*}` |
| Keywords | ✅ `IEEEkeywords` environment, 6 index terms, all inside the first column |
| Bibliography | ✅ `\bibliographystyle{IEEEtran}` + `\bibliography{References}`, 15 entries, all cited |
| No external assets | ✅ the dataflow figure is an inline TikZ picture, so the source bundle is self-contained |

## Submission package

| Artefact | Status |
|---|---|
| `INDICON2026_submit/INDICON2026_850.pdf` | ✅ final camera-ready PDF, 8 pages |
| `INDICON2026_submit/INDICON2026_850_source.zip` | ✅ `INDICON2026Conference.tex` + `References.bib` + the kit's `IEEEtran.cls` + a compile README; **verified to compile standalone in a clean directory to the identical 8-page PDF** |
| `INDICON2026_submit.zip` | ✅ folder package with the PDF, the source bundle, the `.tex` and the `.bib` |
| `dist/indicon2026_supplementary.zip` | ✅ code + benchmarks + results + paper source (1.3 MB); `--with-data` variant adds clips and FAISS indices (18.3 MB) |
| Build scripts | `backend/scripts/package_submission.py`, `backend/scripts/package_release.py` (re-runnable) |

## Independent final review (Muse Spark 1.3 Contributor, xhigh)

An independent reviewer agent audited the finished paper in its own worktree
(`review/indicon-2026-final-paper`) and committed its fixes as `fcf0f37`
(applied on `Enhance-chat-and-retrieval-processes`). Verdict:
**accept with minor fixes**; every fix is merged and re-verified.

| Finding | Severity | Resolution |
|---|---|---|
| `§C3` heading had a mojibake em-dash from a Windows encoding round-trip (rendered as "C3 A-third-party-char alert authoring") | blocker | fixed to `C3 -- alert authoring`; the PDF now renders "E. C3 – alert authoring"; a repo-wide scan of 272 tracked text files finds no remaining mojibake, and the `.tex` is pure ASCII with no BOM |
| Limitations (iii) still asserted an LLM alert extractor and a "6-pp drop with local 7B models" that no artefact supports | major | reworded: the 0.91 deployment result is the recorded measurement of a held-out study whose per-utterance export was not preserved; the 0.91 claim itself is unchanged |
| `RESPONSE_TO_REVIEWERS.md` R1-4 claimed the paper reports the z-score variant; the paper only states per-query min-max normalisation | major | corrected: the paired min-max/z-score deltas are reported in `results/retrieval_control_tables.md` |
| "14 soft-weight triples" in two docs; the sweep has 16 points | major (docs) | corrected to 16 in `RESPONSE_TO_REVIEWERS.md` and `results/README.md`; the paper already said 16 |
| raw-benchmark "best fixed 0.568" was ambiguous (tune-selected vs oracle 0.578) | minor | reworded in `PAPER_CHANGES.md` |
| conclusion said the deployment numbers "can be checked" | minor | reworded to "the code paths behind the deployment numbers can be exercised end-to-end" |

The reviewer also re-ran every offline check from a clean state (63 tests, all
six evaluation scripts, the release manifest, the LaTeX build and a clean-room
build of the source bundle), reproduced every quoted subset number, confirmed
that the deployment-only fusion ordering is absent from the PDF text, and
confirmed that the release contains no stale drafts. Its explicit
unverifiable list (all deployment-only measurements, human IAA, cross-camera
Re-ID, cross-site/night-rain generalisation, 312k latency percentiles) matches
the limitations printed in the paper.

## Repository

| Item | Status |
|---|---|
| `git diff` contains no accidental deletion of user work | ✅ reviewed before commit |
| `results/` files committed | ✅ |
| Final commit message summarises the revision and test status | ✅ |

## Cannot be produced with the shipped artifact (state, do not fabricate)

- Human inter-annotator agreement for the deployment benchmark.
- The deployment per-query export, so grouped resampling cannot be recomputed
  for the 4-camera/312k numbers.
- Cross-camera Re-ID, cross-site / night / rain generalisation.
- 312k-descriptor latency percentiles.
- A re-run of the deployment 150-query benchmark: its query file, annotator
  roster and tune/test partition were not preserved.
- The provider's free-tier quota: the free tier of
  `opencode/deepseek-v4.1-flash-free` is quota-limited, so a parse run can
  stop with a provider usage-limit error.  The default
  `opencode-go/beam-free` and the first fallback
  `opencode-go/minimax-m3.1-flash-free` are not quota-limited, and
  `opencode-go/deepseek-v4.1-flash` is a paid tier.  This is a property of the
  provider account, not of the released artifact: it decides which model
  answers a query, not any number in the paper.
- The end-to-end released-subset track does not reproduce the deployment
  benchmark: it is single-camera (123 clips, `camera_0`, 242 scene frames,
  1,778 person descriptors).
