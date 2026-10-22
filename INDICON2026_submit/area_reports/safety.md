# Area report — safety (responsible-deployment defaults: E4–E6)

Scope: code + tests for gender-off default, approval gate, MMR-off default;
paper-ready sentences only (no edit to `INDICON2026_submit/INDICON2026Conference.tex`;
parent merges). No parser-gap language in this report's paper text (that budget
belongs to Evaluation + future scope, owned elsewhere). No per-group fairness
claim. No new section/table/figure.

Source of truth: `INDICON2026_submit/REVISION_PLAN_850.md` §§7 (E4–E6), 6
(Discussion), 8 (G8). Supporting evidence files: `PAPER_EVIDENCE_AUDIT.md`,
`VERIFICATION.md` (repo root).

---

## 1. Exact LaTeX-ready replacement text (for parent merge)

### 1a. Limitations, privacy/fairness item (Discussion, limitation (v)) — REPLACE the existing gender/approval sentences with:

```latex
Gender is excluded from the person descriptor by default: stored
attribute texts are rebuilt without gender terms and new captions omit
them; no per-group error rate is claimed. Consequential alerts are staged
for operator approval by default: the rule schema carries a
\texttt{requires\_approval} flag alongside the 9-tuple, the engine writes a
\texttt{pending\_approval} log and runs no side effects until approval, and
the artifact exposes the flag with an approve/reject path. Access-control,
audit and retention switches are supplied but off by default.
```

Notes: uses "person descriptor", never "Person Re-ID"; no deployment
numbers; no `verified`; no fairness claim; 9-tuple stays the 9-tuple (the
flag rides alongside, it is not a tenth field).

### 1b. MMR default (C3 post-ranking line / limitation (ii)) — proposed single sentence (coordinate with C3 owner to avoid duplication):

```latex
Recency is inert on the released control; MMR at $\lambda{=}0.3$ costs
0.138 P@5 (0.538$\rightarrow$0.400), so it is off by default and reported
as an ablation.
```

This is plan §6-C3 verbatim; numbers map to `results/ablation_tables.md`
(base 0.538, MMR-only 0.400, $\Delta -0.138$, $p{<}0.0007$).

## 2. Table/figure LaTeX

None. No new table or figure; the released ablation table
(`results/ablation_tables.md`) is the evidence and is unchanged.

## 3. Number → evidence-source map

| # | Paper/code claim | Value | Source (file + key/row) |
|---|---|---:|---|
| 1 | Gender default | off (`false`) | `backend/app/config.py` (`INCLUDE_GENDER_ATTRIBUTES`, default `"false"`); test `test_gender_attributes_default_off` |
| 2 | Gender rows stripped in rebuild | 418 → 0 of 1,778 | `results/safety_run/person_rebuild_report.json`: `gender_rows_before: 418`, `gender_rows_after_strip: 0`, `texts_changed_by_strip: 418`, `n_total: 1778` |
| 3 | Reconstruction fidelity | max diff 2.09e-07 < 1e-4 | same report: `reconstruction_max_abs_diff: 2.086e-07`, `reconstruction_ok_tol_1e_4: true`; degenerate stored rows `[1, 143]` (both blocks zero) carried, not dropped |
| 4 | New gender-free index | 1,778 rows, 1152-d, L2 unit, 0 gender hits | `data/index/faiss_person_nogender/{index.bin,meta.json}` (verified: ntotal 1778, dim 1152, norms 0.9999974–0.9999987, regex hits 0); note: `data/index/` is gitignored — reproduce with §4 command |
| 5 | Rebuild drift (old vs new vectors) | mean cosine 0.9967 (unchanged-text rows 0.9993, changed rows 0.9886) | same report: `drift.*` |
| 6 | Approval default | true for crowd/loiter/fight/enter/zone; false for count/time | `backend/app/services/alert_validator.py`: `CONSEQUENTIAL_EVENTS`, `default_requires_approval`; tests `test_consequential_events_default_to_approval`, `test_validator_accepts_missing_flag_and_rejects_silent_opt_out` |
| 7 | Opt-out rule | `requires_approval=false` without `approval_authorized` is rejected | `alert_validator.py` (`requires_approval=false needs explicit approval_authorized flag`); engine `_fire_or_stage` + `approve_alert`/`reject_alert`; no-side-effect proof `test_no_side_effect_before_approval_then_fires_after` (1 pending log, 0 normal logs pre-approval; 1 normal log post-approval) |
| 8 | MMR default | off (`ENABLE_RESULT_DIVERSITY=false`), option kept (`MMR_DIVERSITY_LAMBDA=0.3`) | `backend/app/config.py`; test `test_mmr_diversity_default_off_but_option_kept`; check `results/safety_run/mmr_default_check.json` |
| 9 | MMR ablation | 0.538 → 0.400, Δ −0.138, p<0.0007; recency-only 0.538 = base | `results/ablation_results.json`: `summary["base (no recency, no MMR)"].macro 0.538`, `summary["MMR only"].macro 0.400`, `significance_vs_base["MMR only"] delta_p5 −0.138`; `results/ablation_tables.md` rows; therefore default (recency-on + MMR-off) reproduces the 0.538 control |
| 10 | End-to-end unaffected | 0.450 vs 0.448 stands | `backend/scripts/eval_end_to_end.py` ll.42, 563–564 (plain alpha blend, no MMR/post-ranking); `results/end_to_end_tables.md` test macro row (adaptive 0.450); `results/retrieval_control_tables.md` per-category table (adaptive 0.538 = fixed_0.7 0.538) |

No deployment numbers (0.828/0.807/91%/312,544/109 ms/6.0 s/12.0 s) appear in
§1; none were touched.

## 4. Code changes (only allowed paths) + test commands/results

Changed:

- `backend/app/config.py` — `INCLUDE_GENDER_ATTRIBUTES` default `"true"`→`"false"`;
  `ENABLE_RESULT_DIVERSITY` default `"true"`→`"false"` (comment updates only).
- `backend/app/services/alert_validator.py` — added `CONSEQUENTIAL_EVENTS`,
  `APPROVAL_AUTH_FLAG="approval_authorized"`, `default_requires_approval()`;
  `validate_rule` rejects non-boolean `requires_approval` and
  `requires_approval=false` without `approval_authorized`; missing flag keeps
  prior validity (backward compatible).
- `backend/app/services/alert_engine.py` — added `rule_requires_approval()`,
  `is_alert_approved()`, `_insert_pending_approval_log()`, `_fire_or_stage()`
  (gate: pending log, cooldown-rate-limited, zero normal logs pre-approval),
  `approve_alert()`/`reject_alert()`; `evaluate_rule` returns
  `status: pending_approval` when gated; all four `evaluate_realtime` firing
  branches route through `_fire_or_stage`.

Created:

- `backend/scripts/rebuild_person_index.py` (new) — verify/rebuild CLI.
  Exact commands run:
  `python backend/scripts/rebuild_person_index.py --verify-only` →
  `recon_max_diff=2.09e-07 ok=True; gender rows 418 -> 0`;
  `python backend/scripts/rebuild_person_index.py` → full 1,778-row rebuild,
  wrote `data/index/faiss_person_nogender/{index.bin,meta.json}` +
  `results/safety_run/person_rebuild_report.json`.
- `backend/tests/test_safety_defaults.py` (new, 9 tests) — defaults, schema,
  engine gate incl. no-side-effect-before-approval, approve/reject, rebuilt
  index gender scan.
- `results/safety_run/person_rebuild_report.json`,
  `results/safety_run/mmr_default_check.json` (new; released
  `results/*tables.md` untouched).

Not run (deliberately): full `eval_fusion.py` / `eval_retrieval_control.py` /
`eval_end_to_end.py` (long, LLM-dependent, risk overwriting released
results). Default-match argued from released JSONs + flag-respecting code
paths (§3 rows 8–10).

Results:

- `python -m pytest backend/tests/test_safety_defaults.py -q` → **9 passed**.
- `python -m pytest backend/tests -q` (full suite, incl. `test_end_to_end.py`)
  → **230 passed** (212 + 18), no failures. Pre-existing warnings only.

## 5. Risks, blockers, page-budget impact

- **Shippability (blocker for parent):** `data/index/` is gitignored, so the
  rebuilt gender-free index exists on disk but is not committable by default.
  Parent must decide: force-add `data/index/faiss_person_nogender/`, extend
  the release-packaging script, or ship the rebuild command as the artifact.
  Until then "descriptor rebuilt" is true in this checkout only.
- **Tables not re-measured (accepted residual, needs parent sign-off):**
  released C2/C3 tables were measured with gendered descriptors. Mean
  old-vs-new cosine is 0.9967, so re-measurement drift is expected to be
  small, but per plan E5's "tables re-run" gate is explicitly NOT done here
  (would overwrite released evidence owned by other areas). Options: accept
  numbers as standing with defaults documented, or schedule a joint re-run.
- **Docstring drift (not editable in this area):**
  `backend/app/core/privacy.py` still states the index is unchanged by
  gender-off; needs a one-line update by whoever owns that file.
- **API surfacing (not editable in this area):** `routers/chat.py` /
  `routers/alerts.py` do not yet expose `requires_approval` /
  `pending_approval` status to clients; engine + validator are ready and the
  pending queue is in `alert_logs`.
- **Page budget:** §1 is replacement text only (limitations items), zero net
  lines; no new section/table/figure proposed.
