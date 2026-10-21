# Paper evidence audit — required changes before submission

Source: an independent read-only trace of every numeric claim in
`INDICON2026_submit/INDICON2026Conference.tex` against `results/`,
`backend/scripts/`, `backend/app/` and the built PDF.  Every item below was
reproduced by the auditor.  The paper is exactly at the 8-page limit, so every
change must be space-neutral or must displace existing prose.

## BLOCKERS

**B1 — the action-query mechanism contradicts Table II.**
`:188` says action queries "skip the structured branch by design
(semantic-only), so weight on the structured branch therefore injects noise,
and P@5 falls from 0.69 at alpha=0.0 to 0.00 at alpha=1.0".  If `R_s` is empty
for action queries then `S_hyb(r) = (1-alpha) S_sem(r)` is a monotone transform
of `S_sem` for every `alpha < 1`, so the ranking — and P@5 — cannot change
with alpha.  Table II nevertheless reports action P@5 = 0.69 / 0.58 / 0.18 /
0.07 / 0.00 for alpha = 0.0 / 0.3 / 0.5 / 0.7 / 1.0.  The deployed path really
does skip the structured branch (`backend/app/services/unified_retrieval.py`
"Skipped (action query uses semantic only)"), and the alpha-sweep harness
instead gives action queries a structured score of `0.35 * time_overlap`
(`backend/scripts/eval_retrieval_control.py`), which is how alpha=1.0
degenerates to pure time ordering.
  * Fix `:188` and `:401` so the mechanism and the table agree, e.g. state that
    the deployed pipeline routes action queries semantic-only while the
    alpha-sweep harness gives them a time-overlap proxy, so the alpha=1.0
    column must not be read as structured-branch noise injection.

**B2 — "reproducing the override gap of Table II" is contradicted by the
paragraph's own numbers.**  `:439` claims the subset reproduces the override
gap, but in the released control a single fixed alpha (0.7) reproduces *every*
per-category value of the adaptive rule (count 0.390, find 0.920, track 0.810,
action 0.090, filter 0.480; macro 0.538 = 0.538, delta 0.000, p = 1.0), which
the same paragraph already states two clauses earlier.  What the subset shows
is only that the *extreme* fixed values collapse on the override categories.
  * Replace with wording that says exactly that: the subset confirms the
    overrides are not harmful, but the per-category best fixed alpha (0.7)
    happens to match the adaptive rule on this subset, so the subset does not
    reproduce the deployment gain.

**B3 — the raw released benchmark contradicts the method and is never
mentioned.**  The paper quotes one released C1 number (the post-hoc control)
and never discloses that the release also ships the *raw* benchmark, whose
numbers run the other way: adaptive 0.486 vs best fixed 0.568, and the action
category is best with pure structured scoring (0.520 at alpha=1.0) and worst
semantic-only (0.030 at alpha=0.0) — the exact inverse of the override story
(`results/retrieval_tables.md`).  The paper calls the control "protocol-matched"
but never says it is *post-hoc*, although `VERIFICATION.md` claims the paper
says so (it does not).
  * Add one sentence in the evaluation section stating that the raw released
    benchmark is a code-path smoke test that is not protocol-matched (count
    questions carry no target count; action queries carry a time window the
    structured branch can exploit), that it ranks adaptive below the best
    fixed alpha (0.486 vs 0.568) and prefers alpha=1.0 on action queries
    (0.520), and that the post-hoc control below removes those two mismatches
    and is the number we quote.  Also call the control "post-hoc" where it is
    described.

**B4 — "loses exactly the override cases the closed-form rule switches on" is
false.**  `:441` says the learned-alpha competitor "collapses to a constant and
loses exactly the override cases".  Its per-category scores are identical to
the closed-form rule on both override categories (count 0.39, action 0.09); the
entire deficit is on find (0.91 vs 0.92) and filter (0.47 vs 0.48).
  * Rewrite as: the fitted model collapses to the constant alpha=0.5; it
    matches the closed-form rule on both override categories (count 0.39,
    action 0.09) and loses only on the soft zone (find 0.91 vs 0.92, filter
    0.47 vs 0.48).  (The "0.5 on 49 of the 50 tune queries" half is correct.)

**B5 — the cited release URL is 404.**  `GET
https://github.com/sujalmh/intent-adaptive-hybrid-surveillance` returns HTTP
404 (the account exists; the repository does not resolve).  Every "released"
claim in the paper points at a dead link (`:90`, `:109`, `:524`).
  * Either make the claim true (the repository exists under that exact name)
    or soften it: "Code will be made public on acceptance (URL withheld from
    this preprint)" and change the abstract's "are released" to "are included
    in the supplementary artifact".  Note the decision in the response letter.

## MAJOR

**M1 — the 91 % is stated to be measured on the released benchmark's clean
rows.**  `:368` says the 120-utterance release contains 60 clean + 60 stress
"the reported 91% rule-correctness is on the 60 clean utterances".  It was not:
the 91 % comes from a separate 60-utterance deployment study that is not in the
release (`results/alert_results.md`), and the release's own numbers are 30/30
clean and 30/30 stress clarified.
  * Rewrite so the 91 % is attributed to the separate deployment study and the
    release's 120 utterances are a coverage check.

**M2 — §C3 describes a deterministic materializer but the 0.91 was produced by
an LLM materializer.**  `:338` says the materializer "is a deterministic
keyword/regex module ... not LLM structured-output", while Table XI's 0.91 came
from the LLM-backed materializer of the deployment system.
  * Say so in the C3 mechanism paragraph: the *released* materializer is
    deterministic; the 0.91 was measured in the deployment study with the
    LLM-backed materializer and is not reproducible from the released
    benchmark (see Limitations).

**M3 — Contribution 4 over-claims.**  `:109` says the open implementation
"validating all three contributions on a 4-camera live deployment".  The
released artifact cannot: the 4-camera data and 312k index are not released,
the released C1 control ties the best fixed alpha, the released C2 subset
reverses the deployment ordering, and the released C3 set is a different
benchmark.  The repo's own `PAPER_CHANGES.md` required this rewrite and it was
not applied.
  * Rewrite as: an end-to-end open implementation evaluated on a 4-camera live
    deployment (whose data are not released), together with a released
    single-camera reproducibility track that exercises the C1-C3 code paths.

**M4 — the worked alert example does not match the shipped materializer.**
`:341-345` shows severity `med` and a scalar threshold 5.  Running
`backend.app.services.alert_materializer.materialize_alert_rule` on that
utterance gives `severity = "info"` (the module's default) and
`threshold = {"op": ">", "value": 5, "unit": "persons"}`.  The paper's own
`:338` says the default is `info`.
  * Fix the example (severity `info`, threshold as an operator/value pair) and
    add one clause that severity, cooldown, side-effect and cadence defaults
    are applied when the utterance does not state them.

**M5 — the 109 ms / 108.6 ms latency provenance conflicts.**  The abstract
claims "a 109 ms median FAISS lookup over 312k indexed person descriptors";
`:326` says "the person index median is 109 ms ... while the scene-index mean
is 108.6 ms"; Table X's caption calls the "Sem. only" column the scene-index
mean, but that column is per category (94/109/121/137/82) and its Track row is
exactly 109 — the value the caption attributes to the person index.
  * State the mix explicitly: the per-category "Sem. only" column mixes the
    512-d scene index and the 1152-d person index, so its mean 108.6 ms is a
    mixed-index average, and the 109 ms person-index median is a separate
    deployment measurement.

**M6 — the released C2 subset reverses the deployment ordering and the paper
withholds it.**  `:444` says the subset "cannot test the deployment fusion
result" but reports none of its numbers and never says the ordering is
inverted (concat P@5 0.451 < visual-only 0.484, because the pseudo-tracks were
formed with visual-only cosine, which biases visual-only upward).
  * Add that sentence.

**M7 — "a full sensitivity sweep is retained as future work" (`:260`)
contradicts the 16-point sweep reported at `:441`.**
  * Replace the future-work clause with a reference to the reported sweep.

**M8 — Eq. (4) as printed is not the implemented function.**  `:248-251`:
`|F|` is described as "the number of non-empty fields in F", but `F` is a
14-field NLFilter, so `min(1,|F|/4)` is 1 for nearly every query; the code
counts six specific keys (camera_id, objects.object_name, objects.color, zone,
location_hint, timestamp).  The code uses `n_s / max(1, n_s + n_v)`, not
`/(n_s+n_v+1e-6)`.  The intent prior has no defined value for the filter
category (30/150 queries) and the code defaults to 0.7, and the "Realized
alpha ranges" sentence omits the filter category although Table II's filter
row equals the alpha=1.0 value.
  * State the six counted keys, the `max(1, .)` denominator, the 0.7 default
    for unmatched intents, and add the filter category to the realized-alpha
    range (or drop the "closed-form" qualifier).

**M9 — the deployment tables present unverifiable CIs and p-values as ordinary
measured results.**  A repo-wide scan finds 0.828, 0.666, 312,544, 108.6,
0.807 and 0.785 only in the .tex and in planning documents, never in
`results/`.  The per-query export, the CIs, the p-values, the Wilcoxon
cross-check and the fusion CIs are therefore not re-computable.  The paper
discloses the export loss in the setup and limitations, but not in the
captions.
  * Add one clause to the captions of the deployment tables, e.g. "Deployment
    measurements; the per-query export was not preserved, so these CIs and
    p-values are not re-computable from the release", and one clause in the
    abstract.

**M10 — `p < 10^-4` is a bootstrap-resolution floor.**  The released
`paired_bootstrap_ci` computes `p = 2 * P(boot_mean <= 0)` with n_boot = 10 000,
so the smallest representable non-zero p is 2e-4; "p < 1e-4" means "zero
non-positive resamples of 10 000".
  * Report it as "p < 0.0002 (bootstrap resolution; zero non-positive resamples
    of 10 000)" where it is stated.

**M11 — `VERIFICATION.md` contains a false checkmark and a stale test count.**
Row "The released C1 control is labelled as a post-hoc protocol-matched
control | OK stated in the paper" is false (B3); the C2 row is likewise not
stated in the paper; "Contribution 4" contradicts the same file's own row; the
pytest row says "63 passed".

## MINOR (apply only if space allows)

* m1 `:191` caption says "italics = worst per column" but the Filter column's
  worst value (0.55 at alpha=0.0) is not italicised.  Italicise it or change
  the caption to "boldface = best".
* m2 Table V (`tab:consolidated`) is never referenced in the text; IEEE
  requires every table to be cited.  Add a one-clause citation or delete the
  table (it restates Tables II and IV).  The label `sec:eval_protocol` is
  unused.
* m3 `:447` "never a structural rewrite" is a universal quantifier on a
  measurement whose per-utterance export was not preserved.  Soften to "no
  structural rewrite was observed".
* m4 abstract "eliminating manual rule authoring" is contradicted by the
  release (30/30 stress utterances are clarified) and by the paper's own 9 %.
  Soften to "reducing manual rule authoring to a clarification round-trip".
* m5 `:524` "showing that schema-aware fusion closes the gap" is absolute and
  single-site.  Soften to "indicating that ... narrows the gap ... on a single
  4-camera premises".
* m6 `:256` "attains the per-category maxima by construction" -> "empirically
  on this benchmark".
* m7 `:304` "a coarse (alpha_v,beta_t) sweep is monotonic" -> three reported
  points cannot show monotonicity; the trend is unimodal.
* m9 `README.md` describes captioning as BLIP-2 while the paper and the code
  say Qwen2-VL-2B with ViT-GPT2/CLIP fallbacks.
* m12 `:276` the illustrative attribute string does not match
  `attribute_encoder.py`'s format.
* m13 `:338` the severity map {info,high,warning,critical} -> {low,med,high}
  is a 4->3 mapping left unspecified.
* m14 `:406` "human-annotated relevant clip sets" for the deployment: the
  annotator roster was not preserved.  Say "author-annotated (the annotation
  records were not preserved)".

## Verified correct (do not touch)

8 pages exactly, 0 Overfull hbox, 0 undefined references, 15/15 bibliography
entries cited, `IEEEtran.cls` byte-identical to the kit, every number the
paper quotes *from the release* is exact (control 0.538, alpha competitors
0.534, the 16-point sweep, the fusion subset 0.436+-0.0004, the alert 30/30
and 30/30, the latency percentiles, the subset inventory), all Table II/IV/V/X
arithmetic re-computed exactly, and the implementation claims (detector
confidence 0.5, 25-colour palette, 7 attribute heads, 12 collections, MMR
lambda 0.3, fight trigger 0.03 and < 24 px, Z-score severities, 14-field
NLFilter, alpha overrides 1.00/0.10 and clipping [0.10, 0.95]) all trace to
the code.
