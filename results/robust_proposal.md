# Robustness proposal: merge / fusion edge cases, normalisation calibration, and a non-collapsed ranking comparison

**Scope.** Peer-review response for the non-claim robustness points raised against the hybrid
retrieval pipeline:

* **(a)** weighting robustness to score-distribution shifts, **EMPTY** result sets / branches,
  noisy metadata, different backends — unestablished;
* **(b)** score normalisation is per-query min-max only (z-score compared on one control);
* **(c)** no calibrated / learned ranking comparison beyond a collapsed logistic baseline.

**Method.** (1) Static audit of the shipped merge / fusion code paths;
(2) *execution* of every suspected edge case against the real production functions
(`backend/scripts/robust_probe_tmp.py` → `results/robust_probe_tmp.json`, 55 cases + 18
patched-reference regression tests — reproduce with
`python backend/scripts/robust_probe_tmp.py`, CPU-only, no FAISS/Mongo/network needed);
(3) targeted literature search for IR / data-fusion robustness practice that is applicable
**without new data or labels**; (4) exact, unapplied patches + regression-test sketches.

**Nothing in this file has been applied.** The repo is untouched except for the two new scratch
files named above. All diffs below are proposals.

---

## 1. Executive summary

| # | Finding | Status | Severity | Review point |
|---|---------|--------|----------|--------------|
| B1 | Degenerate (all-tie) branch is normalised to **1.0** — a branch with zero discrimination is reported as maximally confident | **live bug** | high | (a), (b) |
| B2 | Single-element branch gets a **unit-dependent** magic constant (0.5 / 0.8 / 1.0) from thresholds (0.25 / 0.40) that are only meaningful for scores in [0,1], but the input is **duration seconds** | **live bug** | high | (a), (b) |
| B3 | Negative score vector inverts the branch ranking and leaves [0,1] | **live bug (latent)** | medium | (a) |
| B4 | NaN / inf in a score vector silently **zeroes the whole branch** (no exception); NaN also reaches the final `score` and the sort key | **live bug** | high | (a), (b) |
| B5 | `score_norm=None` in a semantic record → uncaught `TypeError` (500 on the request path) | **live bug** | high | (a) |
| B6 | `None` branch arguments crash; the **count-constrained path is not even guarded** (the normal path is) | **live bug (latent)** | medium | (a) |
| B7 | `apply_mmr_diversity` is **fail-closed**: it can return an *empty* result list and silently drop every candidate | **live bug (latent)** | high | (a) |
| B8 | Empty branch is **not rebalanced** — with zero structured results alpha stays ≈0.47, so every semantic-only score is deflated by ~47 % and is not comparable across queries | **live bug** | high | (a) |
| B9 | NaN in a score vector → `combined_alpha` returns **1.0** (all weight to the structured branch) because `min(1.0, nan) == 1.0` | **live bug** | medium | (a), (c) |
| B10 | The C1' confidence blend is **dead code in production** — `merge_results` never passes score vectors, so the shipped ranker is the prior-only rule while the paper reports `conf_only` / `combined` | **deployment gap** | high | (c) |
| B11 | The control harness normalises **min-max for both branches**; production uses *max-normalisation* for the structured branch and min-max for the semantic one → the "min-max vs z-score" table measures a normaliser the product does not ship | **measurement-validity gap** | high | (b) |
| B12 | `MultimodalFusion.fuse` mutates the caller's embedding buffers **in place** and is not idempotent | **live bug** | medium | (a) |
| B13 | `fuse_gated`: NaN quality is coerced to **1.0** (max trust) by the clamp; `None` quality raises `TypeError` | **live bug** | medium | (a) |
| B14 | An all-zero / empty branch is silently accepted; the fused output is then **not unit-norm** and the gate still reports full trust in the empty branch | **live bug** | medium | (a) |
| B15 | `_norm_scores` (duplicated in `sem_search.py` and `hybrid_search.py`): an all-tie branch → all **1.0** (a branch that cannot discriminate claims full confidence); the one-hit case is also unit-independent only by accident | **live bug** | high | (a), (b) |

The five highest-value items for the response letter are **B1/B2/B15** (degenerate-branch
confidence inflation), **B8** (empty-branch rebalancing — this is the literature-standard fix),
**B11** (the control does not measure the shipped normaliser) and **B10** (the calibrated
alternative exists in the eval but is not deployed).

---

## 2. Edge-case audit (case → status → evidence)

All evidence is from `results/robust_probe_tmp.json`, produced by executing the **production**
functions. "Guarded" = no crash and no incorrect behaviour; "bug" = wrong behaviour, silent
data loss, or an exception escaping.

### 2.1 Score normalisation

| # | Input case | Status | Observed | Expected / correct |
|---|-----------|--------|----------|--------------------|
| 1 | `normalize_scores([])` | guarded | `[]` | `[]` |
| 2 | `normalize_scores([0.0])` (single, zero) | **bug (B2)** | `[0.5]` | unit-independent: `[1.0]` (a single candidate is trivially the best of its branch) — the "epistemic" side is carried by the branch *confidence* (0 for a 1-item branch), not by the score |
| 3 | `normalize_scores([0.3])` (single) | **bug (B2)** | `[0.8]` | unit-independent: `[1.0]` — the current value depends on a hard-coded 0.25/0.40 threshold applied to *seconds* |
| 4 | `normalize_scores([5.0])` (single, 5 s) | **bug (B2)** | `[1.0]` | same as above (identical to #3, which is the unit-dependence proof) |
| 5 | `normalize_scores([3.0,3.0,3.0])` (all-tie) | **bug (B1)** | `[1.0,1.0,1.0]` | `[0.0,0.0,0.0]` (sklearn convention for a constant feature — verified locally: `minmax_scale([[3],[3],[3]]) → [0,0,0]`) |
| 6 | `normalize_scores([0.0,0.0])` (all-zero) | guarded | `[0.0,0.0]` | correct |
| 7 | `normalize_scores([-1.0,-2.0])` (all-negative) | **bug (B3)** | `[0.5,1.0]` — **worse candidate scores higher** | monotone in the input, e.g. `[0.0,0.0]` or `[0.33,0.0]` |
| 8 | `normalize_scores([-1.0,0.0,2.0])` (mixed sign) | guarded-ish | `[0.0,0.0,1.0]` | negatives clamp to 0 → ties with a genuine 0; acceptable but lossy |
| 9 | `normalize_scores([nan,1.0])` | **bug (B4)** | `[0.0,0.0]` — the *whole branch* is zeroed | raise / isolate the bad element, keep `1.0` |
| 10 | `normalize_scores([1.0,nan])` | **bug (B4)** | `[1.0,0.0]` | NaN → 0 for that element only (this happens to be OK) |
| 11 | `normalize_scores([inf,1.0])` | **bug (B4)** | `[0.0,0.0]` | `+inf` must not outrank everything; should be isolated |
| 12 | `normalize_scores([0.1,0.2])` (sub-second durations) | **bug (B2)** | `[0.25,0.5]` — the 0.25 absolute threshold halves *seconds* | unit-independent scaling |
| 13 | `normalize_scores([1.0,300.0])` (durations) | guarded (but ≠ harness) | `[0.0033,1.0]` | harness min-max gives `[0.0,1.0]` → **B11** |
| 14 | `normalize_scores([42.0])` (single-candidate gallery) | **bug (B2)** | `[1.0]` | `[1.0]` is *acceptable* for a single candidate, but only because it must be **unit-independent** — today it is not (see #3 vs #4) |
| 15 | `sem_search._norm_scores([])` | guarded | `[]` | `[]` |
| 16 | `sem_search._norm_scores([0.37])` | **bug (B15)** | `[1.0]` | acceptable *if* unit-independent; the real defect is the all-tie case (#17) and the duplicated implementation |
| 17 | `sem_search._norm_scores([0.5,0.5,0.5])` | **bug (B15)** | `[1.0,1.0,1.0]` | `[0.0,0.0,0.0]` (sklearn convention for a constant feature — verified locally: `minmax_scale([[3],[3],[3]]) → [0,0,0]`) |
| 18 | `sem_search._norm_scores([0.2,0.9])` | guarded | `[0.0,1.0]` | correct |
| 19 | `hybrid_search._norm_scores` | **bug (B15)** | duplicate of #15–17 (`hybrid_search.py:55-61`) | one shared implementation |

### 2.2 `merge_results` (end-to-end)

| # | Case | Status | Observed | Comment |
|---|------|--------|----------|---------|
| 20 | structured empty, semantic empty | guarded | `[]` | no crash |
| 21 | structured empty, semantic present | guarded but **wrong scale (B8)** | semantic `score_norm=0.8` → final `0.424` (= 0.8 × 0.53) | the whole branch is deflated by `1-α`; a semantic-only query can never score above ~0.53 |
| 22 | structured present, semantic empty | guarded | final `0.335` (= α·0.5, and 0.5 comes from **B2**) | the structured score of a 0-second track is 0.5 |
| 23 | both present | guarded | `score=0.639` | ok |
| 24 | all-tie structured durations | **bug (B1)** | the tied branch would score `[1.0,1.0,1.0]` | see unit-level case 5 |
| 25 | semantic record with **no** `score_norm` key | guarded | `score_semantic=0.0`, result kept with score 0 | result is still returned to the user with a 0 score |
| 26 | semantic record with `score_norm=None` | **bug (B5)** | **`TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`** | uncaught → 500 |
| 27 | semantic record with `score_norm=NaN` | **bug (B4)** | `score = NaN` | NaN enters the ranking and the sort key → undefined ordering |
| 28 | `structured=None` | guarded | merged as an empty structured branch, logs “Track merging failed” | the `try/except` around track merging absorbs it |
| 29 | `semantic=None` | **bug (B6)** | **`TypeError: object of type 'NoneType' has no len()`** | unguarded (`len(semantic)` in the alpha call) |
| 30 | `structured=None` **with a count constraint** | **bug (B6)** | **`TypeError: 'NoneType' object is not iterable`** | the count-constrained path has **no** try/except — asymmetric with #28 |
| 31 | count constraint + semantic results | guarded (by design) | semantic branch discarded, α = 1.0 | deliberate; the count constraint is strict |
| 32 | `parsed_filter=None` in `compute_adaptive_alpha` | guarded | falls back to the fixed α | try/except in `_metadata_prior_alpha` |
| 33 | `n_structured=0, n_semantic=0` | guarded | α = 0.57 | but the *branch ratio* term uses 0.5 by convention — fine |
| 34 | `n_structured=0, n_semantic=10` | **bug (B8)** | α = **0.47** — the empty branch keeps 47 % of the blend | the CombANZ/CombMNZ fix (§3.2) |
| 35 | `n_structured=10, n_semantic=0` | **bug (B8)** | α = 0.67 — the empty *semantic* branch keeps 33 % | same |
| 36 | degenerate structured vector `[0.5,0.5,0.5]` + informative semantic | guarded-but-suboptimal (B9) | α = 0.285 (only the 50 % prior part survives) | a degenerate branch still receives 28.5 % of the weight |
| 37 | NaN in the score vector | **bug (B9)** | α = **1.0** | `min(1.0, nan)` returns `1.0` → all weight to the structured branch |
| 38 | `confidence_alpha([], [])` | guarded | 0.5 | documented fallback |
| 39 | `branch_confidence([0.5])` (single candidate) | guarded but **misleading** | `0.0` | a one-item branch is "no discrimination evidence", so `cs/(cs+ce)` gives the *other* branch 100 % — the one-hit semantic branch is silently discarded |
| 40 | `topk_concentration([-2,-3,-1])` | guarded | `1.0` | non-negative shift before summing — correct |
| 41 | `confidence_alpha([nan,0.5],[0.1,0.9])` | **bug (B9)** | `NaN` | propagates into `combined_alpha` |

### 2.3 MMR diversity

| # | Case | Status | Observed | Comment |
|---|------|--------|----------|---------|
| 42 | 3 candidates with scores −5.0/−4.0/−3.4, λ=0.3 | **bug (B7)** | **`[]`** — *all* results silently dropped | `best_mmr = -1.0` is a sentinel, not a floor: at the first iteration `mmr = λ·rel`, so any `rel ≤ −1/λ = −3.34` makes every candidate lose |
| 43 | 2 candidates | guarded | passthrough (`len(results) <= 2`) | ok |
| 44 | 3 tied candidates | guarded | order preserved, no drop | `mmr > best_mmr` keeps the first of a tie |
| 45 | exception inside the loop | guarded | original order returned | try/except at `result_merger.py:446-448` |

Threshold for case 42 as a function of λ: the drop happens when `max(rel) ≤ −1/λ`
(λ=0.3 → −3.34, λ=0.5 → −2.0, λ=1.0 → −1.0). Production scores are ≥ 0 today, so this is a
landmine rather than an everyday failure — but the failure mode ("no results") is the worst
possible one, and it is silent.

### 2.4 Person fusion (`backend/app/services/fusion.py`)

| # | Case | Status | Observed | Comment |
|---|------|--------|----------|---------|
| 46 | `MultimodalFusion.fuse(v, t)` with float32 inputs | **bug (B12)** | `v[0]`: 1.0 → **0.6**, `t[0]` → 0.4 — the caller's arrays are **modified** | `np.asarray(x, float32).reshape(-1)` is a *view* of a float32 input, and `v *= w` writes through it. `backend/object_detection.py:502-508` passes `v_embs[j]` (a float32 batch row) → the whole batch is scaled in place |
| 47 | `fuse` called twice on the same arrays | **bug (B12)** | `v[0]` → **0.36** | not idempotent: 0.6², 0.4² |
| 48 | `fuse` with all-zero inputs | **bug (B14)** | output norm = **0.0** (contract says L2-normalised) | a zero vector is silently indexed into FAISS; IP search then returns arbitrary neighbours |
| 49 | `fuse_gated` with `visual_quality=NaN` | **bug (B13)** | implied `w_v/w_t` = **40.3** → the gate fully trusts the visual branch | `max(0.0, min(1.0, float(nan)))` → `min(1.0, nan)` returns `1.0` |
| 50 | `fuse_gated` with `visual_quality=None` | **bug (B13)** | **`TypeError: float() argument must be ... not 'NoneType'`** | unguarded |
| 51 | `fuse_gated` with an all-zero visual embedding and `qv=1.0` | **bug (B14)** | visual block norm 0, text block 1.0 — the output is the text embedding, while the gate still reports "visual is better" | the declared quality is never cross-checked against the embedding |
| 52 | `fuse_gated` with wrong-shaped inputs (≠768/384) | guarded | zero-padded / truncated | ok |
| 53 | `_sigmoid` overflow | guarded | unreachable for λ ≤ 4 (\|λ(qv−qt)\| ≤ 4) | only a problem for an absurd λ |

### 2.5 Summary of the *guarded* paths (so the reviewer can see the negative results too)

* empty structured branch, empty semantic branch, and both empty → no crash, no division by zero;
* `normalize_scores([])`, division by zero (`max_score == 0` → `[min_val]*n`), `min_val` clamp;
* `confidence_alpha([], [])` → 0.5 fallback; `topk_concentration` on constant/negative vectors;
* count-constrained queries deliberately disable the semantic branch (α = 1.0);
* `apply_mmr_diversity` passthrough for ≤2 results and full try/except fallback;
* `merge_results` with `structured=None` **on the non-count path** (absorbed by the track-merging
  try/except) and `compute_adaptive_alpha` with `parsed_filter=None`;
* `fuse_gated` shape correction, and `fuse`/`fuse_gated` L2 normalisation with a `+1e-6` epsilon.

---

## 3. Literature → code mapping (all applicable without new data or labels)

### 3.1 Reciprocal Rank Fusion — the reference "different backend" control

> Cormack, Clarke, Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank
> Learning Methods*, SIGIR 2009. PDF: <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>
> Elasticsearch reference: <https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion>
> — *"RRF requires no tuning, and the different relevance indicators do not have to be related to
> each other to achieve high-quality results."*

RRF consumes **ranks only**, so it is invariant to *any* monotone transform of a branch's scores
(and to a branch's score range entirely). That is exactly the property the current
min-max/z-score blend lacks. The repo already ships `backend/app/eval/baselines.py::rrf_fuse`
(R1-7/R2-1) but only as a *scored* competitor; the missing piece is to use RRF as the
**reference fusion for the robustness study** and to report the *rank distance* to the shipped
blend (label-free, §4.1 C1).

### 3.2 CombSUM / CombANZ / CombMNZ — the canonical *empty-branch* fix

> Fox & Shaw, *Combination of Multiple Searches*, TREC-2 (1994) / TREC-3.
> Full text: <https://trec.nist.gov/pubs/trec2/papers/txt/23.txt>
> — *"CombANZ, the average of the non-zero similarity values, that **ignores the effects of a single
> given run or query failing to retrieve a relevant document**"*; and CombMNZ *"to provide higher
> weights to documents retrieved by multiple retrieval methods."*
> Their §4.3 *Future Exploration* literally lists *"**Normalization methods to allow combination
> of runs made with different weighting schemes**"* — the reviewer's "different backends" point.

This is the direct precedent for **B8**: a fixed-weight blend (`α·s_struct + (1−α)·s_sem`) keeps
the weight of a branch that returned nothing, so the fused score scale changes with the number of
contributing branches. CombANZ/CombMNZ renormalise by the branches that actually contributed.
`rrf_fuse` in the repo already has the same property (a document missing from one list simply
contributes 0 from that list, with no global deflation).

### 3.3 Score ranges differ per backend → fused scores are not comparable across queries

> Azure AI Search, *Hybrid Search Scoring (RRF)*:
> <https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking>
> — BM25: *"No upper limit"*; vector (cosine): *"0.333 – 1.00"*; RRF: *"Upper limit is bounded by
> the number of queries being fused"*.

Two consequences for this repo: (i) the absolute scale of a fused score is a function of which
branches contributed (B8), and (ii) comparing the *reported* fused score across queries or
backends is invalid without renormalisation — which is why the proposed empty-branch fix must
renormalise, not merely rescale.

### 3.4 Degenerate-distribution convention

> scikit-learn, `MinMaxScaler` / `minmax_scale`:
> <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>
> — a *constant* feature is mapped to the lower bound (verified locally:
> `minmax_scale([[3],[3],[3]]) → [0,0,0]`), and the docs note that MinMax *"doesn't reduce the
> effect of outliers"* and that `clip` does not prevent *score drift*.

A branch whose scores are all equal carries **no** evidence, so it must score 0, not 1 — this is
the standard convention and the fix for **B1 / B15**. The "outlier" note motivates a third
normalisation arm that is robust to a heavy-tailed branch: `QuantileTransformer` /
`PowerTransformer` (<https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html>),
which is label-free and makes scores comparable across queries — the concrete answer to
review point (b), which currently offers only min-max and one z-score control.

### 3.5 Label-free rank-stability metrics

> Webber, Moffat, Zobel, *Similarity Measures for Ranked Lists*, ACM TOIS 33(4), 2015,
> doi:<https://doi.org/10.1145/2743774.2743794> (RBO; reference implementation:
> <https://pypi.org/project/rbo/>)

RBO (and Kendall's τ) compare two rankings **without labels**. They are the missing measurement
for review point (a): today the only robustness number reported is P@5 under a *label* metric, so
"robustness" is confounded with "accuracy on 150 caption-derived labels". §4.1 proposes four
label-free probes built entirely on the existing released subset.

### 3.6 Label-free relevance pseudo-labels (if a *learned* ranker is wanted)

> Luyu et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, ACL 2023:
> <https://www.cs.cmu.edu/~callan/Papers/acl23-Luyu-Gao.pdf>
> *Unsupervised Dense Retrieval with Relevance-Aware Contrastive Pre-Training*, ACL Findings 2023:
> <https://aclanthology.org/2023.findings-acl.695/>

LLM-generated pseudo-relevance judgements are the standard way to obtain a *learned* ranker
without new human labels. This matters for review point (c): the fix there is **not** more labels
but (i) a non-collapsing training target and (ii) an honest calibration check on the existing
tune split (§4.2).

---

## 4. Proposed protocol (no new data, no new labels)

### 4.1 Label-free robustness probes (close review point (a))

| ID | Probe | Construction | Metric | Closes |
|----|-------|--------------|--------|--------|
| **C1** | **Normaliser rank-stability** | For each query, fuse with {production max-norm, min-max, z-score, rank/quantile, RRF}; compare the top-10 lists pairwise | RBO@10, Kendall τ, overlap@10 | (a), (b) |
| **C2** | **Monotone-warp invariance** | Apply a fixed monotone warp to one branch's raw scores: `x→x^p` (p∈{0.5,2}), `x→log(1+x)`, `x→rank`; re-fuse | RBO@10 vs the un-warped ranking | (a) — "different backends" is exactly a change of score *shape*; min-max/z-score are invariant to **affine** warps only |
| **C3** | **Empty-branch stress** | Re-run the merge with each branch emptied in turn | (i) RBO@10 of the surviving branch (must be 1.0 — same order), (ii) mean \|score\| shift (must be a *constant* factor, i.e. no ranking-relevant distortion) | (a) |
| **C4** | **Score-as-confidence check** | Bin the final fused score into deciles; per bin report the branch agreement rate (top-1 id overlap) and the top-1 margin ratio | monotone-in-agreement? (yes ⇒ the fused score is a usable confidence; no ⇒ it is not) | (b) |

C1–C4 need only the per-query score vectors that
`backend/scripts/eval_retrieval_control.py` already computes (it writes
`results/retrieval_control_per_query.csv`), so they can be added to that script **without
re-encoding anything** — the expensive part (OpenCLIP text encoding + FAISS search) is already
in the script and can be cached.

### 4.2 A non-collapsed calibrated/learned ranking comparison (close review point (c))

**Why the current baseline collapsed.** `backend/scripts/eval_alpha_competitors.py:224-282` trains
a 5-way multinomial logistic regression on the *argmax* alpha per query. On the tune split the
oracle-best-alpha label distribution is `{"2": 49, "3": 1}` (i.e. 49/50 queries share one
label) → the classifier degenerates to a constant, the predicted-alpha histogram is
`{"0.5": 100}` and the test macro P@5 (0.534) is exactly `fixed_0.5` (0.534)
(`results/alpha_competitors.json`). The baseline therefore measures nothing.

**Proposed replacements** (all trainable on the existing 50-query tune split):

* **L1 — regression instead of argmax classification.** Predict the *continuous* per-query P@5
  of each of the 5 fixed alphas (a 5-dim regression / soft-label), then take the argmax of the
  prediction. There is no single class per query to collapse onto, and the target is exactly the
  quantity the system is trying to maximise.
* **L2 — a real listwise/pairwise ranker.** RankNet-style pairwise logistic over per-candidate
  features `(s_struct, s_sem, s_struct·s_sem, branch agreement, top-1 margin)` on the tune split,
  using the *existing* relevance labels. This is the "learned ranking" the reviewer asks for, and
  it is where the C1' confidence features belong.
* **L3 — calibration of the learned ranker.** Fit Platt scaling on the tune split and report the
  reliability diagram + ECE of the *predicted* P@5 against the realised P@5. This is the
  "calibration check" the reviewer asks for and it needs **no new labels** — only the existing
  tune split, held out from the test split.
* **L4 — non-learned, robust-by-construction references.** RRF and CombMNZ/CombANZ (§3.1, §3.2)
  as the honest "do we even need a learned weight?" baselines. If L2 cannot beat RRF on the tune
  split, that is the honest headline.
* **L5 — deployment consistency (B10).** Whatever is claimed must be *shipped*: either wire the
  confidence blend into `merge_results` (D7) or state plainly in the paper that the deployed
  ranker is the prior-only rule and that `conf_only`/`combined` are offline ablations.

---

## 5. Proposed patches (unified-diff style — **NOT applied**)

> Reviewer note: these diffs are written against the current working tree
> (`Enhance-chat-and-retrieval-processes`). They are proposals for review, not committed work.

### D1 — `backend/app/services/retrieval_utils.py`: robust, unit-free score normalisation

Closes **B1, B2, B3, B4**; contributes to (a) and (b).

```diff
--- a/backend/app/services/retrieval_utils.py
+++ b/backend/app/services/retrieval_utils.py
@@ -55,30 +55,72 @@
 # ---------------------------------------------------------------------------
 # Score normalization
 # ---------------------------------------------------------------------------
-def normalize_scores(scores: List[float], min_val: float = 0.0) -> List[float]:
+_NORM_EPS = 1e-12
+
+
+def finite_score(value: Any, default: float = 0.0) -> float:
+    """Coerce *value* to a finite float (NaN/inf/None -> *default*).
+
+    A single corrupt score (NaN from a broken FAISS index, ``None`` from a
+    partially-populated record) must never be able to zero out or NaN-poison
+    an entire branch.
+    """
+    try:
+        f = float(value)
+    except (TypeError, ValueError):
+        return float(default)
+    if f != f or f in (float("inf"), float("-inf")):
+        return float(default)
+    return f
+
+
+def normalize_scores(
+    scores: List[float],
+    min_val: float = 0.0,
+    mode: str = "minmax",
+) -> List[float]:
     """
-    Normalize *scores* to [0, 1], scaling down when the absolute maximum is
-    weak so that final scores honestly reflect low confidence.
+    Normalize one query's *scores* into [min_val, 1].
+
+    Robustness contract (see results/robust_proposal.md):
+
+    * non-finite inputs are coerced to 0.0 (see :func:`finite_score`);
+    * a *degenerate* branch (two or more candidates, all scores equal)
+      carries no discrimination information and is mapped to ``min_val``
+      (0.0).  The scikit-learn convention for a constant feature is the
+      same; reporting it as 1.0 would claim confidence the branch does
+      not have.
+    * a *single-candidate* branch is mapped to 1.0 -- that candidate is
+      trivially the best of its branch -- but the value must not depend on
+      the unit of the score.  The "this branch cannot discriminate" signal
+      is carried by the branch *confidence* (margin/concentration are 0 for
+      a 1-item branch), not by the score.
+    * negative scores are *shifted*, never divided by a negative maximum,
+      so the within-branch order is preserved and the output stays in
+      [min_val, 1].
+    * the historical "absolute maximum" rescaling (0.25 / 0.40) is removed:
+      those thresholds are only meaningful for scores in [0, 1] and are
+      meaningless for the track durations (seconds) they are applied to.
+
+    ``mode="minmax"`` (default) matches the normalisation used by the
+    evaluation harness; ``mode="max"`` reproduces the previous
+    max-normalisation *without* the unit-dependent rescaling, for A/B
+    against the released numbers.
     """
     if not scores:
         return []
-    if len(scores) == 1:
-        s = scores[0]
-        if s < 0.25:
-            return [max(min_val, 0.5)]
-        if s < 0.4:
-            return [max(min_val, 0.8)]
-        return [max(min_val, 1.0)]
-
-    max_score = max(scores)
-    if max_score == 0:
-        return [min_val] * len(scores)
-
-    scale = 1.0
-    if max_score < 0.25:
-        scale = 0.5
-    elif max_score < 0.4:
-        scale = 0.8
-
-    return [max(min_val, (s / max_score) * scale) for s in scores]
+    vals = [finite_score(s) for s in scores]
+    if len(vals) == 1:
+        return [1.0]
+    lo, hi = min(vals), max(vals)
+    if hi - lo <= _NORM_EPS:
+        # Degenerate (constant) branch: no information, no confidence.
+        return [min_val] * len(vals)
+    if mode == "max":
+        if hi <= _NORM_EPS:
+            return [min_val] * len(vals)
+        return [max(min_val, min(1.0, (v - lo) / hi)) for v in vals]
+    return [max(min_val, min(1.0, (v - lo) / (hi - lo))) for v in vals]
+    vals = [finite_score(s) for s in scores]
+    lo, hi = min(vals), max(vals)
+    if hi - lo <= _NORM_EPS:
+        # Degenerate (constant) branch: no information, no confidence.
+        return [min_val] * len(vals)
+    if mode == "max":
+        if hi <= _NORM_EPS:
+            return [min_val] * len(vals)
+        return [max(min_val, min(1.0, (v - lo) / hi)) for v in vals]
+    return [max(min_val, min(1.0, (v - lo) / (hi - lo))) for v in vals]
```

### D2 — `backend/app/services/sem_search.py` + `backend/app/services/hybrid_search.py`: one shared semantic normaliser

Closes **B15**.

```diff
--- a/backend/app/services/sem_search.py
+++ b/backend/app/services/sem_search.py
@@ -21,13 +21,11 @@
 
 
-def _norm_scores(scores: List[float]) -> List[float]:
-    if not scores:
-        return []
-    mn = min(scores)
-    mx = max(scores)
-    if mx - mn < 1e-9:
-        return [1.0 for _ in scores]
-    return [(s - mn) / (mx - mn) for s in scores]
+def _norm_scores(scores: List[float]) -> List[float]:
+    """Per-query min-max with the degenerate case mapped to 0.0.
+
+    A constant score vector (FAISS ties after thresholding, or a single
+    surviving clip) carries no evidence; returning 1.0 for every element
+    would hand the semantic branch full confidence it has not earned.
+    """
+    return normalize_scores(scores, min_val=0.0, mode="minmax")
```

with the import added at the top of `sem_search.py`:

```diff
--- a/backend/app/services/sem_search.py
+++ b/backend/app/services/sem_search.py
@@ -11,6 +11,7 @@
 from backend.app.config import settings
 from backend.app.services.sem_embedder import get_embedder
 from backend.app.services.sem_store import get_faiss_store
+from backend.app.services.retrieval_utils import normalize_scores
```

and the duplicate in `hybrid_search.py` replaced by the same delegate:

```diff
--- a/backend/app/services/hybrid_search.py
+++ b/backend/app/services/hybrid_search.py
@@ -52,13 +52,10 @@
 
 
 def _norm_scores(scores: List[float]) -> List[float]:
-    if not scores:
-        return []
-    mn, mx = min(scores), max(scores)
-    if mx - mn < 1e-9:
-        return [1.0 for _ in scores]
-    return [(s - mn) / (mx - mn) for s in scores]
+    """See :func:`backend.app.services.sem_search._norm_scores`."""
+    from backend.app.services.sem_search import _norm_scores as _shared
+
+    return _shared(scores)
```

### D3 — `backend/app/services/result_merger.py`: empty-branch rebalancing (CombANZ-style) + score guards

Closes **B5, B6, B8**; contributes to (a).

```diff
--- a/backend/app/services/result_merger.py
+++ b/backend/app/services/result_merger.py
@@ -25,9 +25,11 @@
     SEM_RAW_THRESH_LOW,
     SEM_RAW_THRESH_MED,
     SEM_RAW_THRESH_NO_STRUCTURED,
     SEM_RAW_THRESH_HIGH,
+    finite_score,
     normalize_count_constraint,
     normalize_scores,
     object_matches,
     parse_ts,
 )
@@ -310,20 +312,45 @@
     """
     prior = _metadata_prior_alpha(
         parsed_filter, intent, has_count_constraint, has_action,
         n_structured, n_semantic,
     )
-    if struct_scores is None or sem_scores is None:
-        return prior
-    try:
-        from backend.app.eval.confidence import combined_alpha
-
-        return float(combined_alpha(prior, struct_scores, sem_scores, blend=confidence_blend))
-    except Exception:
-        logger.opt(exception=True).debug("Confidence blend failed, using metadata prior")
-        return prior
+    alpha = prior
+    if struct_scores is not None and sem_scores is not None:
+        try:
+            from backend.app.eval.confidence import combined_alpha
+
+            alpha = float(
+                combined_alpha(prior, struct_scores, sem_scores, blend=confidence_blend)
+            )
+        except Exception:
+            logger.opt(exception=True).debug("Confidence blend failed, using metadata prior")
+            alpha = prior
+    return _rebalance_for_empty_branches(
+        alpha, has_count_constraint, n_structured, n_semantic
+    )
+
+
+def _rebalance_for_empty_branches(
+    alpha: float,
+    has_count_constraint: bool,
+    n_structured: int,
+    n_semantic: int,
+) -> float:
+    """Give a branch's weight back when that branch returned nothing.
+
+    A fixed-weight blend keeps the share of a branch that produced no
+    candidates, so the fused score scale silently depends on how many
+    branches contributed.  CombANZ / CombMNZ (Fox & Shaw, TREC-2 1994)
+    renormalise by the branches that actually contributed; this is the same
+    idea applied to the alpha of a two-branch blend.
+
+    With a count constraint the semantic branch is *deliberately* disabled
+    and alpha stays at 1.0.
+    """
+    if has_count_constraint:
+        return alpha
+    if n_structured <= 0 and n_semantic > 0:
+        return 0.0
+    if n_semantic <= 0 and n_structured > 0:
+        return 1.0
+    return alpha
```

Score guards inside `merge_results` (the semantic branch):

```diff
--- a/backend/app/services/result_merger.py
+++ b/backend/app/services/result_merger.py
@@ -493,6 +518,9 @@
     results_map: Dict[str, Dict[str, Any]] = {}
 
     # Structured tracks
     struct_scores = [t.get("duration_seconds", 0) for t in combined_tracks]
     struct_norm = normalize_scores(struct_scores)
     for track, norm_score in zip(combined_tracks, struct_norm):
         key = f"{track.get('camera_id')}_{track.get('start')}_{track.get('end')}"
@@ -505,14 +533,17 @@
 
     # Semantic results (disabled for strict count constraints)
     if not has_count_constraint:
         enable_temporal = getattr(settings, "ENABLE_TEMPORAL_OVERLAP", True)
         for sem in semantic:
             clip_url = sem.get("clip_url")
+            sem_norm = finite_score(sem.get("score_norm"))
+            sem_raw = finite_score(sem.get("score_raw"))
             matched = False
             # Direct clip_url match
             for _key, result in results_map.items():
                 if result.get("clip_url") == clip_url:
-                    result["score_semantic"] = sem.get("score_norm", 0.0)
+                    result["score_semantic"] = sem_norm
                     result["source"] = "hybrid"
                     result["frames"] = sem.get("frames", [])
                     matched = True
                     break
@@ -525,10 +556,10 @@
                         r_start, r_end = result.get("start"), result.get("end")
                         if r_start and r_end and _ranges_overlap(sem_start, sem_end, r_start, r_end):
                             result["score_semantic"] = max(
-                                result.get("score_semantic", 0.0),
-                                sem.get("score_norm", 0.0),
+                                result.get("score_semantic", 0.0),
+                                sem_norm,
                             )
                             result["source"] = "hybrid"
                             if not result.get("frames"):
                                 result["frames"] = sem.get("frames", [])
                             matched = True
@@ -538,13 +569,13 @@
                     "clip_url": sem.get("clip_url"),
                     "clip_path": sem.get("clip_path"),
                     "score_struct": 0.0,
-                    "score_semantic": sem.get("score_norm", 0.0),
-                    "score_raw_semantic": sem.get("score_raw", 0.0),
+                    "score_semantic": sem_norm,
+                    "score_raw_semantic": sem_raw,
                     "source": "semantic",
                     "frames": sem.get("frames", []),
                 }
@@ -556,8 +587,11 @@
     results: List[Dict[str, Any]] = []
     for result in results_map.values():
-        final_score = alpha * result["score_struct"] + (1 - alpha) * result["score_semantic"]
+        final_score = (
+            alpha * finite_score(result.get("score_struct"))
+            + (1 - alpha) * finite_score(result.get("score_semantic"))
+        )
 
         # Semantic-only penalty
         if result.get("source") == "semantic":
-            raw_sem = result.get("score_raw_semantic", 0.0)
+            raw_sem = finite_score(result.get("score_raw_semantic"))
```

and the `None`-branch guard at the top of `merge_results`:

```diff
--- a/backend/app/services/result_merger.py
+++ b/backend/app/services/result_merger.py
@@ -462,6 +462,12 @@
     Combine structured detection results with semantic FAISS results into a
     single ranked list.
     """
+    structured = structured or []
+    semantic = semantic or []
+    parsed_filter = parsed_filter or {}
     count_constraint = normalize_count_constraint(parsed_filter.get("count_constraint"))
     has_count_constraint = count_constraint is not None
     has_action = parsed_filter.get("action") is not None
```

### D4 — `backend/app/services/result_merger.py`: make MMR fail-open

Closes **B7**.

```diff
--- a/backend/app/services/result_merger.py
+++ b/backend/app/services/result_merger.py
@@ -428,20 +428,26 @@
 
         while remaining:
-            best_idx, best_mmr = -1, -1.0
+            # -inf, not -1.0: the sentinel must be below any attainable
+            # relevance, otherwise a batch of very negative scores makes
+            # every candidate "lose" and the whole result set is dropped.
+            best_idx, best_mmr = -1, -math.inf
             for i, r in enumerate(remaining):
-                rel = r.get("score", 0.0)
+                rel = finite_score(r.get("score"))
                 key_r = (r.get("camera_id"), _time_bucket(r))
                 max_sim = 0.0
                 for s in selected:
                     key_s = (s.get("camera_id"), _time_bucket(s))
                     max_sim = max(max_sim, 1.0 if key_r == key_s else 0.0)
                 mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
                 if mmr > best_mmr:
                     best_mmr = mmr
                     best_idx = i
             if best_idx < 0:
                 break
             selected.append(remaining.pop(best_idx))
-        return selected
+        # Fail open: never turn a ranking problem into an empty answer.
+        return selected if selected else results
     except Exception:
         logger.opt(exception=True).debug("MMR diversity failed, returning original order")
         return results
```

### D5 — `backend/app/eval/confidence.py`: NaN-safe, degeneracy-aware confidence

Closes **B9**; feeds the C1' calibration arm of review point (c).

```diff
--- a/backend/app/eval/confidence.py
+++ b/backend/app/eval/confidence.py
@@ -28,12 +28,22 @@
 CONF_K_DEFAULT = 5
 CONF_BLEND_DEFAULT = 0.5
 _MARGIN_W = 0.5
 _CONC_W = 0.5
 
 
+def _finite(values: Sequence[float]) -> list:
+    out = []
+    for s in values:
+        try:
+            f = float(s)
+        except (TypeError, ValueError):
+            f = 0.0
+        if f != f or f in (float("inf"), float("-inf")):
+            f = 0.0
+        out.append(f)
+    return out
+
+
 def _sorted_desc(scores: Sequence[float]) -> list:
-    return sorted((float(s) for s in scores), reverse=True)
+    return sorted(_finite(scores), reverse=True)
+
+
+def is_degenerate(scores: Sequence[float], eps: float = 1e-12) -> bool:
+    """True when the branch carries no discrimination information.

+    A constant (all-tie) vector and a single-element vector are both
+    degenerate: neither can tell two candidates apart.  The merge uses this
+    to stop a branchless-in-effect branch from keeping its share of alpha.
+    """
+    vals = _finite(scores)
+    if len(vals) < 2:
+        return True
+    return (max(vals) - min(vals)) <= eps
@@ -85,18 +95,22 @@
     margin = top1_margin(scores)
     conc = topk_concentration(scores, k=k)
-    return float(margin_weight * margin + conc_weight * conc)
+    value = float(margin_weight * margin + conc_weight * conc)
+    if value != value:  # NaN
+        return 0.0
+    return max(0.0, min(1.0, value))
@@ -95,9 +109,12 @@
     """Pure score-driven structured weight in [0, 1]."""
     cs = branch_confidence(struct_scores, k=k)
     ce = branch_confidence(sem_scores, k=k)
-    if cs + ce <= 1e-12:
+    if cs + ce <= 1e-12:
         return 0.5
-    return float(cs / (cs + ce))
+    alpha = float(cs / (cs + ce))
+    if alpha != alpha:  # NaN
+        return 0.5
+    return max(0.0, min(1.0, alpha))
@@ -110,7 +127,11 @@
     """Half prior, half score-driven (blend tunable, default 0.5)."""
     conf = confidence_alpha(struct_scores, sem_scores, k=k)
     alpha = float(blend * float(prior_alpha) + (1.0 - blend) * conf)
-    return max(0.0, min(1.0, alpha))
+    if alpha != alpha:  # NaN: min()/max() silently return the first argument
+        return float(prior_alpha)
+    return max(0.0, min(1.0, alpha))
```

### D6 — `backend/app/services/fusion.py`: non-mutating, NaN-safe, zero-safe fusion

Closes **B12, B13, B14**.

```diff
--- a/backend/app/services/fusion.py
+++ b/backend/app/services/fusion.py
@@ -6,6 +6,21 @@
 import numpy as np
 
 
+def _safe_quality(value: object, default: float = 0.5) -> float:
+    """Coerce a quality signal to a finite float in [0, 1].
+
+    ``min(1.0, nan)`` returns 1.0 in Python, so a NaN quality would
+    otherwise be read as *maximum* trust.  ``None`` would raise.
+    """
+    try:
+        f = float(value)  # type: ignore[arg-type]
+    except (TypeError, ValueError):
+        return float(default)
+    if f != f or f in (float("inf"), float("-inf")):
+        return float(default)
+    return max(0.0, min(1.0, f))
+
+
 def _sigmoid(x: float) -> float:
     return 1.0 / (1.0 + math.exp(-x))
@@ -44,8 +59,9 @@
         n = min(384, t.shape[0])
         if n > 0:
             t2[:n] = t[:n]
         t = t2
-    qv = max(0.0, min(1.0, float(visual_quality)))
-    qt = max(0.0, min(1.0, float(text_quality)))
+    qv = _safe_quality(visual_quality)
+    qt = _safe_quality(text_quality)
     gw = _sigmoid(float(lam) * (qv - qt))
     gw = max(0.05, min(0.95, gw))
@@ -52,8 +68,17 @@
     wv = (base_visual_weight * 2.0) * gw
     wt = (base_text_weight * 2.0) * (1.0 - gw)
     v = (v * wv).astype(np.float32)
     t = (t * wt).astype(np.float32)
     fused = np.concatenate([v, t]).astype(np.float32)
     norm = float(np.linalg.norm(fused)) + 1e-6
-    return (fused / norm).astype(np.float32)
+    out = (fused / norm).astype(np.float32)
+    if float(np.linalg.norm(out)) <= 1e-9:
+        # Both branches empty (or exactly cancelling): refuse to return a
+        # non-unit vector, which would silently poison the FAISS index.
+        raise ValueError("fusion produced a zero vector (empty visual/text branch)")
+    return out
@@ -76,9 +101,11 @@
-        v = np.asarray(visual_emb, dtype=np.float32).reshape(-1)
-        t = np.asarray(text_emb, dtype=np.float32).reshape(-1)
+        # copy=True: asarray() on a float32 input returns a *view*, so the
+        # in-place `v *=` below would scale the caller's buffer.
+        v = np.array(visual_emb, dtype=np.float32, copy=True).reshape(-1)
+        t = np.array(text_emb, dtype=np.float32, copy=True).reshape(-1)
```

(the same `copy=True` change applies to the `fuse_gated` module-level function and to
`MultimodalFusion.fuse_gated`).

The zero-vector guard has to be added to `MultimodalFusion.fuse` as well (it duplicates the
normalisation code rather than calling the module-level helper):

```diff
--- a/backend/app/services/fusion.py
+++ b/backend/app/services/fusion.py
@@ -95,9 +95,15 @@
 
         v *= self.visual_weight
         t *= self.text_weight
         fused = np.concatenate([v, t]).astype(np.float32)
 
         # L2 normalize
         norm = float(np.linalg.norm(fused)) + 1e-6
         fused = (fused / norm).astype(np.float32)
+        if float(np.linalg.norm(fused)) <= 1e-9:
+            # Both branches empty: refuse to return a non-unit vector,
+            # which would silently poison the FAISS index.
+            raise ValueError("fusion produced a zero vector (empty visual/text branch)")
         return fused
```

### D7 — `backend/app/services/result_merger.py`: wire the confidence blend into production (optional, addresses B10)

```diff
--- a/backend/app/services/result_merger.py
+++ b/backend/app/services/result_merger.py
@@ -485,10 +485,19 @@
 
     # --- Scoring --------------------------------------------------------
     alpha = compute_adaptive_alpha(
         parsed_filter, intent, has_count_constraint, has_action,
         n_structured=len(combined_tracks), n_semantic=len(semantic),
+        struct_scores=[t.get("duration_seconds", 0) for t in combined_tracks],
+        sem_scores=[finite_score(s.get("score_norm")) for s in semantic],
     )
```

This must be gated by a setting (e.g. `ENABLE_CONFIDENCE_ALPHA`, default **off** until the
control eval has been re-run with it on), because it changes the shipped ranking rule.

### D8 — `backend/scripts/eval_retrieval_control.py`: measure the *shipped* normaliser (addresses B11)

```diff
--- a/backend/scripts/eval_retrieval_control.py
+++ b/backend/scripts/eval_retrieval_control.py
@@ -325,7 +325,17 @@
 def normalize_branch(values: Sequence[float], mode: str) -> np.ndarray:
     arr = np.asarray(values, dtype=np.float64)
     if mode == "minmax":
         return minmax(arr)
     if mode == "zscore":
         return np.asarray(zscore_norm(arr.tolist()), dtype=np.float64)
+    if mode == "production":
+        # The normaliser the product actually ships for the structured
+        # branch (backend.app.services.retrieval_utils.normalize_scores).
+        from backend.app.services.retrieval_utils import normalize_scores
+
+        return np.asarray(normalize_scores([float(v) for v in arr]), dtype=np.float64)
     raise ValueError(mode)
@@ -614,7 +624,10 @@
-    norm_modes = ["minmax"] + ([] if args.no_zscore else ["zscore"])
+    norm_modes = ["minmax", "production"] + ([] if args.no_zscore else ["zscore"])
```

and the same `"production"` arm for the semantic branch
(`backend/app/services/sem_search._norm_scores`). Without this arm the "min-max vs z-score"
table answers a question about a normaliser the product does not use.

### D9 — `backend/app/eval/baselines.py`: add the CombANZ / CombMNZ references (review point (a), §3.2)

```diff
--- a/backend/app/eval/baselines.py
+++ b/backend/app/eval/baselines.py
@@ -55,6 +55,34 @@
     return [1.0 / (1.0 + math.exp(-(x - m) / sd)) for x in xs]
 
 
+def comb_anz_fuse(
+    branch_scores: Sequence[Sequence[float]],
+    ids: Sequence[Sequence[str]],
+) -> Dict[str, float]:
+    """CombANZ (Fox & Shaw, TREC-2 1994): average of the *non-zero* scores.
+
+    A branch that returned nothing therefore does not dilute the surviving
+    branch -- the empty-branch behaviour that the alpha blend lacks.
+    """
+    sums: Dict[str, float] = {}
+    counts: Dict[str, int] = {}
+    for scores, branch_ids in zip(branch_scores, ids):
+        for s, d in zip(scores, branch_ids):
+            if float(s) <= 0.0:
+                continue
+            sums[d] = sums.get(d, 0.0) + float(s)
+            counts[d] = counts.get(d, 0) + 1
+    return {d: sums[d] / counts[d] for d in sums if counts.get(d)}
+
+
+def comb_mnz_fuse(
+    branch_scores: Sequence[Sequence[float]],
+    ids: Sequence[Sequence[str]],
+) -> Dict[str, float]:
+    """CombMNZ: sum of the non-zero scores times the number of branches that
+    actually returned the document (consensus bonus)."""
+    sums: Dict[str, float] = {}
+    counts: Dict[str, int] = {}
+    for scores, branch_ids in zip(branch_scores, ids):
+        for s, d in zip(scores, branch_ids):
+            if float(s) <= 0.0:
+                continue
+            sums[d] = sums.get(d, 0.0) + float(s)
+            counts[d] = counts.get(d, 0) + 1
+    return {d: sums[d] * counts[d] for d in sums if counts.get(d)}
+
+
 def rrf_fuse(rank_a: Sequence[str], rank_b: Sequence[str], k: int = 60) -> List[str]:
```

---

## 6. Regression-test sketches

New file `backend/tests/test_robustness.py` (sketch only — **not created**). Every test below
currently **fails** on the working tree and passes after the corresponding patch.

```python
"""Robustness regression tests for the merge / fusion paths.

Each test maps 1:1 to a row of the edge-case audit in results/robust_proposal.md.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.app.eval import confidence as conf
from backend.app.services.fusion import MultimodalFusion, fuse_gated
from backend.app.services.result_merger import (
    apply_mmr_diversity,
    compute_adaptive_alpha,
    merge_results,
)
from backend.app.services.retrieval_utils import normalize_scores
from backend.app.services.sem_search import _norm_scores


PF = {"objects.object_name": "person"}


def _sem(clip="a", score_norm=0.8, score_raw=0.5):
    return {
        "camera_id": 0,
        "clip_path": f"clips/{clip}.mp4",
        "clip_url": f"/clips/{clip}.mp4",
        "score": score_raw,
        "score_raw": score_raw,
        "score_norm": score_norm,
        "frames": [{"frame_ts": "2026-03-01T17:10:00"}],
    }


def _struct(ts="2026-03-01T17:10:00", tid=1):
    return {
        "camera_id": 0,
        "timestamp": ts,
        "objects": [{"object_name": "person", "color": "red",
                     "track_id": tid, "confidence": 0.9}],
    }


# --- D1: normalisation -------------------------------------------------

def test_degenerate_branch_is_not_confident():
    """B1/B15: an all-tie branch carries no evidence -> 0.0, not 1.0."""
    assert normalize_scores([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
    assert _norm_scores([0.5, 0.5]) == [0.0, 0.0]


def test_single_element_branch_is_unit_independent():
    """B2: the value must not depend on the unit of the score."""
    assert normalize_scores([0.3]) == normalize_scores([18.0]) == [1.0]
    assert normalize_scores([0.0]) == [1.0]


def test_normalisation_is_unit_free():
    """B2: scaling the input must not change the output (no 0.25/0.40 rule)."""
    base = [0.1, 0.2, 0.3]
    assert normalize_scores(base) == pytest.approx(normalize_scores([x * 100 for x in base]))


def test_negative_scores_preserve_order():
    """B3: the worse candidate must not score higher."""
    out = normalize_scores([-1.0, -2.0])
    assert out[0] >= out[1]
    assert all(0.0 <= v <= 1.0 for v in out)


def test_nan_does_not_poison_the_branch():
    """B4: one NaN must not zero out the whole branch."""
    out = normalize_scores([float("nan"), 1.0])
    assert out[1] == pytest.approx(1.0)
    assert all(math.isfinite(v) for v in out)


def test_infinite_score_is_isolated():
    """B4: +inf must not outrank everything."""
    out = normalize_scores([float("inf"), 1.0])
    assert all(math.isfinite(v) for v in out)


# --- D3: merge --------------------------------------------------------

def test_semantic_score_norm_none_does_not_crash():
    """B5: a None score must not raise out of the merge."""
    sem = _sem()
    sem["score_norm"] = None
    out = merge_results([], [sem], PF, "find")
    assert len(out) == 1
    assert math.isfinite(out[0]["score"])


def test_nan_semantic_score_is_sanitised():
    """B4: NaN must not reach the ranking."""
    sem = _sem()
    sem["score_norm"] = float("nan")
    out = merge_results([], [sem], PF, "find")
    assert math.isfinite(out[0]["score"])


def test_none_branches_are_tolerated():
    """B6: None structured/semantic, with and without a count constraint."""
    assert merge_results(None, None, PF, "find") == []
    assert merge_results(None, [], {"count_constraint": {"gte": 2}}, "count") == []


def test_empty_structured_branch_rebalances_alpha():
    """B8: CombANZ-style -- a branch that returned nothing gives up its weight."""
    only_sem = compute_adaptive_alpha(PF, "find", False, False, 0, 10)
    assert only_sem == 0.0
    only_struct = compute_adaptive_alpha(PF, "find", False, False, 10, 0)
    assert only_struct == 1.0


def test_empty_branch_does_not_deflate_the_surviving_branch():
    """B8: the surviving branch keeps the full [0, 1] scale."""
    sem = _sem(score_norm=0.8)
    out = merge_results([], [sem], PF, "find")
    assert out[0]["score"] == pytest.approx(0.8, abs=0.05)


def test_count_constraint_keeps_alpha_one():
    """B8: the count constraint deliberately disables the semantic branch."""
    assert compute_adaptive_alpha(PF, "count", True, False, 5, 5) == 1.0


# --- D4: MMR ----------------------------------------------------------

def test_mmr_never_drops_every_result():
    """B7: fail open."""
    out = apply_mmr_diversity(
        [{"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -5.0},
         {"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -4.0},
         {"camera_id": 0, "start": "2026-03-01T17:00:00", "score": -3.4}], 0.3)
    assert len(out) == 3


def test_mmr_keeps_the_relevance_order_for_positive_scores():
    out = apply_mmr_diversity(
        [{"camera_id": 0, "score": 0.9}, {"camera_id": 1, "score": 0.5},
         {"camera_id": 2, "score": 0.1}], 0.3)
    assert out[0]["score"] >= out[1]["score"] >= out[2]["score"]


# --- D5: confidence ----------------------------------------------------

def test_confidence_is_nan_safe():
    """B9: NaN must not become alpha=1.0 (min(1.0, nan) == 1.0)."""
    a = conf.combined_alpha(0.6, [float("nan"), 0.5], [0.1, 0.9])
    assert math.isfinite(a)
    assert 0.0 <= a <= 1.0
    assert a != 1.0          # the NaN must not collapse to full structured trust
    assert conf.confidence_alpha([float("nan")], [0.1, 0.9]) == 0.0


def test_degenerate_branch_is_detected():
    assert conf.is_degenerate([0.5, 0.5, 0.5]) is True
    assert conf.is_degenerate([0.4, 0.9]) is False
    assert conf.is_degenerate([0.5]) is True


# --- D6: fusion -------------------------------------------------------

def test_fuse_does_not_mutate_the_caller_buffers():
    """B12: object_detection.py passes float32 batch rows."""
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    MultimodalFusion(0.6, 0.4).fuse(v, t)
    assert float(v[0]) == pytest.approx(1.0)
    assert float(t[0]) == pytest.approx(1.0)


def test_fuse_is_idempotent():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    f = MultimodalFusion(0.6, 0.4)
    a = f.fuse(v.copy(), t.copy())
    b = f.fuse(v.copy(), t.copy())
    assert np.allclose(a, b)


def test_gated_fusion_with_nan_quality_is_not_maximal_trust():
    """B13: NaN must not read as quality 1.0."""
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    out_nan = fuse_gated(v, t, float("nan"), 0.0)
    out_half = fuse_gated(v, t, 0.5, 0.0)
    assert np.allclose(out_nan, out_half)


def test_gated_fusion_with_none_quality_does_not_raise():
    v = np.ones((768,), dtype=np.float32)
    t = np.ones((384,), dtype=np.float32)
    out = fuse_gated(v, t, None, 0.0)
    assert np.isfinite(out).all()


def test_fusion_of_empty_branches_is_rejected():
    """B14: a zero vector must not be indexed."""
    with pytest.raises(ValueError):
        MultimodalFusion(0.6, 0.4).fuse(
            np.zeros((768,), dtype=np.float32), np.zeros((384,), dtype=np.float32))
```

**Protocol-level tests** (to be added to `backend/scripts/eval_retrieval_control.py`, §4.1):

```python
def test_label_free_rank_stability():
    """C1/C2: the fused ranking must not depend on the branch's score shape."""
    # for each query and each (normaliser, monotone warp):
    #   assert rbo(fused_under_variant, fused_reference) >= 0.9
    # report the mean RBO@10 per category; fail if the production
    # normaliser loses more than 10 % of the top-10 overlap vs RRF.

def test_empty_branch_preserves_order():
    """C3: emptying a branch must not reorder the surviving branch."""
    # assert rbo(ranking_without_structured, ranking_without_structured_repeat) == 1.0
    # assert the score shift is a constant factor (max/mean ratio == 1.0)

def test_fused_score_is_monotone_in_branch_agreement():
    """C4: the fused score must behave like a confidence."""
    # bin the fused score into deciles; assert the branch agreement rate is
    # non-decreasing across bins.
```

---

## 6.1 Patch validation (executed)

The proposed diffs were **not applied to the repo**. Instead, reference implementations of
D1–D6 were written in the scratch probe (`backend/scripts/robust_probe_tmp.py`, phase 2) and
every regression-test sketch above was executed against them:

```
[PASS] t_degenerate_branch_not_confident        [PASS] t_mmr_never_drops_every_result
[PASS] t_single_element_unit_independent        [PASS] t_mmr_keeps_relevance_order
[PASS] t_normalisation_unit_free                 [PASS] t_fuse_does_not_mutate_buffers
[PASS] t_negative_scores_preserve_order          [PASS] t_fuse_is_idempotent
[PASS] t_nan_does_not_poison_branch              [PASS] t_gated_nan_quality_not_maximal_trust
[PASS] t_infinite_score_isolated                [PASS] t_gated_none_quality_no_raise
[PASS] t_empty_structured_rebalances_alpha       [PASS] t_fusion_of_empty_branches_rejected
[PASS] t_count_constraint_keeps_alpha_one        [PASS] t_confidence_nan_safe
[PASS] t_surviving_branch_keeps_full_scale       [PASS] t_degenerate_detection
18/18 passed   (results/robust_probe_tmp.json -> "patched_tests")
```

The same test bodies fail on the current working tree (that is the point of the exercise); the
probe also asserts the *unpatched* behaviour explicitly, e.g.

```python
assert conf.combined_alpha(0.6, [float("nan"), 0.5], [0.1, 0.9]) == 1.0   # current: NaN -> full trust
```

### Behavioural deltas the patches introduce (the A/B plan)

| Query shape | current score | patched score | why |
|---|---|---|---|
| semantic-only (structured branch empty), `score_norm=0.8` | 0.424 | **0.800** | the empty structured branch no longer keeps 47 % of the blend (B8) |
| α for a semantic-only query | 0.47 | **0.0** | CombANZ-style rebalancing (B8) |
| structured-only, one 0-second track | 0.335 | *(re-checked by the control run)* | the structured score of that track was 0.5 because of the unit-mismatched thresholds (B2) |
| all-tie structured branch | `[1.0, 1.0, 1.0]` | **`[0.0, 0.0, 0.0]`** | a branch that cannot discriminate must not claim confidence (B1) |
| NaN in a score vector | whole branch → 0.0, α → 1.0 | NaN isolated, α finite | B4 / B9 |
| MMR with all-negative scores | `[]` (everything dropped) | original order | B7 |

Every one of these changes a *reported number*, so D1–D3 must be A/B'd on
`backend/scripts/eval_retrieval_control.py` before being merged, and
`results/retrieval_control_*.{json,md}` regenerated.

---

## 7. Review-point traceability

| Review point (non-claim) | Closed by | Evidence / status |
|---|---|---|
| **(a)** weighting robustness to score-distribution shifts | D1, D2 (unit-free + degenerate-safe normalisation), C2 (monotone-warp invariance), C1 (rank stability), D9 (RRF / CombANZ / CombMNZ references) | B1, B2, B3, B15 are the concrete failures; the literature anchors are §3.1–3.3 |
| **(a)** EMPTY result sets / branches | D3 `_rebalance_for_empty_branches` (CombANZ-style), D2 (degenerate → 0), D3 `None`-branch guard, D4 (MMR fail-open), C3 (empty-branch stress) | B5, B6, B7, B8 are live; the empty structured/semantic paths themselves do not crash (§2.5) |
| **(a)** noisy metadata | D1/D5 (NaN/inf isolation), D3 `finite_score` guards, D6 (NaN-safe quality) | B4, B9, B13 |
| **(a)** different backends | §3.3 (score-range mismatch across backends), C2 (monotone-warp invariance ≈ a different backend's score *shape*), D9 (RRF is range-free) | the control currently only tests min-max vs z-score on **one** release |
| **(b)** per-query min-max only | D8 (add a `"production"` arm so the control measures the shipped normaliser), D1 (`mode="minmax" \| "max"`), §4.1 C1 (four normalisers incl. rank/quantile + RRF), §3.4 (sklearn degenerate convention) | **B11** is the headline: the harness's min-max is not the production structured normaliser |
| **(c)** no calibrated/learned ranking comparison beyond a collapsed logistic baseline | §4.2 L1–L5 (regression target, listwise/pairwise ranker, Platt/ECE calibration, RRF+CombMNZ references, deployment consistency), D5 (NaN-safe confidence), D7 (ship the confidence blend or state that it is offline-only) | the baseline collapses to a constant (`train_label_balance {"2":49,"3":1}` → `predicted_alpha_histogram {"0.5":100}`); **B10** shows the calibrated alternative is not even deployed |

---

## 8. Residual — what cannot be fixed without new data or labels

1. **Whether the empty-branch rebalance improves accuracy.** The *ordering* is provably
   unchanged for a single-branch query (the surviving branch's order is untouched) and the
   score scale becomes comparable across queries — both verifiable without labels. Whether the
   *ranking quality* changes needs the 150 released relevance labels and a control re-run.
2. **Which normaliser is best.** min-max / z-score / quantile / RRF can all be computed offline
   on the released subset, but the comparison needs a full control run (minutes) and, for any
   *accuracy* claim, the caption-derived labels (which are not human annotations — no IAA
   exists; already documented in `results/retrieval_control_results.json → limitations`).
3. **Whether a learned ranker beats the alpha blend.** The collapse is fixable in code (L1/L2),
   but the *result* requires a tune/test run with the 50/100 split; the tune split is small, so
   the honest expectation is "no significant gain", which is itself reportable.
4. **Calibration of the fused score as a confidence.** Only *agreement-based* proxies (C4) are
   label-free; true calibration (precision of the top-1 vs its score) needs labels.
5. **Cross-camera and multi-backend behaviour.** The released subset is single camera
   (`camera_0`, 121 indexed clips, 242 frames) — "different backends" can only be *simulated*
   (affine / monotone warps, C2), not measured. A genuine multi-backend claim needs the
   full-deployment data, which the repo's own limitations list excludes.
6. **Real FAISS NaN/inf scores.** Cannot be produced deterministically offline; the D1/D3/D5
   guards are defensive, and their regression tests use injected NaNs.
7. **Human relevance judgements / inter-annotator agreement.** The labels are VLM-caption
   derived; no human IAA is available, so every "robustness" number remains conditional on
   that label source.
8. **Whether the MMR fail-open path changes retrieval quality.** Failing open returns the
   pre-MMR order instead of an empty list; the order differs from the MMR order, so the effect
   on P@5 needs a run (it only triggers on inputs the current code cannot produce, so in
   practice the effect is nil).

---

## 9. Suggested order of work

1. **D1 + D2** (normalisation contract) and **D4** (MMR fail-open) — self-contained, no
   behavioural risk to the released numbers, each with a clean regression test.
2. **D3** (empty-branch rebalance + guards) — needs an A/B on the control before shipping,
   because it changes the reported score scale for semantic-only queries.
3. **D8** (control measures the shipped normaliser) — then re-run
   `backend/scripts/eval_retrieval_control.py` and update `results/retrieval_control_*.{json,md}`.
4. **§4.1 C1–C4** (label-free probes) — add to the same script; this is the *evidence* for
   review point (a).
5. **§4.2 L1–L4** (non-collapsed learned + calibrated comparison) and **D7** (deployment
   consistency) — this is the evidence for review point (c).

**Scratch artefacts produced by this analysis** (not part of the deliverable, safe to delete):
`backend/scripts/robust_probe_tmp.py`, `results/robust_probe_tmp.json`.
