# Safe-fixes proposal — privacy/fairness safeguards, threat model, and benchmark representativeness

**Scope of this document.** Peer-review issues assigned to this round (non-claim issues only):
(1) privacy/fairness safeguards incomplete; (2) no threat model, access restrictions, or
retention policy distinguishing **ACTIVE** vs **SUPPLIED** safeguards; (3) thin
benchmark-representativeness caveat.

**Paper under edit:** `INDICON2026_submit/INDICON2026Conference.tex` (6 pages, verified).
**Method:** (a) code audit of `backend/app/` with file:line evidence — claims in the paper
were *verified against the code*, not taken from comments; (b) web search of current
top-venue requirements; (c) exact replacement prose with a measured line budget.
**No tracked file was modified.** This document is the only file written.

**Bottom line (3 sentences).** The paper's Limitations (iv) is *honest* but *imprecise*:
it calls all four safeguards "disabled by default", which is true for access control /
audit / retention / face blur but **false for the gender-off filter** (gender is **ON** by
default; only the filter is off), and it does not say that the audit trail is *gated on*
access control, that the retention prune covers **media files only** (MongoDB metadata,
both FAISS indices and `audit.jsonl` have no TTL and no rotation), or that the access
control's camera scope is enforced on **query-parameter identifiers only**. The paper also
has **10.5 free body lines** on page 6 (measured from the compiled PDF), which is enough
for a compressed in-body disclosure with **zero cuts** — or **~12 lines of named cuts**
buy the full disclosure. No demographic-bias number can be produced (no demographic
labels exist); what *can* be stated is the attribute head's own output distribution
(76.5% abstention over the 1,778 released descriptors), which is a model output and not a
demographic fact.

---

## PART 1 — Code audit: active vs supplied vs utility-only (verified)

Verdict vocabulary used in this document:

| Verdict | Meaning |
| --- | --- |
| **ACTIVE** | Enforced on the default request path of the released configuration. |
| **SUPPLIED** | Implemented, unit-tested, but **off by default**; arming it is an operator step. |
| **UTILITY** | Implemented and tested, but **not on any live path**; a callable helper only. |
| **NOT PRESENT** | No mechanism exists in the repo. |

### 1.1 Role / camera / zone access control — **SUPPLIED** (opt-in)

| Fact | Evidence |
| --- | --- |
| Default is **off**: `ACCESS_CONTROL_ENABLED` defaults to `"false"` | `backend/app/config.py:192-194` |
| Middleware is registered unconditionally but **no-ops** when the flag is false | `backend/app/main.py:277`; `backend/app/core/security.py:313-315` |
| Role is derived from **HTTP method + path class**, not declared per route (GET→viewer, non-GET on `/api/settings`→admin, other non-GET→operator) | `backend/app/core/security.py:290-296` |
| Identity comes from **trusted-proxy headers** (`X-Role`, `X-Actor`, `X-Allowed-Cameras`, `X-Allowed-Zones`); the app validates **no bearer token** | `backend/app/core/security.py:134-158`, docstring `security.py:5-19` |
| Camera/zone scope is checked **only** on `request.query_params["camera_id"|"zone"]` and on a regex-extracted camera id from the **path** | `backend/app/core/security.py:280-288`, `359-375` |
| `require_role(...)` (per-route dependency) and `filter_authorized_results(...)` (record-level filtering) are **never called by any router** — only by `backend/tests/test_privacy.py` | grep over `backend/app/routers/*.py`: zero hits |
| **Consequence:** the natural-language camera filter in `/api/chat/send` arrives in the **JSON body** and is parsed server-side (`backend/app/routers/chat.py:296-310`), so a camera named inside a query is **not** scope-checked | `chat.py:296-310`, `security.py:359-375` |
| Static mounts `/media/recordings|clips|snapshots` are served by `StaticFiles` | `backend/app/main.py:297-299` (mounts bypass router-level logic; the middleware does see `/media` paths) |

**Verdict:** SUPPLIED. A paper sentence that says "the release ships access control with
audit logging" is true, but a sentence that says "camera-scoped access control is
enforced" would be **an overclaim** unless it is qualified as "on query-parameter
identifiers only".

### 1.2 Audit logging — **SUPPLIED**, and *gated on access control*

| Fact | Evidence |
| --- | --- |
| The **only** callers of `audit_log(...)` are inside `AccessControlMiddleware` | `backend/app/core/security.py:324, 343, 381, 399, 416` |
| With `ACCESS_CONTROL_ENABLED=false` **no audit record is ever written** | `security.py:313-315` (early return) — asserted by `backend/tests/test_privacy.py:115-127` (`assert not (tmp_path/"audit.jsonl").exists()`) |
| Audit trail is a JSONL file `data/logs/audit.jsonl`, append-only, best-effort, never raises | `security.py:215-274` |
| It is **not** a loguru sink, so it has **no rotation and no retention** (the app log has `retention="30 days"`, the audit trail has none) | `backend/app/core/logging_config.py:72-96` vs `security.py:258-262` (plain `open("a")`) |
| Mongo mirroring of the trail is off by default (`AUDIT_MONGO_ENABLED=false`) | `backend/app/config.py:205-207`; `security.py:267-273` |

**Verdict:** SUPPLIED, and **not independent of access control** — it is a *sub-feature* of
it. "Audit logging is shipped" is true; "audit logging is active" is false.

### 1.3 Retention prune — **SUPPLIED** (callable, never automatic) and **media-files-only**

| Fact | Evidence |
| --- | --- |
| `RETENTION_ENABLED` defaults to `"false"`; `RETENTION_HOURS` default `8` | `backend/app/config.py:211-214` |
| `run_retention_policy()` refuses to act unless the flag is true (or `force=True`) | `backend/app/core/privacy.py:161-167` |
| **Nothing in the app calls it.** No startup hook, no scheduler, no CLI in `backend/scripts/` | grep: only `privacy.py` and `backend/tests/test_privacy.py`; module docstring `privacy.py:5-8` |
| It prunes **only files** under `CLIPS_DIR`, `SNAPSHOTS_DIR`, `RECORDINGS_DIR` | `privacy.py:35-42` (`_media_dirs`) |
| It does **not** touch MongoDB collections (`detections`, `tracks`, `alerts`, `alert_logs`, `chat_messages`, `vlm_frames`), **not** the two FAISS indices, and **not** `audit.jsonl` | `privacy.py:35-42`; `backend/app/db/mongo.py:40-109` (index list contains **no** `expireAfterSeconds`) |
| The only automatic expiry mechanism in the repo is a **full wipe** (`DELETE /api/settings/reset`), which drops all Mongo collections and clears media + indices | `backend/app/routers/settings.py:113-124`; `backend/app/services/cleanup.py:17-68` (admin-only when access control is on) |
| Dry-run is the default for the destructive call (`dry_run=True`) | `privacy.py:65, 149-150` |

**Verdict:** SUPPLIED, **operator-invoked**, **files only**. The paper's current phrasing
"an 8-h retention prune" is right; it must not be read as "the system expires data after
8 h" — the metadata store and both indices do **not** expire.

### 1.4 Face blur — **UTILITY** (not wired into live ingestion)

| Fact | Evidence |
| --- | --- |
| `FACE_BLUR_ENABLED` defaults to `"false"`; `blur_faces_in_file()` refuses to modify a file unless the flag is true or `force=True` is passed | `backend/app/config.py:215-217`; `backend/app/core/privacy.py:286-293` |
| The implementation is OpenCV's **bundled Haar cascade** (frontal faces only) | `privacy.py:180-216` |
| **Zero calls anywhere in the runtime.** The only `blur` references in `backend/` are inside `privacy.py` itself and inside `backend/tests/test_privacy.py` | grep `blur` over `backend/` → `privacy.py` + `test_privacy.py` only |
| In particular the **live capture path** (`backend/object_detection.py`, `process_live_stream`) and the **clip builder** (`backend/app/services/clip_builder.py`) never call it | grep confirms |
| The repo's own note already says it is "not wired into the live ingestion/streaming pipeline or into clip generation" | `results/privacy_notes.md:88-90` |

**Verdict:** UTILITY. The paper's "face-blur utility … not wired into live ingestion" is
**accurate**. Note the consequence that must be stated: because blur is not on the frame
path, **exported clips and the stored person gallery are not anonymised**.

### 1.5 Gender-off filter — **ACTIVE on output only**; gender itself is **ON by default**

| Fact | Evidence |
| --- | --- |
| `INCLUDE_GENDER_ATTRIBUTES` defaults to **`true`** → gender text is produced by default | `backend/app/config.py:218-223` |
| Index/caption gate: `attributes_to_text()` appends `"male"`/`"female"` when the OpenVINO `male_confidence` is ≥0.65 or ≤0.35, and only if the flag is true | `backend/app/services/attribute_encoder.py:61-67` |
| Live ingestion writes that text into MongoDB detection documents (`objects[].attribute_text`) | `backend/object_detection.py:1101-1114`, `1455-1467` |
| Output filter: `hybrid_search.py` strips gender tokens from person-search results (recursively into `objects`, `person_hits`, `frames`, `detection`) **only when the flag is false** | `backend/app/services/hybrid_search.py:458-461`; `backend/app/core/privacy.py:359-392` |
| It **does not rewrite or rebuild** the stored index; the stored text (and hence the MiniLM embedding) still carries gender | `backend/app/core/privacy.py:12-14` (docstring); `results/privacy_notes.md:102-105` |
| The gender-off path is **not** applied to the raw `/api/detections` response, which returns `objects[].person_attributes` (including `male_confidence`) and `attribute_text` verbatim | `backend/app/routers/detections.py:18-30, 63-82` |

**Verdict:** the *filter* is SUPPLIED (off by default ⇒ **gender is retained by default**);
the *capability to strip* is ACTIVE **on the person-search output path only**. The
current paper sentence groups it under "disabled by default", which is technically true of
the filter but misleading: the safe state is *not* the default state.

**Measured facts about the released gallery** (from `data/index/faiss_person/meta.json`,
1,778 descriptors, camera 0, 2026-03-01 17:01→18:56, i.e. a **1.92-hour** window):

| Quantity | Value |
| --- | --- |
| descriptors carrying a predicted `male` term | **64 / 1,778 (3.60%)** |
| descriptors carrying a predicted `female` term | **354 / 1,778 (19.91%)** |
| descriptors where the head **abstained** (0.35 < p < 0.65) | **1,360 / 1,778 (76.49%)** |

These are **model outputs, not demographic facts**, and are *not* a fairness metric — but
they are the only demographic-adjacent quantity that is actually computable from the
release, and they are directly relevant: the head abstains on three quarters of the ROIs,
which is evidence that the attribute is a weak, low-coverage cue rather than a load-bearing
demographic claim.

### 1.6 Alert clarify/abstain validator — **ACTIVE**

Deterministic validator, always on, refuses to emit a rule it cannot validate. The one
safeguard in the release that is on by default. It is the paper's C1 contribution, so
calling it an "active safeguard" is also a nice framing of C1's safety property.

### 1.7 What is **NOT PRESENT** in the repository

| Missing item | Evidence |
| --- | --- |
| Any **threat model** (assets / adversaries / trust boundaries / STRIDE-style analysis) | grep for `threat model|threat_model|STRIDE|insider|spoof` over all `*.md` → one hit, `results/privacy_notes.md:48` ("would be spoofable, so this is a policy hook, not a complete authn system") — i.e. a caveat, not a model |
| Any **retention policy document** (per-store retention rules) | `results/privacy_notes.md` documents the *switches*, not a policy; no `retention_policy.md` exists |
| Any **access-restriction specification** beyond the header contract | `security.py:5-19` is the only statement, and it is a docstring |
| **MongoDB TTL** indexes on any collection | `backend/app/db/mongo.py:40-109` — no `expireAfterSeconds` anywhere in the repo |
| **Audit-log rotation** | `security.py:258-262` appends to a plain file; the 30-day rotation in `logging_config.py:78,91` applies only to `app_*.log` |
| **Consent / notice artefacts** (signage, consent log, DPIA, IRB record) | nothing in the repo; the paper says "notified premises" only |

---

## PART 2 — What top-tier venues now require (web research, with sources)

### 2.1 Checklist

| # | Requirement | Where it comes from | Does this paper meet it? |
| --- | --- | --- | --- |
| R1 | **Ethical-considerations statement is mandatory** for papers that "describe experiments with human subjects and/or deployed systems (e.g., websites or apps)" or "rely on sensitive user data"; must cover "compliance with applicable laws, respect for privacy, secure storage of sensitive data, voluntary and informed consent when appropriate, … risk mitigation; and post-hoc disclosure of audits" | ACM FAccT 2026/2027 Author Guide, *Endmatter Sections* / *Ethical Considerations Statement* | **Partial.** The paper has Limitations (iv) but no statement that names consent basis, applicable law, or what was disclosed post-hoc. Also: at ACM this statement is **endmatter that does not count towards the page limit** — INDICON has no such slot, which is why the appendix route (Part 4, Route 1) matters. |
| R2 | **Papers are flagged for review** on "improper treatment of human subjects", "privacy concerns", "discrimination, bias, and fairness issues", "legal concerns"; authors are "encouraged to proactively address any such concerns directly in their paper" | ACM FAccT 2026 Author Guide, *Policies and Research Integrity* | **Partial.** Needs proactive, specific disclosure (not a generic caveat). |
| R3 | **Error rates must be disaggregated by sex, race and other context-dependent demographic features, as appropriate**; accuracy must be "fully auditable"; risk management must include "multi-factor impact assessment and risk analysis"; no FR/biometric system should be deployed "prior to establishing appropriate policies governing its use and the management of data collected by the system" | ACM USTPC, *Statement on Principles and Prerequisites for the Development, Evaluation and Use of Unbiased Facial Recognition Technologies* | **Not met, and cannot be met** (no demographic labels). The reviewable part is the *disclosure*: state that disaggregation is the standard, state that no labels exist, state what is measurable instead, and state the protocol that would produce the missing numbers. |
| R4 | Retention policies must be "legally compliant, transparent to the public, and limit data retention to what is strictly necessary"; "data retention policies and practices should be… context-specific", with **context-aware deletion**; systems should minimise "the quantity and richness of any data retained" | USTPC (Governance); NIST AI RMF *Application to Surveillance Technology* (2024), arXiv:2403.15646 | **Not met as stated.** No per-store retention policy; the 8-h prune covers media files only; metadata/indices/audit trail never expire. Fixable in prose (state the policy as it actually is). |
| R5 | **Automatic logging** of system operation, retained "for a period appropriate to the intended purpose… of at least six months" (Art. 12/19/26 EU AI Act) | EU AI Act Art. 12 (record-keeping), Art. 19 (provider retention), Art. 26(6) (deployer retention) | **Not met.** Audit trail is off by default and has no rotation; this is the exact opposite of "logs retained at least six months" — the paper must not imply otherwise. |
| R6 | **Threat modelling** is the standard method for identifying risks: scope the application, identify entry/exit points, trust boundaries and assets; enumerate threats; determine countermeasures; assess residual risk | AIES 2025, *Secure Human Oversight of AI: Threat Modeling in a Socio-Technical Context* (doi 10.1609/aies.v9i2.43187), STRIDE-based | **Not met.** No threat model anywhere. Fixable in prose (Part 4). |
| R7 | Accuracy measurements "should always be paired with clearly defined and realistic test sets – that are representative of conditions of expected use"; "may include **disaggregation** of results for different data segments" | NIST AI RMF 1.0 §3.1 (Valid and Reliable) | **Partial.** The deployment benchmark's per-query export, annotator roster and split were not preserved, so the representativeness claim is unverifiable. |
| R8 | Video-surveillance-specific expectations: **privacy impact assessment**, **default privacy settings**, **purpose limitation**, **data minimisation**, **data-storage limits** ("only store video footage for as long as is reasonably necessary"), **data access restriction** ("define rules stating who has access and when and for what purpose"), **transparency/notice** | Security Industry Association, *Data Privacy Code of Practice: Video Surveillance* (2025) | **Partial.** The release has defaults (all-off) and access primitives; no PIA, no notice artefact, no storage-limit policy for metadata/indices. |
| R9 | Lawful basis, **consent or specified legitimate use**, purpose limitation, security safeguards, and **erasure once the specified purpose is no longer served** (§8(7)) | India, *Digital Personal Data Protection Act 2023* (MeitY), §4, §7(i), §8(5), §8(7) — directly applicable (capture in notified premises in India) | **Partial.** §7(i) (employer safeguarding) is a plausible basis for the notified-premises capture; erasure-on-purpose-ended is only partially met (files yes, metadata no). |
| R10 | When demographic labels are unavailable, the fairness literature's explicit response is **not** to skip the analysis but to use proxies, auxiliary data, **or complementary mechanisms** (documentation, model cards/datasheets, exploratory fairness analysis, audits, complaints) — and to state the assumptions | Ashurst & Weller, *Fairness Without Demographic Data: A Survey of Approaches*, EAAMO '23 (doi 10.1145/3617694.3623234); Gebru et al., *Datasheets for Datasets*, CACM 2021; Mitchell et al., *Model Cards*, FAT* '19 | **Not met.** The paper says "no demographic-bias evaluation is possible" and stops. The surveyable answer: document composition, state the label-absence explicitly, and specify the collection protocol. |
| R11 | A binary gender label "is a more complex construct" reduced to two classes; the standard audit reports **intersectional** error rates (darker/lighter × female/male) | Buolamwini & Gebru, *Gender Shades*, FAccT '18, PMLR 81 | **Not met** — and the citation is what makes the paper's "coarse visual proxy" claim citable rather than an assertion. |

### 2.2 INDICON 2026 constraints that bind the fix

| Constraint | Source | Consequence here |
| --- | --- | --- |
| Max **6 pages** for the full-registration paper; **up to 2 extra pages at ₹1000 (~$18) per page** at camera-ready | indicon2026.com *Call For Papers* (registration) | **The appendix route is affordable.** A 7th page costs ₹1000 and needs zero cuts. |
| Submission limit is 8 pages (4–8 inclusive) | indicon2026.com *Guidelines & Policies* | A camera-ready 7-page version was always admissible. |
| "Ensures a minimum of 5 pages and **10 quality references**"; "**Reference not to be cited in the Abstract and Conclusion part**" | INDICON 2026 camera-ready notification | All proposed citations must sit in the body or Limitations — **never** the abstract or the Conclusion. The repo's `References.bib` header says "15 references max" and all 15 are cited, so **each new reference must displace one** (Part 6). |
| Camera-ready: "Ensure the ethical guidelines in the final camera-ready paper submission" | INDICON 2026 camera-ready notification | Direct instruction to add the ethics/safeguards content — the reviewers' request aligns with the conference checklist. |

---

## PART 3 — Measured page budget (from the compiled PDF, not estimated)

`INDICON2026_submit/INDICON2026Conference.pdf` — 6 pages, US Letter 612×792 pt.
Measured with PyMuPDF:

| Page | Left column last text bottom (pt) | Right column last text bottom (pt) | Free lines in right column |
| --- | --- | --- | --- |
| 1 | 700.45 | 718.85 | 0 |
| 2 | 718.37 | 718.85 | 0 |
| 3 | 719.62 | 718.85 | 0 |
| 4 | 719.46 | 718.85 | 0 |
| 5 | 718.85 | 718.85 | 0 |
| 6 | **718.85** | **593.44** | **≈10.5** |

* Text-area bottom = 718.85 pt; body line pitch = 11.95 pt (measured from consecutive line
  baselines on page 6).
* Page 6's right column ends at the **last bibliography entry** ([15] MiniLM, y=593.44).
  Everything added to the body pushes the bibliography down by the same number of lines.
* **⇒ The paper already has ≈10.5 free body lines before any cut.**
* Line budget convention used below: **1 body line ≈ 56 characters**. Calibrated on this
  exact document, not guessed: the median body-line width extracted from the compiled PDF
  is **56.5 characters**, and the current 11-line Limitations item (iv) contains 618
  characters → 618 / 11 = 56.2 chars/line, i.e. the two independent measurements agree.
  (The *mean* over all pages is lower — 42.8 — because it is dragged down by table rows,
  headings and short last lines; the median is the correct statistic for flowing text.)
* Character counts are given for every block so the numbers can be re-derived.
* **Caveat:** this arithmetic is exact for *prose*. A **float** (e.g. a table) is placed at
  a column top and re-flows the page; it must be validated by recompiling, not by
  arithmetic.

---

## PART 4 — Proposed prose (exact LaTeX), with placement anchors and line counts

### 4.1 BLOCK-B1C — replacement of Limitations (iv) [15 lines, replaces the current 11]

**Placement anchor:** in `INDICON2026Conference.tex`, inside the `\textbf{Limitations.}`
paragraph (line 295), replace the text from `(iv)~\emph{Demographic bias, privacy and
consent.}` to the end of that paragraph (`... demographic-bias measurement is claimed.`)
with the following. It becomes items **(iv)** and **(v)**. Note the current item (iv) is
**11 lines long** in the PDF (6 lines at the bottom of page 6's left column + 5 at the top
of the right column) and contains 618 characters, so this block is a **net +4 lines**, not
+15.

```latex
(iv)~\emph{Benchmark representativeness.} The 150-query deployment benchmark covers one premises, four cameras and one $\sim$8-h daytime capture; its per-query export, annotator roster and split were not preserved, so the Table~\ref{tab:alpha_baseline} macro-means are single-run estimates with non-recomputable CIs ($n{=}30$ per category); the released single-camera control is the only re-computable benchmark and it does not reproduce the gap. (v)~\emph{Privacy and fairness.} Accuracy degrades outside adult daylight imagery. No demographic labels or second annotator exist, so no inter-annotator agreement, cross-camera generalisation or per-group error rate is claimed; gender remains a soft cue only (76.5\% abstention over 1{,}778 released descriptors). Live capture was in notified premises and face imagery was pruned per camera at 8~h.
```

*819 characters → 15 lines (at 56 chars/line).*

**What changed and why**

| Change | Reason (audit ref) |
| --- | --- |
| "no demographic-bias evaluation is possible without demographic labels" → "No demographic labels or second annotator exist, so no inter-annotator agreement, cross-camera generalisation or per-group error rate is claimed" | Same honesty, but names the *three* things not claimed and preserves the original "cross-camera generalisation" caveat verbatim. |
| **Adds a measurable number**: 76.5% abstention over 1,778 released descriptors | §1.5 — the only demographic-adjacent quantity computable from the release; converts "we cannot measure bias" into "here is what we *can* measure, and here is why it is not a fairness metric". |
| "face imagery retained per-camera for at most 8 h" → "face imagery was pruned per camera at 8 h" | The verb "pruned" makes the *mechanism* explicit (operator-invoked prune), matching Part 1 §1.3. |
| Splits the item into **(iv) representativeness** and **(v) privacy/fairness** | Gives the reviewer two addressable items instead of one mixed caveat; the representativeness caveat is no longer buried inside a privacy paragraph. |
| Drops from (iv): "The release ships access control with audit logging, an 8 h retention prune, a face-blur utility and a gender-off filter—disabled by default and not wired into live ingestion." | This content **moves** to BLOCK-B2C, where it can be stated precisely (ACTIVE vs SUPPLIED vs UTILITY). Dropping it here is not a loss of caveat. |
| Keeps "Accuracy degrades outside adult daylight imagery" and "Gender is a soft cue only" (as "gender remains a soft cue only") | **No caveat is lost.** Verified sentence by sentence against the current item (iv). |

### 4.2 BLOCK-B2C — new run-in paragraph: safeguards, threat model, retention [21 lines, new text]

**Placement anchor:** insert immediately **after** the `\paragraph*{Post-ranking}` block
(i.e. after line 239, `...returning top-$k$ clips plus an LLM answer.`) and **before**
`\section{Evaluation}\label{sec:eval}` (line 241). This puts the safeguards statement
after the system description and before the evaluation, which is where a reviewer expects
it.

```latex
\paragraph*{Safeguards, threat model, and retention} The system ranks and flags; a human reviews every returned clip. We treat frames, person descriptors, attribute text and query logs as personal data, and model an unauthenticated API user, a reader of an exported index and an insider with filesystem access. The alert clarify/abstain validator is \emph{active}. Five safeguards are \emph{supplied}: role and camera/zone access control, its JSONL audit trail, the 8-h media prune, the face-blur utility and the gender-off filter---each implemented and unit-tested, each off in the released configuration. The retention policy is per store: media files are pruned on an 8-h window by an operator call that never runs automatically, while detection metadata, the FAISS indices and the audit trail have no TTL. Identity is asserted by trusted-proxy headers the app does not validate, so the control is a policy hook, and camera scope is checked on query-parameter identifiers only. Face blur is not on the live frame path, so exported clips are not anonymised. The binary gender attribute is a soft cue: ablatable on output, retained by default, not removed from the stored index.
```

*1,179 characters → 21 lines (at 56 chars/line).*

**Audit mapping — every sentence to its file:line evidence**

| Sentence in B2C | Evidence |
| --- | --- |
| "The system ranks and flags; a human reviews every returned clip." | The system's outputs are ranked clips + LLM text (`routers/chat.py`); no automatic action. Honest characterisation of a search/alert aid. |
| "treat frames, person descriptors, attribute text and query logs as personal data" | `config.py:38-42` (media dirs), `faiss_person/meta.json` (`attribute_text` per ROI), `security.py:215-216` (audit log). |
| "an unauthenticated API user, a reader of an exported index and an insider with filesystem access" | `security.py:313-315` (open by default) + `security.py:5-19` (header trust) + `privacy.py`/index files on disk. |
| "The alert clarify/abstain validator is *active*." | `alert_validator.py` always runs; `§C1 Validation safeguards`. |
| "Five safeguards are *supplied*: … each off in the released configuration" | `config.py:192-194` (access), `205-207` (audit mirror; the trail itself is gated on access control), `211-213` (retention), `215-217` (blur), `218-223` (gender-off). |
| "media files are pruned on an 8-h window by an operator call that never runs automatically, while detection metadata, the FAISS indices and the audit trail have no TTL" | `privacy.py:35-42`, `149-174`; `db/mongo.py:40-109` (no TTL); `security.py:258-262` (no rotation). |
| "Identity is asserted by trusted-proxy headers the app does not validate, so the control is a policy hook" | `security.py:5-19`, `134-158`. |
| "camera scope is checked on query-parameter identifiers only" | `security.py:359-375` vs `routers/chat.py:296-310` (camera parsed from the NL body). **This is the single most important precision fix**: the current paper text ("ships access control …") invites the reading that camera scoping is complete. |
| "Face blur is not on the live frame path, so exported clips are not anonymised." | `privacy.py` (no runtime caller); `results/privacy_notes.md:88-90`. |
| "The binary gender attribute is a soft cue: ablatable on output, retained by default, not removed from the stored index." | `config.py:218-223` (default true), `attribute_encoder.py:61-67`, `hybrid_search.py:458-461`, `privacy.py:12-14`. |

### 4.3 BLOCK-B3 — optional: one clause in the Evaluation setup [+2 lines]

**Placement anchor:** `INDICON2026Conference.tex` line 244, the `\textbf{Deployment
benchmark.}` sentence in `\paragraph*{Setup}`. Replace
`Its per-query export, annotator roster and tune/test partition were not preserved.` (82
characters) with:

```latex
Its per-query export, annotator roster and tune/test partition were not preserved, so it is not re-computable and is not representative beyond that premises and window (Limitations (iv)).
```

*Adds 2 lines (82 → 187 characters).* Optional — it makes the caveat findable from the
setup rather than only from the limitations, and it points at the new item (iv).

### 4.4 BLOCK-B4 — the Table (only for the 7-page route; ~12 lines)

```latex
\begin{table}[!t]
\caption{Safeguard status in the released configuration. \emph{Active}: enforced on the default path. \emph{Supplied}: implemented and unit-tested but disabled by default (operator opt-in). \emph{Utility}: implemented but on no live path.}
\label{tab:safeguards}
\centering
\footnotesize
\begin{tabular}{@{}lll@{}}
\toprule
Safeguard & Status & Enforcement point \\
\midrule
Alert clarify/abstain validator & Active & materializer (always) \\
Role / camera / zone control & Supplied & HTTP middleware \\
JSONL audit trail & Supplied & same middleware (off with it) \\
Retention prune, 8 h, media files & Supplied & operator/admin call \\
Face blur & Utility & standalone still-image helper \\
Gender-off filter & Supplied & query-time output strip \\
\bottomrule
\end{tabular}
\end{table}
```

**Do not place this in the 6-page build**: it is ~12 lines, it is a float (so the
arithmetic in Part 3 does not apply), and it would exceed the 10.5 free lines on its own.
It belongs on the 7th page (Route 1).

### 4.5 BLOCK-B2THREAT and BLOCK-B1FULL — the long forms (7-page route only)

`BLOCK-B2THREAT` — the full body paragraph (1,791 chars → **32 lines**). Same placement
anchor as `BLOCK-B2C`, but use this instead of B2C on the 7th page:

```latex
\paragraph*{Safeguards, threat model, and retention} The system is an operator-facing search and alerting aid, not an autonomous decision maker: it ranks and flags, and a human reviews every returned clip. We model the assets as camera frames, person descriptors, attribute text and query logs; the adversaries as an unauthenticated API user, a reader of an exported index, and an insider with filesystem access; and the trust boundaries as camera-to-store, store-to-API, and API-to-screen. Two roles are distinguished. The alert clarify/abstain validator is \emph{active}: it refuses to emit an unvalidated rule, so no unverified rule reaches an operator. Five safeguards are \emph{supplied}: role and camera/zone access control, its JSONL audit trail, the 8-h media retention prune, the face-blur utility and the gender-off filter; each is implemented and unit-tested, each is off in the released configuration, and each requires an operator step to arm. The audit trail is not independent of access control: with the default configuration no audit record is written at all. The retention policy is per store, not per system: media files (recordings, clips, snapshots) are pruned on an 8-h window by an operator-invoked call that never runs automatically, while detection, track and attribute metadata, the two FAISS indices and the audit trail have no TTL and expire only on a full reset. Access control is a policy hook, not an authentication system: identity is asserted by trusted-proxy headers that the application does not validate, roles follow the HTTP method and path class rather than per-route declarations, and camera scope is enforced on query-parameter and path identifiers only, so a camera named inside a natural-language query is not scope-checked. Face blur is a standalone still-image helper that is never applied on the live frame path, so exported clips are not anonymised. The binary gender attribute is a soft retrieval cue: it is ablatable on output, it is not removed from the stored index, and it is retained by default.
```

`BLOCK-B1FULL` — the full replacement for Limitations item (iv) (1,951 chars → **35 lines**,
i.e. +24 over the current 11 lines):

```latex
(iv)~\emph{Benchmark representativeness.} The deployment benchmark is 150 queries (30 per category) from one premises, four cameras and a single $\sim$8-h daytime capture; its per-query export, annotator roster and tune/test split were not preserved, so the Table~\ref{tab:alpha_baseline} macro-means are single-run point estimates whose bootstrap CIs cannot be recomputed ($n{=}30$ per category gives half-widths of $0.04$--$0.08$); the released single-camera control is the only re-computable benchmark, and it does not reproduce the gap. (v)~\emph{Privacy, safeguards, and fairness.} Accuracy degrades outside adult daylight imagery. Our threat model treats frames, descriptors, attribute text and query logs as personal data and admits an unauthenticated API user, a reader of an exported index and an insider with filesystem access; a human reviews every clip, so the system advises rather than decides. Safeguards are \emph{active} (the clarify/abstain validator) or \emph{supplied and disabled} (role and camera/zone control with its JSONL audit trail, the 8-h media prune, the face-blur utility, and the gender-off filter, with gender retained by default). Face blur is not on the live frame path, so exported clips are not anonymised; the prune covers media files only, while detection metadata, the FAISS indices and the audit trail have no automatic expiry. Identity arrives in a trusted-proxy header, so the shipped control is a policy hook rather than an authentication system, and camera scope is checked on query-parameter identifiers only. The binary gender attribute is a coarse visual proxy and a soft cue only: on the released gallery the head abstains on 76.5\% of 1{,}778 descriptors and predicts female on 19.9\% and male on 3.6\%---model output, not demographic fact---and with no demographic labels, no second annotator and no per-group error rate, no inter-annotator agreement, cross-camera generalisation or bias measurement is claimed.
```

---

## PART 5 — Paired cut / add budget

### 5.1 Route 1 — **7-page camera-ready appendix (RECOMMENDED, ₹1000)**

Rationale: the INDICON 2026 registration policy allows up to 2 extra pages at ₹1000/page.
A 7th page costs ~$18 and removes the entire page-budget constraint, letting the
reviewer's three issues be answered properly instead of telegraphed. **No cut is needed and
no measured value or caveat is lost.**

Contents of the 7th page (≈100 lines available; a full IEEEtran page holds ≈100 body lines
at 56 chars/line):

1. `BLOCK-B2THREAT` body paragraph (1,791 chars → **32 lines**), the long form of B2C —
   it adds the trust boundaries (camera→store, store→API, API→screen), the per-store
   retention detail, and the explicit statement that the audit trail is *gated on* access
   control. Full text is in §4.5 below.
2. `BLOCK-B4` safeguard-status table (≈12 lines).
3. The full threat model as a short table (see §5.4).
4. `BLOCK-B1FULL` in place of the compressed Limitations item (1,951 chars → **35 lines**,
   i.e. +24 over the current 11-line item (iv)): the full representativeness caveat, the
   full fairness statement with the measured 76.5% / 19.9% / 3.6% figures, the explicit
   statement that these are model outputs and not demographic facts, the statement of the
   standard audit (disaggregated/intersectional error rates) with a citation, and the
   concrete protocol that would produce the missing numbers.
5. A datasheet-style composition block (composition, intended use, out-of-scope use,
   maintenance) for the released track, per Gebru et al. / Mitchell et al.
6. A one-paragraph "Ethical Considerations" statement in the ACM FAccT shape (consent basis
   = notified premises, applicable law = DPDP 2023, secure storage, risk mitigation,
   post-hoc disclosure of audits), even though INDICON has no such slot — it is what a
   top-tier reviewer will look for.

Total ≈ 32 + 12 + 12 + 24 + 10 + 8 ≈ **98 lines** → exactly one extra page.

### 5.2 Route 2 — **strict 6 pages, zero cuts** (safest fallback)

| Add | Lines | Cut | Lines |
| --- | --- | --- | --- |
| `BLOCK-B1C` (15 lines, replaces the current 11-line item (iv)) | **+4** | — | — |
| `BLOCK-B3` (setup clause, +105 chars) | **+2** | — | — |
| **Total** | **+6** | **0** | **0** |

Net +6 lines against **10.5 free lines** ⇒ **stays at 6 pages with ~4.5 lines of slack.**
This route answers issue (3) fully, corrects the four factual imprecisions in the current
Limitations (iv), and adds the two computable fairness facts — but it does **not** add a
threat model to the body; it only states the assets/adversaries in one clause. If a threat
model is non-negotiable, use Route 3.

### 5.3 Route 3 — **6 pages, full in-body disclosure, ≈16.5 lines of named cuts**

| Add | Lines |
| --- | --- |
| `BLOCK-B2C` (new paragraph) | +21 |
| `BLOCK-B1C` (15 lines, replaces the current 11-line item (iv)) | +4 |
| `BLOCK-B3` (setup clause) | +2 |
| **Total additions** | **+27** |
| Free space already on page 6 | −10.5 |
| **Cuts required** | **≈16.5** |

**Tier-1 cuts (safe: each removes a duplication; no measured value and no caveat leaves
the paper).** Line savings are computed from the measured character count of the text
removed (÷56 chars/line); the raw character count is given so the number can be re-derived.

| # | Location | Action | Chars | Lines | Risk |
| --- | --- | --- | --- | --- | --- |
| C1 | `§Related Work`, line 89 | Delete the first sentence *"We compare with the closest published work along the three axes below; only this one provides all three, and none combines override-routing with heterogeneous-space Re-ID fusion."* — the same novelty statement survives at line 95 ("Override-routing with category hard overrides plus heterogeneous single-index fusion is undemonstrated for surveillance video") and in the Contributions list. | 178 | 3.2 | Low |
| C2 | `§C1 Grammar`, line 109 | Replace the parenthetical *"(the 0.91 rule-correctness of\S\ref{sec:eval} came from the deployment study with the LLM-backed materializer and is not reproducible from the released benchmark (Limitations))"* (169 chars) with *"(Limitations (iii))"* (19 chars) — the same disclosure is made at line 254 and in Limitations (iii). | 150 net | 2.7 | Low |
| C3 | `§C1 Release probes`, line 122 | Replace the `\paragraph*{Release probes}` body (529 chars) with *"Three release probes (§\ref{sec:eval}) report held-out rule-correctness, oracle-dialogue turns and a seeded fuzz sweep."* (121 chars) — the superset description (incl. the 20,000-input adversarial round and the always-ask-threshold variant) is at line 254. | 408 net | 7.3 | **Medium** — §C1 currently carries the numbers first; keep the pointer phrased so no number is lost (all of 9/9, 51/51, 1.08, 2.58, 2,000, 200 still appear at line 254). |
| C4 | `§C2 Ablation`, line 165 | Delete the parenthetical *"(the $(0.5,0.5){=}0.785$ tie with RRF needs multi-seed re-verification)"* — the identical fact is in §Eval C2 ("$F_1@10$ 0.785/0.799/0.774 at (0.5,0.5)/(0.7,0.3)/(0.4,0.6): the RRF tie is weight-specific"). | 69 | 1.2 | Low |
| C5 | `§C3 Soft-zone behavior`, line 234 | Delete *"The 16-point soft-weight sweep in\S\ref{sec:eval} shows the result does not depend on the exact soft weights."* — duplicated in §Eval Competitors (ii). | 102 | 1.8 | Low |
| C6 | `§C3 Failure of fixed-α`, line 182 | Delete *"(exact-count EM/MAE (0.10/2.20, \texttt{eval\_alpha\_competitors.py}) is the primary counting-ability measure, in the release)"* — the same numbers are in §Eval Competitors (iii) ("the window-mode caption estimate reaches EM 0.10/MAE 2.20"). | 116 | 2.1 | Low |
| C7 | `§Eval Setup`, line 244 | Delete *"Its per-query export, annotator roster and tune/test partition were not preserved."* (82 chars) — it moves into `BLOCK-B3`/Limitations (iv), where it belongs. | 82 | 1.5 | Low |
| | | **Tier-1 total** | | **≈19.8** | |

**Tier-2 cuts (reserve; use only if Tier 1 is insufficient).**

| # | Location | Action | Chars | Lines | Risk |
| --- | --- | --- | --- | --- | --- |
| C8 | Abstract, line 65 | Delete *"and the release adds three reproducible probes: compositional held-out 9/9 vs.\ 51/51, oracle dialogue at 1.08 vs.\ 2.58 mean turns, and seeded fuzz with zero crashes, zero soundness violations and zero false materializations, plus a fixed adversarial round."* — the probes remain in §Eval. | 256 | 4.6 | **Medium** — `SUBMISSION_TODO.md` item F11 records an abstract-string check (`abstract 0x 'releas'`); re-run the repo's `audit_compliance.py` / `verify_review.py` / `crosscheck2.py` (E29) before and after. Also brings the abstract closer to the 250-word CMT limit (it is currently **303 words** by a LaTeX-stripped count). |
| C9 | Abstract, line 65 | Delete *"They run in a nine-model pipeline at 2~FPS per camera, with 109~ms median FAISS lookup over 312k descriptors."* — §Background line 99 and §Eval line 291 carry both facts. | 109 | 1.9 | Low |
| C10 | `§C3 Post-ranking`, line 239 | Delete the `\emph{Query resolution.}` sentence (332 chars) — it restates §Background + §C3's own opening. | 332 | 5.9 | Medium |
| C11 | `§Introduction`, line 77 | Delete *"Two gaps compound this: hand-translation of behavior into typed rules, and same-space-only Re-ID fusion needing matched backbones deployments rarely have."* — the Contributions list states both gaps. | 154 | 2.8 | Medium |
| | | **Tier-2 total** | | **≈15.2** | |

Tier-1 + Tier-2 ≈ **35.0 lines**. Tier-1 alone (19.8) already covers Route 3's 16.5-line
requirement, so the cuts can be chosen for reviewer-safety rather than for maximum yield.

### 5.4 Threat model, table form (Route 1 appendix)

```latex
\begin{table}[!t]
\caption{Threat model. Assets are camera frames, person descriptors, attribute text and query logs.}
\label{tab:threats}
\centering
\footnotesize
\begin{tabular}{@{}lll@{}}
\toprule
Adversary / boundary & Capability & Mitigation in the release \\
\midrule
Unauthenticated API user & Query any camera, any time & Access control (opt-in); no default \\
Index reader & Copy the person gallery & None (indices are files on disk) \\
Filesystem insider & Read media, MongoDB, logs & Retention prune (opt-in, files only) \\
Header spoofing & Forge X-Role / X-Allowed-Cameras & Trusted reverse proxy required \\
Scope bypass via NL & Name a camera inside a query & Not scope-checked \\
\bottomrule
\end{tabular}
\end{table}
```

### 5.5 Three routes at a glance

| Route | Pages | Cost | Cuts required | Answers the threat-model ask? |
| --- | --- | --- | --- | --- |
| 1 — appendix | 7 | ₹1000 | 0 | Yes, fully (body + tables) |
| 2 — minimal | 6 | ₹0 | 0 | Partially (assets/adversaries in one clause) |
| 3 — in-body | 6 | ₹0 | ≈16.5 (35.0 available) | Yes (one paragraph) |

---

## PART 6 — Reference budget (15 refs, all currently cited)

`INDICON2026_submit/References.bib` holds 15 entries and **all 15 are cited**; the repo
treats 15 as the cap (`References.bib:2`). The INDICON camera-ready notification asks for
"10 quality references" as a *minimum*, so 15→15 with a swap is the safe move. Current
citation counts (measured): `b7`=3, `b12`=3, `b27`=5, `b25`=2, `b29`=2, `b32`=2,
`b14`=1, `b18b`=1, `b22b`=1, `b23`=1, `b24`=1, `b30`=1, `b33`=1, `b34`=1, `b35`=1.

**Single-use (cheapest) drop candidates, in order:** `b30` (HyST) → `b34` (DAT) →
`b35` (CLIP-ReID) → `b32` (person search) → `b33` (Bruch). Do **not** drop `b7`/`b12`
(cited 3× each, and they carry the "commercial alert engines" positioning) or `b27`
(cited 5×).

| Priority | New reference | Supports | Drop |
| --- | --- | --- | --- |
| 1 | Buolamwini & Gebru, *Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification*, FAccT '18, PMLR 81 (`proceedings.mlr.press/v81/buolamwini18a`) | R11, R3 — the binary gender label is a reduced view of a complex construct; the standard audit is a **disaggregated/intersectional error rate** | `b30` |
| 2 | Security Industry Association, *Data Privacy Code of Practice: Video Surveillance*, 2025 (`securityindustry.org/.../SIA-Video-Code-of-Practice-2025.pdf`) | R8 — the surveillance-specific checklist (PIA, default privacy settings, purpose limitation, storage limits, access restriction, notice) | `b34` |
| 3 | Ashurst & Weller, *Fairness Without Demographic Data: A Survey of Approaches*, EAAMO '23 (doi `10.1145/3617694.3623234`) | R10 — the literature's answer to "no demographic labels" (proxies, auxiliary data, complementary mechanisms such as documentation/model cards) | `b35` |
| 4 | NIST AI RMF 1.0, doi `10.6028/NIST.AI.100-1` | R6, R7 — threat modelling and disaggregation of results by data segment | `b32` |

**Constraint check:** INDICON's camera-ready rule says *"Reference not to be cited in the
Abstract and Conclusion part."* None of the blocks in Part 4 cite anything in the abstract
or the Conclusion. If the authors prefer **zero reference churn**, use the *uncited* form
of B2C and B1C as written above (they are self-contained) and add the citations only on
the 7th-page appendix (Route 1), where the reference list can grow — but note the 15-cap
convention still applies to the .bib, so a swap is required there too.

---

## PART 7 — What remains **unfixable**, and the honest sentence for each

These cannot be repaired by prose or by the page budget. For each, the paper should say
what **can** be stated, not a vague apology.

### 7.1 Demographic-bias evaluation — **unfixable** (no demographic labels exist)

**Cannot be done:** per-group error rates, per-group FPR/TPR, intersectional analysis, or a
fairness metric of any kind. The released subset and the deployment carry **no
ground-truth demographic labels**, and the paper's own benchmark relevance labels are
model-derived (§Eval C2: "metadata-formed pseudo-tracks", "caption-derived labels"). The
paper's own repository states this: `results/privacy_notes.md:123-126` ("No
demographic-bias measurement… reporting accuracy/false-positive rates by gender, skin tone,
or age would require labelled data and an IRB-style protocol").

**CAN be stated (all three are computed from the release and are verifiable):**

1. **Coverage / abstention:** the gender head abstains on **1,360 / 1,778 = 76.5%** of the
   released person descriptors (the OpenVINO `male_confidence` gate is 0.65/0.35,
   `attribute_encoder.py:63-67`). The attribute is therefore a *low-coverage* cue, not a
   dense per-person label.
2. **The predicted output distribution:** **354 (19.9%)** predicted female, **64 (3.6%)**
   predicted male. This is the *model's* output distribution over one camera's imagery —
   a statement about the attribute head's behaviour, **not** a demographic fact about the
   scene, and **not** a fairness metric.
3. **The removal path is real and verifiable:** the attribute text is produced only when
   `INCLUDE_GENDER_ATTRIBUTES` is true (`attribute_encoder.py:62`), so a deployment can
   rebuild the text side of the index without gender. The paper should add the honest
   caveat that **the cost of doing so was not measured** — the gender token is part of the
   MiniLM text embedding, so removing it changes the descriptors and hence Table III's
   numbers; we did not re-measure, so no retrieval cost is claimed.

**Suggested limitation sentence (uncited form, if no reference churn is wanted):**
> The binary gender attribute is a coarse visual proxy and a soft retrieval cue only: the
> head abstains on 76.5% of the released 1,778 descriptors and its predicted output
> distribution is 19.9% female and 3.6% male. These are model outputs, not demographic
> facts; with no demographic labels and no second annotator, no per-group error rate,
> inter-annotator agreement or bias measurement is claimed, and a standard audit would
> report disaggregated error rates we cannot compute.

### 7.2 Threat model of the **deployed** installation — **unfixable** in the camera-ready window

No penetration test, no adversarial evaluation of the API, no audit of the notified site.
The paper can and should state the model (assets, adversaries, boundaries) and the
residual risks, which is what §7.3 enumerates.

### 7.3 Access-control residual risks — **unfixable without a code change** (and must be disclosed, not hidden)

* Identity is asserted by **headers the application does not validate**; without a trusted
  reverse proxy that authenticates and strips them, role and camera grants are
  **spoofable** (`security.py:5-19`).
* Camera/zone scope is enforced on **query-parameter and path identifiers only**; a camera
  named inside a natural-language query is **not** scope-checked, and the record-level
  filter `filter_authorized_results` is not called by any router
  (`security.py:359-375`; `routers/chat.py:296-310`).
* The audit trail is **gated on access control** — with the default configuration, **no
  audit record is written at all** (`security.py:313-315`).
* `/api/detections` returns `person_attributes` and `attribute_text` **verbatim**
  (`routers/detections.py:18-30`), i.e. the gender-off filter is not applied on that path.

### 7.4 Retention residual risks — **unfixable without a code change**

* The 8-h prune covers **media files only**; detection/track/attribute metadata, both FAISS
  indices and `audit.jsonl` have **no TTL and no rotation** (`privacy.py:35-42`;
  `db/mongo.py:40-109`; `security.py:258-262`).
* The prune is **never automatic** — an operator (or cron) must call it
  (`privacy.py:149-174`).
* Consequently the paper's "face imagery retained per-camera for at most 8 h" is a
  **deployment fact, not a system guarantee**, and should be worded that way.

### 7.5 Benchmark representativeness — **unfixable** (no more data, no preserved export)

Unfixable facts: one premises, four cameras, a single ~8-h daytime capture, 150 queries
(30 per category), and no preserved per-query export / annotator roster / tune-test split.
The released track is **one camera over a 1.92-hour window** (`data/index/faiss_person/
meta.json`: 2026-03-01 17:01→18:56, camera 0) with metadata-derived labels, and it
**ties** the best fixed α (0.538 vs 0.538) rather than reproducing the deployment gap.

**What CAN be stated (this is the honest fix for issue 3):**
* the deployment macro-means are **single-run point estimates** with **non-recomputable**
  bootstrap CIs (already stated at line 246 and in the Table II caption);
* per-category CIs at n=30 have half-widths of **0.04–0.08** (read off Table II);
* the **only re-computable** benchmark in the release is the 150-query control
  (50 tune / 100 test) on a single camera, and it does **not** reproduce the deployment gap;
* the release is therefore the reproducible track and the deployment is the
  one-site, one-window measurement — a distinction the reader can act on.

---

## PART 8 — Pre-submission verification checklist for this edit

1. Recompile and confirm **6 pages** (or 7 for Route 1) and **0 overfull boxes**
   (`INDICON2026_submit/INDICON2026Conference.log`).
2. Re-run the repo's audit scripts — `SUBMISSION_TODO.md` item **E29** records
   `audit_compliance.py`, `verify_review.py`, `crosscheck2.py` in a scratch directory
   (not in the repo) and pins several exact strings, including the C10 latency-provenance
   strings, *"no structural rewrite was observed"*, *"reducing manual rule authoring"* and
   *"Crucially, no single fixed α"*. **None of the proposed cuts touch those strings**,
   but the abstract cut C8 must be re-checked against the F11 abstract check.
3. Re-check the **distinct-value diff** used by the 8→6 page reduction (`SUBMISSION_TODO.md`
   E23/E29): no measured value may disappear. The new text *adds* values (76.5%, 19.9%,
   3.6%, 1,778, 1,360, 354, 64) — verify they are all reproducible from
   `data/index/faiss_person/meta.json` (the counting recipe is in Part 9).
4. Confirm **all 15 references remain cited** (or that each new reference has a swap
   per Part 6), and that **no citation appears in the abstract or the Conclusion**
   (INDICON camera-ready rule).
5. Confirm the two statements the paper now makes about the deployment remain true:
   *"face imagery was pruned per camera at 8 h"* (deployment fact) and *"the released
   single-camera control … does not reproduce the gap"* (0.538 vs 0.538, §Eval).
6. If Route 1 is used: confirm the 7th page carries the ₹1000 overlength payment per the
   INDICON 2026 registration policy and that the page is inside the 8-page submission
   maximum.

---

## PART 9 — Reproduction recipe for the two new fairness numbers

Both are computed from the released person index metadata; no new capture is needed.

```powershell
# backend/app/services/attribute_encoder.py gates on male_confidence >= 0.65 (male) / <= 0.35 (female)
$m = Get-Content "data\index\faiss_person\meta.json" -Raw | ConvertFrom-Json
$tot = $m.Count                                   # 1778
$mal = ($m | Where-Object { $_.attribute_text -match '(^|[,\s])male($|[,\s])' }).Count    # 64
$fem = ($m | Where-Object { $_.attribute_text -match '(^|[,\s])female($|[,\s])' }).Count  # 354
$none = $tot - $mal - $fem                        # 1360  (abstained)
"male $mal ($([math]::Round(100*$mal/$tot,2))%) | female $fem ($([math]::Round(100*$fem/$tot,2))%) | abstained $none ($([math]::Round(100*$none/$tot,2))%)"
```

Supporting facts about the released track (both are already stated in the paper and are
worth quoting in the representativeness caveat):

* person descriptors: **1,778**; scene frames: **242**; clips: **123**; camera: **0 only**.
* timestamp span: **2026-03-01 17:01 → 18:56 = 1.92 h**.
* control benchmark: **150 queries, 30 per category** (`backend/benchmarks/
  queries_control.jsonl`), split 50 tune / 100 test.

---

## PART 10 — One-paragraph summary for the response letter

> We thank the reviewers for the privacy/fairness and threat-model points. We audited the
> released safeguards against the code and found that the shipped posture separates one
> *active* safeguard (the deterministic clarify/abstain alert validator) from five
> *supplied* ones (role/camera/zone access control with its JSONL audit trail, the 8-h
> media retention prune, the face-blur utility, and the gender-off filter), all of which
> are implemented and unit-tested but off in the released configuration; we now state this
> split explicitly, add a threat model (assets, adversaries, trust boundaries) and a
> per-store retention policy, and disclose the residual gaps we found — that the audit
> trail is gated on access control, that the prune covers media files only while detection
> metadata, the FAISS indices and the audit trail have no TTL or rotation, that camera
> scope is enforced on query-parameter identifiers only, and that face blur is not on the
> live frame path so exported clips are not anonymised. We also corrected the earlier
> phrasing that grouped the gender-off filter with the disabled switches: gender is
> retained by default. No demographic-bias measurement is possible without demographic
> labels, so we now report what *is* computable — the attribute head abstains on 76.5% of
> the 1,778 released descriptors and its predicted output distribution is 19.9% female and
> 3.6% male — and state plainly that these are model outputs, not demographic facts, and
> that no per-group error rate, inter-annotator agreement or cross-camera generalisation
> is claimed. Finally, we consolidated the benchmark-representativeness caveat: one
> premises, four cameras, a single ~8-h daytime capture, 150 queries with no preserved
> per-query export or annotator roster, per-category CI half-widths of 0.04–0.08 at n=30,
> and a released single-camera control that ties rather than reproduces the deployment gap.
