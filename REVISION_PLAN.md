# INDICON 2026 revision — working plan

## Ground truth about the released artifact

The repository is **not** the full deployment described in the paper.  The
released subset is:

- `data/clips/`: 123 clips, `camera_0` only, 2026-03-01 17:01–18:56.
  Clips are 640×480, 5 FPS, ~3.6 s each (18 frames); filenames span ~15 s
  source windows.  The selected clip footage is ~7.2 min, not a continuous
  2 h recording.
- `data/snapshots/`: 2,071 full-frame JPEGs, `camera_0` only.
- `data/recordings/`: 11 recordings (~33 min total, one corrupt file),
  `camera_0` only.
- `data/index/faiss/`: 242 scene vectors (OpenCLIP ViT-B/32, 512-d).
- `data/index/faiss_person/`: 1,778 person descriptors (SigLIP 768 + MiniLM 384,
  1152-d), `track_id=-1` for all.

MongoDB is not running.  OpenCLIP and MiniLM are cached and work offline on
CPU.  There is no GPU (`torch 2.10.0+cpu`).  The OpenAI key in `backend/.env`
works; parsing one query takes ~14 s on CPU/network and is not suitable for
large offline sweeps.

The paper's headline numbers (0.666 → 0.828, 0.785 → 0.807, 91%, 6 s/12 s)
come from the full 4-camera / 312k-descriptor deployment.  That data is not in
the repository.  We must **not fabricate** a reproduction of those numbers,
and we must **not replace** them with single-camera numbers as if they were the
same experiment.

## Claims policy

1. Keep the full-deployment numbers as the paper's primary results, explicitly
   labelled as full-deployment measurements.
2. Add a **released-subset reproducibility track** with real numbers computed
   from the shipped 123-clip / 242-frame / 1,778-descriptor subset.  Every
   number must be produced by a script in this repository.
3. Never claim that the subset reproduces 4-camera / 312k results, human IAA,
   cross-camera generalisation, or full-deployment latency percentiles.
4. Where a reviewer request cannot be satisfied from the release, state the
   missing artifact/provenance instead of inventing it.
5. Prefer improving the code so a claim becomes true (e.g. wire the validator,
   implement role/zone access control, measure real latency) over deleting the
   claim.

## Workstreams

### A. Alerts (owner: alert subagent)

Files:

- `backend/benchmarks/alert_utterances.jsonl`
- `backend/benchmarks/alert_schema.json`
- `backend/app/services/alert_validator.py`
- `backend/app/services/alert_materializer.py` (new)
- `backend/app/routers/chat.py` (alert-creation path only)
- `backend/scripts/eval_alerts.py`
- `backend/tests/test_alerts.py`

Deliverables:

- 120 unique utterances: 60 clean + 60 stress (ambiguous / conflicting /
  missing / unsupported), each with 9-field reference rules.
- Deterministic rule-based materializer for the full 9-field grammar.
- Dev/test split; tune on dev only, report test.
- Real evaluation: rule correctness, per-field accuracy, clarify/abstain
  precision/recall, per-stress-kind table.
- Wire `needs_clarification` into the live chat alert path.
- `results/alert_results.json`, `results/alert_results.md`.

### B. Retrieval + fusion (owner: retrieval subagent)

Files:

- `backend/benchmarks/queries.jsonl`
- `backend/benchmarks/splits.json`
- `backend/benchmarks/person_pairs.jsonl`
- `backend/benchmarks/README.md`
- `backend/scripts/eval_retrieval.py`
- `backend/scripts/eval_counting.py` (new)
- `backend/scripts/eval_fusion.py`
- `backend/app/eval/*`
- `backend/tests/test_metrics.py`, `test_stats.py`, `test_alpha_fusion.py`

Deliverables:

- 150-query benchmark built from the released metadata, with deterministic
  relevance labels; 5 categories if defensible, otherwise document the
  category that the subset cannot support and use the maximum honest set.
- Runnable evaluator using the real FAISS indices and cached OpenCLIP/MiniLM;
  no MongoDB and no synthetic Gaussian scores.
- Fixed-alpha sweep, adaptive alpha, per-intent alpha, z-score normalisation,
  EM/MAE for count queries, grouped bootstrap + Holm.
- Person-pair/track benchmark from the 1,778 descriptors; identity-disjoint
  split; concat vs visual/text/zero-pad/norm-matched/random-projection/RRF/
  learned-projection baselines; P@5/R@10/F1@10.
- `results/retrieval_*`, `results/fusion_*` plus per-query CSV/JSON.

### C. Systems / privacy / latency (owner: systems subagent)

Files:

- `backend/app/config.py`
- `backend/app/db/mongo.py`
- `backend/app/services/nl_parser.py`
- `backend/app/services/unified_retrieval.py`
- `backend/app/services/hybrid_search.py`
- `backend/app/core/security.py` (new)
- `backend/app/core/privacy.py` (new)
- `backend/app/main.py`
- `backend/scripts/measure_latency.py`
- `backend/tests/test_privacy.py`, `test_person_routing.py`

Deliverables:

- Fail-fast Mongo/offline startup (foundational patch already applied; verify).
- Date-grounded NL parsing and count-query classification (foundational patch
  already applied; verify and add tests).
- Person-scene routing: wire `HybridSearchEngine` into `UnifiedRetrieval`
  behind `ENABLE_PERSON_ROUTING`, with a documented fallback.
- Role/zone access control, audit logging, retention cleanup, face blur,
  gender-off switch, with tests.
- Real latency harness: component timings, p50/p95, concurrency; no simulated
  lognormal numbers.
- `results/latency_*`, `results/privacy_notes.md`.

### D. Paper (owner: paper subagent, after A–C)

Files:

- `INDICON2026_submit/INDICON2026Conference.tex`
- `INDICON2026_submit/References.bib`
- figures under `INDICON2026_submit/`

Deliverables:

- Update text/tables using `results/*`.
- Keep full-deployment numbers primary; add released-subset reproducibility
  table(s), clearly labelled.
- Fix LLM-vs-regex contradiction, overfull boxes, page limit, terminology.
- Add alpha sensitivity, normalised baselines, EM/MAE, grouped stats, fusion
  baselines, alert stress results, latency percentiles, privacy implementation.
- No invented numbers; every number traceable to a result file.

### E. Verification (owner: verification subagent, after D)

- Run `pytest`, every eval script, and the LaTeX build.
- Check page count, overfull boxes, undefined references.
- Cross-check each paper number against `results/*`.
- Commit the final state.

## Baseline commands

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m pytest backend/tests -q
python backend/scripts/eval_alerts.py
python backend/scripts/eval_retrieval.py
python backend/scripts/eval_fusion.py
python backend/scripts/measure_latency.py
```
