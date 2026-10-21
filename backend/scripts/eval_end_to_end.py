"""End-to-end evaluation of the released subset with the *real* NL parser.

This is the query-path track: the natural-language query is parsed by the
deployed LLM backend (the local opencode runtime with a free model by default;
any OpenAI-compatible endpoint is supported), the parsed filter drives the
structured branch, the semantic branch runs over the released OpenCLIP/FAISS
scene index, and the production adaptive-alpha rule merges the two.

``eval_retrieval_control.py`` replays *metadata-derived* filters, i.e. an
oracle parse.  This script measures the whole query path instead.  Parses are
cached on disk, so the run is reproducible and resumable.

Protocol
--------
The default benchmark is ``backend/benchmarks/queries_control.jsonl``: the
protocol-matched query file used by the control track -- same 150 queries, same
50 tune / 100 test split, same relevance labels and the same per-category
wording (count queries carry an explicit numeric target, action queries carry
no time cue).  Running both tracks on the same file is what makes the two
numbers comparable; they then differ only in where the filter comes from.

``backend/benchmarks/queries.jsonl`` is the raw released benchmark.  Its action
queries keep a 65--90 s time window, which gives the structured branch a time
cue the protocol-matched control deliberately removes, so the production action
override (alpha = 0.1) loses signal there.  It can be run with ``--queries``
for a protocol-difference study, but its numbers are *not* comparable with the
control track.

What this track does and does not establish
-------------------------------------------
* The relevance labels are used only to score the final ranking.  The
  structured branch sees the parsed filter and the query window, never the
  label list.
* The query window comes from the parser when it produced one (falling back to
  the benchmark window only when the parse has no time), so the time filter is
  part of the measured query path.
* The alpha rule is the production function
  (``backend.app.services.result_merger.compute_adaptive_alpha``) called with
  the same arguments as in the control track.
* It is not the deployed fusion stack: the track uses the plain
  ``alpha * struct + (1 - alpha) * sem`` blend on min-max normalised per-clip
  scores, without the recency boost, MMR and track merging of the live
  pipeline, and it fixes ``n_semantic = 20`` rather than the live top-k.
* The count EM/MAE helper reads the caption count at the benchmark's anchor
  timestamp, so it measures whether the anchor clip ranked first rather than
  counting ability; it is reported but not to be read as count accuracy.

Run:
    python backend/scripts/eval_end_to_end.py --limit 150
    python backend/scripts/eval_end_to_end.py --queries backend/benchmarks/queries.jsonl \
        --out-prefix end_to_end_raw
    python backend/scripts/eval_end_to_end.py --refresh --model opencode-go/minimax-m3.1-flash-free
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.eval import subset  # noqa: E402
from backend.app.services import llm_client, nl_parser  # noqa: E402
from backend.scripts import eval_retrieval_control as erc  # noqa: E402

RESULTS = ROOT / "results"
CACHE_DIR = ROOT / "backend" / "benchmarks" / "parsed_cache"
DEFAULT_QUERIES = ROOT / "backend" / "benchmarks" / "queries_control.jsonl"
FIXED_ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
CATEGORIES = ["count", "find", "track", "action", "filter"]
N_BOOT = 3000
BOOT_SEED = 11


def _heuristic_filter(text: str) -> Dict[str, Any]:
    """Deterministic fallback used when the LLM backend is unreachable.

    Extracts only what can be read off the query text: camera id, colour,
    clock window and an explicit numeric count.  It is deliberately thin
    rather than clever, so a failed parse stays visible in the results
    instead of being silently patched up.
    """
    import re as _re

    from backend.app.config import settings

    epoch = settings.DATA_EPOCH_DATE or "2026-03-01"

    out: Dict[str, Any] = {}
    low = text.lower()
    cam = _re.search(r"camera\s*(\d+)", low)
    if cam:
        out["camera_id"] = int(cam.group(1))
    for colour in ("olive", "red", "brown"):
        if colour in low:
            out["objects.color"] = colour.title()
            break
    times = _re.findall(r"\b(\d{1,2}:\d{2})\b", text)
    if len(times) >= 2:
        out["timestamp"] = {"$gte": f"{epoch}T{times[0]}", "$lte": f"{epoch}T{times[1]}"}
    elif times:
        out["timestamp"] = {"$gte": f"{epoch}T{times[0]}", "$lte": f"{epoch}T{times[0]}"}
    count = _re.search(r"\bexactly\s+(\d+)\b", low) or _re.search(
        r"\b(\d+)\s*(?:people|persons|person)\b", low
    )
    if count:
        out["count_constraint"] = {"eq": int(count.group(1))}
    out["__raw"] = text
    return out


def cache_key(qid: str, text: str, model: str) -> str:
    """Cache key that also covers the query text and the model.

    A stale parse must never be reused after the benchmark is rebuilt or the
    model changes, so the key hashes the query text and the model id as well.
    """
    digest = hashlib.sha1(f"{text}\x00{model}".encode("utf-8")).hexdigest()[:12]
    return f"{qid}:{digest}"


def parse_queries(
    queries: List[Dict[str, Any]], model: str, limit: Optional[int], refresh: bool
) -> Dict[str, Any]:
    """Parse every query with the configured LLM backend (cached on disk)."""
    cache_path = CACHE_DIR / (model.replace("/", "_").replace(":", "_") + ".json")
    cache = llm_client.ParseCache(cache_path)
    rows: List[Dict[str, Any]] = []
    subset_rows = queries[:limit] if limit is not None else queries
    for i, q in enumerate(subset_rows, 1):
        key = cache_key(q["qid"], q["text"], model)
        entry = cache.get(key) if not refresh else None
        if entry is None:
            t0 = time.time()
            try:
                f = nl_parser.parse_nl_with_llm(q["text"])
                entry = {
                    "parsed": {k: v for k, v in f.items() if not k.startswith("__")},
                    "ok": True,
                    "seconds": round(time.time() - t0, 2),
                }
            except Exception as exc:
                entry = {
                    "parsed": _heuristic_filter(q["text"]),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                    "seconds": round(time.time() - t0, 2),
                }
                if any(tok in str(exc).lower() for tok in
                           ("usage limit", "rate limit", "quota", "429",
                            "too many requests")):
                    # The provider quota is exhausted: stop cleanly, keep the
                    # parses made so far (the cache is written after every
                    # query) and let the run be resumed later.
                    cache.put(key, entry)
                    cache.save()
                    print(f"  [stop] provider quota reached at {q['qid']}: {exc}", flush=True)
                    raise SystemExit(2)
            cache.put(key, entry)
            cache.save()  # keep the run resumable
        row = dict(q)
        row["parsed"] = entry["parsed"]
        row["parse_ok"] = entry["ok"]
        row["parse_seconds"] = entry.get("seconds")
        rows.append(row)
        print(f"  [{i}/{len(subset_rows)}] {q['qid']}: {'llm' if entry['ok'] else 'fallback'}"
              f" ({entry.get('seconds')}s)", flush=True)
    cache.save()
    return {"rows": rows, "cache": str(cache_path.relative_to(ROOT))}


def normalise_for_eval(row: Dict[str, Any], use_parsed_time: bool = True) -> Dict[str, Any]:
    """Map the parser output onto the fields the evaluation expects.

    The parser returns the *filter* shape used by the live pipeline
    (``timestamp`` with ``$gte``/``$lte``, ``objects.color``,
    ``count_constraint``, ...); the control track's context builder reads a
    flatter shape.  Three details matter for fidelity:

    * the query window used by the structured branch comes from the parse when
      the parser produced one -- an end-to-end track must not silently fall
      back to benchmark metadata for the time filter -- and from the benchmark
      only when the parse has no window;
    * ``attributes`` and ``event`` are carried through, because dropping them
      changes the structured score;
    * the count constraint keeps its bare-integer and ``{"eq": n}`` forms.
    """
    p = row.get("parsed") or {}
    out = dict(row)
    stamp = p.get("timestamp") if isinstance(p.get("timestamp"), dict) else {}
    parsed_start = stamp.get("$gte")
    parsed_end = stamp.get("$lte")
    if use_parsed_time and parsed_start and parsed_end:
        out["time_start"] = parsed_start
        out["time_end"] = parsed_end
    count_constraint = p.get("count_constraint")
    if isinstance(count_constraint, (int, float)) and not isinstance(count_constraint, bool):
        count_constraint = {"eq": int(count_constraint)}
    action = p.get("action")
    event = p.get("event")
    if not event and action in {"fighting", "smoke"}:
        event = action
    out["parsed"] = {
        "time_start": parsed_start,
        "time_end": parsed_end,
        "camera_id": p.get("camera_id"),
        "count_target_in_text": (count_constraint or {}).get("eq")
        if isinstance(count_constraint, dict) else None,
        "count_constraint": count_constraint,
        "objects.color": p.get("objects.color"),
        "objects.object_name": p.get("objects.object_name"),
        "attributes": list(p.get("attributes") or []),
        "zone": p.get("zone"),
        "action": action,
        "event": event,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model id override")
    ap.add_argument("--provider", default=None, help="provider override")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--refresh", action="store_true", help="ignore the parse cache")
    ap.add_argument("--queries", default=str(DEFAULT_QUERIES),
                    help="query file (default: the protocol-matched control file)")
    ap.add_argument("--out-prefix", default="end_to_end",
                    help="prefix of the result files written under results/")
    ap.add_argument("--benchmark-time", action="store_true",
                    help="use the benchmark's query window instead of the parsed one")
    args = ap.parse_args()

    from backend.app.config import settings

    if args.provider:
        settings.LLM_PROVIDER = args.provider
    if args.model:
        settings.NL_DEFAULT_MODEL = args.model
    cfg = settings.get_active_llm_config()
    model = args.model or cfg.get("model") or settings.OPENCODE_MODEL
    queries_path = Path(args.queries)
    print(f"provider={cfg.get('provider')} model={model}")
    print(f"queries={queries_path.name}")

    queries = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"parsing {min(args.limit, len(queries))} queries with the deployed parser...")
    parsed = parse_queries(queries, model, args.limit, args.refresh)
    rows = [normalise_for_eval(r, use_parsed_time=not args.benchmark_time)
            for r in parsed["rows"]]

    # --- retrieval over the released scene index -----------------------------
    index, frames, clips, clip_ids = erc.load_scene_index()
    dets, clip_of, dets_by_clip = erc.load_clip_detections(clips)
    motion = subset.compute_motion_energy(clips)
    motion_max = max((v["mean"] for v in motion.values()), default=1.0)

    from backend.app.services.sem_embedder import get_embedder

    embedder = get_embedder()
    q_emb = embedder.text_embed([r["text"] for r in rows])
    D, I = index.search(q_emb.astype(np.float32), index.ntotal)
    clip_index = {cid: i for i, cid in enumerate(clip_ids)}
    sem_raw: Dict[str, np.ndarray] = {}
    for qi, r in enumerate(rows):
        per_clip = np.full(len(clip_ids), -1.0, dtype=np.float64)
        for dist, fi in zip(D[qi], I[qi]):
            cid = frames[int(fi)]["clip_id"]
            j = clip_index[cid]
            per_clip[j] = max(per_clip[j], float(dist))
        per_clip[per_clip < 0] = float(np.min(D[qi]))
        sem_raw[r["qid"]] = per_clip

    contexts = {
        r["qid"]: erc.build_query_context(r, clips, dets_by_clip, motion, motion_max)
        for r in rows
    }
    struct_raw = {
        qid: np.asarray([ctx["raw"][cid] for cid in clip_ids], dtype=np.float64)
        for qid, ctx in contexts.items()
    }
    sem_norm = {qid: erc.normalize_branch(v, "minmax") for qid, v in sem_raw.items()}
    struct_norm = {qid: erc.normalize_branch(v, "minmax") for qid, v in struct_raw.items()}

    systems = [f"fixed_{a:.1f}" for a in FIXED_ALPHAS] + [
        "adaptive", "per_intent_tuned", "paper_lookup"
    ]
    tuned, _ = erc.tune_per_intent(rows, clip_ids, sem_norm, struct_norm, contexts)
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for method in systems:
        fused = erc.build_fused(rows, clip_ids, sem_norm, struct_norm, method, tuned, contexts)
        scored = erc.score_rows(rows, fused, clips, sem_norm, struct_norm, clip_index)
        for r, q in zip(scored, rows):
            r["group"] = q.get("group")
        all_rows[method] = scored

    summary: Dict[str, Any] = {}
    for method in systems:
        test_rows = [r for r in all_rows[method] if r["split"] == "test"]
        p5 = erc.aggregate(test_rows, "p5")
        summary[method] = {
            "macro": p5["macro"],
            "per_category": p5["per_category"],
            "count": erc.count_scores(test_rows),
        }

    # --- parse fidelity: how the LLM parse compares with the reference parse --
    # The control track replays the metadata-derived reference filter, which
    # carries information a text-only parser cannot recover (the anchor
    # person's attribute list) and a time window that follows the subject
    # past the window the query states.  Quantifying that difference is what
    # makes the two numbers comparable.
    reference_rows = {
        json.loads(line)["qid"]: json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    def _ts(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None

    fidelity: Dict[str, Any] = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        label_side_attrs = 0
        window_differs = 0
        colour_hit = colour_total = 0
        target_hit = target_total = 0
        for r in cat_rows:
            ref = (reference_rows.get(r["qid"], {}).get("parsed") or {})
            p_filter = r.get("parsed") or {}
            if ref.get("attributes"):
                label_side_attrs += 1
            rs, re_ = _ts(ref.get("time_start")), _ts(ref.get("time_end"))
            ps, pe = _ts(p_filter.get("time_start")), _ts(p_filter.get("time_end"))
            if None not in (rs, re_, ps, pe) and abs((re_ - rs) - (pe - ps)) > 1:
                window_differs += 1
            ref_colour = ref.get("objects.color")
            if ref_colour:
                colour_total += 1
                if str(p_filter.get("objects.color") or "").lower() == str(ref_colour).lower():
                    colour_hit += 1
            ref_target = ref.get("count_target_in_text")
            if ref_target is not None:
                target_total += 1
                if p_filter.get("count_target_in_text") == ref_target:
                    target_hit += 1
        fidelity[cat] = {
            "n": len(cat_rows),
            "colour_agreement": [colour_hit, colour_total],
            "count_target_agreement": [target_hit, target_total],
            "reference_has_label_side_attributes": label_side_attrs,
            "reference_window_differs_from_parsed": window_differs,
        }

    control_path = RESULTS / "retrieval_control_results.json"
    oracle_delta: Optional[Dict[str, Any]] = None
    if control_path.exists():
        try:
            ctrl = json.loads(control_path.read_text(encoding="utf-8"))
            cc = ctrl["test"]["minmax"]["adaptive"]
            oracle_delta = {
                "control_macro_p5": cc["macro"]["p5"],
                "end_to_end_macro_p5": summary["adaptive"]["macro"],
                "delta": summary["adaptive"]["macro"] - cc["macro"]["p5"],
                "per_category": {
                    cat: {
                        "control": cc["per_category"][cat]["p5"],
                        "end_to_end": summary["adaptive"]["per_category"][cat],
                        "delta": summary["adaptive"]["per_category"][cat]
                        - cc["per_category"][cat]["p5"],
                    }
                    for cat in CATEGORIES if cat in cc["per_category"]
                },
                "note": (
                    "The control replays the metadata-derived reference filter, which "
                    "carries the anchor person's attribute list and a time window that "
                    "follows the subject past the window the query states; a text-only "
                    "parser cannot recover either.  The gap is therefore the price of "
                    "not having an oracle parse, not a difference in the fusion rule."
                ),
            }
        except (KeyError, TypeError, ValueError):
            oracle_delta = None

    # --- significance: adaptive vs the tune-selected best fixed alpha ---------
    tune_scores = {
        m: erc.aggregate([r for r in all_rows[m] if r["split"] == "tune"], "p5")["macro"]
        for m in systems if m.startswith("fixed_")
    }
    best_fixed = max(tune_scores, key=lambda m: (tune_scores[m], m))
    a_rows = {r["qid"]: r for r in all_rows["adaptive"] if r["split"] == "test"}
    b_rows = {r["qid"]: r for r in all_rows[best_fixed] if r["split"] == "test"}
    qids = [r["qid"] for r in rows if r["split"] == "test"]
    a_vals = [a_rows[q]["p5"] for q in qids]
    b_vals = [b_rows[q]["p5"] for q in qids]
    groups = [a_rows[q]["group"] for q in qids]
    dm, lo, hi, p = erc.paired_bootstrap_ci(a_vals, b_vals, n_boot=N_BOOT, seed=BOOT_SEED)
    gm, glo, ghi, gp = erc.paired_grouped_bootstrap_ci(
        a_vals, b_vals, groups, n_boot=N_BOOT, seed=BOOT_SEED
    )

    # --- per-category paired tests with Holm correction ----------------------
    per_category: Dict[str, Any] = {}
    raw_p: Dict[str, float] = {}
    for cat in CATEGORIES:
        cat_qids = [q for q in qids if a_rows[q]["category"] == cat]
        if not cat_qids:
            continue
        av = [a_rows[q]["p5"] for q in cat_qids]
        bv = [b_rows[q]["p5"] for q in cat_qids]
        cdm, clo, chi, cp = erc.paired_bootstrap_ci(av, bv, n_boot=N_BOOT, seed=BOOT_SEED)
        raw_p[cat] = float(cp)
        per_category[cat] = {
            "n": len(cat_qids),
            "adaptive": float(np.mean(av)),
            "best_fixed": float(np.mean(bv)),
            "delta": float(cdm),
            "lo": float(clo),
            "hi": float(chi),
            "p": float(cp),
        }
    holm = erc.holm_adjusted(raw_p)
    for cat, value in per_category.items():
        value["p_holm"] = float(holm.get(cat, value["p"]))

    # --- parse fidelity ------------------------------------------------------
    parse_ok = sum(1 for r in rows if r.get("parse_ok"))
    parse_times = [r.get("parse_seconds") or 0.0 for r in rows]
    count_rows = [r for r in rows if r["category"] == "count"]
    count_with_target = sum(
        1 for r in count_rows
        if (r.get("parsed") or {}).get("count_target_in_text") is not None
    )
    query_file = queries_path.name

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "track": "end-to-end released-subset evaluation (real LLM parse)",
            "model": model,
            "provider": cfg.get("provider"),
            "queries": query_file,
            "protocol": (
                "protocol-matched with eval_retrieval_control.py: same queries, "
                "split, labels and per-category wording"
                if query_file == DEFAULT_QUERIES.name else
                "raw released benchmark; NOT protocol-matched with the control track"
            ),
            "window_source": "benchmark metadata" if args.benchmark_time else "parsed by the LLM",
            "note": (
                "Single-camera released subset; not the paper's 4-camera / 312k "
                "deployment benchmark."
            ),
        },
        "parse": {
            "n": len(rows),
            "llm_ok": parse_ok,
            "llm_ok_rate": parse_ok / max(1, len(rows)),
            "mean_seconds": float(np.mean(parse_times)) if parse_times else 0.0,
            "p95_seconds": float(np.percentile(parse_times, 95)) if parse_times else 0.0,
            "count_queries": len(count_rows),
            "count_queries_with_parsed_target": count_with_target,
            "cache": parsed["cache"],
        },
        "test": summary,
        "best_fixed_on_tune": {"method": best_fixed, "tune_macro_p5": tune_scores[best_fixed]},
        "significance": {
            "comparison": f"adaptive vs {best_fixed} (fixed alpha selected on the tune split)",
            "method": "paired percentile bootstrap on per-query P@5 deltas",
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
            "delta_p5": float(dm), "lo": float(lo), "hi": float(hi), "p": float(p),
            "grouped_delta_p5": float(gm), "grouped_lo": float(glo),
            "grouped_hi": float(ghi), "grouped_p": float(gp),
        },
        "per_category_vs_best_fixed": per_category,
        "parse_fidelity_vs_reference": fidelity,
        "oracle_delta": oracle_delta,
        "per_query": [
            {
                "qid": r["qid"], "category": r["category"], "split": r["split"],
                "parse_ok": r.get("parse_ok"),
                "text": r["text"],
                "parsed": r.get("parsed"),
                "p5": {m: next((x["p5"] for x in all_rows[m] if x["qid"] == r["qid"]), None)
                       for m in systems},
            }
            for r in rows
        ],
        "limitations": [
            "Single camera, 123 clips, ~7.2 min of selected footage; the deployment "
            "numbers are not reproduced here.",
            "Query relevance labels are metadata-derived, not human-annotated.",
            "The MongoDB structured branch is replaced by an in-memory metadata store "
            "built from the released sidecar; the scoring code is the production code.",
            "The fusion is the plain alpha blend, not the live stack (no recency boost, "
            "no MMR, no track merging); n_semantic is fixed at 20 rather than the live top-k.",
            "The tune and test splits share relevant clips (the split is stratified by "
            "category, not by clip or time bucket).",
            "The count EM/MAE helper reads the caption count at the benchmark's anchor "
            "timestamp, so it measures whether the anchor clip ranked first rather than "
            "counting ability.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.out_prefix}_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# End-to-end released-subset evaluation (real LLM parser)",
        "",
        f"Model: `{model}` via provider `{cfg.get('provider')}`.",
        f"Queries: `{query_file}` -- {results['scope']['protocol']}.",
        f"Query window used by the structured branch: {results['scope']['window_source']}.",
        "",
        f"Parse: {parse_ok}/{len(rows)} answered by the LLM "
        f"({parse_ok / max(1, len(rows)) * 100:.0f}%), "
        f"mean {np.mean(parse_times):.1f}s, p95 {np.percentile(parse_times, 95):.1f}s per query.",
        f"Count queries whose parsed filter carries an explicit numeric target: "
        f"{count_with_target}/{len(count_rows)}.",
        "",
        "## Test macro P@5",
        "",
        "| Method | Macro P@5 | Count | Find | Track | Action | Filter |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in systems:
        s = summary[m]
        pc = s["per_category"]
        lines.append(
            f"| {m} | {s['macro']:.3f} | {pc['count']:.3f} | {pc['find']:.3f} | "
            f"{pc['track']:.3f} | {pc['action']:.3f} | {pc['filter']:.3f} |"
        )
    sig = results["significance"]
    lines += [
        "",
        f"Adaptive vs {best_fixed} (tune-selected): delta P@5 = {sig['delta_p5']:+.3f} "
        f"[{sig['lo']:+.3f}, {sig['hi']:+.3f}], p = {sig['p']:.4f}; grouped "
        f"[{sig['grouped_lo']:+.3f}, {sig['grouped_hi']:+.3f}], p = {sig['grouped_p']:.4f} "
        f"({N_BOOT} resamples, seed {BOOT_SEED}).",
        "",
        "## Per category (test split, vs the best tune-selected fixed alpha)",
        "",
        "| Category | n | Adaptive | Best fixed | Delta | 95% CI | p | p (Holm) |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for cat in CATEGORIES:
        c = per_category.get(cat)
        if not c:
            continue
        lines.append(
            f"| {cat} | {c['n']} | {c['adaptive']:.3f} | {c['best_fixed']:.3f} | "
            f"{c['delta']:+.3f} | [{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p']:.4f} | "
            f"{c['p_holm']:.4f} |"
        )
    lines += [
        "",
        "## What this track does not establish",
        "",
    ]
    for item in results["limitations"]:
        lines.append(f"- {item}")
    if oracle_delta:
        lines += [
            "",
            "## Comparison with the oracle-parse control",
            "",
            "The control track replays the metadata-derived reference filter; this track "
            "uses the parser's own answer.  The difference is the price of not having an "
            "oracle parse: the reference filter carries the anchor person's attribute "
            "list and a time window that follows the subject past the window the query "
            "states, and neither is recoverable from the query text.",
            "",
            "| Category | Control (oracle parse) | End-to-end (LLM parse) | Delta |",
            "|---|---:|---:|---:|",
        ]
        for cat, v in oracle_delta["per_category"].items():
            lines.append(
                f"| {cat} | {v['control']:.3f} | {v['end_to_end']:.3f} | {v['delta']:+.3f} |"
            )
        lines.append(
            f"| **macro** | **{oracle_delta['control_macro_p5']:.3f}** | "
            f"**{oracle_delta['end_to_end_macro_p5']:.3f}** | "
            f"**{oracle_delta['delta']:+.3f}** |"
        )
        lines += [
            "",
            "Parse fidelity against the reference parse:",
            "",
            "| Category | n | Colour agreement | Count target | Reference has label-side attributes | Reference window differs |",
            "|---|---:|---|---|---:|---:|",
        ]
        for cat in CATEGORIES:
            f = fidelity.get(cat)
            if not f:
                continue
            ch, ct = f["colour_agreement"]
            th, tt = f["count_target_agreement"]
            lines.append(
                f"| {cat} | {f['n']} | {ch}/{ct if ct else '-'} | "
                f"{th}/{tt if tt else '-'} | "
                f"{f['reference_has_label_side_attributes']} | "
                f"{f['reference_window_differs_from_parsed']} |"
            )
    lines.append("")
    (RESULTS / f"{args.out_prefix}_tables.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"parse ok: {parse_ok}/{len(rows)}  mean {np.mean(parse_times):.1f}s")
    for m in systems:
        print(f"  {m:18s} macro P@5 = {summary[m]['macro']:.3f}")
    print(f"  adaptive vs {best_fixed}: {sig['delta_p5']:+.3f} p={sig['p']:.4f}")
    print(f"wrote {RESULTS / (args.out_prefix + '_results.json')}")
    print(f"wrote {RESULTS / (args.out_prefix + '_tables.md')}")


if __name__ == "__main__":
    main()
