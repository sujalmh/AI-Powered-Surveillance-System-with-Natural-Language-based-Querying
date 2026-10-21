"""End-to-end evaluation of the released subset with the *real* NL parser.

This is the track the paper can quote as a system measurement: the natural
language query is parsed by the deployed LLM backend (opencode + DeepSeek
V4.1 Flash by default, OpenAI-compatible endpoints supported), the parsed
filter drives the structured branch, the semantic branch runs over the
released OpenCLIP/FAISS scene index, and the production adaptive-alpha rule
merges the two.

Unlike ``eval_retrieval*.py``, which replays metadata-derived filters, this
script measures the whole query path.  Parses are cached on disk so the run
is reproducible and can be repeated without the model.

Run:
    python backend/scripts/eval_end_to_end.py --limit 150
    python backend/scripts/eval_end_to_end.py --model opencode/deepseek-v4.1-flash-free
"""
from __future__ import annotations

import argparse
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
FIXED_ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
CATEGORIES = ["count", "find", "track", "action", "filter"]


def _heuristic_filter(text: str) -> Dict[str, Any]:
    """Deterministic fallback used when the LLM backend is unreachable.

    Extracts only what can be read off the query text: camera id, colour,
    clock window and an explicit numeric count.  It is deliberately empty
    rather than clever, so that a failed parse is visible in the results
    instead of being silently patched up.
    """
    import re as _re

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
        out["timestamp"] = {"$gte": f"2026-03-01T{times[0]}", "$lte": f"2026-03-01T{times[1]}"}
    elif times:
        out["timestamp"] = {"$gte": f"2026-03-01T{times[0]}", "$lte": f"2026-03-01T{times[0]}"}
    count = _re.search(r"\b(\d+)\s*(?:people|persons|person)\b", low)
    if count:
        out["count_constraint"] = {"eq": int(count.group(1))}
    out["__raw"] = text
    return out


def parse_queries(
    queries: List[Dict[str, Any]], model: str, limit: Optional[int], refresh: bool
) -> Dict[str, Any]:
    """Parse every query with the configured LLM backend (cached on disk)."""
    cache_path = CACHE_DIR / (model.replace("/", "_").replace(":", "_") + ".json")
    cache = llm_client.ParseCache(cache_path)
    rows: List[Dict[str, Any]] = []
    backend = llm_client.build_backend()
    subset_rows = queries[:limit] if limit else queries
    for i, q in enumerate(subset_rows, 1):
        key = q["qid"]
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
                if "usage limit" in str(exc).lower() or "rate limit" in str(exc).lower():
                    # The provider quota is exhausted: stop cleanly, keep the
                    # parses made so far (the cache is written after every
                    # query) and let the run be resumed later.
                    cache.put(key, entry)
                    cache.save()
                    print(f"  [stop] provider quota reached at {key}: {exc}", flush=True)
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


def normalise_for_eval(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map the parser output onto the fields the evaluation expects."""
    p = row.get("parsed") or {}
    out = dict(row)
    out["parsed"] = {
        "time_start": p.get("timestamp", {}).get("$gte") if isinstance(p.get("timestamp"), dict) else None,
        "time_end": p.get("timestamp", {}).get("$lte") if isinstance(p.get("timestamp"), dict) else None,
        "camera_id": p.get("camera_id"),
        "count_target_in_text": (p.get("count_constraint") or {}).get("eq")
        if isinstance(p.get("count_constraint"), dict) else p.get("count_constraint"),
        "count_constraint": p.get("count_constraint"),
        "objects.color": p.get("objects.color"),
        "objects.object_name": p.get("objects.object_name"),
        "zone": p.get("zone"),
        "action": p.get("action"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model id override")
    ap.add_argument("--provider", default=None, help="provider override")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--refresh", action="store_true", help="ignore the parse cache")
    ap.add_argument("--queries", default=str(ROOT / "backend" / "benchmarks" / "queries.jsonl"))
    args = ap.parse_args()

    from backend.app.config import settings

    if args.provider:
        settings.LLM_PROVIDER = args.provider
    if args.model:
        settings.NL_DEFAULT_MODEL = args.model
    cfg = settings.get_active_llm_config()
    model = args.model or cfg.get("model") or settings.OPENCODE_MODEL
    print(f"provider={cfg.get('provider')} model={model}")

    queries = [
        json.loads(line)
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"parsing {min(args.limit, len(queries))} queries with the deployed parser...")
    parsed = parse_queries(queries, model, args.limit, args.refresh)
    rows = [normalise_for_eval(r) for r in parsed["rows"]]

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
    dm, lo, hi, p = erc.paired_bootstrap_ci(a_vals, b_vals, n_boot=3000, seed=11)
    gm, glo, ghi, gp = erc.paired_grouped_bootstrap_ci(a_vals, b_vals, groups, n_boot=3000, seed=11)

    parse_ok = sum(1 for r in rows if r.get("parse_ok"))
    parse_times = [r.get("parse_seconds") or 0.0 for r in rows]

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "track": "end-to-end released-subset evaluation (real LLM parse)",
            "model": model,
            "provider": cfg.get("provider"),
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
            "cache": parsed["cache"],
        },
        "test": summary,
        "best_fixed_on_tune": {"method": best_fixed, "tune_macro_p5": tune_scores[best_fixed]},
        "significance": {
            "comparison": f"adaptive vs {best_fixed} (fixed alpha selected on the tune split)",
            "delta_p5": float(dm), "lo": float(lo), "hi": float(hi), "p": float(p),
            "grouped_delta_p5": float(gm), "grouped_lo": float(glo),
            "grouped_hi": float(ghi), "grouped_p": float(gp),
        },
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
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "end_to_end_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# End-to-end released-subset evaluation (real LLM parser)",
        "",
        f"Model: `{model}` via provider `{cfg.get('provider')}`.",
        "",
        f"Parse: {parse_ok}/{len(rows)} answered by the LLM "
        f"({parse_ok / max(1, len(rows)) * 100:.0f}%), "
        f"mean {np.mean(parse_times):.1f}s, p95 {np.percentile(parse_times, 95):.1f}s per query.",
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
        f"[{sig['grouped_lo']:+.3f}, {sig['grouped_hi']:+.3f}], p = {sig['grouped_p']:.4f}.",
        "",
        "## What cannot be reproduced",
        "",
    ]
    for item in results["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    (RESULTS / "end_to_end_tables.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"parse ok: {parse_ok}/{len(rows)}  mean {np.mean(parse_times):.1f}s")
    for m in systems:
        print(f"  {m:18s} macro P@5 = {summary[m]['macro']:.3f}")
    print(f"  adaptive vs {best_fixed}: {sig['delta_p5']:+.3f} p={sig['p']:.4f}")
    print(f"wrote {RESULTS / 'end_to_end_results.json'}")
    print(f"wrote {RESULTS / 'end_to_end_tables.md'}")


if __name__ == "__main__":
    main()
