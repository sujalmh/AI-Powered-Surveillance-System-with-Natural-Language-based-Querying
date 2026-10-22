"""Real component latency measurements on the released single-camera subset.

This script replaces the earlier simulated/lognormal probe.  It measures only
operations it actually executes on this machine:

* cold (in-process) load of the cached OpenCLIP ViT-B/32 and MiniLM-L12-v2;
* text encoding and FAISS search over the released 242 scene vectors and
  1,778 person descriptors;
* clip-score aggregation, adaptive-alpha computation, and result merging over
  the real candidate lists returned by those searches;
* a ThreadPoolExecutor concurrency probe (1/2/4/8 workers) on the OpenCLIP
  text-encoder + scene-FAISS path;
* optionally ``--live-llm N`` real NL-parser calls (network/CPU, ~14 s each).

Excluded and labelled as such: LLM answer generation, clip stitching/ffmpeg,
MongoDB structured retrieval, detection/embedding ingestion, the frontend, and
the paper's 4-camera/312k-descriptor deployment.  The released subset is
camera_0, 2026-03-01 17:01-18:56, so these numbers are a reproducibility
component measurement, **not** full-deployment latency percentiles.

Run:
    python backend/scripts/measure_latency.py
    python backend/scripts/measure_latency.py --live-llm 3
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENE_META = ROOT / "data" / "index" / "faiss" / "meta.json"
SCENE_INDEX = ROOT / "data" / "index" / "faiss" / "index.bin"
PERSON_META = ROOT / "data" / "index" / "faiss_person" / "meta.json"
PERSON_INDEX = ROOT / "data" / "index" / "faiss_person" / "index.bin"
RESULTS_DIR = ROOT / "results"

# Queries are intentionally descriptive of the released subset (single
# corridor/entrance camera).  They are not relevance-labelled; this script
# measures latency, not retrieval quality.
QUERIES: List[str] = [
    "person wearing red clothing",
    "person wearing blue clothing carrying a bag",
    "person in a dark jacket walking through the corridor",
    "person wearing a hat",
    "person with long sleeves and long pants",
    "person wearing a coat near the entrance",
    "people standing in a corridor",
    "person carrying a backpack",
    "person wearing light coloured clothing",
    "person walking past the camera",
    "group of people in the scene",
    "person wearing brown clothing",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                "min_ms": 0.0, "max_ms": 0.0}
    return {
        "n": len(vals),
        "mean_ms": float(statistics.mean(vals)),
        "p50_ms": _pct(vals, 50),
        "p95_ms": _pct(vals, 95),
        "min_ms": float(min(vals)),
        "max_ms": float(max(vals)),
    }


def _timed(fn: Callable[[], Any]) -> Tuple[float, Any]:
    started = time.perf_counter()
    result = fn()
    return (time.perf_counter() - started) * 1000.0, result


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-9:
        return [1.0 for _ in scores]
    return [(float(s) - low) / (high - low) for s in scores]


def _aggregate_scene_hits(
    indices: Sequence[int],
    distances: Sequence[float],
    scene_meta: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate frame hits to clip max, matching the production aggregation."""
    by_clip: Dict[str, float] = {}
    for idx, score in zip(indices, distances):
        idx = int(idx)
        if idx < 0 or idx >= len(scene_meta):
            continue
        row = scene_meta[idx]
        clip = row.get("clip_path")
        if not clip:
            continue
        by_clip[clip] = max(by_clip.get(clip, float("-inf")), float(score))
    clips = [
        {"clip_path": clip, "score": score}
        for clip, score in by_clip.items()
    ]
    norms = _normalize_scores([c["score"] for c in clips])
    for clip, norm in zip(clips, norms):
        clip["score_norm"] = norm
    clips.sort(key=lambda c: c["score"], reverse=True)
    return clips


def _load_scene_meta() -> List[Dict[str, Any]]:
    rows = json.loads(SCENE_META.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected scene metadata shape in {SCENE_META}")
    return rows


def _env_info() -> Dict[str, Any]:
    import faiss
    import open_clip
    import sentence_transformers
    import torch

    cpu = (
        os.getenv("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or platform.machine()
    )
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "open_clip": getattr(open_clip, "__version__", "unknown"),
        "sentence_transformers": getattr(sentence_transformers, "__version__", "unknown"),
        "gpu_available": bool(torch.cuda.is_available()),
        "device": "cpu",
    }


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-llm",
        type=int,
        default=0,
        metavar="N",
        help="make N real NL-parser LLM calls (default 0; each may take ~14 s)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=24,
        help="repetitions for encode/search component timings (default 24)",
    )
    parser.add_argument(
        "--skip-concurrency",
        action="store_true",
        help="skip the ThreadPoolExecutor probe",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    command = "python " + " ".join(
        ["backend/scripts/measure_latency.py"] + sys.argv[1:]
    )
    components: Dict[str, Dict[str, Any]] = {}
    cold_loads: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    excluded = [
        "LLM answer generation (live chat answer path)",
        "clip stitching / ffmpeg clip generation",
        "MongoDB structured detection retrieval (MongoDB is not running)",
        "detection + embedding ingestion (YOLO/OpenVINO/SigLIP indexing)",
        "frontend/network round-trip",
        "4-camera, ~312k-descriptor full deployment; 123-clip camera_0 subset only",
    ]

    # ------------------------------------------------------------------
    # Cold model/index loads
    # ------------------------------------------------------------------
    print("[latency] loading cached OpenCLIP ViT-B/32 (CPU) ...", flush=True)

    def _load_scene_encoder():
        from backend.app.services.sem_embedder import get_embedder

        return get_embedder()

    cold_ms, scene_encoder = _timed(_load_scene_encoder)
    cold_loads["openclip_vit_b32"] = {
        "n": 1, "ms": cold_ms,
        "note": "in-process cold load from local cache; includes module import",
    }

    print("[latency] loading cached MiniLM-L12-v2 ...", flush=True)

    def _load_attribute_encoder():
        from backend.app.services.attribute_encoder import get_attribute_encoder

        return get_attribute_encoder()

    cold_ms, attr_encoder = _timed(_load_attribute_encoder)
    cold_loads["minilm_l12_v2"] = {
        "n": 1, "ms": cold_ms,
        "note": "in-process cold load from local cache; includes module import",
    }

    print("[latency] loading FAISS indices ...", flush=True)
    import faiss

    cold_ms, scene_index = _timed(lambda: faiss.read_index(str(SCENE_INDEX)))
    cold_loads["scene_faiss_index"] = {
        "n": 1, "ms": cold_ms,
        "note": f"{scene_index.ntotal} vectors, d={scene_index.d}",
    }
    def _load_person_store():
        from backend.app.services.person_store import PersonFaissStore

        return PersonFaissStore(dim=1152)

    cold_ms, person_store = _timed(_load_person_store)
    cold_loads["person_faiss_store"] = {
        "n": 1, "ms": cold_ms,
        "note": (
            f"{person_store.count()} descriptors, d=1152; includes meta.json load"
        ),
    }
    scene_meta = _load_scene_meta()
    scene_by_clip = {
        row.get("clip_path"): row for row in scene_meta if row.get("clip_path")
    }
    if scene_index.ntotal != len(scene_meta):
        warnings.append(
            f"scene index/meta length mismatch: {scene_index.ntotal} vs {len(scene_meta)}"
        )

    # Pre-load the production scene store outside the timed region so the
    # first pipeline timing does not include module/index startup.
    from backend.app.services.sem_store import get_faiss_store

    get_faiss_store(dim=512)
    from backend.app.services.sem_search import _cached_text_embed, search_unstructured
    from backend.app.services.result_merger import (
        compute_adaptive_alpha,
        merge_results,
    )
    from backend.app.services.fusion import MultimodalFusion
    from backend.app.services.hybrid_search import HybridSearchEngine

    fusion = MultimodalFusion()
    person_engine = HybridSearchEngine(
        person_store=person_store,
        attr_encoder=attr_encoder,
        scene_meta_path=SCENE_META,
        person_meta_path=PERSON_META,
        enable_mongo=False,
    )

    # ------------------------------------------------------------------
    # Warm-up (excluded from component samples)
    # ------------------------------------------------------------------
    print("[latency] warming encoders/search ...", flush=True)
    scene_encoder.text_embed([QUERIES[0]])
    attr_encoder.encode_text(QUERIES[0])
    _cached_text_embed.cache_clear()
    search_unstructured(QUERIES[0], top_k=10)
    _cached_text_embed.cache_clear()
    person_engine.search_meta(QUERIES[0], top_k=10)

    reps = max(4, int(args.reps))
    scene_encode_ms: List[float] = []
    scene_faiss_ms: List[float] = []
    scene_encode_search_ms: List[float] = []
    scene_aggregate_ms: List[float] = []
    person_text_ms: List[float] = []
    person_fuse_ms: List[float] = []
    person_faiss_ms: List[float] = []
    person_encode_search_ms: List[float] = []
    scene_raw_candidates: List[List[Dict[str, Any]]] = []
    zero_visual = np.zeros((768,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Scene: encode, FAISS, aggregation
    # ------------------------------------------------------------------
    print("[latency] measuring scene encode/search ...", flush=True)
    for i in range(reps):
        query = QUERIES[i % len(QUERIES)]
        ms, embedding = _timed(lambda q=query: scene_encoder.text_embed([q])[0])
        scene_encode_ms.append(ms)
        emb = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, -1)
        ms, (distances, indices) = _timed(
            lambda e=emb: scene_index.search(e, 10)
        )
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        scene_faiss_ms.append(ms)
        scene_encode_search_ms.append(scene_encode_ms[-1] + ms)
        ms, aggregated = _timed(
            lambda d=distances, ix=indices: _aggregate_scene_hits(ix, d, scene_meta)
        )
        scene_aggregate_ms.append(ms)
        sem_candidates = []
        for clip in aggregated:
            row = scene_by_clip.get(clip["clip_path"], {})
            sem_candidates.append(
                {
                    "camera_id": row.get("camera_id"),
                    "clip_path": clip["clip_path"],
                    "clip_url": row.get("clip_url"),
                    "score": clip["score"],
                    "score_raw": clip["score"],
                    "score_norm": clip["score_norm"],
                    "frames": [
                        {"frame_ts": row.get("frame_ts"), "score": clip["score"]}
                    ],
                }
            )
        scene_raw_candidates.append(sem_candidates)

    # ------------------------------------------------------------------
    # Person: MiniLM text encode + zero-visual fusion + FAISS
    # ------------------------------------------------------------------
    print("[latency] measuring person encode/search ...", flush=True)
    for i in range(reps):
        query = QUERIES[i % len(QUERIES)]
        ms, text_emb = _timed(lambda q=query: attr_encoder.encode_text(q))
        person_text_ms.append(ms)
        ms, fused = _timed(lambda t=text_emb: fusion.fuse(zero_visual, t))
        person_fuse_ms.append(ms)
        fv = np.ascontiguousarray(fused, dtype=np.float32).reshape(1, -1)
        ms, _ = _timed(lambda v=fv: person_store.vector_search(v, top_k=50))
        person_faiss_ms.append(ms)
        person_encode_search_ms.append(
            person_text_ms[-1] + person_fuse_ms[-1] + person_faiss_ms[-1]
        )

    # ------------------------------------------------------------------
    # Production scene pipeline + merge on real candidate lists
    # ------------------------------------------------------------------
    print("[latency] measuring production scene pipeline + merge ...", flush=True)
    _cached_text_embed.cache_clear()
    scene_pipeline_ms: List[float] = []
    merge_scene_ms: List[float] = []
    merged_scene_counts: List[int] = []
    for idx, query in enumerate(QUERIES):
        parsed = {"query_type": "visual", "__raw": query, "__semantic_query": query}
        ms, payload = _timed(lambda q=query: search_unstructured(q, top_k=10))
        scene_pipeline_ms.append(ms)
        results = payload.get("semantic_results", [])
        ms, merged = _timed(
            lambda r=results, p=parsed: merge_results([], r, p, "find")
        )
        merge_scene_ms.append(ms)
        merged_scene_counts.append(len(merged))

    # Merge over the *unfiltered* top-10 real FAISS candidate lists as well:
    # the production threshold legitimately drops several subset queries, which
    # would otherwise leave merge_results with empty input for those samples.
    scene_merge_raw_ms: List[float] = []
    for idx, candidates in enumerate(scene_raw_candidates):
        parsed = {
            "query_type": "visual",
            "__raw": QUERIES[idx % len(QUERIES)],
            "__semantic_query": QUERIES[idx % len(QUERIES)],
        }
        ms, _ = _timed(
            lambda c=candidates, p=parsed: merge_results([], c, p, "find")
        )
        scene_merge_raw_ms.append(ms)

    adaptive_alpha_ms: List[float] = []
    for _ in range(50):
        ms, _ = _timed(
            lambda: compute_adaptive_alpha(
                {"query_type": "visual", "objects.object_name": "person"},
                "find",
                False,
                False,
                n_structured=0,
                n_semantic=max(1, len(QUERIES)),
            )
        )
        adaptive_alpha_ms.append(ms)

    # ------------------------------------------------------------------
    # Person routing pipeline + merge on real candidate lists
    # ------------------------------------------------------------------
    print("[latency] measuring person routing + merge ...", flush=True)
    person_search_meta_ms: List[float] = []
    merge_person_ms: List[float] = []
    person_clip_counts: List[int] = []
    person_scores: List[float] = []
    for query in QUERIES:
        parsed = {
            "query_type": "visual",
            "objects.object_name": "person",
            "__raw": query,
            "__semantic_query": query,
        }
        ms, results = _timed(
            lambda q=query, p=parsed: person_engine.search_meta(
                q, top_k=10, parsed_filter=p
            )
        )
        person_search_meta_ms.append(ms)
        person_clip_counts.append(len(results))
        person_scores.extend(float(r.get("score_raw") or 0.0) for r in results)
        ms, _ = _timed(
            lambda r=results, p=parsed: merge_results([], r, p, "find")
        )
        merge_person_ms.append(ms)

    components["scene_text_encode"] = _summary(scene_encode_ms)
    components["scene_faiss_search_k10"] = _summary(scene_faiss_ms)
    components["scene_encode_plus_faiss"] = _summary(scene_encode_search_ms)
    components["scene_clip_aggregate_normalize"] = _summary(scene_aggregate_ms)
    components["scene_pipeline_search"] = _summary(scene_pipeline_ms)
    components["scene_merge"] = _summary(merge_scene_ms)
    components["scene_merge_raw_candidates"] = _summary(scene_merge_raw_ms)
    components["adaptive_alpha"] = _summary(adaptive_alpha_ms)
    components["person_text_encode"] = _summary(person_text_ms)
    components["person_zero_visual_fusion"] = _summary(person_fuse_ms)
    components["person_faiss_search_k50"] = _summary(person_faiss_ms)
    components["person_encode_plus_faiss"] = _summary(person_encode_search_ms)
    components["person_search_meta_to_clip"] = _summary(person_search_meta_ms)
    components["person_merge"] = _summary(merge_person_ms)

    scene_core = [a + b for a, b in zip(scene_pipeline_ms, merge_scene_ms)]
    person_core = [a + b for a, b in zip(person_search_meta_ms, merge_person_ms)]
    end_to_end = {
        "scene_retrieval_core": _summary(scene_core),
        "scene_core_definition": (
            "search_unstructured (OpenCLIP text encode + FAISS + clip aggregation "
            "+ adaptive threshold) + merge_results, per query"
        ),
        "person_retrieval_core": _summary(person_core),
        "person_core_definition": (
            "person search_meta (MiniLM encode + zero-visual fusion + person FAISS "
            "+ timestamp-to-scene-clip mapping + aggregation) + merge_results, "
            "per query"
        ),
        "scene_component_sum_ms": (
            components["scene_encode_plus_faiss"]["mean_ms"]
            + components["scene_clip_aggregate_normalize"]["mean_ms"]
            + components["scene_merge_raw_candidates"]["mean_ms"]
        ),
        "scene_production_component_sum_ms": (
            components["scene_encode_plus_faiss"]["mean_ms"]
            + components["scene_clip_aggregate_normalize"]["mean_ms"]
            + components["scene_merge"]["mean_ms"]
        ),
        "person_component_sum_ms": (
            components["person_search_meta_to_clip"]["mean_ms"]
            + components["person_merge"]["mean_ms"]
        ),
        "excluded_components": excluded,
    }

    # ------------------------------------------------------------------
    # Concurrency probe: OpenCLIP text encode + scene FAISS
    # ------------------------------------------------------------------
    concurrency: List[Dict[str, Any]] = []
    if not args.skip_concurrency:
        print("[latency] measuring concurrency (1/2/4/8 workers) ...", flush=True)
        scene_encoder.text_embed([QUERIES[0]])

        def job(i: int) -> float:
            query = QUERIES[i % len(QUERIES)]
            started = time.perf_counter()
            emb = scene_encoder.text_embed([query])[0]
            emb = np.ascontiguousarray(emb, dtype=np.float32).reshape(1, -1)
            scene_index.search(emb, 10)
            return (time.perf_counter() - started) * 1000.0

        for workers in (1, 2, 4, 8):
            jobs = max(8, workers * 4)
            wall_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                latencies = list(pool.map(job, range(jobs)))
            wall = time.perf_counter() - wall_started
            summary = _summary(latencies)
            concurrency.append(
                {
                    "workers": workers,
                    "jobs": jobs,
                    "wall_seconds": wall,
                    "throughput_jobs_per_s": jobs / wall if wall > 0 else 0.0,
                    "job_latency": summary,
                }
            )

    # ------------------------------------------------------------------
    # Optional live LLM parser calls
    # ------------------------------------------------------------------
    live_llm: Dict[str, Any] = {
        "requested": int(args.live_llm),
        "calls": [],
        "summary": None,
        "note": "excluded from component composition unless --live-llm > 0",
    }
    if args.live_llm > 0:
        print(f"[latency] making {args.live_llm} live LLM parser call(s) ...", flush=True)
        from backend.app.services.nl_parser import parse_nl_with_llm

        llm_ms: List[float] = []
        for query in QUERIES[: max(0, int(args.live_llm))]:
            started = time.perf_counter()
            try:
                parsed = parse_nl_with_llm(query)
                elapsed = (time.perf_counter() - started) * 1000.0
                llm_ms.append(elapsed)
                live_llm["calls"].append(
                    {
                        "query": query,
                        "latency_ms": elapsed,
                        "ok": True,
                        "query_type": parsed.get("query_type"),
                        "timestamp": parsed.get("timestamp"),
                        "count_constraint": parsed.get("count_constraint"),
                    }
                )
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000.0
                live_llm["calls"].append(
                    {
                        "query": query,
                        "latency_ms": elapsed,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        live_llm["summary"] = _summary(llm_ms)
        if not llm_ms:
            warnings.append("all live LLM parser calls failed")

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    payload: Dict[str, Any] = {
        "generated_at": _now_iso(),
        "command": command,
        "environment": _env_info(),
        "subset": {
            "source": "released repository subset",
            "cameras": [0],
            "clips": 123,
            "scene_vectors": int(scene_index.ntotal),
            "person_vectors": int(person_store.count()),
            "time_range": "2026-03-01T17:01:00-18:56:18 (camera_0)",
            "note": (
                "Single-camera subset. These are component measurements, not "
                "full 4-camera/312k-deployment latency percentiles."
            ),
        },
        "cold_loads": cold_loads,
        "components": components,
        "end_to_end": end_to_end,
        "concurrency": concurrency,
        "live_llm": live_llm,
        "diagnostics": {
            "scene_queries": len(QUERIES),
            "reps": reps,
            "merged_scene_counts": merged_scene_counts,
            "person_clip_counts": person_clip_counts,
            "person_score_raw_min": min(person_scores) if person_scores else None,
            "person_score_raw_max": max(person_scores) if person_scores else None,
            "warnings": warnings,
        },
        "excluded": excluded,
    }
    json_path = RESULTS_DIR / "latency_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    component_rows = []
    labels = {
        "scene_text_encode": "Scene OpenCLIP text encode",
        "scene_faiss_search_k10": "Scene FAISS search (242 vectors, k=10)",
        "scene_encode_plus_faiss": "Scene encode + FAISS",
        "scene_clip_aggregate_normalize": "Scene clip aggregation + normalise",
        "scene_pipeline_search": "Scene production search_unstructured",
        "scene_merge": "Scene merge_results (production candidates post-threshold)",
        "scene_merge_raw_candidates": "Scene merge_results (real top-10 FAISS candidates)",
        "adaptive_alpha": "Adaptive alpha computation",
        "person_text_encode": "Person MiniLM text encode",
        "person_zero_visual_fusion": "Person zero-visual fusion (1152-d)",
        "person_faiss_search_k50": "Person FAISS search (1,778 vectors, k=50)",
        "person_encode_plus_faiss": "Person encode + fuse + FAISS",
        "person_search_meta_to_clip": "Person search_meta + nearest-clip mapping",
        "person_merge": "Person merge_results (real candidates)",
    }
    for key, label in labels.items():
        row = components[key]
        component_rows.append(
            [
                label,
                row["n"],
                _fmt_ms(row["mean_ms"]),
                _fmt_ms(row["p50_ms"]),
                _fmt_ms(row["p95_ms"]),
                _fmt_ms(row["min_ms"]),
                _fmt_ms(row["max_ms"]),
            ]
        )

    cold_rows = []
    for key, value in cold_loads.items():
        cold_rows.append([key, _fmt_ms(value["ms"]), value.get("note", "")])

    e2e_rows = [
        [
            "Scene retrieval core (encode+FAISS+aggregate+merge)",
            end_to_end["scene_retrieval_core"]["n"],
            _fmt_ms(end_to_end["scene_retrieval_core"]["mean_ms"]),
            _fmt_ms(end_to_end["scene_retrieval_core"]["p50_ms"]),
            _fmt_ms(end_to_end["scene_retrieval_core"]["p95_ms"]),
        ],
        [
            "Person retrieval core (search_meta+merge)",
            end_to_end["person_retrieval_core"]["n"],
            _fmt_ms(end_to_end["person_retrieval_core"]["mean_ms"]),
            _fmt_ms(end_to_end["person_retrieval_core"]["p50_ms"]),
            _fmt_ms(end_to_end["person_retrieval_core"]["p95_ms"]),
        ],
    ]

    conc_rows = []
    for row in concurrency:
        lat = row["job_latency"]
        conc_rows.append(
            [
                row["workers"],
                row["jobs"],
                f"{row['throughput_jobs_per_s']:.2f}",
                _fmt_ms(lat["p50_ms"]),
                _fmt_ms(lat["p95_ms"]),
                _fmt_ms(lat["mean_ms"]),
            ]
        )

    llm_section = ""
    if args.live_llm > 0:
        llm_summary = live_llm["summary"] or _summary([])
        llm_section = (
            "\n## Live NL-parser calls\n\n"
            f"`--live-llm {args.live_llm}`; n={llm_summary['n']}, "
            f"mean={_fmt_ms(llm_summary['mean_ms'])} ms, "
            f"p50={_fmt_ms(llm_summary['p50_ms'])} ms, "
            f"p95={_fmt_ms(llm_summary['p95_ms'])} ms.  "
            "This is network + LLM inference, not local CPU only.\n"
        )

    md_lines = [
        "# Real latency measurements — released single-camera subset",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Command: `{command}`",
        f"- Environment: CPU-only (`gpu_available={payload['environment']['gpu_available']}`), "
        f"{payload['environment']['cpu_count']} logical CPUs, "
        f"Python {payload['environment']['python']}, "
        f"PyTorch {payload['environment']['torch']}, "
        f"FAISS {payload['environment']['faiss']}.",
        f"- Subset: {payload['subset']['clips']} clips, "
        f"{payload['subset']['scene_vectors']} scene vectors, "
        f"{payload['subset']['person_vectors']} person descriptors, "
        "camera_0 only (2026-03-01 17:01–18:56).",
        "",
        "Every value below is measured from operations executed by this script. "
        "No lognormal/simulated values are used. This is a **reproducibility "
        "component measurement**, not a reproduction of the paper's 4-camera / "
        "312k-descriptor latency percentiles.",
        "",
        "## Cold loads (in-process, local cache)",
        "",
        _md_table(["Component", "ms", "Note"], cold_rows),
        "",
        "## Component timings",
        "",
        _md_table(
            ["Component", "n", "Mean ms", "p50 ms", "p95 ms", "Min ms", "Max ms"],
            component_rows,
        ),
        "",
        "Percentiles use NumPy linear interpolation over the measured samples. "
        "Scene FAISS is an exact `IndexFlatIP` over 242 vectors; person FAISS "
        "over 1,778 descriptors. Text-only person queries use zero visual + "
        "MiniLM text, matching the paper's text-only person branch.",
        "",
        "## End-to-end component composition",
        "",
        _md_table(
            ["Composed path", "n", "Mean ms", "p50 ms", "p95 ms"], e2e_rows
        ),
        "",
        f"- Scene component sum (encode+FAISS + aggregation + merge over real "
        f"top-10 FAISS candidates): "
        f"**{_fmt_ms(end_to_end['scene_component_sum_ms'])} ms** (mean components).",
        f"- Scene production sum (post-threshold candidates; several subset "
        f"queries yield an empty semantic list): "
        f"**{_fmt_ms(end_to_end['scene_production_component_sum_ms'])} ms**.",
        f"- Person component sum (search_meta + merge): "
        f"**{_fmt_ms(end_to_end['person_component_sum_ms'])} ms** (mean components).",
        "",
        "Excluded from these numbers: "
        + "; ".join(excluded)
        + ".",
        "",
    ]
    if args.skip_concurrency:
        md_lines.append("## Concurrency probe\n\nSkipped (`--skip-concurrency`).\n")
    else:
        md_lines += [
            "## Concurrency probe (OpenCLIP text encode + scene FAISS)",
            "",
            _md_table(
                ["Workers", "Jobs", "Throughput jobs/s", "p50 ms", "p95 ms", "Mean ms"],
                conc_rows,
            ),
            "",
            "CPU-only, shared model/index. Throughput is end-to-end wall-clock "
            "for this job mix and includes CPU contention; it is not a 4-camera "
            "ingestion throughput claim.",
            "",
        ]
    md_lines += [
        llm_section,
        "## Routing, multi-scale retrieval, and generalisation implications",
        "",
        "- The released subset is single-camera and short. The measured scene "
        "branch uses 512-d frame-level OpenCLIP vectors aggregated to clips; the "
        "person branch uses 1152-d (SigLIP 768 + MiniLM 384) descriptors with a "
        "text-only query using zero visual, then maps the person timestamp to the "
        "nearest released scene clip. These are different index scales and are "
        "reported separately, not mixed.",
        "- Person routing is intent-gated (`ENABLE_PERSON_ROUTING=true`): explicit "
        "person/attribute/colour queries with no action or count constraint go to "
        "the person index; missing/empty person index falls back to the scene "
        "index. Action and count queries stay on the scene/structured branches.",
        "- The subset cannot estimate cross-camera domain shift, demographic "
        "bias, or 4-camera/312k scaling. Those require the unreleased deployment "
        "data or a new labelled capture; this report makes no such claim.",
        "- The per-component numbers are reproducible on CPU from the shipped "
        "indices. Full-deployment 6 s/12 s totals remain the paper's primary "
        "measurements and are not replaced by this table.",
        "",
    ]
    md_path = RESULTS_DIR / "latency_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[latency] wrote {json_path}", flush=True)
    print(f"[latency] wrote {md_path}", flush=True)
    for key in (
        "scene_encode_plus_faiss",
        "person_search_meta_to_clip",
        "scene_merge",
        "person_merge",
    ):
        row = components[key]
        print(
            f"  {key:32s} n={row['n']:3d} mean={row['mean_ms']:8.2f} "
            f"p50={row['p50_ms']:8.2f} p95={row['p95_ms']:8.2f} ms",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
