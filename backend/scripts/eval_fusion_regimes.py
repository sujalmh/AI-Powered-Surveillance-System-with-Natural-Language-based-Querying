"""Regime slices + frozen photometric/descriptor stress for person fusion.

Released-subset only.  Reuses helpers from ``backend/scripts/eval_fusion.py``.

Modes:
  --mode regimes   metadata-proxy stratum tables (P@5 + pair AUC per method)
  --mode stress    descriptor-level gallery stress curves
  --mode all       both (default)

Outputs (never overwrite an existing results file without --overwrite):
  results/regime_slices.json / .md
  results/regime_stress.json / .md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.scripts.eval_fusion as ef
from backend.app.services import fusion as fusion_mod

# Safe fusion: allow zero-branch descriptors (rows 1, 143 in the release) to
# fuse to a zero vector instead of raising, matching the pre-hardening
# behaviour under which results/fusion_results.json was produced.
def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))

def _safe_fuse(visual_emb, text_emb, vw=0.6, tw=0.4):
    v = np.array(visual_emb, dtype=np.float32, copy=True).reshape(-1)
    t = np.array(text_emb, dtype=np.float32, copy=True).reshape(-1)
    if v.shape[0] != 768:
        v2 = np.zeros((768,), dtype=np.float32); n = min(768, v.shape[0])
        if n > 0: v2[:n] = v[:n]
        v = v2
    if t.shape[0] != 384:
        t2 = np.zeros((384,), dtype=np.float32); n = min(384, t.shape[0])
        if n > 0: t2[:n] = t[:n]
        t = t2
    v = (v * float(vw)).astype(np.float32)
    t = (t * float(tw)).astype(np.float32)
    fused = np.concatenate([v, t]).astype(np.float32)
    norm = float(np.linalg.norm(fused)) + 1e-6
    return (fused / norm).astype(np.float32)

def _safe_fuse_gated(visual_emb, text_emb, visual_quality, text_quality, lam=4.0,
                     base_visual_weight=0.6, base_text_weight=0.4):
    v = np.array(visual_emb, dtype=np.float32, copy=True).reshape(-1)
    t = np.array(text_emb, dtype=np.float32, copy=True).reshape(-1)
    if v.shape[0] != 768:
        v2 = np.zeros((768,), dtype=np.float32); n = min(768, v.shape[0])
        if n > 0: v2[:n] = v[:n]
        v = v2
    if t.shape[0] != 384:
        t2 = np.zeros((384,), dtype=np.float32); n = min(384, t.shape[0])
        if n > 0: t2[:n] = t[:n]
        t = t2
    qv = fusion_mod._safe_quality(visual_quality)
    qt = fusion_mod._safe_quality(text_quality)
    gw = _sigmoid(float(lam) * (qv - qt))
    gw = max(0.05, min(0.95, gw))
    wv = (base_visual_weight * 2.0) * gw
    wt = (base_text_weight * 2.0) * (1.0 - gw)
    v = (v * wv).astype(np.float32)
    t = (t * wt).astype(np.float32)
    fused = np.concatenate([v, t]).astype(np.float32)
    norm = float(np.linalg.norm(fused)) + 1e-6
    return (fused / norm).astype(np.float32)

class _SafeMMF(ef.MultimodalFusion):
    def fuse(self, visual_emb, text_emb):
        return _safe_fuse(visual_emb, text_emb, self.visual_weight, self.text_weight)
    def fuse_gated(self, visual_emb, text_emb, visual_quality, text_quality, lam=4.0):
        return _safe_fuse_gated(visual_emb, text_emb, visual_quality, text_quality, lam=lam,
                                base_visual_weight=self.visual_weight,
                                base_text_weight=self.text_weight)

ef.MultimodalFusion = _SafeMMF
ef.fuse_gated = _safe_fuse_gated
fusion_mod.MultimodalFusion = _SafeMMF
fusion_mod.fuse_gated = _safe_fuse_gated

from backend.app.eval import subset
from backend.app.eval.stats import bootstrap_ci, grouped_bootstrap_ci

RESULTS = ROOT / "results"
N_BOOT = 2000

# ============================================================
# PRE-REGISTRATION (frozen before any computation)
# ============================================================
SEED = 20260301
STRESS_SEEDS = {
    "gaussian": [1001, 1002, 1003],
    "dropout": [2001, 2002, 2003],
}

PRE_REGISTRATION = {
    "artifact_lock": "released subset only: backend/benchmarks/*, data/index/*, subset metadata",
    "seed_lock": SEED,
    "strata": {
        "illumination": {
            "proxy": "anchor-clip start hour (VLM caption lighting words: none present in release)",
            "rule": "stratum='early' if clip_start.hour <= 17 else 'late'",
            "split": "fixed by hour value (no search)",
        },
        "scene_density": {
            "proxy": "max person count across frames of anchor clip (released count field)",
            "rule": "median split over the evaluation set; ties to lower stratum",
            "split": "median over evaluation set",
        },
        "motion_energy": {
            "proxy": "subset.compute_motion_energy(clip)['mean']",
            "rule": "median split over the evaluation set; ties to lower stratum",
            "split": "median over evaluation set",
        },
    },
    "stress_grid": {
        "clean": "identity (no perturbation)",
        "gaussian_noise": {
            "sigmas": [0.03, 0.08, 0.15],
            "seeds": STRESS_SEEDS["gaussian"],
            "op": "v <- normalize(v + N(0,sigma)); t <- normalize(t + N(0,sigma))",
        },
        "dropout": {
            "rates": [0.05, 0.10, 0.20],
            "seeds": STRESS_SEEDS["dropout"],
            "op": "mask <- Bernoulli(1-p) per dimension; v <- normalize(v*mask); t <- normalize(t*mask)",
        },
        "magnitude_scale": {
            "scales": [0.8, 0.5],
            "op": "v <- normalize(scale * v); t <- normalize(scale * t)  (norm shrinkage)",
        },
    },
    "methods_evaluated": ["visual", "text", "concat", "gated", "zero_pad",
                          "norm_matched", "learned_proj", "rrf", "randproj"],
    "metrics": {
        "retrieval": "macro P@5 over queries in stratum/condition",
        "pair": "threshold-free pair AUC over test pairs in stratum/condition",
    },
    "ci": "query-level / pair-stratified bootstrap, n_boot=%d, seed=%d" % (N_BOOT, SEED),
    "evaluation": "gallery-side degradation only: gallery vectors perturbed, query vectors kept clean; "
                  "track formation, query set, and labels unchanged.",
}


def write_json(path: Path, obj: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_md(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ============================================================
# Shared setup
# ============================================================

def setup_base() -> Dict[str, Any]:
    """Reproduce eval_fusion setup (clean). Returns context dict."""
    dets, vectors, visual, text, vis_track_of, vis_tracks, split = ef.load_tracks()
    meta_track_of, meta_tracks, meta_edges = ef.build_tracks_metadata(dets)
    meta_cal_tracks, meta_test_tracks = ef.split_tracks_by_time(meta_tracks)
    meta_split_of = {tid: ("calibration" if tid in meta_cal_tracks else "test")
                     for tid in range(len(meta_tracks))}
    frames = subset.load_scene_frames()
    qv, qt, quality_info = ef.estimate_qualities(dets, frames)

    track_of, tracks = meta_track_of, meta_tracks
    cal_tracks, test_tracks = meta_cal_tracks, meta_test_tracks
    track_name = {tid: f"{meta_split_of[tid]}_t{tid:05d}" for tid in range(len(meta_tracks))}
    track_meta = {
        "n_tracks": len(meta_tracks),
        "n_test_tracks": len(meta_test_tracks),
        "n_calibration_tracks": len(meta_cal_tracks),
    }

    cal_idx = [i for i, t in enumerate(track_of) if t in cal_tracks]
    W = ef.fit_ridge_projection(visual, text, cal_idx, lam=1.0)

    fuse = ef.MultimodalFusion(0.6, 0.4)
    concat = np.vstack([ef.fuse_detached(fuse, v, t) for v, t in zip(visual, text)])

    test_idx = [i for i, t in enumerate(track_of) if t in test_tracks]
    relevant_by_query: Dict[int, set] = {}
    candidates_by_query: Dict[int, List[int]] = {}
    query_indices: List[int] = []
    for qi in test_idx:
        tid = track_of[qi]
        rel = {j for j in test_idx if j != qi and track_of[j] == tid
               and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) > 0.5}
        if not rel:
            continue
        cand = [j for j in test_idx if j != qi and not (
            track_of[j] == tid and abs((dets[j]["timestamp"] - dets[qi]["timestamp"]).total_seconds()) <= 0.5)]
        relevant_by_query[qi] = rel
        candidates_by_query[qi] = cand
        query_indices.append(qi)

    # pairs
    pair_rows = ef.build_pairs_from_tracks(track_of, tracks, dets, meta_split_of, seed=0)
    cal_pairs = [p for p in pair_rows if p["split"] == "calibration"]
    test_pairs = [p for p in pair_rows if p["split"] == "test"]
    cal_labels = np.asarray([1 if p["same"] else 0 for p in cal_pairs], dtype=np.int64)
    test_labels = np.asarray([1 if p["same"] else 0 for p in test_pairs], dtype=np.int64)

    rrf_cache = ef.build_rrf_pair_cache(visual, text, {"calibration": cal_idx, "test": test_idx})

    # clip assignment for proxies
    clips = subset.build_clip_table(frames)
    clip_ids = subset.assign_detections_to_clips(dets, clips)
    motion = subset.compute_motion_energy(clips)
    # max count per clip
    clip_max_count: Dict[str, int] = {}
    for cid, clip in clips.items():
        counts = clip.get("counts") or []
        clip_max_count[cid] = int(max(counts)) if counts else 1

    det_proxy: List[Dict[str, Any]] = []
    for i, d in enumerate(dets):
        cid = clip_ids[i]
        clip = clips.get(cid) if cid else None
        hour = (clip["start"].hour if clip else d["timestamp"].hour)
        mc = clip_max_count.get(cid, 1) if cid else 1
        me = 0.0
        if cid and cid in motion:
            me = float(motion[cid].get("mean", 0.0))
        det_proxy.append({"clip_id": cid, "hour": int(hour),
                         "max_count": int(mc), "motion_mean": float(me)})

    return {
        "dets": dets, "visual": visual, "text": text, "W": W, "qv": qv, "qt": qt,
        "track_of": track_of, "tracks": tracks, "meta_split_of": meta_split_of,
        "query_indices": query_indices, "candidates_by_query": candidates_by_query,
        "relevant_by_query": relevant_by_query, "test_pairs": test_pairs,
        "test_labels": test_labels, "cal_pairs": cal_pairs, "cal_labels": cal_labels,
        "rrf_cache": rrf_cache, "det_proxy": det_proxy, "frames": frames, "clips": clips,
        "quality_info": quality_info, "concat_clean": concat,
    }


def build_galleries(ctx: Dict[str, Any], visual: np.ndarray, text: np.ndarray) -> Dict[str, np.ndarray]:
    W = ctx["W"]
    qv = ctx["qv"]
    qt = ctx["qt"]
    fuse = ef.MultimodalFusion(0.6, 0.4)
    concat = np.vstack([ef.fuse_detached(fuse, v, t) for v, t in zip(visual, text)])
    galleries = ef.build_method_galleries(visual, text, concat, W, qv, qt)
    # rrf has no gallery matrix; placeholder visual for pair_scores_for_method fallback
    galleries.setdefault("rrf", visual.astype(np.float32))
    return galleries


def method_list() -> List[str]:
    return ["visual", "text", "concat", "gated", "zero_pad", "norm_matched",
            "learned_proj", "rrf", "randproj"]


# ============================================================
# Proxy helpers
# ============================================================

def proxy_value(ctx: Dict[str, Any], det_idx: int, proxy: str) -> float:
    p = ctx["det_proxy"][det_idx]
    if proxy == "illumination":
        return float(p["hour"])
    if proxy == "scene_density":
        return float(p["max_count"])
    if proxy == "motion_energy":
        return float(p["motion_mean"])
    raise ValueError(proxy)


def stratum_label_for_value(proxy: str, value: float, threshold: float) -> str:
    if proxy == "illumination":
        return "early" if value <= 17 else "late"
    return "low" if value <= threshold else "high"


def compute_thresholds(ctx: Dict[str, Any], proxy: str, items: Sequence[int], item_to_det: Callable[[int], int]) -> float:
    vals = [proxy_value(ctx, item_to_det(i), proxy) for i in items]
    return float(np.median(vals)) if vals else 0.0


# ============================================================
# Retrieval evaluation (gallery/query may differ)
# ============================================================

def eval_retrieval_for_condition(
    ctx: Dict[str, Any],
    galleries_p: Dict[str, np.ndarray],   # perturbed gallery matrices
    visual_clean: np.ndarray,
    text_clean: np.ndarray,
    visual_p: np.ndarray,
    text_p: np.ndarray,
    clean_query: bool = True,
) -> Dict[int, Dict[str, float]]:
    """Returns per-query p5 for each method? We'll return dict method -> list of (qid, p5)."""
    dets = ctx["dets"]
    qis = ctx["query_indices"]
    cand_map = ctx["candidates_by_query"]
    rel_map = ctx["relevant_by_query"]
    visual_c = visual_clean
    text_c = text_clean

    out: Dict[str, List[Tuple[int, float]]] = {m: [] for m in method_list()}

    # deterministic methods except rrf
    for name in ["visual", "text", "concat", "gated", "zero_pad", "norm_matched", "learned_proj"]:
        g_p = galleries_p[name]
        for qi in qis:
            cand = cand_map[qi]
            carr = np.asarray(cand, dtype=np.int64)
            if name == "visual":
                qv = visual_c[qi]
                gv = g_p  # visual space
                # gallery is visual_p matrix
                scores = g_p @ qv if False else None
            # general: gallery matrix @ query vector (clean query)
            if name == "visual":
                qvec = visual_c[qi]
                smat = g_p
            elif name == "text":
                qvec = text_c[qi]
                smat = g_p
            else:
                # fused spaces: gallery and query both in fused space; query clean vector not available in fused space.
                # We approximate by using clean gallery vector of query from clean galleries.
                qvec = None
            # For fused methods, query vector must be clean fused vector of qi.
            # We'll handle separately below.
            _ = None
        # We'll instead do a unified loop per method using a query vector resolver.
    # Unified approach:
    out = {m: [] for m in method_list()}
    # Precompute clean fused query vectors for each method from clean galleries
    # (galleries built from clean visual/text) -- but galleries_p only has perturbed.
    # We need clean galleries too. Build them from ctx clean vectors:
    galleries_c = build_galleries(ctx, visual_c, text_c)

    for name in ["visual", "text", "concat", "gated", "zero_pad", "norm_matched", "learned_proj"]:
        g_p = galleries_p[name]
        g_c = galleries_c[name]
        for qi in qis:
            cand = cand_map[qi]
            carr = np.asarray(cand, dtype=np.int64)
            qvec = g_c[qi]  # clean query vector in method space
            scores = g_p[carr] @ qvec if False else None
            # need scores over candidates for ranking; evaluate by dot of each candidate with qvec
            s = g_p[carr] @ qvec
            # build full score vector over all docs for evaluate? retrieval_metrics needs scores array indexed by doc id.
            # We'll compute ranking directly.
            full_scores = np.zeros(len(g_p), dtype=np.float64)
            full_scores[carr] = s
            # but candidates only; ranking_from_scores uses scores[cand]. fine.
            m = ef.retrieval_metrics(full_scores, cand, rel_map[qi])
            out[name].append((qi, float(m["p5"])))

    # rrf
    gvp = visual_p
    gvc = visual_c
    gtp = text_p
    gtc = text_c
    for qi in qis:
        cand = cand_map[qi]
        carr = np.asarray(cand, dtype=np.int64)
        sv = gvp[carr] @ gvc[qi]
        st = gtp[carr] @ gtc[qi]
        order_v = np.argsort(-sv, kind="stable")
        order_t = np.argsort(-st, kind="stable")
        rank_v = [str(carr[i]) for i in order_v]
        rank_t = [str(carr[i]) for i in order_t]
        fused = ef.rrf_fuse(rank_v, rank_t, k=60)[:10]
        rank = [int(x) for x in fused]
        p5 = ef.precision_at_k(rank, rel_map[qi], 5)
        out["rrf"].append((qi, float(p5)))

    # randproj: average over seeds
    rp_p5_by_qi = {qi: [] for qi in qis}
    for seed in range(5):
        # perturbed gallery for randproj
        g_p_rp = ef.random_proj_add_batch(visual_p, text_p, seed=seed)
        g_c_rp = ef.random_proj_add_batch(visual_c, text_c, seed=seed)
        for qi in qis:
            cand = cand_map[qi]
            carr = np.asarray(cand, dtype=np.int64)
            qvec = g_c_rp[qi]
            s = g_p_rp[carr] @ qvec
            full = np.zeros(len(g_p_rp), dtype=np.float64)
            full[carr] = s
            m = ef.retrieval_metrics(full, cand, rel_map[qi])
            rp_p5_by_qi[qi].append(float(m["p5"]))
    for qi in qis:
        out["randproj"].append((qi, float(np.mean(rp_p5_by_qi[qi]))))

    return out


def eval_pairs_for_condition(
    ctx: Dict[str, Any],
    galleries_p: Dict[str, np.ndarray],
    visual_clean: np.ndarray,
    text_clean: np.ndarray,
    visual_p: np.ndarray,
    text_p: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return method -> array of scores over test_pairs."""
    test_pairs = ctx["test_pairs"]
    visual_c = visual_clean
    text_c = text_clean
    galleries_c = build_galleries(ctx, visual_c, text_c)
    out: Dict[str, np.ndarray] = {}

    # rrf cache for perturbed
    # build rrf pair scores on the fly
    for name in ["visual", "text", "concat", "gated", "zero_pad", "norm_matched", "learned_proj"]:
        g_p = galleries_p[name]
        g_c = galleries_c[name]
        sc = []
        for p in test_pairs:
            qi = int(p["query_row_index"])
            gj = int(p["gallery_row_index"])
            sc.append(float(g_p[gj] @ g_c[qi]))
        out[name] = np.asarray(sc, dtype=np.float64)

    # rrf: rank over the full held-out test split for parity with eval_fusion.
    pool = list(ctx["test_idx"])
    sc = []
    for p in test_pairs:
        qi = int(p["query_row_index"])
        gj = int(p["gallery_row_index"])
        cand = [j for j in pool if j != qi]
        if gj not in cand:
            cand = cand + [gj]
        sv = visual_p[cand] @ visual_c[qi]
        st = text_p[cand] @ text_c[qi]
        order_v = np.argsort(-sv, kind="stable")
        order_t = np.argsort(-st, kind="stable")
        rank_v = {cand[i]: k + 1 for k, i in enumerate(order_v)}
        rank_t = {cand[i]: k + 1 for k, i in enumerate(order_t)}
        if gj in rank_v:
            sc.append(1.0 / (60 + rank_v[gj]) + 1.0 / (60 + rank_t[gj]))
        else:
            sc.append(0.0)
    out["rrf"] = np.asarray(sc, dtype=np.float64)

    # randproj averaged over seeds
    rp_scores = []
    for seed in range(5):
        g_p_rp = ef.random_proj_add_batch(visual_p, text_p, seed=seed)
        g_c_rp = ef.random_proj_add_batch(visual_c, text_c, seed=seed)
        sc = []
        for p in test_pairs:
            qi = int(p["query_row_index"])
            gj = int(p["gallery_row_index"])
            sc.append(float(g_p_rp[gj] @ g_c_rp[qi]))
        rp_scores.append(sc)
    out["randproj"] = np.asarray(rp_scores, dtype=np.float64).mean(axis=0)

    return out


# Fix context to include cal_tracks/test_tracks/test_idx
def enrich_ctx(ctx: Dict[str, Any]) -> None:
    # ctx already has track_of, tracks; meta_split_of maps track->split
    ctx["cal_tracks"] = {t for t in range(len(ctx["tracks"])) if ctx["meta_split_of"].get(t) == "calibration"}
    ctx["test_tracks"] = {t for t in range(len(ctx["tracks"])) if ctx["meta_split_of"].get(t) == "test"}
    ctx["test_idx"] = [i for i, t in enumerate(ctx["track_of"]) if t in ctx["test_tracks"]]
    ctx["cal_idx"] = [i for i, t in enumerate(ctx["track_of"]) if t in ctx["cal_tracks"]]


# ============================================================
# Regime slices
# ============================================================

def run_regimes(ctx: Dict[str, Any], overwrite: bool) -> Dict[str, Any]:
    enrich_ctx(ctx)
    dets = ctx["dets"]
    qis = ctx["query_indices"]
    test_pairs = ctx["test_pairs"]
    test_labels = ctx["test_labels"]

    # clean galleries (visual=text original)
    galleries_c = build_galleries(ctx, ctx["visual"], ctx["text"])
    # retrieval clean per query p5 per method
    clean_ret = eval_retrieval_for_condition(ctx, galleries_c, ctx["visual"], ctx["text"],
                                            ctx["visual"], ctx["text"])
    # pair scores clean
    clean_pair = eval_pairs_for_condition(ctx, galleries_c, ctx["visual"], ctx["text"],
                                         ctx["visual"], ctx["text"])

    proxies = ["illumination", "scene_density", "motion_energy"]

    # thresholds for retrieval queries
    ret_thresholds = {}
    for proxy in proxies:
        if proxy == "illumination":
            ret_thresholds[proxy] = 17.0
        else:
            ret_thresholds[proxy] = compute_thresholds(ctx, proxy, qis, lambda i: i)

    # thresholds for pairs (by query det)
    pair_thresholds = {}
    pair_qis = [int(p["query_row_index"]) for p in test_pairs]
    for proxy in proxies:
        if proxy == "illumination":
            pair_thresholds[proxy] = 17.0
        else:
            pair_thresholds[proxy] = compute_thresholds(ctx, proxy, range(len(test_pairs)),
                                                       lambda i: pair_qis[i])

    def ret_stratum(qi: int, proxy: str) -> str:
        v = proxy_value(ctx, qi, proxy)
        thr = ret_thresholds[proxy]
        return stratum_label_for_value(proxy, v, thr)

    def pair_stratum(k: int, proxy: str) -> str:
        qi = int(test_pairs[k]["query_row_index"])
        v = proxy_value(ctx, qi, proxy)
        thr = pair_thresholds[proxy]
        return stratum_label_for_value(proxy, v, thr)

    results = {
        "scope": {
            "name": "released single-camera person-descriptor subset",
            "not": "full 4-camera / 312k-descriptor identity benchmark",
        },
        "pre_registration": PRE_REGISTRATION,
        "retrieval": {},
        "pairs": {},
    }

    # Retrieval strata
    for proxy in proxies:
        strata = sorted(set(ret_stratum(qi, proxy) for qi in qis))
        results["retrieval"][proxy] = {}
        for stratum in strata:
            qs = [qi for qi in qis if ret_stratum(qi, proxy) == stratum]
            n = len(qs)
            stratum_out = {"n_queries": n, "methods": {}}
            for method in method_list():
                vals = [dict(clean_ret[method])[qi] for qi in qs]
                # bootstrap CI over queries
                if n > 0:
                    m, lo, hi = bootstrap_ci(vals, n_boot=N_BOOT, seed=SEED)
                else:
                    m, lo, hi = 0.0, 0.0, 0.0
                stratum_out["methods"][method] = {
                    "p5": float(np.mean(vals)) if vals else 0.0,
                    "p5_ci": [float(lo), float(hi)],
                    "n": n,
                }
            # group info
            groups = [str(ctx["det_proxy"][qi]["clip_id"]) for qi in qs]
            stratum_out["n_clips"] = len(set(groups))
            results["retrieval"][proxy][stratum] = stratum_out

    # Pair strata
    for proxy in proxies:
        strata = sorted(set(pair_stratum(k, proxy) for k in range(len(test_pairs))))
        results["pairs"][proxy] = {}
        for stratum in strata:
            idxs = [k for k in range(len(test_pairs)) if pair_stratum(k, proxy) == stratum]
            n = len(idxs)
            lab = test_labels[idxs] if idxs else np.asarray([], dtype=np.int64)
            stratum_out = {"n_pairs": n, "n_pos": int(lab.sum()) if len(lab) else 0, "methods": {}}
            for method in method_list():
                sc = clean_pair[method][idxs] if idxs else np.asarray([], dtype=np.float64)
                if len(sc) > 0 and len(set(lab.tolist())) > 1:
                    auc = ef.auc_score(sc, lab)
                    # bootstrap CI (stratified by label)
                    pos = np.flatnonzero(lab == 1)
                    neg = np.flatnonzero(lab == 0)
                    boots = []
                    rng = random.Random(SEED)
                    for _ in range(N_BOOT):
                        ii = np.concatenate([
                            np.asarray([pos[rng.randrange(len(pos))] for _ in range(len(pos))]) if len(pos) else np.asarray([], dtype=int),
                            np.asarray([neg[rng.randrange(len(neg))] for _ in range(len(neg))]) if len(neg) else np.asarray([], dtype=int),
                        ])
                        if len(ii) == 0 or len(set(lab[ii].tolist())) < 2:
                            continue
                        boots.append(ef.auc_score(sc[ii], lab[ii]))
                    boots.sort()
                    lo = boots[int(0.025 * len(boots))] if boots else 0.0
                    hi = boots[int(0.975 * len(boots)) - 1] if boots else 0.0
                else:
                    auc = 0.0
                    lo = hi = 0.0
                stratum_out["methods"][method] = {
                    "auc": float(auc), "auc_ci": [float(lo), float(hi)], "n": n,
                }
            results["pairs"][proxy][stratum] = stratum_out

    # MD
    md = []
    md.append("# Regime slices — person fusion (released subset)\n")
    md.append("**Scope.** released single-camera subset; not the 4-camera/312k deployment.\n")
    md.append("## Pre-registration\n")
    md.append("Frozen strata definitions are embedded in `results/regime_slices.json` under `pre_registration`.\n")
    md.append("## Retrieval P@5 by stratum\n")
    for proxy in proxies:
        md.append(f"### {proxy}\n")
        md.append("| stratum | n | " + " | ".join(method_list()) + " |")
        md.append("|---|" + "---|" * (len(method_list()) + 1))
        strata = sorted(results["retrieval"][proxy].keys())
        for st in strata:
            row = results["retrieval"][proxy][st]
            vals = [f"{row['methods'][m]['p5']:.3f}" for m in method_list()]
            md.append(f"| {st} | {row['n_queries']} | " + " | ".join(vals) + " |")
        md.append("")
    md.append("## Pair AUC by stratum\n")
    for proxy in proxies:
        md.append(f"### {proxy}\n")
        md.append("| stratum | n | " + " | ".join(method_list()) + " |")
        md.append("|---|" + "---|" * (len(method_list()) + 1))
        strata = sorted(results["pairs"][proxy].keys())
        for st in strata:
            row = results["pairs"][proxy][st]
            vals = [f"{row['methods'][m]['auc']:.3f}" for m in method_list()]
            md.append(f"| {st} | {row['n_pairs']} | " + " | ".join(vals) + " |")
        md.append("")

    write_json(RESULTS / "regime_slices.json", results, overwrite)
    write_md(RESULTS / "regime_slices.md", "\n".join(md), overwrite)
    return results


# ============================================================
# Stress
# ============================================================

def apply_perturbation(visual: np.ndarray, text: np.ndarray, spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    v = visual.astype(np.float32).copy()
    t = text.astype(np.float32).copy()
    kind = spec["kind"]
    if kind == "clean":
        return v, t
    if kind == "gaussian":
        sigma = spec["sigma"]
        rng = np.random.RandomState(spec["seed"])
        v = v + rng.normal(0.0, sigma, v.shape).astype(np.float32)
        t = t + rng.normal(0.0, sigma, t.shape).astype(np.float32)
        # renormalize rows
        vn = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        tn = np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        return (v / vn).astype(np.float32), (t / tn).astype(np.float32)
    if kind == "dropout":
        p = spec["rate"]
        rng = np.random.RandomState(spec["seed"])
        mv = (rng.random(v.shape) >= p).astype(np.float32)
        mt = (rng.random(t.shape) >= p).astype(np.float32)
        v = v * mv
        t = t * mt
        vn = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        tn = np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        return (v / vn).astype(np.float32), (t / tn).astype(np.float32)
    if kind == "magnitude":
        scale = spec["scale"]
        v = scale * v
        t = scale * t
        vn = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        tn = np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        return (v / vn).astype(np.float32), (t / tn).astype(np.float32)
    raise ValueError(kind)


def stress_conditions() -> List[Dict[str, Any]]:
    out = [{"id": "clean", "kind": "clean", "label": "clean (identity)"}]
    for sigma, seed in zip([0.03, 0.08, 0.15], STRESS_SEEDS["gaussian"]):
        out.append({"id": f"gauss_{sigma:.2f}", "kind": "gaussian", "sigma": sigma, "seed": seed,
                    "label": f"gaussian noise sigma={sigma}"})
    for rate, seed in zip([0.05, 0.10, 0.20], STRESS_SEEDS["dropout"]):
        out.append({"id": f"dropout_{rate:.2f}", "kind": "dropout", "rate": rate, "seed": seed,
                    "label": f"dropout p={rate}"})
    for scale in [0.8, 0.5]:
        out.append({"id": f"mag_{scale:.1f}", "kind": "magnitude", "scale": scale,
                    "label": f"magnitude scale={scale}"})
    return out


def run_stress(ctx: Dict[str, Any], overwrite: bool) -> Dict[str, Any]:
    enrich_ctx(ctx)
    dets = ctx["dets"]
    qis = ctx["query_indices"]
    test_pairs = ctx["test_pairs"]
    test_labels = ctx["test_labels"]
    visual_c = ctx["visual"]
    text_c = ctx["text"]

    # clean baseline metrics
    galleries_c = build_galleries(ctx, visual_c, text_c)
    clean_ret = eval_retrieval_for_condition(ctx, galleries_c, visual_c, text_c, visual_c, text_c)
    clean_pair = eval_pairs_for_condition(ctx, galleries_c, visual_c, text_c, visual_c, text_c)

    clean_ret_mean = {m: float(np.mean([v for (_, v) in clean_ret[m]])) for m in method_list()}
    clean_pair_auc = {m: float(ef.auc_score(clean_pair[m], test_labels)) for m in method_list()}

    conds = stress_conditions()
    results = {
        "scope": {"name": "released single-camera person-descriptor subset",
                  "not": "full 4-camera / 312k-descriptor identity benchmark"},
        "pre_registration": PRE_REGISTRATION,
        "clean_baseline": {
            "retrieval_p5": clean_ret_mean,
            "pair_auc": clean_pair_auc,
        },
        "conditions": {},
    }

    for spec in conds:
        vp, tp = apply_perturbation(visual_c, text_c, spec)
        galleries_p = build_galleries(ctx, vp, tp)
        ret = eval_retrieval_for_condition(ctx, galleries_p, visual_c, text_c, vp, tp)
        pair = eval_pairs_for_condition(ctx, galleries_p, visual_c, text_c, vp, tp)

        cond_out = {"id": spec["id"], "label": spec["label"], "kind": spec["kind"], "methods": {}}
        for method in method_list():
            rvals = [v for (_, v) in ret[method]]
            p5 = float(np.mean(rvals)) if rvals else 0.0
            auc = float(ef.auc_score(pair[method], test_labels)) if len(pair[method]) else 0.0
            # retention
            ret_retention = (p5 / clean_ret_mean[method]) if clean_ret_mean[method] > 0 else 0.0
            auc_retention = (auc / clean_pair_auc[method]) if clean_pair_auc[method] > 0 else 0.0
            # bootstrap CI for p5
            m, lo, hi = bootstrap_ci(rvals, n_boot=N_BOOT, seed=SEED) if rvals else (0.0, 0.0, 0.0)
            # bootstrap CI for retention (ratio) over queries
            pos = None
            boots_ret = []
            rng = random.Random(SEED)
            nq = len(rvals)
            for _ in range(N_BOOT):
                idx = [rng.randrange(nq) for _ in range(nq)]
                rp = float(np.mean([rvals[i] for i in idx]))
                # clean values aligned
                cvals = [dict(clean_ret[method])[qis[i]] for i in range(nq)] if False else None
            # simpler: retention CI over queries using paired clean/perturbed p5 per query
            clean_vals_list = [dict(clean_ret[method])[qi] for qi in qis]
            pert_vals_list = [v for (_, v) in ret[method]]
            boots_ratio = []
            rng2 = random.Random(SEED + 1)
            for _ in range(N_BOOT):
                idx = [rng2.randrange(nq) for _ in range(nq)]
                pm = float(np.mean([pert_vals_list[i] for i in idx]))
                cm = float(np.mean([clean_vals_list[i] for i in idx]))
                boots_ratio.append((pm / cm) if cm > 0 else 0.0)
            boots_ratio.sort()
            rlo = boots_ratio[int(0.025 * N_BOOT)]
            rhi = boots_ratio[int(0.975 * N_BOOT) - 1]
            cond_out["methods"][method] = {
                "p5": p5, "p5_ci": [float(lo), float(hi)],
                "auc": auc,
                "retention_p5": float(ret_retention),
                "retention_p5_ci": [float(rlo), float(rhi)],
                "retention_auc": float(auc_retention),
                "n_queries": len(rvals),
            }
        results["conditions"][spec["id"]] = cond_out

    # MD
    md = []
    md.append("# Descriptor stress — person fusion (released subset)\n")
    md.append("**Scope.** released single-camera subset; gallery-side degradation only.\n")
    md.append("## Pre-registration\nFrozen grid embedded in JSON.\n")
    md.append("## Retrieval P@5 retention (perturbed/clean)\n")
    md.append("| condition | " + " | ".join(method_list()) + " |")
    md.append("|---|" + "---|" * len(method_list()))
    for spec in conds:
        cid = spec["id"]
        row = results["conditions"][cid]
        vals = [f"{row['methods'][m]['retention_p5']:.3f}" for m in method_list()]
        md.append(f"| {cid} | " + " | ".join(vals) + " |")
    md.append("")
    md.append("## Pair AUC (perturbed gallery)\n")
    md.append("| condition | " + " | ".join(method_list()) + " |")
    md.append("|---|" + "---|" * len(method_list()))
    for spec in conds:
        cid = spec["id"]
        row = results["conditions"][cid]
        vals = [f"{row['methods'][m]['auc']:.3f}" for m in method_list()]
        md.append(f"| {cid} | " + " | ".join(vals) + " |")
    md.append("")

    write_json(RESULTS / "regime_stress.json", results, overwrite)
    write_md(RESULTS / "regime_stress.md", "\n".join(md), overwrite)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["regimes", "stress", "all"], default="all")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting existing results/regime_* files")
    args = ap.parse_args()

    ctx = setup_base()
    if args.mode in ("regimes", "all"):
        run_regimes(ctx, args.overwrite)
        print("regimes done")
    if args.mode in ("stress", "all"):
        run_stress(ctx, args.overwrite)
        print("stress done")


if __name__ == "__main__":
    main()
