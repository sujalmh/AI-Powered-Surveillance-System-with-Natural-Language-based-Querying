from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.config import settings
from backend.app.services.sem_embedder import get_embedder
from backend.app.services.sem_store import get_faiss_store


def _norm_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx - mn < 1e-9:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _adaptive_min_confidence(
    clip_max_scores: List[float],
    has_action: bool = False,
    base: float = 0.15,
) -> float:
    """
    Compute minimum confidence threshold from score distribution.
    Action queries get lower threshold; otherwise use percentile so we keep reasonable recall.
    """
    if not getattr(settings, "ENABLE_ADAPTIVE_CONFIDENCE", True):
        return 0.10 if has_action else base
    try:
        if has_action:
            return max(0.08, min(0.25, base - 0.05))
        if not clip_max_scores:
            return base
        arr = np.array(clip_max_scores, dtype=float)
        # Use 25th percentile so we keep top 75% of clips; floor 0.08, ceiling 0.35
        p25 = float(np.percentile(arr, 25))
        adaptive = max(0.08, min(0.35, p25))
        return adaptive
    except Exception:
        return 0.10 if has_action else base


def search_unstructured(
    query: str,
    top_k: int = 50,
    camera_id: Optional[int] = None,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    min_confidence: float = 0.15,
    has_action: bool = False,
    expanded_queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Semantic (vector-only) search using CLIP text embedding + FAISS.
    Returns per-clip aggregated results with representative frames.
    
    Args:
        query: Text query to search for
        top_k: Maximum number of results to return
        camera_id: Filter by specific camera
        from_iso: Start time filter (ISO format)
        to_iso: End time filter (ISO format)
        min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.25 filters out low-quality matches.
        expanded_queries: Optional list of query variants for embedding fusion (first = primary).
    """
    if not settings.ENABLE_SEMANTIC:
        return {"mode": "unstructured", "semantic_results": []}

    use_expansion = getattr(settings, "ENABLE_CLIP_EXPANSION", True) and expanded_queries and len(expanded_queries) > 1
    if use_expansion:
        try:
            embedder = get_embedder()
            embs = embedder.text_embed(expanded_queries)
            n = len(embs)
            w_primary = 0.6
            w_rest = (1.0 - w_primary) / max(1, n - 1)
            weights = [w_primary] + [w_rest] * (n - 1)
            combined = np.average(embs, axis=0, weights=weights).astype(np.float32)
            norm = np.linalg.norm(combined)
            if norm > 1e-12:
                combined = combined / norm
            query_emb = combined
        except Exception:
            query_emb = get_embedder().text_embed([query])[0]
    else:
        query_emb = get_embedder().text_embed([query])[0]
    store = get_faiss_store(dim=512)

    def _filter_pred(meta: Dict[str, Any]) -> bool:
        if camera_id is not None and int(meta.get("camera_id", -1)) != int(camera_id):
            return False
        # Time window filter if available
        try:
            ts = meta.get("frame_ts")
            if (from_iso or to_iso) and not ts:
                # Strict time-bounded queries should not accept rows without timestamps
                return False
            if ts and (from_iso or to_iso):
                dt = datetime.fromisoformat(ts)
                if from_iso and dt < datetime.fromisoformat(from_iso):
                    return False
                if to_iso and dt > datetime.fromisoformat(to_iso):
                    return False
        except Exception:
            pass
        return True

    # Over-fetch to compensate for post-filtering losses (camera/time filters)
    fetch_multiplier = 4 if (camera_id is not None or from_iso or to_iso) else 1
    hits = store.vector_search(query_emb, top_k=top_k * fetch_multiplier, filter_pred=_filter_pred)

    # Group by clip to form clip-level results
    by_clip: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        meta = h["meta"]
        clip = meta.get("clip_path") or ""
        if not clip:
            # Skip entries lacking clip metadata
            continue
        entry = by_clip.setdefault(
            clip,
            {
                "camera_id": meta.get("camera_id"),
                "clip_path": meta.get("clip_path"),
                "clip_url": meta.get("clip_url"),
                "score": 0.0,
                "frames": [],  # representative frames
            },
        )
        entry["frames"].append(
            {
                "frame_ts": meta.get("frame_ts"),
                "frame_index": meta.get("frame_index"),
                "score": h["score"],
                "caption": meta.get("caption"),
            }
        )

    # Aggregate clip scores: max over frames; then apply adaptive or fixed threshold
    clip_scores: List[float] = []
    for clip, entry in by_clip.items():
        frames = sorted(entry["frames"], key=lambda x: float(x.get("score") or 0.0), reverse=True)
        max_score = float(frames[0]["score"]) if frames else 0.0
        entry["score"] = max_score
        entry["frames"] = frames[:5]
        clip_scores.append(max_score)
    # Adaptive threshold from distribution (or use fixed min_confidence)
    if getattr(settings, "ENABLE_ADAPTIVE_CONFIDENCE", True) and clip_scores:
        threshold = _adaptive_min_confidence(clip_scores, has_action=has_action, base=min_confidence)
    else:
        threshold = 0.10 if has_action else min_confidence
    results = [entry for entry in by_clip.values() if entry["score"] >= threshold]

    # Normalize scores for consistent merging later
    normed = _norm_scores([r["score"] for r in results])
    for r, s in zip(results, normed):
        r["score_norm"] = s

    # Sort by normalized score desc
    results.sort(key=lambda x: x.get("score_norm", 0.0), reverse=True)
    
    print(f"[SemanticSearch] Filtered results: {len(results)} clips with confidence >= {threshold}")
    
    return {"mode": "unstructured", "semantic_results": results}


def combine_hybrid(
    structured: List[Dict[str, Any]],
    semantic: List[Dict[str, Any]],
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Combine structured clips and semantic clips via weighted normalized scoring.
    structured: list of dicts with at least {"clip_url"?, "score_struct" or fallbacks}
    semantic: output from search_unstructured()["semantic_results"]
    alpha: weight for structured; (1-alpha) for semantic.
    Returns merged ranked list.
    """
    # Build maps by clip_url or clip_path if clip_url missing
    s_map: Dict[str, Dict[str, Any]] = {}
    for s in structured:
        key = s.get("clip_url") or s.get("clip_path") or ""
        if not key:
            continue
        s_map[key] = s

    # Normalize structured scores
    s_scores = []
    for s in structured:
        sc = float(s.get("score_struct") or s.get("duration_seconds") or 0.0)
        s["_score_struct_raw"] = sc
        s_scores.append(sc)
    s_norm = _norm_scores(s_scores)
    for s, ns in zip(structured, s_norm):
        s["score_struct_norm"] = ns

    out_map: Dict[str, Dict[str, Any]] = {}
    # Seed with semantic entries
    for m in semantic:
        key = m.get("clip_url") or m.get("clip_path") or ""
        if not key:
            continue
        out_map[key] = {
            "clip_url": m.get("clip_url"),
            "clip_path": m.get("clip_path"),
            "camera_id": m.get("camera_id"),
            "score_sem_norm": float(m.get("score_norm") or 0.0),
            "score_struct_norm": 0.0,
            "frames": m.get("frames", []),
        }

    # Merge structured
    for s in structured:
        key = s.get("clip_url") or s.get("clip_path") or ""
        if not key:
            continue
        if key not in out_map:
            out_map[key] = {
                "clip_url": s.get("clip_url"),
                "clip_path": s.get("clip_path"),
                "camera_id": s.get("camera_id"),
                "score_sem_norm": 0.0,
                "score_struct_norm": float(s.get("score_struct_norm") or 0.0),
                "frames": [],
            }
        else:
            out_map[key]["score_struct_norm"] = float(s.get("score_struct_norm") or 0.0)

    # Final score
    merged: List[Dict[str, Any]] = []
    for v in out_map.values():
        fs = alpha * float(v.get("score_struct_norm") or 0.0) + (1.0 - alpha) * float(v.get("score_sem_norm") or 0.0)
        v["score"] = fs
        merged.append(v)

    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged
