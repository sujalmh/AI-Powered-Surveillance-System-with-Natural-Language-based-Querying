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


def search_unstructured(
    query: str,
    top_k: int = 50,
    camera_id: Optional[int] = None,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    min_confidence: float = 0.25,
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
    """
    if not settings.ENABLE_SEMANTIC:
        return {"mode": "unstructured", "semantic_results": []}

    emb = get_embedder().text_embed([query])
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

    hits = store.vector_search(emb[0], top_k=top_k, filter_pred=_filter_pred)

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

    # Aggregate clip scores: max over frames and keep top few frames
    results: List[Dict[str, Any]] = []
    for clip, entry in by_clip.items():
        frames = sorted(entry["frames"], key=lambda x: float(x.get("score") or 0.0), reverse=True)
        max_score = float(frames[0]["score"]) if frames else 0.0
        
        # Filter by minimum confidence threshold
        if max_score < min_confidence:
            continue
            
        entry["score"] = max_score
        entry["frames"] = frames[:5]
        results.append(entry)

    # Normalize scores for consistent merging later
    normed = _norm_scores([r["score"] for r in results])
    for r, s in zip(results, normed):
        r["score_norm"] = s

    # Sort by normalized score desc
    results.sort(key=lambda x: x.get("score_norm", 0.0), reverse=True)
    
    print(f"[SemanticSearch] Filtered results: {len(results)} clips with confidence >= {min_confidence}")
    
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
