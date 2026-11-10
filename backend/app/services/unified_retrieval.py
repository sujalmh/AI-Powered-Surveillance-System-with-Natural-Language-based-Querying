"""
Unified hybrid retrieval orchestrator.

Coordinates:
1. Informational queries (alerts, cameras) via ``query_handlers``
2. Structured MongoDB search via ``structured_search``
3. Semantic FAISS vector search via ``sem_search``
4. Result merging, scoring, diversity via ``result_merger``
5. Clip building + enrichment

This module is intentionally kept thin -- all heavy lifting lives in
dedicated helper modules.
"""
from __future__ import annotations

from loguru import logger
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib

from backend.app.config import settings
from backend.app.db.mongo import (
    cameras as cameras_col,
    vlm_frames as vlm_frames_col,
)
from backend.app.services.answer_generator import AnswerGenerator
from backend.app.services.clip_builder import build_clip_from_snapshots
from backend.app.services.query_handlers import execute_alert_query, execute_camera_query
from backend.app.services.result_merger import merge_results
from backend.app.services.retrieval_utils import normalize_count_constraint, parse_ts
from backend.app.services.sem_search import search_unstructured
from backend.app.services.structured_search import execute_structured_query


_PERSON_OBJECT_ALIASES = {"person", "people", "persons", "pedestrian", "human"}
_PERSON_TEXT_HINTS = (
    "person",
    "people",
    "pedestrian",
    "man ",
    "woman ",
    "wearing",
    "jacket",
    "shirt",
    "clothing",
    "coat",
    "trousers",
    "pants",
    "backpack",
    "bag",
    "hat",
    "uniform",
    "male",
    "female",
)
_NON_PERSON_OBJECTS = {
    "car", "cars", "vehicle", "vehicles", "truck", "bus", "bicycle",
    "motorcycle", "dog", "cat",
}


def should_route_person(parsed_filter: Dict[str, Any]) -> bool:
    """Return True when a query belongs on the 1152-d person index.

    Routing is deliberately conservative: explicit person objects, or an
    attribute/clothing-style text with no competing object, and no action or
    count constraint (those require the scene/structured branches).
    """
    if not getattr(settings, "ENABLE_PERSON_ROUTING", True):
        return False
    if parsed_filter.get("action") is not None:
        return False
    if normalize_count_constraint(parsed_filter.get("count_constraint")):
        return False
    if str(parsed_filter.get("query_type", "visual")).lower() != "visual":
        return False

    obj = str(parsed_filter.get("objects.object_name") or "").strip().lower()
    if obj in _PERSON_OBJECT_ALIASES:
        return True
    if obj in _NON_PERSON_OBJECTS:
        return False

    text = " ".join(
        str(parsed_filter.get(key) or "")
        for key in ("__semantic_query", "__raw", "__embedding_text")
    ).lower()
    return any(hint in text for hint in _PERSON_TEXT_HINTS)


# ======================================================================
# Processing-step tracker (used for tracing in API responses)
# ======================================================================

def _step(steps: list, name: str, status: str, details: str) -> None:
    steps.append({
        "name": name,
        "status": status,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ======================================================================
# Orchestrator
# ======================================================================

class UnifiedRetrieval:
    """
    Unified hybrid retrieval engine that combines:
    1. Structured MongoDB filtering (time, camera, object type, color)
    2. Semantic FAISS vector search (CLIP embeddings)

    Executes both, intelligently merges results, and generates an LLM answer.
    """

    def __init__(self) -> None:
        self.max_gap_seconds: float = 3.0
        self.join_gap_seconds: float = 10.0
        self._person_engine: Any = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def search(
        self,
        parsed_filter: Dict[str, Any],
        semantic_query: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute unified hybrid search.

        Returns ``{results, answer, metadata, processing_steps}``.
        """
        steps: list = []

        # --- Extract intent / flags -------------------------------------
        query_type = parsed_filter.get("query_type", "visual")
        intent = parsed_filter.get("__intent", "find")
        query_subtype = (parsed_filter.get("__query_subtype") or "").strip().lower() or None
        has_action = parsed_filter.get("action") is not None
        count_constraint = normalize_count_constraint(parsed_filter.get("count_constraint"))
        is_zero_count = bool(count_constraint and count_constraint.get("eq") == 0)

        # Effective result limit (respect NL hints like "only 1 clip")
        effective_limit = max(1, int(limit))
        try:
            rl = parsed_filter.get("__result_limit")
            if rl is not None and int(rl) > 0:
                effective_limit = min(effective_limit, int(rl))
        except Exception:
            pass
        requested_limit = parsed_filter.get("__result_limit")

        # --- Informational short-circuits --------------------------------
        if query_subtype == "alerts":
            return self._handle_informational(
                "alerts", parsed_filter, intent, limit, steps,
            )
        if query_subtype == "cameras":
            return self._handle_informational(
                "cameras", parsed_filter, intent, limit, steps,
            )

        # --- Step 1: Structured MongoDB search ---------------------------
        structured_results: List[Dict[str, Any]] = []
        if has_action:
            _step(steps, "Parse Query", "complete", f"Action query: {parsed_filter.get('action')}")
            _step(steps, "MongoDB Query", "complete", "Skipped (action query uses semantic only)")
            logger.opt(exception=True).info("Skipping structured query for action '{}'", parsed_filter.get("action"))
        else:
            _step(steps, "Parse Query", "complete", f"Extracted filters (intent: {intent})")
            _step(steps, "MongoDB Query", "in-progress", "Searching detections...")
            structured_results = execute_structured_query(parsed_filter, limit)
            steps[-1]["status"] = "complete"
            steps[-1]["details"] = f"{len(structured_results)} matches"
            logger.info("Structured query returned {} results", len(structured_results))

        # --- Step 2: Semantic FAISS search --------------------------------
        semantic_results: List[Dict[str, Any]] = []
        retrieval_branch = "scene"
        person_fallback: Optional[str] = None
        should_run_semantic = bool(
            settings.ENABLE_SEMANTIC
            and semantic_query
            and (not is_zero_count or has_action)
        )
        if should_run_semantic:
            if should_route_person(parsed_filter):
                _step(steps, "Vector Search", "in-progress", "Searching person index...")
                person_results, person_reason = self._execute_person_query(
                    semantic_query, parsed_filter, limit,
                )
                if person_results:
                    semantic_results = person_results
                    retrieval_branch = "person"
                    steps[-1]["status"] = "complete"
                    steps[-1]["details"] = (
                        f"{len(person_results)} person-index matches "
                        "(mapped to scene clips)"
                    )
                    logger.info(
                        "Person-index search returned {} clip matches",
                        len(person_results),
                    )
                else:
                    # Documented fallback: missing/empty index or no matches.
                    # The scene OpenCLIP branch is always available.
                    person_fallback = person_reason or "no person-index matches"
                    semantic_results = self._execute_semantic_query(
                        semantic_query, parsed_filter, limit,
                    )
                    retrieval_branch = "scene"
                    steps[-1]["status"] = "complete"
                    steps[-1]["details"] = (
                        f"{len(semantic_results)} scene matches "
                        f"(person fallback: {person_fallback})"
                    )
                    logger.info(
                        "Person routing fell back to scene index ({})",
                        person_fallback,
                    )
            else:
                _step(steps, "Vector Search", "in-progress", "Searching embeddings...")
                semantic_results = self._execute_semantic_query(
                    semantic_query, parsed_filter, limit,
                )
                steps[-1]["status"] = "complete"
                steps[-1]["details"] = f"{len(semantic_results)} semantic matches"
                logger.info("Semantic search returned {} results", len(semantic_results))
        else:
            _step(steps, "Vector Search", "complete", "Skipped")

        # Action queries REQUIRE semantic results
        if has_action and not semantic_results:
            _step(steps, "Return Results", "complete", "No matches for action query")
            logger.warning("Action query '{}' found no semantic matches.", parsed_filter.get("action"))
            return self._build_response(
                [], parsed_filter, intent, query_type, steps,
                structured_count=len(structured_results),
                requested_limit=requested_limit,
                effective_limit=effective_limit,
                retrieval_branch=retrieval_branch,
                person_fallback=person_fallback,
            )

        # --- Step 3: Merge & rank ----------------------------------------
        _step(steps, "Fusion & Ranking", "in-progress", "Combining results...")
        merged = merge_results(
            structured_results,
            semantic_results,
            parsed_filter,
            intent,
            self.max_gap_seconds,
            self.join_gap_seconds,
        )
        # Semantic-only entries carry captions but no object list.  Attach the
        # detections that fall inside the clip window so the UI can show what
        # was actually seen (and so colour/object filters remain auditable).
        self._attach_objects_to_semantic(merged)
        steps[-1]["status"] = "complete"
        steps[-1]["details"] = f"{len(merged)} unified results"
        logger.info("Merged results: {}", len(merged))

        # --- Step 4: Clip generation (conditional) -----------------------
        clips: List[Dict[str, Any]] = []
        if is_zero_count:
            _step(steps, "Build Clips", "complete", "Skipped (zero-count informational)")
            query_type = "informational"
        elif query_type == "visual" and merged:
            _step(steps, "Build Clips", "in-progress", "Creating video clips...")
            # Exclude merged segments that carry no matched detections
            # (zero-object results would produce empty / misleading clips).
            MAX_RESULTS = 10
            valid_merged = [
                r for r in merged
                if r.get("source") == "semantic"
                or (r.get("matched_object_count") or len(r.get("objects") or [])) > 0
            ][:MAX_RESULTS]
            logger.info(
                "Clip building: %d valid results (from %d merged, capped at %d)",
                len(valid_merged), len(merged), MAX_RESULTS,
            )
            clips = self._build_clips(valid_merged[:effective_limit], parsed_filter)
            steps[-1]["status"] = "complete"
            steps[-1]["details"] = f"Generated {len(clips)} clips"
            logger.info("Generated {} clips", len(clips))
        else:
            _step(steps, "Build Clips", "complete", "Skipped")

        # --- Step 5: LLM answer -------------------------------------------
        _step(steps, "Return Results", "complete", f"{len(clips)} results")
        return self._build_response(
            clips if clips else merged[:10],
            parsed_filter, intent, query_type, steps,
            clips=clips,
            structured_count=len(structured_results),
            semantic_count=len(semantic_results),
            merged_count=len(merged),
            is_zero_count=is_zero_count,
            requested_limit=requested_limit,
            effective_limit=effective_limit,
            retrieval_branch=retrieval_branch,
            person_fallback=person_fallback,
        )

    # ------------------------------------------------------------------
    # Informational queries (alerts / cameras)
    # ------------------------------------------------------------------

    def _handle_informational(
        self,
        subtype: str,
        parsed_filter: Dict[str, Any],
        intent: str,
        limit: int,
        steps: list,
    ) -> Dict[str, Any]:
        handler = execute_alert_query if subtype == "alerts" else execute_camera_query
        _step(steps, "Parse Query", "complete", f"Detected {subtype} query (intent: {intent})")
        _step(steps, "MongoDB Query", "complete", f"Querying {subtype}")
        logger.info("Handling query as {} informational query", subtype.upper())

        results = handler(parsed_filter, limit)
        _step(steps, "Return Results", "complete", f"Found {len(results)} {subtype}")

        answer = AnswerGenerator().generate(
            query=parsed_filter.get("__raw", ""),
            query_type="informational",
            results=results,
            parsed_filter=parsed_filter,
            metadata={
                "intent": intent,
                "query_subtype": subtype,
                "structured_count": len(results),
                "semantic_count": 0,
                "final_count": len(results),
            },
        )
        logger.info("Final {} result count: {}", subtype.upper(), len(results))
        return {
            "results": [],
            "answer": answer,
            "metadata": {
                "query_type": "informational",
                "intent": intent,
                "query_subtype": subtype,
                "structured_count": len(results),
                "semantic_count": 0,
                "final_count": 0,
                "info_data": results,
                "retrieval_branch": "none",
            },
            "processing_steps": steps,
        }

    # ------------------------------------------------------------------
    # Person / semantic query executors
    # ------------------------------------------------------------------

    def _execute_person_query(
        self,
        semantic_query: str,
        parsed_filter: Dict[str, Any],
        limit: int,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Search the person index, mapping hits to scene clips.

        Returns ``(results, reason)``.  A non-empty reason on an empty result
        means the caller should use the scene fallback; no Mongo connection is
        required for this path.
        """
        try:
            from backend.app.services.hybrid_search import HybridSearchEngine

            if not HybridSearchEngine.person_index_available():
                return [], "person index unavailable"
            if self._person_engine is None:
                self._person_engine = HybridSearchEngine()
            results = self._person_engine.search_meta(
                semantic_query,
                top_k=max(1, int(limit)),
                parsed_filter=parsed_filter,
            )
            if not results:
                return [], "no person-index matches"
            return results, "ok"
        except Exception as exc:
            logger.opt(exception=True).warning(
                "Person-index search failed; using scene fallback"
            )
            return [], f"person search error ({type(exc).__name__})"

    @staticmethod
    def _attach_objects_to_semantic(results: List[Dict[str, Any]], max_docs: int = 20) -> None:
        """Populate ``objects`` on semantic-only results from nearby detections.

        Scene matches are caption-driven and carry no object list; without
        this the UI shows a clip with no detected objects and object/colour
        filters cannot be audited against the result.
        """
        try:
            from backend.app.db.mongo import detections as detections_col
            from backend.app.services.retrieval_utils import aggregate_objects
        except Exception:  # pragma: no cover - import guard
            return

        for r in results:
            if r.get("source") != "semantic" or r.get("objects"):
                continue
            start, end = r.get("start"), r.get("end")
            cam = r.get("camera_id")
            if not start or not end or cam is None:
                continue
            try:
                docs = list(
                    detections_col.find(
                        {"camera_id": cam, "timestamp": {"$gte": str(start), "$lte": str(end)}},
                        {"_id": 0, "objects": 1},
                    ).limit(max_docs)
                )
            except Exception:
                continue
            objs = [o for d in docs for o in (d.get("objects") or [])]
            if objs:
                r["objects"] = aggregate_objects(objs)

    def _execute_semantic_query(
        self,
        semantic_query: str,
        parsed_filter: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        camera_id: Optional[int] = (
            parsed_filter["camera_id"]
            if isinstance(parsed_filter.get("camera_id"), int)
            else None
        )
        from_iso = to_iso = None
        ts_filter = parsed_filter.get("timestamp", {})
        if isinstance(ts_filter, dict):
            from_iso = ts_filter.get("$gte")
            to_iso = ts_filter.get("$lte")

        has_action = parsed_filter.get("action") is not None
        min_confidence = 0.20 if has_action else 0.25

        # Use the CLIP-optimized embedding text when available.
        # __embedding_text is a short visual description (e.g. "car",
        # "person wearing red clothing") whereas semantic_query is often
        # verbose LLM output that CLIP struggles with.
        embedding_text = parsed_filter.get("__embedding_text")
        if isinstance(embedding_text, str) and embedding_text:
            clip_query = embedding_text
        else:
            clip_query = semantic_query or ""
        
        # Safety: truncate verbose queries — CLIP retrieval degrades above ~8 words
        clip_words = clip_query.split()
        semantic_query_hash = hashlib.sha256((semantic_query or "").encode("utf-8")).hexdigest()[:12]
        if len(clip_words) > 8:
            logger.warning(
                "CLIP query too verbose (words={}) - truncating to 8 words",
                len(clip_words),
            )
            logger.debug("Original verbose CLIP query text before truncation: {}", clip_query)
            clip_query = " ".join(clip_words[:8])
        
        # Update hash after truncation so it reflects final query
        clip_query_hash = hashlib.sha256((clip_query or "").encode("utf-8")).hexdigest()[:12]

        logger.info(
            "Semantic CLIP query metadata clip_words={} semantic_words={} clip_hash={} semantic_hash={}",
            len(clip_query.split()),
            len((semantic_query or "").split()),
            clip_query_hash,
            semantic_query_hash,
        )
        logger.debug("Semantic CLIP query text clip_query='{}' semantic_query='{}'", clip_query, semantic_query)

        # Query expansion
        expanded_queries: Optional[List[str]] = None
        if getattr(settings, "ENABLE_CLIP_EXPANSION", True):
            expanded_queries = self._build_expanded_queries(clip_query, parsed_filter)

        try:
            result = search_unstructured(
                query=clip_query,
                top_k=limit,
                camera_id=camera_id,
                from_iso=from_iso,
                to_iso=to_iso,
                min_confidence=min_confidence,
                has_action=has_action,
                expanded_queries=expanded_queries,
                requested_object=parsed_filter.get("objects.object_name"),
            )
            return result.get("semantic_results", [])
        except Exception:
            logger.error("Semantic query error")
            return []

    @staticmethod
    def _build_expanded_queries(
        semantic_query: str,
        parsed_filter: Dict[str, Any],
    ) -> Optional[List[str]]:
        terms = parsed_filter.get("__expanded_terms") or {}
        variants = [semantic_query]
        for field in ("objects", "action"):
            items = terms.get(field)
            if items and len(items) > 1:
                try:
                    v = semantic_query.replace(items[0], items[1], 1)
                    if v != semantic_query and v not in variants:
                        variants.append(v)
                except Exception:
                    pass
        return variants if len(variants) > 1 else None

    # ------------------------------------------------------------------
    # Dense-window timestamp trimming
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to_dense_window(
        timestamps: List[str],
        min_gap_abs: float = 5.0,
        gap_factor: float = 4.0,
    ) -> Tuple[str, str]:
        """
        Given a list of ISO timestamp strings, find the densest contiguous
        cluster and return ``(start, end)`` covering only that cluster.

        Filler at the beginning or end of a clip is caused by outlier
        timestamps stretching the window far beyond the core detection
        activity.  This helper identifies the largest internal gap that
        exceeds both *min_gap_abs* and *gap_factor* × median-gap, then
        keeps the cluster that contains more timestamps.
        """
        if not timestamps:
            raise ValueError("Empty timestamp list")

        parsed_dts = [parse_ts(t) for t in timestamps]
        dts = sorted(dt for dt in parsed_dts if dt != datetime.min)
        if not dts:
            # Fallback to original string bounds if all timestamps are unparseable.
            sorted_ts = sorted(timestamps)
            return sorted_ts[0], sorted_ts[-1]
        if len(dts) == 1:
            return dts[0].isoformat(), dts[0].isoformat()
        if len(dts) == 2:
            return dts[0].isoformat(), dts[-1].isoformat()

        gaps = [(dts[i + 1] - dts[i]).total_seconds() for i in range(len(dts) - 1)]
        nonzero = [g for g in gaps if g > 0]
        if not nonzero:
            return dts[0].isoformat(), dts[-1].isoformat()

        median_g = float(sorted(nonzero)[len(nonzero) // 2])
        threshold = max(min_gap_abs, median_g * gap_factor)

        # Iteratively trim: find the largest gap that exceeds threshold,
        # keep the larger cluster, repeat until no more outlier gaps.
        while len(dts) > 2:
            gaps = [(dts[i + 1] - dts[i]).total_seconds() for i in range(len(dts) - 1)]
            split_idx = -1
            max_gap_val = 0.0
            for i, g in enumerate(gaps):
                if g > threshold and g > max_gap_val:
                    max_gap_val = g
                    split_idx = i
            if split_idx < 0:
                break
            left = dts[: split_idx + 1]
            right = dts[split_idx + 1:]
            dts = left if len(left) >= len(right) else right

        return dts[0].isoformat(), dts[-1].isoformat()

    # ------------------------------------------------------------------
    # Clip building + enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_buffer_seconds(parsed_filter: Dict[str, Any]) -> float:
        """
        Determine clip buffer (seconds added before/after the event window)
        based on query type so clips are tightly cropped around relevant content.

        - Object/person queries: ±1s (minimal context, precision needed)
        - Action queries: ±2s (needs motion context before/after)
        - Count queries: ±1s (precise counting windows)
        - General / fallback: ±2s (balanced)
        """
        has_action = parsed_filter.get("action") is not None
        has_count = parsed_filter.get("count_constraint") is not None
        has_object = parsed_filter.get("objects.object_name") is not None

        if has_action:
            return 2.0
        if has_count:
            return 1.0
        if has_object:
            return 1.0
        return 2.0

    def _build_clips(
        self,
        results: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        buffer_sec = self._compute_buffer_seconds(parsed_filter)
        logger.debug("Clip buffer: ±%.1fs (action=%s, count=%s, object=%s)",
                      buffer_sec,
                      parsed_filter.get("action") is not None,
                      parsed_filter.get("count_constraint") is not None,
                      parsed_filter.get("objects.object_name") is not None)

        # Hard cap: never build more than MAX_RESULTS clips
        MAX_RESULTS = 10
        results = results[:MAX_RESULTS]

        enriched: List[Dict[str, Any]] = []
        for result in results:
            rc = dict(result)

            # --- Re-trim semantic-only clips that have frame timestamps ---
            # Pre-indexed clips (with clip_url) may be full-length recording
            # chunks.  If we have frame-level timestamps, compute the dense
            # window and re-build a tighter clip when it saves >40% duration.
            if rc.get("clip_url") and rc.get("frames"):
                try:
                    frame_ts = [
                        f["frame_ts"]
                        for f in rc["frames"]
                        if f.get("frame_ts")
                    ]
                    if len(frame_ts) >= 2:
                        dense_start, dense_end = self._trim_to_dense_window(frame_ts)
                        ds = parse_ts(dense_start)
                        de = parse_ts(dense_end)
                        dense_dur = (de - ds).total_seconds()
                        duration_raw = rc.get("duration_seconds")
                        try:
                            full_dur = float(duration_raw) if duration_raw else 0.0
                        except Exception:
                            full_dur = 0.0
                        if full_dur > 0 and dense_dur < full_dur * 0.6:
                            # Worthwhile to re-build a trimmer clip
                            cam = rc.get("camera_id")
                            if cam is not None:
                                s_dt = ds - timedelta(seconds=buffer_sec)
                                e_dt = de + timedelta(seconds=buffer_sec)
                                clip = build_clip_from_snapshots(
                                    int(cam),
                                    s_dt.isoformat(),
                                    e_dt.isoformat(),
                                    fps=5.0,
                                )
                                rc["clip_url"] = clip.url
                                rc["clip_frames"] = clip.frame_count
                                rc["clip_path"] = clip.path
                                rc["start"] = dense_start
                                rc["end"] = dense_end
                                logger.debug(
                                    "Re-trimmed semantic clip cam=%s: %.1fs → %.1fs",
                                    cam, full_dur, dense_dur + 2 * buffer_sec,
                                )
                        elif full_dur <= 0:
                            logger.debug(
                                "Skipping semantic clip re-trim decision due to missing/invalid duration_seconds"
                            )
                except Exception as e:
                    logger.debug("Semantic clip re-trim skipped: %s", e)

            if rc.get("clip_url"):
                enriched.append(rc)
                continue
            try:
                cam = rc.get("camera_id")

                # Derive tightest possible window from matched_timestamps.
                # Use _trim_to_dense_window to discard outlier timestamps
                # that would stretch the clip with irrelevant filler content.
                matched_ts = rc.get("matched_timestamps")
                if matched_ts and len(matched_ts) >= 3:
                    try:
                        start, end = self._trim_to_dense_window(matched_ts)
                    except Exception:
                        sorted_ts = sorted(matched_ts)
                        start, end = sorted_ts[0], sorted_ts[-1]
                elif matched_ts and len(matched_ts) > 0:
                    sorted_ts = sorted(matched_ts)
                    start = sorted_ts[0]
                    end = sorted_ts[-1]
                else:
                    start = rc.get("start")
                    end = rc.get("end")

                if cam is not None and start and end:
                    allowed_ts = rc.get("matched_timestamps")
                    try:
                        s_dt = parse_ts(start) - timedelta(seconds=buffer_sec)
                        e_dt = parse_ts(end) + timedelta(seconds=buffer_sec)
                        start_iso, end_iso = s_dt.isoformat(), e_dt.isoformat()
                    except Exception:
                        start_iso, end_iso = start, end
                    clip = build_clip_from_snapshots(
                        int(cam), start_iso, end_iso,
                        fps=5.0, allowed_timestamps=allowed_ts,
                    )
                    rc["clip_url"] = clip.url
                    rc["clip_frames"] = clip.frame_count
                    rc["clip_path"] = clip.path
            except Exception as e:
                rc["clip_error"] = str(e)
            enriched.append(rc)

        enriched = self._enrich_with_camera_metadata(enriched)
        enriched = self._enrich_with_vlm_frames(enriched, parsed_filter)
        return enriched

    @staticmethod
    def _enrich_with_camera_metadata(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not results:
            return results
        camera_ids = list({r.get("camera_id") for r in results if r.get("camera_id") is not None})
        if not camera_ids:
            return results
        try:
            cameras = list(cameras_col.find(
                {"camera_id": {"$in": camera_ids}},
                {"camera_id": 1, "location": 1, "status": 1, "_id": 0},
            ))
            cam_map = {c["camera_id"]: c for c in cameras}
            for r in results:
                info = cam_map.get(r.get("camera_id"))
                if info:
                    r["location"] = info.get("location", "Unknown")
                    r["camera_status"] = info.get("status", "unknown")
            logger.opt(exception=True).debug("Enriched {} results with camera metadata", len(results))
        except Exception:
            logger.error("Error enriching camera metadata")
        return results

    @staticmethod
    def _enrich_with_vlm_frames(
        results: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not settings.ENABLE_SEMANTIC:
            return results
        if parsed_filter.get("query_type", "visual") != "visual":
            return results
        if results and results[0].get("frames"):
            return results
        try:
            clip_keys = [
                (r.get("camera_id"), r.get("clip_path"))
                for r in results
                if r.get("clip_path") and r.get("camera_id") is not None
            ]
            if not clip_keys:
                return results
            unique_keys = list(dict.fromkeys(clip_keys))
            or_conditions = [{"camera_id": cam, "clip_path": cp} for cam, cp in unique_keys]
            vlm_docs = list(vlm_frames_col.find(
                {"$or": or_conditions},
                {"_id": 0, "camera_id": 1, "clip_path": 1, "caption": 1,
                 "object_captions": 1, "frame_index": 1, "frame_ts": 1},
            ).limit(100))
            by_key: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
            for d in vlm_docs:
                k = (d.get("camera_id"), d.get("clip_path"))
                if len(by_key[k]) < 10:
                    by_key[k].append(d)
            for r in results:
                docs = by_key.get((r.get("camera_id"), r.get("clip_path")), [])
                if docs:
                    r["vlm_frames"] = docs
                    if docs[0].get("caption"):
                        r["scene_caption"] = docs[0]["caption"]
            enriched_count = sum(1 for r in results if r.get("vlm_frames"))
            if enriched_count:
                logger.opt(exception=True).debug("Enriched {} results with VLM frame data", enriched_count)
        except Exception:
            logger.error("Error enriching VLM frames")
        return results

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response(
        answer_context: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any],
        intent: str,
        query_type: str,
        steps: list,
        *,
        clips: Optional[List[Dict[str, Any]]] = None,
        structured_count: int = 0,
        semantic_count: int = 0,
        merged_count: int = 0,
        is_zero_count: bool = False,
        requested_limit: Any = None,
        effective_limit: int = 50,
        retrieval_branch: str = "scene",
        person_fallback: Optional[str] = None,
    ) -> Dict[str, Any]:
        final_clips = clips if clips else []
        answer = AnswerGenerator().generate(
            query=parsed_filter.get("__raw", ""),
            query_type=query_type,
            results=answer_context,
            parsed_filter=parsed_filter,
            metadata={
                "intent": intent,
                "is_zero_count": is_zero_count,
                "structured_count": structured_count,
                "semantic_count": semantic_count,
                "final_count": len(final_clips) if final_clips else merged_count,
            },
        )
        return {
            "results": final_clips,
            "answer": answer,
            "metadata": {
                "query_type": query_type,
                "intent": intent,
                "requested_limit": requested_limit,
                "effective_limit": effective_limit,
                "structured_count": structured_count,
                "semantic_count": semantic_count,
                "final_count": len(final_clips),
                "retrieval_branch": retrieval_branch,
                "person_fallback": person_fallback,
            },
            "processing_steps": steps,
        }
