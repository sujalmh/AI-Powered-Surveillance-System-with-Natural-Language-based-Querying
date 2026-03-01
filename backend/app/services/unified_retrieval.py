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
        should_run_semantic = bool(
            settings.ENABLE_SEMANTIC
            and semantic_query
            and (not is_zero_count or has_action)
        )
        if should_run_semantic:
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
            },
            "processing_steps": steps,
        }

    # ------------------------------------------------------------------
    # Semantic query executor
    # ------------------------------------------------------------------

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

        # Query expansion
        expanded_queries: Optional[List[str]] = None
        if getattr(settings, "ENABLE_CLIP_EXPANSION", True):
            expanded_queries = self._build_expanded_queries(semantic_query, parsed_filter)

        try:
            result = search_unstructured(
                query=semantic_query,
                top_k=limit,
                camera_id=camera_id,
                from_iso=from_iso,
                to_iso=to_iso,
                min_confidence=min_confidence,
                has_action=has_action,
                expanded_queries=expanded_queries,
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
            if rc.get("clip_url"):
                enriched.append(rc)
                continue
            try:
                cam = rc.get("camera_id")

                # Derive tightest possible window from matched_timestamps so
                # the clip starts at the first real detection and ends at the
                # last – then pad with ±buffer_sec to avoid cutting the event.
                matched_ts = rc.get("matched_timestamps")
                if matched_ts and len(matched_ts) > 0:
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
        effective_limit: int = 10,
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
            },
            "processing_steps": steps,
        }
