from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from backend.app.db.mongo import (
    detections as detections_col,
    cameras as cameras_col,
    vlm_frames as vlm_frames_col
)
from backend.app.services.clip_builder import build_clip_from_snapshots
from backend.app.services.sem_search import search_unstructured
from backend.app.services.answer_generator import AnswerGenerator
from backend.app.config import settings
import logging
import re

logger = logging.getLogger(__name__)


class UnifiedRetrieval:
    """
    Unified hybrid retrieval engine that combines:
    1. Structured MongoDB filtering (time, camera, object type, color)
    2. Semantic FAISS vector search (CLIP embeddings)
    
    Executes both in parallel and intelligently merges results based on query intent.
    """
    
    def __init__(self):
        self.max_gap_seconds = 3  # Gap for merging contiguous track detections
        self.join_gap_seconds = 10  # Gap for coalescing across track IDs
    
    def search(
        self,
        parsed_filter: Dict[str, Any],
        semantic_query: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Execute unified hybrid search.
        
        Args:
            parsed_filter: Structured filters from NL parser (camera_id, timestamp, object filters, etc.)
            semantic_query: Natural language query for semantic search
            limit: Maximum results to return
            
        Returns:
            Dictionary containing:
            - results: Unified ranked list of video clips with metadata
            - answer: Natural language answer
            - metadata: Info about filters applied
        """
        # Extract query type and intent
        query_type = parsed_filter.get("query_type", "visual")
        intent = parsed_filter.get("__intent", "find")
        ask_color = parsed_filter.get("__ask_color", False)
        has_action = parsed_filter.get("action") is not None
        count_constraint_norm = self._normalize_count_constraint(parsed_filter.get("count_constraint"))
        has_count_constraint = count_constraint_norm is not None
        is_zero_count = bool(has_count_constraint and count_constraint_norm.get("eq") == 0)

        # Respect both API limit and optional NL result-limit hint (e.g., "only one clip")
        effective_limit = max(1, int(limit))
        try:
            requested_limit = int(parsed_filter.get("__result_limit")) if parsed_filter.get("__result_limit") is not None else None
            if requested_limit is not None and requested_limit > 0:
                effective_limit = min(effective_limit, requested_limit)
        except Exception:
            requested_limit = None
        
        # 1. Execute structured MongoDB query (SKIP for action queries)
        structured_results = []
        if has_action:
            # Action queries only use semantic search - structured search returns irrelevant results
            print(f"[UnifiedRetrieval] Step 1: Skipping structured query for action '{parsed_filter.get('action')}' (semantic-only)")
            logger.info(f"Skipping structured query for action '{parsed_filter.get('action')}' (semantic-only)")
        else:
            print(f"[UnifiedRetrieval] Step 1: Executing structured query. Intent: {intent}")
            logger.info(f"Step 1: Executing structured query. Intent: {intent}")
            structured_results = self._execute_structured_query(parsed_filter, limit)
            print(f"[UnifiedRetrieval] Structured query returned {len(structured_results)} results")
            logger.info(f"Structured query returned {len(structured_results)} results")
        
        # 2. Execute semantic search (ALWAYS if enabled and query provided)
        semantic_results = []
        # For zero-count queries (e.g. "no person"), semantic retrieval would introduce
        # false positives. For non-zero count queries, semantic can still help when structured
        # returns nothing.
        should_run_semantic = bool(settings.ENABLE_SEMANTIC and semantic_query and (not is_zero_count or has_action))
        if should_run_semantic:
            print(f"[UnifiedRetrieval] Step 2: Executing semantic search for: {semantic_query}")
            semantic_results = self._execute_semantic_query(
                semantic_query,
                parsed_filter,
                limit
            )
            print(f"[UnifiedRetrieval] Semantic search returned {len(semantic_results)} results")
            logger.info(f"Semantic search returned {len(semantic_results)} results")
        else:
            print("[UnifiedRetrieval] Skipping semantic search (disabled or no query)")
            logger.info("Skipping semantic search (disabled or no query)")
        
        # Check if this is an action query with no semantic results
        has_action = parsed_filter.get("action") is not None
        if has_action and not semantic_results:
            # Action queries REQUIRE semantic results - return empty if none found
            print(f"[UnifiedRetrieval] Action query '{parsed_filter.get('action')}' found no semantic matches. Returning empty results.")
            logger.warning(f"Action query '{parsed_filter.get('action')}' found no semantic matches.")
            
            answer_gen = AnswerGenerator()
            answer = answer_gen.generate(
                query=parsed_filter.get("__raw", ""),
                query_type=query_type,
                results=[],
                parsed_filter=parsed_filter,
                metadata={
                    "intent": intent,
                    "structured_count": len(structured_results),
                    "semantic_count": 0,
                    "final_count": 0,
                }
            )
            
            return {
                "results": [],
                "answer": answer,
                "metadata": {
                    "query_type": query_type,
                    "intent": intent,
                    "structured_count": len(structured_results),
                    "semantic_count": 0,
                    "final_count": 0,
                }
            }
        
        # 3. Merge and rank results based on intent
        print(f"[UnifiedRetrieval] Step 3: Merging results. Intent: {intent}")
        merged_results = self._merge_results(
            structured_results,
            semantic_results,
            parsed_filter,
            intent
        )
        print(f"[UnifiedRetrieval] Step 3: Merged results count: {len(merged_results)}")
        logger.info(f"Step 3: Merged results count: {len(merged_results)}")
        
        # 4. CONDITIONAL clip generation based on query_type
        # Zero-count queries ("no person") should NOT generate clips:
        # - Clips of empty scenes have no visual value
        # - The clip builder grabs ALL snapshots in a time range, including ones with persons
        # - A text answer ("No persons detected between X and Y") is the correct response
        clips = []
        if is_zero_count:
            # Treat as informational — return structured segments as metadata without clips
            print("[UnifiedRetrieval] Step 4: Skipping clip generation for zero-count query (informational answer)")
            logger.info("Step 4: Zero-count query — returning informational answer, no clips")
            query_type = "informational"  # Override for answer generator
            clips = []  # Ensure no clips
        elif query_type == "visual":
            # Visual query: build video clips for top results
            print("[UnifiedRetrieval] Step 4: Building video clips for visual query")
            logger.info("Step 4: Building video clips for visual query")
            clips = self._build_clips(merged_results[:effective_limit], parsed_filter)
            print(f"[UnifiedRetrieval] Generated {len(clips)} clips")
            logger.info(f"Generated {len(clips)} clips")
        else:
            # Informational query: NO clip generation, return merged results as-is
            print("[UnifiedRetrieval] Step 4: Skipping clip generation for informational query")
            logger.info("Step 4: Skipping clip generation for informational query")
            clips = []
        
        # 5. LLM-generated answer using AnswerGenerator
        answer_gen = AnswerGenerator()
        # For zero-count queries, pass merged segments as context so the LLM can describe
        # WHEN and WHERE no persons were found, rather than trying to describe clips
        answer_context = clips if clips else merged_results[:10]
        answer = answer_gen.generate(
            query=parsed_filter.get("__raw", ""),
            query_type=query_type,
            results=answer_context,
            parsed_filter=parsed_filter,
            metadata={
                "intent": intent,
                "is_zero_count": is_zero_count,
                "structured_count": len(structured_results),
                "semantic_count": len(semantic_results),
                "final_count": len(clips) if clips else len(merged_results),
            }
        )
        
        print(f"[UnifiedRetrieval] Final Result: {len(clips)} clips, Answer: {answer[:50]}...")
        return {
            "results": clips,
            "answer": answer,
            "metadata": {
                "query_type": query_type,
                "intent": intent,
                "requested_limit": requested_limit,
                "effective_limit": effective_limit,
                "structured_count": len(structured_results),
                "semantic_count": len(semantic_results),
                "final_count": len(clips),
            }
        }
    
    def _resolve_camera_from_location(self, location_hint: str) -> Optional[List[int]]:
        """
        Resolve camera IDs from location hint by querying cameras collection.
        
        Args:
            location_hint: Location name/hint from user query (e.g., "Gate 3", "Main Entrance")
            
        Returns:
            List of camera_ids matching the location, or None if no matches
        """
        try:
            # Case-insensitive regex search on location field
            cameras = list(cameras_col.find(
                {"location": {"$regex": location_hint, "$options": "i"}},
                {"camera_id": 1, "location": 1, "_id": 0}
            ))
            
            if cameras:
                camera_ids = [cam["camera_id"] for cam in cameras]
                locations = [cam["location"] for cam in cameras]
                print(f"[UnifiedRetrieval] Location '{location_hint}' resolved to cameras: {camera_ids} ({locations})")
                logger.info(f"Location '{location_hint}' resolved to cameras: {camera_ids} ({locations})")
                return camera_ids
            else:
                print(f"[UnifiedRetrieval] No cameras found for location: {location_hint}")
                logger.warning(f"No cameras found for location: {location_hint}")
                return None
        except Exception as e:
            print(f"[UnifiedRetrieval] Error resolving location: {e}")
            logger.error(f"Error resolving location: {e}")
            return None
    
    def _execute_structured_query(
        self,
        parsed_filter: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute MongoDB query with structured filters."""
        # Build MongoDB filter (remove metadata fields and control fields)
        # Exclude fields starting with __ and other metadata fields like query_type, action, zone, count_constraint
        metadata_fields = {"query_type", "action", "zone", "count_constraint"}
        mongo_filter = {
            k: v for k, v in parsed_filter.items()
            if not k.startswith("__") and k not in metadata_fields
        }

        count_constraint = self._normalize_count_constraint(parsed_filter.get("count_constraint"))
        is_zero_count_query = bool(count_constraint and count_constraint.get("eq") == 0)

        # For "no person" style queries, object-level Mongo filters would eliminate valid docs.
        # We broaden the DB candidate set and enforce the count constraint in Python.
        if is_zero_count_query:
            mongo_filter.pop("objects.object_name", None)
            mongo_filter.pop("objects.color", None)

        # Use person_count index for fast server-side pre-filtering (non-zero count queries)
        obj_name = parsed_filter.get("objects.object_name", "")
        obj_color = parsed_filter.get("objects.color")
        if count_constraint and obj_name and str(obj_name).lower() == "person" and not obj_color:
            pc_filter: Dict[str, Any] = {}
            if "eq" in count_constraint:
                pc_filter = {"person_count": int(count_constraint["eq"])}
            elif "gt" in count_constraint:
                pc_filter = {"person_count": {"$gt": int(count_constraint["gt"])}}
            elif "gte" in count_constraint:
                pc_filter = {"person_count": {"$gte": int(count_constraint["gte"])}}
            elif "lt" in count_constraint:
                pc_filter = {"person_count": {"$lt": int(count_constraint["lt"])}}
            elif "lte" in count_constraint:
                pc_filter = {"person_count": {"$lte": int(count_constraint["lte"])}}
            if pc_filter:
                mongo_filter.update(pc_filter)
                print(f"[UnifiedRetrieval] Using person_count index: {pc_filter}")
        
        # Step 1: Handle location-based camera resolution
        location_hint = parsed_filter.get("__location_hint")
        if location_hint:
            camera_ids = self._resolve_camera_from_location(location_hint)
            if camera_ids:
                # Update both mongo_filter AND parsed_filter with resolved camera_ids
                # This ensures semantic search also uses the correct camera filter
                if len(camera_ids) == 1:
                    mongo_filter["camera_id"] = camera_ids[0]
                    parsed_filter["camera_id"] = camera_ids[0]  # Share with semantic search
                else:
                    mongo_filter["camera_id"] = {"$in": camera_ids}
                    parsed_filter["camera_id"] = {"$in": camera_ids}  # Share with semantic search
            # If no cameras found for location, the query will return empty results
        
        print(f"[UnifiedRetrieval] MongoDB filter: {mongo_filter}")
        logger.debug(f"MongoDB filter: {mongo_filter}")

        # Normalize object name to lowercase (YOLO stores lowercase names)
        if "objects.object_name" in mongo_filter:
            mongo_filter["objects.object_name"] = str(mongo_filter["objects.object_name"]).lower()

        # Ensure timestamp filter values are consistent ISO strings
        if "timestamp" in mongo_filter:
            ts_f = mongo_filter["timestamp"]
            if isinstance(ts_f, dict):
                for op in ("$gte", "$lte", "$gt", "$lt"):
                    if op in ts_f and isinstance(ts_f[op], str):
                        # Normalize: strip trailing Z or +/-HH:MM timezone offset
                        ts_f[op] = re.sub(r"(Z|[+-]\d{2}:\d{2})$", "", ts_f[op])
        
        if "timestamp" in mongo_filter:
            ts_f = mongo_filter["timestamp"]
            if isinstance(ts_f, dict):
                for op in ("$gte", "$lte", "$gt", "$lt"):
                    if op in ts_f and isinstance(ts_f[op], str):
                        # Normalize: strip trailing Z or +/-HH:MM timezone offset
                        ts_f[op] = re.sub(r"(Z|[+-]\d{2}:\d{2})$", "", ts_f[op])
        # Initialize query_limit to avoid UnboundLocalError
        query_limit = limit
        if parsed_filter.get("__ask_color"):
            query_limit = max(limit, 200)
        if count_constraint:
            query_limit = max(query_limit, 500)
        
        try:
            cursor = detections_col.find(
                mongo_filter,
                {"_id": 0}
            ).sort("timestamp", -1).limit(query_limit)
            results = list(cursor)

            # Strictly enforce count constraints in-memory.
            results = self._filter_docs_by_count_constraint(results, parsed_filter)
            
            # FALLBACK: If detections are empty or insufficient, query vlm_frames
            # This handles uploaded videos that are indexed into vlm_frames but not detections
            # Disabled for strict count-constraint queries to avoid false positives.
            allow_vlm_fallback = count_constraint is None
            if allow_vlm_fallback and len(results) < 5:  # Threshold for fallback
                print(f"[UnifiedRetrieval] Few/no detections found ({len(results)}), checking vlm_frames...")
                logger.info(f"Checking vlm_frames as fallback, detections count: {len(results)}")
                
                # Build vlm_frames filter (use same camera_id and timestamp if provided)
                vlm_filter = {}
                if "camera_id" in mongo_filter:
                    vlm_filter["camera_id"] = mongo_filter["camera_id"]
                
                # Time range filter for vlm_frames uses frame_ts instead of timestamp
                if "timestamp" in mongo_filter:
                    ts_filter = mongo_filter["timestamp"]
                    if isinstance(ts_filter, dict):
                        vlm_filter["frame_ts"] = {}
                        if "$gte" in ts_filter:
                            vlm_filter["frame_ts"]["$gte"] = ts_filter["$gte"]
                        if "$lte" in ts_filter:
                            vlm_filter["frame_ts"]["$lte"] = ts_filter["$lte"]
                    else:
                        vlm_filter["frame_ts"] = ts_filter
                
                try:
                    # Aggregate vlm_frames by clip_path to get clip-level results
                    pipeline = [
                        {"$match": vlm_filter},
                        {"$sort": {"frame_ts": -1}},
                        {"$group": {
                            "_id": "$clip_path",
                            "camera_id": {"$first": "$camera_id"},
                            "clip_path": {"$first": "$clip_path"},
                            "clip_url": {"$first": "$clip_url"},
                            "first_frame_ts": {"$first": "$frame_ts"},
                            "last_frame_ts": {"$last": "$frame_ts"},
                            "frame_count": {"$sum": 1},
                            "captions": {"$push": "$caption"},
                            "object_captions": {"$push": "$object_captions"},
                        }},
                        {"$limit": query_limit}
                    ]
                    
                    vlm_results = list(vlm_frames_col.aggregate(pipeline))
                    
                    # Transform vlm results to return existing clips directly
                    # These already have clip_url from the uploaded video, no need to build new clips
                    for vlm_doc in vlm_results:
                        flat_caps = self._flatten_object_captions(vlm_doc.get("object_captions", []))
                        if not self._vlm_matches_object_filter(flat_caps, parsed_filter):
                            continue
                        objects_set = set(self._extract_object_tokens(flat_caps))
                        
                        # Create result that will bypass clip building
                        # Since clip_url already exists, _build_clips will skip it
                        result_doc = {
                            "camera_id": vlm_doc.get("camera_id"),
                            "clip_path": vlm_doc.get("clip_path"),
                            "clip_url": vlm_doc.get("clip_url"),  # Existing clip URL
                            "timestamp": vlm_doc.get("first_frame_ts"),
                            "start": vlm_doc.get("first_frame_ts"),
                            "end": vlm_doc.get("last_frame_ts"),
                            "duration_seconds": vlm_doc.get("frame_count", 0),
                            "object_name": ", ".join(sorted(objects_set)) if objects_set else "unknown",
                            "objects": [{"object_name": obj} for obj in sorted(objects_set)],
                            "source": "vlm_frames",
                            "score_struct": 1.0,  # High score since it's an exact match
                            "score_semantic": 0.0,
                        }
                        results.append(result_doc)
                    
                    print(f"[UnifiedRetrieval] Added {len(vlm_results)} clips from vlm_frames")
                    logger.info(f"Added {len(vlm_results)} clips from vlm_frames")
                    
                except Exception as vlm_err:
                    print(f"[UnifiedRetrieval] vlm_frames fallback error: {vlm_err}")
                    logger.error(f"vlm_frames fallback error: {vlm_err}")
            
            return results
        except Exception as e:
            print(f"Structured query error: {e}")
            return []
    
    def _execute_semantic_query(
        self,
        semantic_query: str,
        parsed_filter: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute semantic FAISS vector search."""
        # Extract time and camera filters for semantic search
        camera_id_raw = parsed_filter.get("camera_id")
        camera_id: Optional[int]
        if isinstance(camera_id_raw, int):
            camera_id = camera_id_raw
        else:
            camera_id = None
        
        from_iso = None
        to_iso = None
        ts_filter = parsed_filter.get("timestamp", {})
        if isinstance(ts_filter, dict):
            from_iso = ts_filter.get("$gte")
            to_iso = ts_filter.get("$lte")
        
        # Lower confidence threshold for action queries (they need more lenient matching)
        has_action = parsed_filter.get("action") is not None
        min_confidence = 0.10 if has_action else 0.15
        
        try:
            result = search_unstructured(
                query=semantic_query,
                top_k=limit,
                camera_id=camera_id,
                from_iso=from_iso,
                to_iso=to_iso,
                min_confidence=min_confidence
            )
            logger.debug(f"Raw semantic results: {len(result.get('semantic_results', []))}")
            return result.get("semantic_results", [])
        except Exception as e:
            print(f"Semantic query error: {e}")
            return []
    
    def _merge_results(
        self,
        structured: List[Dict[str, Any]],
        semantic: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any],
        intent: str
    ) -> List[Dict[str, Any]]:
        """
        Intelligently merge structured and semantic results.
        
        For 'count' or 'find' intents: prioritize structured results
        For 'track' or semantic-heavy queries: blend both equally
        """
        count_constraint = self._normalize_count_constraint(parsed_filter.get("count_constraint"))
        has_count_constraint = count_constraint is not None

        if has_count_constraint:
            combined_tracks = self._build_count_constrained_segments(structured, parsed_filter)
            print(f"[UnifiedRetrieval] Count-constrained segments: {len(combined_tracks)}")
        else:
            # Merge tracks from structured results
            merged_tracks = self._merge_structured_tracks(structured, parsed_filter)
            print(f"[UnifiedRetrieval] Merged structured tracks: {len(merged_tracks)}")

            # Combine across track IDs into continuous segments
            combined_tracks = self._coalesce_tracks(merged_tracks)
            print(f"[UnifiedRetrieval] Coalesced tracks: {len(combined_tracks)}")
        
        # Score weighting based on intent and query characteristics
        # Check if query requires semantic understanding (actions, complex behaviors)
        has_action = parsed_filter.get("action") is not None
        
        if has_count_constraint:
            # Strict count queries should be governed by structured evidence only
            alpha = 1.0
        elif has_action:
            # Action queries MUST use semantic search since detections don't have action labels
            alpha = 0.1  # 10% structured, 90% semantic
        elif intent == "count":
            # Heavily favor structured
            alpha = 0.9  # 90% structured, 10% semantic
        elif intent == "track":
            # Balanced
            alpha = 0.5
        else:  # find
            # Slightly favor structured but use semantic
            alpha = 0.7
        
        # Build unified result set
        results_map: Dict[str, Dict[str, Any]] = {}
        
        # Add structured tracks with normalized scores
        struct_scores = [t.get("duration_seconds", 0) for t in combined_tracks]
        struct_norm = self._normalize_scores(struct_scores)
        
        for track, norm_score in zip(combined_tracks, struct_norm):
            key = f"{track.get('camera_id')}_{track.get('start')}_{track.get('end')}"
            results_map[key] = {
                **track,
                "score_struct": norm_score,
                "score_semantic": 0.0,
                "source": "structured"
            }
        
        # Add/merge semantic results (disabled for strict count constraints)
        if not has_count_constraint:
            for sem in semantic:
                # Try to match with existing structured results by clip
                clip_url = sem.get("clip_url")
                matched = False

                for key, result in results_map.items():
                    if result.get("clip_url") == clip_url:
                        # Merge semantic score
                        result["score_semantic"] = sem.get("score_norm", 0.0)
                        result["source"] = "hybrid"
                        result["frames"] = sem.get("frames", [])
                        matched = True
                        break

                if not matched:
                    # Add as new semantic-only result
                    key = f"sem_{sem.get('camera_id')}_{sem.get('clip_path', '')}"
                    results_map[key] = {
                        "camera_id": sem.get("camera_id"),
                        "clip_url": sem.get("clip_url"),
                        "clip_path": sem.get("clip_path"),
                        "score_struct": 0.0,
                        "score_semantic": sem.get("score_norm", 0.0),
                        "source": "semantic",
                        "frames": sem.get("frames", [])
                    }
        
        # Calculate final weighted scores
        results = []
        for result in results_map.values():
            final_score = (
                alpha * result["score_struct"] +
                (1 - alpha) * result["score_semantic"]
            )
            result["score"] = final_score
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def _merge_structured_tracks(
        self,
        results: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Merge contiguous detections belonging to the same track.
        Returns a flat list of merged intervals.
        """
        per_track: Dict[Tuple, List[datetime]] = {}
        meta: Dict[Tuple, Dict[str, Any]] = {}
        objects_by_track: Dict[Tuple, List[Dict[str, Any]]] = {}  # Store all object instances
        
        for doc in results:
            cam = doc.get("camera_id")
            ts = self._parse_ts(doc.get("timestamp", ""))
            
            for obj in doc.get("objects", []):
                if not self._object_matches(obj, parsed_filter):
                    continue
                
                tid = obj.get("track_id", -1)
                # Use a synthetic track key for trackless detections instead of dropping them
                if tid is None or (isinstance(tid, (int, float)) and tid < 0):
                    tid = f"notrak_{cam}_{ts.isoformat()}_{id(obj)}"  # synthetic unique key
                
                key = (cam, tid)
                per_track.setdefault(key, []).append(ts)
                
                # Store full object information for later aggregation
                objects_by_track.setdefault(key, []).append(obj)
                
                if key not in meta:
                    meta[key] = {
                        "camera_id": cam,
                        "track_id": tid,
                        "object_name": obj.get("object_name"),
                        "color": obj.get("color"),
                    }
        
        merged = []
        for key, times in per_track.items():
            times.sort()
            if not times:
                continue
            
            seg_start_idx = 0
            start = times[0]
            prev = times[0]
            
            # Aggregate object details for this track
            track_objects = objects_by_track.get(key, [])
            # Create a representative object with aggregated attributes
            aggregated_objects = self._aggregate_objects(track_objects)
            
            for i, t in enumerate(times[1:], start=1):
                gap = (t - prev).total_seconds()
                if gap <= self.max_gap_seconds:
                    prev = t
                else:
                    # Close segment
                    m = dict(meta[key])
                    m["start"] = start.isoformat()
                    m["end"] = prev.isoformat()
                    m["duration_seconds"] = max(0, int((prev - start).total_seconds()))
                    m["objects"] = aggregated_objects  # Attach full object details
                    # Collect timestamps for this segment to whitelist frames in clip builder
                    m["matched_timestamps"] = [t.isoformat() for t in times[seg_start_idx:i]]
                    merged.append(m)
                    start = t
                    prev = t
                    seg_start_idx = i
            
            # Close last segment
            m = dict(meta[key])
            m["start"] = start.isoformat()
            m["end"] = prev.isoformat()
            m["duration_seconds"] = max(0, int((prev - start).total_seconds()))
            m["objects"] = aggregated_objects  # Attach full object details
            m["matched_timestamps"] = [t.isoformat() for t in times[seg_start_idx:]]
            merged.append(m)
        
        merged.sort(key=lambda x: x.get("start", ""), reverse=True)
        return merged
    
    def _coalesce_tracks(
        self,
        merged: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Coalesce segments across different track_ids when temporally adjacent.
        Groups by (camera_id, object_name, color).
        """
        groups: Dict[Tuple, List[Dict[str, Any]]] = {}
        
        for seg in merged:
            key = (
                seg.get("camera_id"),
                seg.get("object_name"),
                seg.get("color")
            )
            groups.setdefault(key, []).append(seg)
        
        output = []
        for key, segs in groups.items():
            segs_sorted = sorted(segs, key=lambda s: s.get("start", ""))
            if not segs_sorted:
                continue
            
            cur_start = self._parse_ts(segs_sorted[0]["start"])
            cur_end = self._parse_ts(segs_sorted[0]["end"])
            cam, obj, col = key
            
            # Start with first segment's objects and timestamps only (not all segments)
            all_segment_objects = list(segs_sorted[0].get("objects", []))
            all_segment_timestamps = list(segs_sorted[0].get("matched_timestamps", []))
            
            for seg in segs_sorted[1:]:
                s = self._parse_ts(seg["start"])
                e = self._parse_ts(seg["end"])
                gap = (s - cur_end).total_seconds()
                
                if gap <= self.join_gap_seconds:
                    if e > cur_end:
                        cur_end = e
                    # Accumulate objects and timestamps from this merged segment
                    if seg.get("objects"):
                        all_segment_objects.extend(seg["objects"])
                    if seg.get("matched_timestamps"):
                        all_segment_timestamps.extend(seg["matched_timestamps"])
                else:
                    # Emit current
                    aggregated_objects = self._aggregate_objects(all_segment_objects)
                    output.append({
                        "camera_id": cam,
                        "object_name": obj,
                        "color": col,
                        "start": cur_start.isoformat(),
                        "end": cur_end.isoformat(),
                        "duration_seconds": max(0, int((cur_end - cur_start).total_seconds())),
                        "objects": aggregated_objects,
                        "matched_timestamps": all_segment_timestamps,
                    })
                    cur_start, cur_end = s, e
                    # Reset for next segment group
                    all_segment_objects = list(seg.get("objects", []))
                    all_segment_timestamps = list(seg.get("matched_timestamps", []))
            
            # Emit final
            aggregated_objects = self._aggregate_objects(all_segment_objects)
            output.append({
                "camera_id": cam,
                "object_name": obj,
                "color": col,
                "start": cur_start.isoformat(),
                "end": cur_end.isoformat(),
                "duration_seconds": max(0, int((cur_end - cur_start).total_seconds())),
                "objects": aggregated_objects,
                "matched_timestamps": all_segment_timestamps,
            })
        
        output.sort(key=lambda x: x.get("start", ""), reverse=True)
        return output
    
    def _enrich_with_camera_metadata(
        self, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich results with camera metadata (location, status) from cameras collection.
        
        Args:
            results: List of result dictionaries containing camera_id
            
        Returns:
            Enriched results with camera location and status
        """
        if not results:
            return results
        
        # Collect unique camera IDs
        camera_ids = list(set(
            r.get("camera_id") for r in results 
            if r.get("camera_id") is not None
        ))
        
        if not camera_ids:
            return results
        
        try:
            # Fetch camera metadata in bulk
            cameras = list(cameras_col.find(
                {"camera_id": {"$in": camera_ids}},
                {"camera_id": 1, "location": 1, "status": 1, "_id": 0}
            ))
            
            # Build lookup map
            camera_map = {cam["camera_id"]: cam for cam in cameras}
            
            # Enrich each result
            for result in results:
                cam_id = result.get("camera_id")
                if cam_id in camera_map:
                    cam_info = camera_map[cam_id]
                    result["location"] = cam_info.get("location", "Unknown")
                    result["camera_status"] = cam_info.get("status", "unknown")
            
            print(f"[UnifiedRetrieval] Enriched {len(results)} results with camera metadata")
            logger.debug(f"Enriched {len(results)} results with camera metadata")
            
        except Exception as e:
            print(f"[UnifiedRetrieval] Error enriching camera metadata: {e}")
            logger.error(f"Error enriching camera metadata: {e}")
        
        return results
    
    def _enrich_with_vlm_frames(
        self,
        results: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Optionally enrich results with VLM frame captions and object information.
        
        This adds detailed scene descriptions and object-level captions from vlm_frames collection
        to provide richer context for the answer generator.
        
        Args:
            results: List of result dictionaries
            parsed_filter: Original parsed filter to determine if enrichment needed
            
        Returns:
            Results enriched with VLM frame data if available
        """
        # Only enrich if semantic search is enabled and we have visual query type
        if not settings.ENABLE_SEMANTIC:
            return results
        
        query_type = parsed_filter.get("query_type", "visual")
        if query_type != "visual":
            return results
        
        # Don't enrich if results already have frames from semantic search
        if results and results[0].get("frames"):
            return results
        
        try:
            for result in results:
                cam_id = result.get("camera_id")
                clip_path = result.get("clip_path")
                
                # Skip if no clip_path or camera_id
                if not clip_path or cam_id is None:
                    continue
                
                # Query vlm_frames for this clip
                vlm_docs = list(vlm_frames_col.find(
                    {
                        "camera_id": cam_id,
                        "clip_path": clip_path
                    },
                    {
                        "_id": 0,
                        "caption": 1,
                        "object_captions": 1,
                        "frame_index": 1,
                        "frame_ts": 1
                    }
                ).limit(10))  # Limit to first 10 frames
                
                if vlm_docs:
                    result["vlm_frames"] = vlm_docs
                    # Add main caption from first frame if available
                    if vlm_docs and vlm_docs[0].get("caption"):
                        result["scene_caption"] = vlm_docs[0]["caption"]
            
            enriched_count = sum(1 for r in results if r.get("vlm_frames"))
            if enriched_count > 0:
                print(f"[UnifiedRetrieval] Enriched {enriched_count} results with VLM frame data")
                logger.debug(f"Enriched {enriched_count} results with VLM frame data")
                
        except Exception as e:
            print(f"[UnifiedRetrieval] Error enriching VLM frames: {e}")
            logger.error(f"Error enriching VLM frames: {e}")
        
        return results
    
    def _build_clips(
        self,
        results: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build video clips for results and enrich with camera metadata and VLM frames."""
        enriched = []
        
        for result in results:
            result_copy = dict(result)
            
            # Skip if already has clip_url from semantic search
            if result_copy.get("clip_url"):
                enriched.append(result_copy)
                continue
            
            # Try to build clip from structured result
            try:
                cam = result_copy.get("camera_id")
                start = result_copy.get("start")
                end = result_copy.get("end")
                
                if cam is not None and start and end:
                    allowed_ts = result_copy.get("matched_timestamps")
                    
                    # Add 3s context padding (buffer) to ensure smooth lead-in/lead-out
                    # This also ensures single-frame detections yield meaningful clips (~6s)
                    try:
                        s_dt = self._parse_ts(start) - timedelta(seconds=3)
                        e_dt = self._parse_ts(end) + timedelta(seconds=3)
                        start_iso = s_dt.isoformat()
                        end_iso = e_dt.isoformat()
                    except Exception:
                        start_iso = start
                        end_iso = end

                    clip = build_clip_from_snapshots(
                        int(cam),
                        start_iso,
                        end_iso,
                        fps=5.0,
                        allowed_timestamps=allowed_ts
                    )
                    result_copy["clip_url"] = clip.url
                    result_copy["clip_frames"] = clip.frame_count
                    result_copy["clip_path"] = clip.path  # Store clip_path for VLM enrichment
            except Exception as e:
                result_copy["clip_error"] = str(e)
            
            enriched.append(result_copy)
        
        # Enrich all results with camera metadata
        enriched = self._enrich_with_camera_metadata(enriched)
        
        # Optionally enrich with VLM frame captions
        enriched = self._enrich_with_vlm_frames(enriched, parsed_filter)
        
        return enriched
    

    
    def _object_matches(
        self,
        obj: Dict[str, Any],
        parsed_filter: Dict[str, Any]
    ) -> bool:
        """Check if object matches parsed filter criteria."""
        if not isinstance(obj, dict):
            return False
        
        name = parsed_filter.get("objects.object_name")
        color = parsed_filter.get("objects.color")
        
        if name and str(obj.get("object_name", "")).lower() != str(name).lower():
            return False
        if color and str(obj.get("color", "")).lower() != str(color).lower():
            return False
        
        return True

    def _normalize_count_constraint(self, raw: Any) -> Optional[Dict[str, int]]:
        if not isinstance(raw, dict):
            return None
        key_map = {
            "eq": "eq",
            "=": "eq",
            "==": "eq",
            "gte": "gte",
            ">=": "gte",
            "gt": "gt",
            ">": "gt",
            "lte": "lte",
            "<=": "lte",
            "lt": "lt",
            "<": "lt",
        }
        out: Dict[str, int] = {}
        for k, v in raw.items():
            norm_k = key_map.get(str(k).strip().lower())
            if not norm_k:
                continue
            try:
                out[norm_k] = int(v)
            except Exception:
                continue
        return out or None

    def _count_matches_constraint(self, count: int, constraint: Dict[str, int]) -> bool:
        if "eq" in constraint and count != int(constraint["eq"]):
            return False
        if "gte" in constraint and count < int(constraint["gte"]):
            return False
        if "gt" in constraint and count <= int(constraint["gt"]):
            return False
        if "lte" in constraint and count > int(constraint["lte"]):
            return False
        if "lt" in constraint and count >= int(constraint["lt"]):
            return False
        return True

    def _count_matching_objects_in_doc(self, doc: Dict[str, Any], parsed_filter: Dict[str, Any]) -> int:
        # Fast-path: use pre-computed person_count when filtering only by name="person" (no color)
        obj_name = parsed_filter.get("objects.object_name")
        color = parsed_filter.get("objects.color")
        if obj_name and str(obj_name).lower() == "person" and not color:
            pc = doc.get("person_count")
            if pc is not None:
                return int(pc)
        # Fast-path: use object_counts dict for other object types (no color filter)
        if obj_name and not color:
            oc = doc.get("object_counts")
            if isinstance(oc, dict):
                cnt = oc.get(str(obj_name).lower())
                if cnt is not None:
                    return int(cnt)
        # Fallback: count matching objects by iterating the array
        objs = doc.get("objects", [])
        if not isinstance(objs, list):
            return 0
        return sum(1 for o in objs if isinstance(o, dict) and self._object_matches(o, parsed_filter))

    def _filter_docs_by_count_constraint(self, docs: List[Dict[str, Any]], parsed_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        constraint = self._normalize_count_constraint(parsed_filter.get("count_constraint"))
        if not constraint:
            return docs

        out: List[Dict[str, Any]] = []
        for d in docs:
            cnt = self._count_matching_objects_in_doc(d, parsed_filter)
            if self._count_matches_constraint(cnt, constraint):
                dd = dict(d)
                dd["matched_object_count"] = cnt
                out.append(dd)
        return out

    def _build_count_constrained_segments(
        self,
        docs: List[Dict[str, Any]],
        parsed_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Build temporal segments directly from per-detection docs that satisfy count constraints
        (e.g., >=2 persons, ==0 persons). This avoids track-level false positives.
        """
        constraint = self._normalize_count_constraint(parsed_filter.get("count_constraint"))
        if not constraint:
            return []

        filtered = self._filter_docs_by_count_constraint(docs, parsed_filter)
        by_cam: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for d in filtered:
            by_cam[d.get("camera_id")].append(d)

        segments: List[Dict[str, Any]] = []
        for cam, cam_docs in by_cam.items():
            cam_docs_sorted = sorted(cam_docs, key=lambda x: x.get("timestamp", ""))
            seg_start: Optional[datetime] = None
            seg_end: Optional[datetime] = None
            seg_peak_count = 0
            seg_objects: List[Dict[str, Any]] = []
            seg_timestamps: List[str] = []

            def _emit_segment() -> None:
                nonlocal seg_start, seg_end, seg_peak_count, seg_objects, seg_timestamps
                if seg_start is None or seg_end is None:
                    return
                segments.append(
                    {
                        "camera_id": cam,
                        "object_name": parsed_filter.get("objects.object_name"),
                        "color": parsed_filter.get("objects.color"),
                        "start": seg_start.isoformat(),
                        "end": seg_end.isoformat(),
                        "duration_seconds": max(0, int((seg_end - seg_start).total_seconds())),
                        "objects": self._aggregate_objects(seg_objects),
                        "match_count_peak": int(seg_peak_count),
                        "count_constraint": constraint,
                        "matched_timestamps": list(seg_timestamps),
                    }
                )

            for d in cam_docs_sorted:
                ts = self._parse_ts(d.get("timestamp", ""))
                cnt = int(d.get("matched_object_count", self._count_matching_objects_in_doc(d, parsed_filter)))
                matched_objs = [
                    o for o in (d.get("objects") or []) if isinstance(o, dict) and self._object_matches(o, parsed_filter)
                ]

                if seg_start is None:
                    seg_start = ts
                    seg_end = ts
                    seg_peak_count = cnt
                    seg_objects = matched_objs.copy()
                    seg_timestamps = [ts.isoformat()]
                    continue

                gap = (ts - seg_end).total_seconds() if seg_end is not None else 0
                if gap <= self.join_gap_seconds:
                    seg_end = ts
                    if cnt > seg_peak_count:
                        seg_peak_count = cnt
                    seg_objects.extend(matched_objs)
                    seg_timestamps.append(ts.isoformat())
                else:
                    _emit_segment()
                    seg_start = ts
                    seg_end = ts
                    seg_peak_count = cnt
                    seg_objects = matched_objs.copy()
                    seg_timestamps = [ts.isoformat()]

            _emit_segment()

        segments.sort(key=lambda x: x.get("start", ""), reverse=True)
        return segments

    def _flatten_object_captions(self, payload: Any) -> List[str]:
        out: List[str] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, list):
                    for x in item:
                        if isinstance(x, str) and x.strip():
                            out.append(x.strip().lower())
                elif isinstance(item, str) and item.strip():
                    out.append(item.strip().lower())
        return out

    def _extract_object_tokens(self, flat_caps: List[str]) -> List[str]:
        out = set()
        for cap in flat_caps:
            low = cap.lower()
            if "person" in low:
                out.add("person")
            if "vehicle" in low or "car" in low:
                out.add("car")
            tok = low.split(" ")[0].strip()
            if tok:
                out.add(tok)
        return sorted(out)

    def _vlm_matches_object_filter(self, flat_caps: List[str], parsed_filter: Dict[str, Any]) -> bool:
        obj = parsed_filter.get("objects.object_name")
        color = parsed_filter.get("objects.color")

        if not obj and not color:
            return True
        if not flat_caps:
            return False

        obj_ok = True
        if obj:
            obj_low = str(obj).lower()
            aliases = {obj_low}
            if obj_low == "person":
                aliases.update({"people", "person"})
            elif obj_low == "car":
                aliases.update({"car", "vehicle", "cars", "vehicles"})
            obj_ok = any(any(a in c for a in aliases) for c in flat_caps)

        color_ok = True
        if color:
            color_low = str(color).lower()
            color_ok = any(color_low in c for c in flat_caps)

        return obj_ok and color_ok
    
    def _parse_ts(self, ts: str) -> datetime:
        """Parse timestamp string to datetime."""
        try:
            # Robust parsing: strip Z and +/-HH:MM timezone offset
            safe = re.sub(r"(Z|[+-]\d{2}:\d{2})$", "", ts)
            return datetime.fromisoformat(safe)
        except Exception:
            try:
                return datetime.strptime(ts.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            except Exception:
                return datetime.utcnow()
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0]  # Single result gets full score
        max_score = max(scores)
        if max_score == 0:
            return [1.0] * len(scores)  # All equal = all get full score
        return [s / max_score for s in scores]
    
    def _aggregate_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate multiple object instances into a representative list.
        Calculates average confidence and collects unique colors and attributes.
        
        Args:
            objects: List of object dictionaries from detections
            
        Returns:
            List with aggregated object information
        """
        if not objects:
            return []
        
        # Group by object_name
        by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for obj in objects:
            name = obj.get("object_name", "unknown")
            by_name[name].append(obj)
        
        aggregated = []
        for obj_name, obj_instances in by_name.items():
            # Calculate average confidence
            confidences = [o.get("confidence", 0) for o in obj_instances if o.get("confidence")]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Collect unique colors
            colors = list(set(o.get("color") for o in obj_instances if o.get("color")))
            
            # Use first instance as template
            representative = obj_instances[0].copy()
            representative["confidence"] = avg_confidence
            
            # Add color information
            if colors:
                representative["color"] = colors[0] if len(colors) == 1 else ", ".join(colors)
                representative["colors"] = colors  # All detected colors
            
            aggregated.append(representative)
        
        return aggregated
