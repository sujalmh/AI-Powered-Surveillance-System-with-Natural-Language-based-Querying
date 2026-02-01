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
        if settings.ENABLE_SEMANTIC and semantic_query:
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
        clips = []
        if query_type == "visual":
            # Visual query: build video clips for top results
            print("[UnifiedRetrieval] Step 4: Building video clips for visual query")
            logger.info("Step 4: Building video clips for visual query")
            clips = self._build_clips(merged_results[:5], parsed_filter)
            print(f"[UnifiedRetrieval] Generated {len(clips)} clips")
            logger.info(f"Generated {len(clips)} clips")
        else:
            # Informational query: NO clip generation, return merged results as-is
            print("[UnifiedRetrieval] Step 4: Skipping clip generation for informational query")
            logger.info("Step 4: Skipping clip generation for informational query")
            clips = []
        
        # 5. LLM-generated answer using AnswerGenerator
        answer_gen = AnswerGenerator()
        answer = answer_gen.generate(
            query=parsed_filter.get("__raw", ""),
            query_type=query_type,
            results=clips if clips else structured_results[:10],
            parsed_filter=parsed_filter,
            metadata={
                "intent": intent,
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
        
        # Increase limit for color queries to get better distribution
        query_limit = limit
        if parsed_filter.get("__ask_color"):
            query_limit = max(limit, 200)
        
        try:
            cursor = detections_col.find(
                mongo_filter,
                {"_id": 0}
            ).sort("timestamp", -1).limit(query_limit)
            results = list(cursor)
            
            # FALLBACK: If detections are empty or insufficient, query vlm_frames
            # This handles uploaded videos that are indexed into vlm_frames but not detections
            if len(results) < 5:  # Threshold for fallback
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
                        # Extract objects from object_captions arrays
                        objects_set = set()
                        for obj_caps in vlm_doc.get("object_captions", []):
                            if isinstance(obj_caps, list):
                                for cap in obj_caps:
                                    if isinstance(cap, str):
                                        objects_set.add(cap.lower().strip())
                        
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
        camera_id = parsed_filter.get("camera_id")
        
        from_iso = None
        to_iso = None
        ts_filter = parsed_filter.get("timestamp", {})
        if isinstance(ts_filter, dict):
            from_iso = ts_filter.get("$gte")
            to_iso = ts_filter.get("$lte")
        
        # Lower confidence threshold for action queries (they need more lenient matching)
        has_action = parsed_filter.get("action") is not None
        min_confidence = 0.15 if has_action else 0.25
        
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
        # Merge tracks from structured results
        merged_tracks = self._merge_structured_tracks(structured, parsed_filter)
        print(f"[UnifiedRetrieval] Merged structured tracks: {len(merged_tracks)}")
        
        # Combine across track IDs into continuous segments
        combined_tracks = self._coalesce_tracks(merged_tracks)
        print(f"[UnifiedRetrieval] Coalesced tracks: {len(combined_tracks)}")
        
        # Score weighting based on intent and query characteristics
        # Check if query requires semantic understanding (actions, complex behaviors)
        has_action = parsed_filter.get("action") is not None
        
        if has_action:
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
        
        # Add/merge semantic results
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
        
        for doc in results:
            cam = doc.get("camera_id")
            ts = self._parse_ts(doc.get("timestamp", ""))
            
            for obj in doc.get("objects", []):
                if not self._object_matches(obj, parsed_filter):
                    continue
                
                tid = obj.get("track_id", -1)
                if tid is None or tid < 0:
                    continue
                
                key = (cam, tid)
                per_track.setdefault(key, []).append(ts)
                
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
            
            start = times[0]
            prev = times[0]
            
            for t in times[1:]:
                gap = (t - prev).total_seconds()
                if gap <= self.max_gap_seconds:
                    prev = t
                else:
                    # Close segment
                    m = dict(meta[key])
                    m["start"] = start.isoformat()
                    m["end"] = prev.isoformat()
                    m["duration_seconds"] = max(0, int((prev - start).total_seconds()))
                    merged.append(m)
                    start = t
                    prev = t
            
            # Close last segment
            m = dict(meta[key])
            m["start"] = start.isoformat()
            m["end"] = prev.isoformat()
            m["duration_seconds"] = max(0, int((prev - start).total_seconds()))
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
            
            for seg in segs_sorted[1:]:
                s = self._parse_ts(seg["start"])
                e = self._parse_ts(seg["end"])
                gap = (s - cur_end).total_seconds()
                
                if gap <= self.join_gap_seconds:
                    if e > cur_end:
                        cur_end = e
                else:
                    # Emit current
                    output.append({
                        "camera_id": cam,
                        "object_name": obj,
                        "color": col,
                        "start": cur_start.isoformat(),
                        "end": cur_end.isoformat(),
                        "duration_seconds": max(0, int((cur_end - cur_start).total_seconds())),
                    })
                    cur_start, cur_end = s, e
            
            # Emit final
            output.append({
                "camera_id": cam,
                "object_name": obj,
                "color": col,
                "start": cur_start.isoformat(),
                "end": cur_end.isoformat(),
                "duration_seconds": max(0, int((cur_end - cur_start).total_seconds())),
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
                    clip = build_clip_from_snapshots(
                        int(cam),
                        start,
                        end,
                        fps=5.0
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
        
        if name and obj.get("object_name") != name:
            return False
        if color and obj.get("color") != color:
            return False
        
        return True
    
    def _parse_ts(self, ts: str) -> datetime:
        """Parse timestamp string to datetime."""
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            try:
                return datetime.strptime(ts.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            except Exception:
                return datetime.utcnow()
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        
        mn = min(scores)
        mx = max(scores)
        
        if mx - mn < 1e-9:
            return [1.0 for _ in scores]
        
        return [(s - mn) / (mx - mn) for s in scores]
