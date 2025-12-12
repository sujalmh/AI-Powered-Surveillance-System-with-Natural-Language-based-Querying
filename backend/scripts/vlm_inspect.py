import argparse
import json
from pathlib import Path

from backend.app.db.mongo import vlm_frames
from backend.app.services.sem_store import get_faiss_store
from backend.app.services.sem_search import search_unstructured


def print_mongo_summary(sample: int = 3) -> None:
    total = vlm_frames.count_documents({})
    print(f"[Mongo] vlm_frames total docs: {total}")

    # Top cameras by frame count
    try:
        pipeline = [
            {"$group": {"_id": "$camera_id", "n": {"$sum": 1}}},
            {"$sort": {"n": -1}},
            {"$limit": 10},
        ]
        rows = list(vlm_frames.aggregate(pipeline))
        if rows:
            print("[Mongo] Top cameras by frames:")
            for r in rows:
                print(f"  camera {r.get('_id')}: {r.get('n')} frames")
    except Exception as e:
        print(f"[Mongo] aggregate error: {e!r}")

    # Sample latest documents
    cur = vlm_frames.find({}, {"_id": 0}).sort("frame_ts", -1).limit(sample)
    print(f"[Mongo] Latest {sample} docs:")
    for doc in cur:
        print(json.dumps(doc, indent=2, ensure_ascii=False))


def print_faiss_summary(sample: int = 3) -> None:
    try:
        store = get_faiss_store(dim=512)
        cnt = store.count()
        print(f"[FAISS] vector count: {cnt}")
        # Inspect meta mapping file (faiss_id -> mongo_id, clip_path, etc.)
        meta_path: Path = store.meta_path  # type: ignore[attr-defined]
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                print(f"[FAISS] meta entries: {len(meta)}")
                print(f"[FAISS] First {min(sample, len(meta))} meta rows:")
                for row in meta[:sample]:
                    # Keep output compact
                    compact = {
                        "faiss_id": row.get("faiss_id"),
                        "mongo_id": row.get("mongo_id"),
                        "camera_id": row.get("camera_id"),
                        "clip_path": row.get("clip_path"),
                        "frame_index": row.get("frame_index"),
                        "frame_ts": row.get("frame_ts"),
                        "model": row.get("model"),
                    }
                    print(json.dumps(compact, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"[FAISS] failed to read meta.json: {e!r}")
        else:
            print("[FAISS] meta.json not found")
    except Exception as e:
        print(f"[FAISS] error opening index: {e!r}")


def run_semantic_query(query: str, camera_id: int | None, top_k: int) -> None:
    try:
        out = search_unstructured(query, top_k=top_k, camera_id=camera_id)
        print("[Semantic Search] query result:")
        print(json.dumps(out, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[Semantic Search] error: {e!r}")


def print_caption_samples(sample: int = 3) -> None:
    try:
        cur = vlm_frames.find(
            {"caption": {"$exists": True, "$nin": [None, ""]}},
            {"_id": 0, "clip_path": 1, "frame_index": 1, "frame_ts": 1, "caption": 1},
        ).sort("frame_ts", -1).limit(sample)
        rows = list(cur)
        if rows:
            print(f"[Mongo] Caption samples (latest {sample}):")
            for r in rows:
                print(json.dumps(r, indent=2, ensure_ascii=False))
        else:
            print("[Mongo] No captions found yet; ensure with_captions=true and ENABLE_CAPTIONS=true, then re-index.")
    except Exception as e:
        print(f"[Mongo] caption query error: {e!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect VLM (CLIP) stored details: Mongo, FAISS, and run a semantic query.")
    parser.add_argument("--sample", type=int, default=3, help="Number of sample docs/meta rows to print")
    parser.add_argument("--query", type=str, default=None, help="Optional text query to test semantic search")
    parser.add_argument("--camera", type=int, default=None, help="Optional camera_id filter for the query")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k for semantic search")
    args = parser.parse_args()

    print_mongo_summary(sample=args.sample)
    print_faiss_summary(sample=args.sample)
    print_caption_samples(sample=args.sample)

    if args.query:
        run_semantic_query(args.query, camera_id=args.camera, top_k=args.top_k)


if __name__ == "__main__":
    main()
