"""Release manifest / integrity check.

This script does not touch MongoDB or the network.  It inventories the
released single-camera subset, the two FAISS indices, the benchmark files, and
the installed package versions, then writes a machine-readable manifest under
``results/``.

Run:
    python backend/scripts/check_release.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return None


def ffprobe_duration(path: Path) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None


def ffprobe_video(path: Path) -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        stream = json.loads(out)["streams"][0]
        return {
            "width": stream.get("width"),
            "height": stream.get("height"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "nb_frames": stream.get("nb_frames"),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def index_info(index_path: Path, meta_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "index_path": str(index_path.relative_to(ROOT)),
        "meta_path": str(meta_path.relative_to(ROOT)),
        "exists": index_path.exists() and meta_path.exists(),
    }
    if not info["exists"]:
        return info
    info["index_sha256"] = sha256(index_path)
    info["meta_sha256"] = sha256(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    info["n_meta"] = len(meta)
    try:
        import faiss

        index = faiss.read_index(str(index_path))
        info["n_index"] = int(index.ntotal)
        info["dim"] = int(index.d)
        info["index_type"] = type(index).__name__
        info["metric"] = "inner_product" if index.metric_type == faiss.METRIC_INNER_PRODUCT else str(index.metric_type)
    except Exception as exc:
        info["faiss_error"] = f"{type(exc).__name__}: {exc}"
    return info


def benchmark_info(path: Path, text_field: str = "text") -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    raw = path.read_text(encoding="utf-8")
    # A few benchmark files (e.g. splits.json) are a single JSON object rather
    # than JSONL.  Report them distinctly so the manifest does not lie about
    # "unique texts".
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {
                "path": str(path.relative_to(ROOT)),
                "exists": True,
                "format": "json_object",
                "sha256": sha256(path),
                "keys": list(parsed.keys())[:10],
            }
    except json.JSONDecodeError:
        pass
    rows: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = {"_invalid": line[:120]}
        rows.append(value if isinstance(value, dict) else {"_value": value})
    texts = [str(r.get(text_field, "")) for r in rows]
    has_text = any(text_field in r for r in rows)
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "format": "jsonl",
        "sha256": sha256(path),
        "rows": len(rows),
        "unique_texts": len(set(texts)) if has_text else None,
        "sample": texts[:3] if has_text else None,
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    clips = sorted((DATA / "clips" / "camera_0").glob("*.mp4"))
    snapshots = sorted((DATA / "snapshots" / "camera_0").glob("*.jpg"))
    recordings = sorted((DATA / "recordings" / "camera_0").glob("*.mp4"))

    clip_durations = [ffprobe_duration(p) for p in clips]
    valid_clip_durations = [d for d in clip_durations if d is not None]
    recording_durations = [ffprobe_duration(p) for p in recordings]
    valid_recording_durations = [d for d in recording_durations if d is not None]

    scene_meta_path = DATA / "index" / "faiss" / "meta.json"
    person_meta_path = DATA / "index" / "faiss_person" / "meta.json"
    scene_meta = json.loads(scene_meta_path.read_text(encoding="utf-8")) if scene_meta_path.exists() else []
    person_meta = json.loads(person_meta_path.read_text(encoding="utf-8")) if person_meta_path.exists() else []

    scene_captions = [str(x.get("caption", "")) for x in scene_meta]
    scene_action_labels = Counter()
    scene_person_counts: List[int] = []
    for caption in scene_captions:
        low = caption.lower()
        for label in ("fighting", "smoke", "security"):
            if label in low:
                scene_action_labels[label] += 1
        match = re.search(r"(\d+)\s+person", caption)
        if match:
            scene_person_counts.append(int(match.group(1)))

    person_colors = Counter(str(x.get("color")) for x in person_meta)
    person_attrs = Counter(str(x.get("attribute_text")) for x in person_meta)
    person_ts = sorted({str(x.get("timestamp")) for x in person_meta if x.get("timestamp")})

    manifest: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "platform": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "open-clip-torch",
                "sentence-transformers",
                "faiss-cpu",
                "numpy",
                "pymongo",
                "fastapi",
            )
        },
        "subset": {
            "camera_ids": sorted({int(x.get("camera_id", -1)) for x in person_meta + scene_meta}),
            "clips": {
                "count": len(clips),
                "format": ffprobe_video(clips[0]) if clips else None,
                "ffprobe_durations_s": [round(d, 3) for d in valid_clip_durations],
                "total_duration_s": round(sum(valid_clip_durations), 3),
                "first": clips[0].name if clips else None,
                "last": clips[-1].name if clips else None,
            },
            "snapshots": {
                "count": len(snapshots),
                "first": snapshots[0].name if snapshots else None,
                "last": snapshots[-1].name if snapshots else None,
            },
            "recordings": {
                "count": len(recordings),
                "total_duration_s": round(sum(valid_recording_durations), 3),
                "unreadable": [p.name for p, d in zip(recordings, recording_durations) if d is None],
            },
            "scene_index_meta": {
                "n": len(scene_meta),
                "timestamp_min": min((x.get("frame_ts") for x in scene_meta), default=None),
                "timestamp_max": max((x.get("frame_ts") for x in scene_meta), default=None),
                "unique_captions": len(set(scene_captions)),
                "person_count_distribution": dict(sorted(Counter(scene_person_counts).items())),
                "caption_action_labels": dict(scene_action_labels),
                "color_mentions": {
                    color: sum(1 for c in scene_captions if color.lower() in c.lower())
                    for color in ("Olive", "Red", "Brown")
                },
            },
            "person_index_meta": {
                "n": len(person_meta),
                "timestamp_min": person_ts[0] if person_ts else None,
                "timestamp_max": person_ts[-1] if person_ts else None,
                "unique_timestamps": len(person_ts),
                "track_ids": sorted({int(x.get("track_id", -1)) for x in person_meta}),
                "colors": dict(person_colors),
                "unique_attribute_texts": len(person_attrs),
                "attribute_texts": dict(person_attrs.most_common(10)),
            },
        },
        "indices": {
            "scene": index_info(DATA / "index" / "faiss" / "index.bin", scene_meta_path),
            "person": index_info(DATA / "index" / "faiss_person" / "index.bin", person_meta_path),
        },
        "benchmarks": {
            "retrieval_queries": benchmark_info(ROOT / "backend" / "benchmarks" / "queries.jsonl"),
            "retrieval_queries_control": benchmark_info(
                ROOT / "backend" / "benchmarks" / "queries_control.jsonl"
            ),
            "alert_utterances": benchmark_info(
                ROOT / "backend" / "benchmarks" / "alert_utterances.jsonl"
            ),
            "person_pairs": benchmark_info(
                ROOT / "backend" / "benchmarks" / "person_pairs.jsonl"
            ),
            "splits": benchmark_info(
                ROOT / "backend" / "benchmarks" / "splits.json", text_field="qid"
            ),
            "splits_control": benchmark_info(
                ROOT / "backend" / "benchmarks" / "splits_control.json", text_field="qid"
            ),
        },
    }

    out_json = RESULTS / "release_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Release manifest",
        "",
        f"Generated: `{manifest['generated_at_utc']}`",
        f"Git: `{manifest['git_commit']}`",
        "",
        "## Subset",
        "",
        f"- Clips: {len(clips)} (~{sum(valid_clip_durations)/60:.1f} min selected "
        f"footage; {manifest['subset']['clips'].get('format', {}).get('width', '?')}x"
        f"{manifest['subset']['clips'].get('format', {}).get('height', '?')} @ "
        f"{manifest['subset']['clips'].get('format', {}).get('r_frame_rate', '?')} fps)",
        f"- Snapshots: {len(snapshots)}",
        f"- Recordings: {len(recordings)} (~{sum(valid_recording_durations)/60:.1f} min readable)",
        f"- Scene index: {len(scene_meta)} frames, "
        f"{manifest['indices']['scene'].get('dim', '?')}-d, "
        f"{manifest['indices']['scene'].get('n_index', '?')} vectors",
        f"- Person index: {len(person_meta)} descriptors, "
        f"{manifest['indices']['person'].get('dim', '?')}-d, "
        f"{manifest['indices']['person'].get('n_index', '?')} vectors",
        f"- Camera IDs: {manifest['subset']['camera_ids']}",
        f"- Timestamp span: {manifest['subset']['scene_index_meta']['timestamp_min']} "
        f"to {manifest['subset']['scene_index_meta']['timestamp_max']}",
        "",
        "## Benchmarks",
        "",
    ]
    for name, info in manifest["benchmarks"].items():
        if not info.get("exists"):
            lines.append(f"- `{info['path']}`: missing")
        elif info.get("format") == "json_object":
            keys = ", ".join(info.get("keys", []))
            lines.append(f"- `{info['path']}`: JSON object (keys: {keys})")
        else:
            unique = info.get("unique_texts")
            suffix = f", {unique} unique texts" if unique is not None else ""
            lines.append(f"- `{info['path']}`: {info.get('rows', '?')} rows{suffix}")
    lines += [
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$env:PYTHONIOENCODING='utf-8'",
        "python backend/scripts/check_release.py",
        "python backend/benchmarks/build_subset.py",
        "python backend/benchmarks/build_subset_control.py",
        "python backend/scripts/eval_retrieval.py",
        "python backend/scripts/eval_retrieval_control.py",
        "python backend/scripts/eval_fusion.py",
        "python backend/scripts/eval_alerts.py",
        "python backend/scripts/measure_latency.py",
        "```",
    ]
    lines += [
        "",
        "## Notes",
        "",
        "- The subset is single-camera and covers ~7.2 min of selected 3.6-s "
        "clips within a 2 h timestamp window; it is not a continuous 2 h "
        "recording.",
        "- The person index stores 1,152-d fused descriptors (768-d visual + "
        "384-d text).  All released person track IDs are -1; track construction "
        "for the person-pair benchmark is documented in the fusion results.",
        "- FAISS indices are flat inner-product indices; no approximate index "
        "is released.",
        "",
    ]
    (RESULTS / "release_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json} and {RESULTS / 'release_manifest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
