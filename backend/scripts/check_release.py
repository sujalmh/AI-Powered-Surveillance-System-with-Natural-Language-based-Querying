"""Release manifest / integrity check.

This script does not touch MongoDB or the network.  It inventories the
released single-camera subset, the two FAISS indices, the benchmark files, and
the installed package versions, then writes a machine-readable manifest under
``results/``.

It also *verifies* the two delivery bundles:

* the supplementary release (``dist/indicon2026_supplementary.zip``) must
  contain the provider-agnostic LLM client, the end-to-end evaluation script
  and the matching tests, must not contain a secret or a regenerable parse
  cache, and must carry the measured results that exist on disk;
* the submission bundle (``INDICON2026_submit/INDICON2026_850.pdf`` and
  ``INDICON2026_850_source.zip``) must be an 8-page camera-ready with a clean
  LaTeX log, and the source bundle must compile standalone into the same
  8-page PDF.

The standalone compile happens in a scratch directory below the system
temporary directory -- never inside the repository.

Run:
    python backend/scripts/check_release.py
    python backend/scripts/check_release.py --strict     # missing artefacts fail too
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMIT = ROOT / "INDICON2026_submit"
DIST = ROOT / "dist"
BUNDLE = DIST / "indicon2026_supplementary.zip"

# The LaTeX helpers live next to this script and are shared with the
# packaging script so that both apply exactly the same rules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_submission import (  # noqa: E402
    CLASS_NAME,
    EXPECTED_PAGES,
    PAPER_BIB,
    PAPER_TEX,
    README_NAME,
    check_latex_log,
    compile_paper,
    find_tool,
    pdf_page_count,
)

# Files the release bundle has to carry.
BUNDLE_REQUIRED = [
    "backend/app/services/llm_client.py",
    "backend/app/services/nl_parser.py",
    "backend/scripts/eval_end_to_end.py",
    "backend/tests/test_llm_client.py",
    "backend/requirements.txt",
    "MANIFEST.txt",
]

# Measured results: shipped when they exist on disk.
BUNDLE_OPTIONAL_RESULTS = [
    "results/end_to_end_results.json",
    "results/end_to_end_tables.md",
    "results/parser_results.json",
    "results/parser_results.md",
]

# Never ship these.
BUNDLE_FORBIDDEN_PARTS = {"parsed_cache", "parser_cache", "__pycache__", ".pytest_cache"}
BUNDLE_FORBIDDEN_SUFFIXES = {".env", ".key", ".pem", ".pfx", ".p12", ".jks", ".keystore"}
BUNDLE_FORBIDDEN_NAMES = {
    ".env",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secret.txt",
}
BUNDLE_FORBIDDEN_PREFIXES = ("secret", "secrets", "credential", "credentials", ".env")


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


# ---------------------------------------------------------------------------
# bundle verification
# ---------------------------------------------------------------------------


def _check(name: str, status: str, detail: str) -> Dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def verify_release_bundle(bundle: Path) -> List[Dict[str, str]]:
    """Check the supplementary release archive."""
    checks: List[Dict[str, str]] = []
    if not bundle.is_file():
        checks.append(
            _check(
                "release archive present",
                "skip",
                f"{bundle} not built yet (run backend/scripts/package_release.py)",
            )
        )
        return checks

    checks.append(
        _check(
            "release archive present",
            "pass",
            f"{bundle.name} ({bundle.stat().st_size/1e6:.2f} MB, sha256 {sha256(bundle)[:16]})",
        )
    )
    with zipfile.ZipFile(bundle) as z:
        names = set(z.namelist())

    missing = [rel for rel in BUNDLE_REQUIRED if rel not in names]
    if missing:
        checks.append(
            _check(
                "release archive contents",
                "fail",
                "missing: " + ", ".join(missing),
            )
        )
    else:
        checks.append(
            _check(
                "release archive contents",
                "pass",
                f"{len(BUNDLE_REQUIRED)} required entries present (llm_client.py, eval_end_to_end.py, test_llm_client.py, ...)",
            )
        )

    # secrets
    offenders: List[str] = []
    for name in sorted(names):
        base = name.rsplit("/", 1)[-1]
        lowered = base.lower()
        if lowered.endswith(tuple(BUNDLE_FORBIDDEN_SUFFIXES)):
            offenders.append(name)
        elif lowered in BUNDLE_FORBIDDEN_NAMES:
            offenders.append(name)
        elif lowered.startswith(BUNDLE_FORBIDDEN_PREFIXES):
            offenders.append(name)
        elif "backend/.env" in name:
            offenders.append(name)
    if offenders:
        checks.append(
            _check(
                "no secrets in the release",
                "fail",
                "secret-like entries: " + ", ".join(offenders),
            )
        )
    else:
        checks.append(
            _check(
                "no secrets in the release",
                "pass",
                "no *.env/*.key/*.pem/credentials*/secrets* entries",
            )
        )

    # regenerable caches
    cached = [name for name in sorted(names) if set(name.split("/")) & BUNDLE_FORBIDDEN_PARTS]
    if cached:
        checks.append(
            _check(
                "no parse caches in the release",
                "fail",
                "regenerable cache entries: " + ", ".join(cached[:10]),
            )
        )
    else:
        checks.append(
            _check(
                "no parse caches in the release",
                "pass",
                "no parsed_cache/parser_cache entries",
            )
        )

    # measured results
    missing_results: List[str] = []
    absent_results: List[str] = []
    for rel in BUNDLE_OPTIONAL_RESULTS:
        if not (ROOT / rel).is_file():
            absent_results.append(rel)
        elif rel not in names:
            missing_results.append(rel)
    if missing_results:
        checks.append(
            _check(
                "measured results shipped",
                "fail",
                "present on disk but absent from the archive: " + ", ".join(missing_results),
            )
        )
    else:
        detail = "all measured results present"
        if absent_results:
            detail += "; not produced yet: " + ", ".join(absent_results)
        checks.append(_check("measured results shipped", "pass", detail))

    return checks


def verify_submission_bundle(
    submit_dir: Path, folder_zip: Path, work_dir: Path, compile_source: bool
) -> List[Dict[str, str]]:
    """Check the camera-ready PDF and the source bundle."""
    checks: List[Dict[str, str]] = []

    pdf = submit_dir / "INDICON2026_850.pdf"
    source_zip = submit_dir / "INDICON2026_850_source.zip"

    # 1. the camera-ready PDF is 8 pages
    if not pdf.is_file():
        checks.append(_check("camera-ready PDF", "skip", f"{pdf.name} not built yet"))
    else:
        pages = pdf_page_count(pdf)
        if pages is None:
            checks.append(_check("camera-ready PDF", "fail", "the page count could not be determined"))
        elif pages != EXPECTED_PAGES:
            checks.append(
                _check("camera-ready PDF", "fail", f"{pages} pages, expected {EXPECTED_PAGES}")
            )
        else:
            checks.append(
                _check(
                    "camera-ready PDF",
                    "pass",
                    f"{pages} pages ({pdf.stat().st_size/1e6:.2f} MB, sha256 {sha256(pdf)[:16]})",
                )
            )

    # 2. the source bundle carries the current sources
    if not source_zip.is_file():
        checks.append(_check("source bundle", "skip", f"{source_zip.name} not built yet"))
        return checks
    with zipfile.ZipFile(source_zip) as z:
        entries = set(z.namelist())
        wanted = {PAPER_TEX, PAPER_BIB, CLASS_NAME, README_NAME}
        absent = sorted(wanted - entries)
        if absent:
            checks.append(
                _check("source bundle", "fail", "missing entries: " + ", ".join(absent))
            )
        else:
            checks.append(
                _check(
                    "source bundle",
                    "pass",
                    f"{len(entries)} entries ({PAPER_TEX}, {PAPER_BIB}, {CLASS_NAME}, {README_NAME})",
                )
            )
        tex_in_zip = z.read(PAPER_TEX)
    reference = submit_dir / PAPER_TEX
    if not reference.is_file():
        reference = SUBMIT / PAPER_TEX
    if not reference.is_file():
        checks.append(
            _check(
                "source bundle carries the current paper",
                "skip",
                f"{PAPER_TEX} is not available to compare against",
            )
        )
    else:
        current_tex = reference.read_bytes()
        if tex_in_zip != current_tex:
            checks.append(
                _check(
                    "source bundle carries the current paper",
                    "fail",
                    f"{PAPER_TEX} inside {source_zip.name} differs from {reference}",
                )
            )
        else:
            checks.append(
                _check(
                    "source bundle carries the current paper",
                    "pass",
                    f"{PAPER_TEX} is byte-identical to {reference.name}",
                )
            )

    # 3. the repository log, when it is around, is clean
    log = submit_dir / "INDICON2026Conference.log"
    if not log.is_file():
        checks.append(
            _check(
                "LaTeX log clean",
                "skip",
                "INDICON2026_submit/INDICON2026Conference.log not present "
                "(it is git-ignored; the standalone compile below checks the same rules)",
            )
        )
    else:
        errors, _warnings, pages = check_latex_log(log.read_text(encoding="utf-8", errors="replace"))
        if errors:
            checks.append(_check("LaTeX log clean", "fail", "; ".join(errors[:5])))
        else:
            checks.append(_check("LaTeX log clean", "pass", f"no errors, {pages} pages reported"))

    # 4. the folder-level package
    if not folder_zip.is_file():
        checks.append(_check("folder package", "skip", "INDICON2026_submit.zip not built yet"))
    else:
        with zipfile.ZipFile(folder_zip) as z:
            entries = set(z.namelist())
        wanted = {
            f"INDICON2026_submit/{pdf.name}",
            f"INDICON2026_submit/{source_zip.name}",
            f"INDICON2026_submit/{PAPER_TEX}",
            f"INDICON2026_submit/{PAPER_BIB}",
        }
        absent = sorted(wanted - entries)
        if absent:
            checks.append(
                _check("folder package", "fail", "missing entries: " + ", ".join(absent))
            )
        else:
            checks.append(
                _check(
                    "folder package",
                    "pass",
                    f"INDICON2026_submit.zip carries the PDF, the source bundle and the sources "
                    f"({folder_zip.stat().st_size/1e6:.2f} MB)",
                )
            )

    # 5. the source bundle compiles standalone
    if not compile_source:
        checks.append(_check("source bundle compiles standalone", "skip", "--no-compile was given"))
        return checks

    root_resolved = ROOT.resolve()
    if work_dir.resolve() == root_resolved or root_resolved in work_dir.resolve().parents:
        raise SystemExit(
            f"error: the scratch directory {work_dir} is inside the repository; "
            f"pick a directory outside it"
        )
    src = work_dir / "src"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as z:
        z.extractall(src)

    try:
        pdflatex = find_tool("pdflatex")
        bibtex = find_tool("bibtex")
        built_pdf, built_log, _transcript = compile_paper(src, pdflatex, bibtex)
    except SystemExit as exc:
        checks.append(
            _check("source bundle compiles standalone", "fail", f"the LaTeX build failed: {exc}")
        )
        return checks

    errors, _warnings, pages = check_latex_log(
        built_log.read_text(encoding="utf-8", errors="replace")
    )
    actual = pdf_page_count(built_pdf)
    if actual is None:
        errors.append("the compiled PDF has no readable page count")
    elif actual != EXPECTED_PAGES:
        errors.append(f"the compiled PDF has {actual} pages, expected {EXPECTED_PAGES}")
    if errors:
        checks.append(
            _check("source bundle compiles standalone", "fail", "; ".join(errors[:5]))
        )
    else:
        checks.append(
            _check(
                "source bundle compiles standalone",
                "pass",
                f"rebuilt from {source_zip.name} into a clean {actual}-page PDF (scratch: {work_dir})",
            )
        )
    return checks


def print_checks(checks: List[Dict[str, str]]) -> None:
    print("bundle checks:")
    for check in checks:
        mark = {"pass": "ok  ", "fail": "FAIL", "skip": "skip"}[check["status"]]
        print(f"  [{mark}] {check['name']}: {check['detail']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Inventory the released subset and verify the two INDICON 2026 delivery bundles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--bundle",
        default=str(BUNDLE),
        help="supplementary release archive to verify (default: dist/indicon2026_supplementary.zip)",
    )
    ap.add_argument(
        "--submit-dir",
        default=str(SUBMIT),
        help="folder holding the camera-ready PDF and the source bundle (default: INDICON2026_submit)",
    )
    ap.add_argument(
        "--folder-zip",
        default=str(ROOT / "INDICON2026_submit.zip"),
        help="folder-level submission package (default: INDICON2026_submit.zip in the repository root)",
    )
    ap.add_argument(
        "--manifest-dir",
        default=str(RESULTS),
        help="directory the release manifest is written to (default: results)",
    )
    ap.add_argument(
        "--work-dir",
        default=str(Path(tempfile.gettempdir()) / "opencode" / "check_release"),
        help="scratch directory for the standalone LaTeX build; must be outside the repository "
        "(default: <temp>/opencode/check_release)",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="do not rebuild the source bundle (skips the standalone compile check)",
    )
    ap.add_argument(
        "--no-bundle-check",
        action="store_true",
        help="only write the inventory manifest, do not verify the bundles",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat skipped checks (missing artefacts) as failures",
    )
    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="keep the scratch directory after the standalone compile",
    )
    args = ap.parse_args(argv)

    manifest_dir = Path(args.manifest_dir).expanduser()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).expanduser()

    checks: List[Dict[str, str]] = []
    if not args.no_bundle_check:
        checks += verify_release_bundle(Path(args.bundle).expanduser())
        checks += verify_submission_bundle(
            Path(args.submit_dir).expanduser(),
            Path(args.folder_zip).expanduser(),
            work_dir,
            compile_source=not args.no_compile,
        )

    manifest_dir.mkdir(parents=True, exist_ok=True)
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

    failed = [c for c in checks if c["status"] == "fail"]
    skipped = [c for c in checks if c["status"] == "skip"]

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
        "release_bundle": {
            "archive": str(Path(args.bundle).expanduser()),
            "checks": checks,
            "failed": len(failed),
            "skipped": len(skipped),
            "passed": len(checks) - len(failed) - len(skipped),
            "ok": not failed and not (skipped and args.strict),
        },
    }

    out_json = manifest_dir / "release_manifest.json"
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
        "## Delivery bundles",
        "",
    ]
    if checks:
        for check in checks:
            mark = {"pass": "ok", "fail": "FAIL", "skip": "skipped"}[check["status"]]
            lines.append(f"- **{mark}** -- {check['name']}: {check['detail']}")
    else:
        lines.append("- bundle verification was skipped (`--no-bundle-check`)")
    lines += [
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$env:PYTHONIOENCODING='utf-8'",
        "python backend/scripts/check_release.py",
        "python backend/benchmarks/build_subset.py",
        "python backend/benchmarks/build_subset_control.py",
        "python backend/scripts/eval_retrieval_control.py",
        "python backend/scripts/eval_alpha_competitors.py",
        "python backend/scripts/eval_fusion.py",
        "python backend/scripts/eval_alerts.py",
        "python backend/scripts/eval_end_to_end.py",
        "python backend/scripts/measure_latency.py",
        "python backend/scripts/eval_retrieval.py   # raw smoke test, not protocol-matched",
        "python backend/scripts/package_submission.py  # camera-ready PDF + source bundle",
        "python backend/scripts/package_release.py    # build the supplementary archive",
        "```",
        "",
        "The C1 number quoted in the paper comes from `eval_retrieval_control.py`;",
        "`eval_retrieval.py` is the raw released benchmark (a code-path smoke test",
        "whose count/action queries are not protocol-matched to the deployment study).",
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
    (manifest_dir / "release_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json} and {manifest_dir / 'release_manifest.md'}")
    if checks:
        print_checks(checks)
    if not args.keep_work and work_dir.exists() and not args.no_compile:
        shutil.rmtree(work_dir, ignore_errors=True)
    if failed:
        sys.stdout.flush()
        print(f"\n{len(failed)} bundle check(s) FAILED", file=sys.stderr)
        for check in failed:
            print(f"  - {check['name']}: {check['detail']}", file=sys.stderr)
        return 1
    if skipped and args.strict:
        sys.stdout.flush()
        print(f"\n{len(skipped)} bundle check(s) skipped (--strict)", file=sys.stderr)
        for check in skipped:
            print(f"  - {check['name']}: {check['detail']}", file=sys.stderr)
        return 1
    if skipped:
        print(f"\n{len(skipped)} bundle check(s) skipped (not a failure without --strict)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
