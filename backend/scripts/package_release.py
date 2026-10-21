"""Package the supplementary release.

Builds a zip from an explicit allow-list so that the archive contains the
code, the benchmarks, the measured results and the paper source -- and only
those.  The repository also holds superseded drafts of the manuscript
(``research_paper*.tex``, an older PDF, slide decks, model weights and older
archives); shipping them would expose conflicting versions of the paper, so
they are excluded by construction rather than by remembering to delete them.

Run:
    python backend/scripts/package_release.py
    python backend/scripts/package_release.py --with-data      # + clips and FAISS indices
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"

# (relative directory, glob) entries that are part of the release.
INCLUDE: List[Tuple[str, str]] = [
    ("backend/app", "**/*.py"),
    ("backend/scripts", "*.py"),
    ("backend/benchmarks", "*"),
    ("backend/tests", "*.py"),
    ("results", "*"),
]

# Single files that are part of the release.
INCLUDE_FILES: List[str] = [
    "README.md",
    "VERIFICATION.md",
    "RESPONSE_TO_REVIEWERS.md",
    "requirements_utf8.txt",
    "backend/requirements.txt",
    "INDICON2026_submit/INDICON2026Conference.tex",
    "INDICON2026_submit/INDICON2026Conference.pdf",
    "INDICON2026_submit/References.bib",
]

# Optional bulk data (19 MB): the released clips and the two FAISS indices.
DATA_INCLUDE: List[Tuple[str, str]] = [
    ("data/clips", "**/*"),
    ("data/index", "**/*"),
]

EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "venv", ".venv"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".onnx", ".env"}
EXCLUDE_NAMES = {".env", "secret.txt"}


def collect(include: Iterable[Tuple[str, str]], include_files: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for base, pattern in include:
        directory = ROOT / base
        if not directory.exists():
            continue
        for path in sorted(directory.glob(pattern)):
            if not path.is_file():
                continue
            if set(path.parts) & EXCLUDE_PARTS:
                continue
            if path.suffix.lower() in EXCLUDE_SUFFIXES or path.name in EXCLUDE_NAMES:
                continue
            files.append(path)
    for rel in include_files:
        path = ROOT / rel
        if path.is_file():
            files.append(path)
    seen = set()
    unique = []
    for path in files:
        key = path.relative_to(ROOT).as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DIST / "indicon2026_supplementary.zip"))
    ap.add_argument("--with-data", action="store_true",
                    help="also include data/clips and data/index (19 MB)")
    args = ap.parse_args()

    files = collect(INCLUDE, INCLUDE_FILES)
    if args.with_data:
        files += collect(DATA_INCLUDE, [])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "INDICON 2026 supplementary release",
        f"generated: {datetime.now(timezone.utc).isoformat()}",
        f"files: {len(files)}",
        "",
        "sha256  path",
    ]
    for path in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix()):
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"{sha256(path)}  {rel}")
    manifest = "\n".join(lines) + "\n"

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix()):
            z.write(path, path.relative_to(ROOT).as_posix())
        z.writestr("MANIFEST.txt", manifest)

    size = out.stat().st_size
    print(f"wrote {out} ({size/1e6:.1f} MB, {len(files)} files)")
    print("top-level entries:")
    for name in sorted({p.relative_to(ROOT).parts[0] for p in files}):
        n = sum(1 for p in files if p.relative_to(ROOT).parts[0] == name)
        print(f"  {name}/ ({n} files)")
    excluded = [
        "superseded manuscript drafts (research_paper*.tex, older PDF, slide deck)",
        "model weights (*.pt, backend/intel)",
        "raw recordings and snapshots (data/recordings, data/snapshots)",
        "secrets and caches (*.env, __pycache__)",
    ]
    print("deliberately excluded:")
    for item in excluded:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
