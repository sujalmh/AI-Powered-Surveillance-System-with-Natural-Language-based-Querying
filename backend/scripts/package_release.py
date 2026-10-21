"""Package the supplementary release.

Builds a zip from an explicit allow-list so that the archive contains the
code, the benchmarks, the measured results and the paper source -- and only
those.  The repository also holds superseded drafts of the manuscript
(``research_paper*.tex``, an older PDF, slide decks, model weights and older
archives); shipping them would expose conflicting versions of the paper, so
they are excluded by construction rather than by remembering to delete them.

The release is meant to be reproducible from a clean checkout:

* every file is taken from an allow-list, never "everything except ...",
* regenerable build artefacts (the LLM parse caches
  ``backend/benchmarks/parsed_cache/`` and ``backend/benchmarks/parser_cache/``)
  are excluded -- they are outputs of a run, not inputs of the release,
* secrets are excluded and the collected set is audited before the archive is
  written, so a stray ``.env``/``*.key`` aborts the build instead of leaking,
* setting ``SOURCE_DATE_EPOCH`` (or ``--source-date-epoch``) makes the archive
  byte-for-byte reproducible.

Run:
    python backend/scripts/package_release.py
    python backend/scripts/package_release.py --with-data      # + clips and FAISS indices
    python backend/scripts/package_release.py --out-dir C:\\tmp\\out
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
ARCHIVE_NAME = "indicon2026_supplementary.zip"

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

# Files that must be in the release: the packaging fails loudly when one of
# them disappears instead of shipping a bundle that cannot run.
REQUIRED_FILES: List[str] = [
    "backend/app/services/llm_client.py",
    "backend/app/services/nl_parser.py",
    "backend/scripts/eval_end_to_end.py",
    "backend/scripts/eval_retrieval_control.py",
    "backend/scripts/package_release.py",
    "backend/scripts/package_submission.py",
    "backend/scripts/check_release.py",
    "backend/tests/test_llm_client.py",
    "backend/requirements.txt",
]

# Measured results that are shipped when they exist.  They are produced by
# separate evaluation runs, so a missing file is not an error -- but it is
# reported, and the file is never silently dropped once it exists.
OPTIONAL_RESULT_FILES: List[str] = [
    "results/end_to_end_results.json",
    "results/end_to_end_tables.md",
    "results/parser_results.json",
    "results/parser_results.md",
]

# Optional bulk data (19 MB): the released clips and the two FAISS indices.
DATA_INCLUDE: List[Tuple[str, str]] = [
    ("data/clips", "**/*"),
    ("data/index", "**/*"),
]

# Regenerable build artefacts: LLM parse outputs, tool caches, virtualenvs.
# ``parsed_cache``/``parser_cache`` hold the cached LLM parses of the released
# benchmark utterances -- they are the *output* of a parse run (and contain
# provider-specific payloads), not source, so they never ship.
EXCLUDE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "parsed_cache",
    "parser_cache",
    "venv",
    ".venv",
    "node_modules",
    ".git",
}

# Secrets and private keys.
EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".onnx",
    ".env",
    ".key",
    ".pem",
    ".pfx",
    ".p12",
    ".jks",
    ".keystore",
    ".crt",
    ".cer",
    ".der",
    ".asc",
    ".gpg",
    ".ppk",
}
EXCLUDE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "secret.txt",
    "secrets.json",
    "credentials.json",
    "credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "service-account.json",
}
EXCLUDE_PREFIXES = ("secret", "secrets", "credential", "credentials", ".env")
EXCLUDE_PATHS = {"backend/.env"}

# Editor/backup leftovers.
EXCLUDE_SUFFIXES |= {".bak", ".tmp", ".swp", ".orig", ".rej"}


def is_secret(path: Path) -> bool:
    """True when the file looks like a secret, a key or a credential store."""
    name = path.name.lower()
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if name in EXCLUDE_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


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
            if any(path.name.lower().startswith(p) for p in EXCLUDE_PREFIXES):
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


def audit(files: List[Path]) -> List[str]:
    """Return the list of problems in a collected file set (empty == clean)."""
    problems: List[str] = []
    root = ROOT.resolve()
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDE_PATHS:
            problems.append(f"{rel}: excluded path was collected")
        if set(path.parts) & EXCLUDE_PARTS:
            problems.append(f"{rel}: cache directory was collected")
        if is_secret(path):
            problems.append(f"{rel}: looks like a secret and must not be shipped")
        try:
            path.resolve().relative_to(root)
        except ValueError:
            problems.append(f"{rel}: resolves outside the repository")
    for required in REQUIRED_FILES:
        if not (ROOT / required).is_file():
            problems.append(f"{required}: required release file is missing")
    return problems


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Package the INDICON 2026 supplementary release (code, benchmarks, results, paper source).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The archive name is always 'indicon2026_supplementary.zip' unless --out says "
            "otherwise; --with-data adds data/clips and data/index but does not rename the "
            "archive, so pass --out to keep the two variants apart."
        ),
    )
    ap.add_argument(
        "--out",
        default=None,
        help=f"path of the release archive (default: {DIST / ARCHIVE_NAME})",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help=f"directory the archive is written to (default: {DIST}); the file name stays {ARCHIVE_NAME}",
    )
    ap.add_argument(
        "--with-data",
        action="store_true",
        help="also include data/clips and data/index (19 MB)",
    )
    ap.add_argument(
        "--source-date-epoch",
        type=int,
        default=None,
        help="timestamp used for the manifest and the zip entries; defaults to $SOURCE_DATE_EPOCH, "
        "otherwise the current time.  Set it to make the archive reproducible.",
    )
    args = ap.parse_args()

    if args.out and args.out_dir:
        ap.error("--out and --out-dir are mutually exclusive")
    if args.out:
        out = Path(args.out).expanduser()
    elif args.out_dir:
        out = Path(args.out_dir).expanduser() / ARCHIVE_NAME
    else:
        out = DIST / ARCHIVE_NAME

    files = collect(INCLUDE, INCLUDE_FILES)
    if args.with_data:
        files += collect(DATA_INCLUDE, [])

    problems = audit(files)
    if problems:
        print("release packaging REJECTED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    epoch = args.source_date_epoch
    if epoch is None:
        env_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        epoch = int(env_epoch) if env_epoch else None
    if epoch is None:
        generated = datetime.now(timezone.utc).isoformat()
        zip_date: Tuple[int, int, int, int, int, int] = time.localtime()[:6]
    else:
        generated = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
        zip_date = time.gmtime(epoch)[:6]
    if zip_date[0] < 1980:  # the zip format cannot store timestamps before 1980
        zip_date = (1980, 1, 1, 0, 0, 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "INDICON 2026 supplementary release",
        f"generated: {generated}",
        f"files: {len(files)}",
        "",
        "sha256  path",
    ]
    for path in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix()):
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"{sha256(path)}  {rel}")
    manifest = "\n".join(lines) + "\n"

    def add(z: zipfile.ZipFile, arcname: str, data: bytes) -> None:
        info = zipfile.ZipInfo(arcname, date_time=zip_date)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, data)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix()):
            add(z, path.relative_to(ROOT).as_posix(), path.read_bytes())
        add(z, "MANIFEST.txt", manifest.encode("utf-8"))

    size = out.stat().st_size
    print(f"wrote {out} ({size/1e6:.1f} MB, {len(files)} files)")
    print("top-level entries:")
    for name in sorted({p.relative_to(ROOT).parts[0] for p in files}):
        n = sum(1 for p in files if p.relative_to(ROOT).parts[0] == name)
        print(f"  {name}/ ({n} files)")
    present = {p.relative_to(ROOT).as_posix() for p in files}
    print("optional measured results:")
    for rel in OPTIONAL_RESULT_FILES:
        state = "included" if rel in present else "absent (optional)"
        print(f"  - {rel}: {state}")
    excluded = [
        "superseded manuscript drafts (research_paper*.tex, older PDF, slide deck)",
        "model weights (*.pt, backend/intel)",
        "raw recordings and snapshots (data/recordings, data/snapshots)",
        "secrets and credentials (*.env, *.key, *.pem, credentials*, secrets*)",
        "regenerable LLM parse caches (backend/benchmarks/parsed_cache, backend/benchmarks/parser_cache)",
        "python/tool caches (__pycache__, .pytest_cache)",
    ]
    print("deliberately excluded:")
    for item in excluded:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
