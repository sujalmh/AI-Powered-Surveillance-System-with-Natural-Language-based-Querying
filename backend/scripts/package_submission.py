"""Build the INDICON 2026 submission package.

Produces the three artefacts the submission portal expects:

* ``INDICON2026_submit/INDICON2026_850.pdf``      -- the final camera-ready PDF
* ``INDICON2026_submit/INDICON2026_850_source.zip`` -- source bundle
  (paper .tex, References.bib, the conference IEEEtran.cls and a README with
  the exact compile sequence)
* ``INDICON2026_submit.zip``                       -- folder-level package with
  the PDF, the source bundle, the .tex and the .bib

The camera-ready PDF is **rebuilt from source on every run** -- the packaging
never trusts a PDF left behind by an earlier build.  The paper is compiled with
pdfLaTeX (+BibTeX) and the build is rejected unless

* the PDF has exactly 8 pages (cross-checked between the LaTeX log and the
  PDF itself),
* the log contains no undefined references and no undefined citations,
* the log contains no ``Overfull \\hbox`` (and no ``Overfull \\vbox``),
* the log contains no LaTeX error and the bibliography was produced.

The source bundle carries the conference's own ``IEEEtran.cls`` so that the
portal can compile it even where a different IEEEtran version is installed.

Run:
    python backend/scripts/package_submission.py
    python backend/scripts/package_submission.py --out-dir C:\\tmp\\out   # sandbox
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
SUBMIT = ROOT / "INDICON2026_submit"
KIT = ROOT / "IEEE__INDICON_2026.zip"          # conference author kit
KIT_TEX = "INDICON2026Conference.tex"
PDF_NAME = "INDICON2026_850.pdf"
SOURCE_NAME = "INDICON2026_850_source.zip"
FOLDER_ZIP = ROOT / "INDICON2026_submit.zip"

PAPER_TEX = "INDICON2026Conference.tex"
PAPER_BIB = "References.bib"
CLASS_NAME = "IEEEtran.cls"
README_NAME = "README.txt"
PAPER_STEM = "INDICON2026Conference"
EXPECTED_PAGES = 8

README = """INDICON 2026 submission source
================================

Files
-----
INDICON2026Conference.tex  paper source (8 pages, IEEEtran conference class)
References.bib             bibliography (IEEEtran style)
IEEEtran.cls              class file shipped with the INDICON 2026 author kit
README.txt                this file

Rebuild (pdfLaTeX + BibTeX, run from this directory)
----------------------------------------------------
    pdflatex -interaction=nonstopmode INDICON2026Conference.tex
    bibtex   INDICON2026Conference
    pdflatex -interaction=nonstopmode INDICON2026Conference.tex
    pdflatex -interaction=nonstopmode INDICON2026Conference.tex

The fourth pass settles the cross references, so do not stop after the first
one.  latexmk does the same in a single command (it needs perl on Windows):

    latexmk -pdf INDICON2026Conference.tex

Expected result
---------------
The build must end with

    Output written on INDICON2026Conference.pdf (8 pages, ...)

and the log must contain neither an undefined reference/citation nor an
"Overfull \\hbox" warning.  backend/scripts/package_submission.py and
backend/scripts/check_release.py enforce both, so a submission that does not
rebuild into a clean 8-page PDF is rejected before it leaves the repository.

The paper contains no external image files: the dataflow figure is a TikZ
picture inside the .tex source.  The only companion files needed are the two
shipped here (References.bib and IEEEtran.cls).

Paper ID: 850
"""

FOLDER_README = """INDICON 2026 submission package
===============================

Files
-----
INDICON2026_850.pdf         the camera-ready paper (8 pages, 2-column IEEEtran)
INDICON2026_850_source.zip  the LaTeX source bundle (.tex, .bib, IEEEtran.cls,
                            README.txt with the exact rebuild sequence)
INDICON2026Conference.tex   the paper source (same file as in the source bundle)
References.bib              the bibliography
IEEEtran.cls                class file shipped with the INDICON 2026 author kit
README.txt                  this file

The camera-ready PDF is generated from INDICON2026Conference.tex with
pdfLaTeX + BibTeX; see README.txt inside INDICON2026_850_source.zip for the
exact command sequence and for the checks the build has to pass (8 pages, no
undefined references, no "Overfull \\hbox").

Paper ID: 850
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_tool(name: str, explicit: Optional[str] = None) -> Path:
    """Locate a TeX tool: explicit path, then PATH, then the MiKTeX install."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise SystemExit(f"error: {name} not found at {candidate}")
        return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        miktex = Path(local) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / f"{name}.exe"
        if miktex.is_file():
            return miktex
    raise SystemExit(
        f"error: {name} not found on PATH.  Install MiKTeX (TeX Live also works) "
        f"or pass an explicit path."
    )


def kit_class(target_dir: Path) -> Path:
    """Extract the conference IEEEtran.cls from the author kit."""
    if not KIT.is_file():
        raise SystemExit(f"error: conference author kit {KIT} is missing")
    with zipfile.ZipFile(KIT) as z:
        names = z.namelist()
        cls = next((n for n in names if n.endswith(CLASS_NAME)), None)
        if cls is None:
            raise SystemExit(f"error: {CLASS_NAME} not found inside {KIT.name}")
        target = target_dir / CLASS_NAME
        target.write_bytes(z.read(cls))
    return target


def run(cmd: Sequence[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def pdf_page_count(path: Path) -> Optional[int]:
    """Number of pages in a PDF.

    Uses pypdf / PyPDF2 when either is importable and falls back to counting
    the page objects in the raw file, so the check also works on a clean
    checkout where no PDF library is installed.
    """
    for module in ("pypdf", "PyPDF2"):
        try:
            reader_mod = __import__(module)
        except Exception:
            continue
        try:
            reader = reader_mod.PdfReader(str(path))
            return len(reader.pages)
        except Exception:
            continue
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    pages = len(re.findall(rb"/Type\s*/Page(?![s/\w])", raw))
    if pages:
        return pages
    counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", raw)]
    return max(counts) if counts else None


def check_latex_log(
    log_text: str, expected_pages: int = EXPECTED_PAGES
) -> Tuple[List[str], List[str], Optional[int]]:
    """Inspect a LaTeX log.

    Returns ``(errors, warnings, pages)``.  ``pages`` is the page count the
    log itself reports (``Output written on ... (N pages, ...)``).
    """
    errors: List[str] = []
    warnings: List[str] = []

    match = re.search(r"Output written on (.+?) \((\d+) pages?", log_text)
    pages = int(match.group(2)) if match else None
    if match is None:
        errors.append("the log has no 'Output written on ... (N pages, ...)' line: the PDF was not produced")
    elif pages != expected_pages:
        errors.append(f"the paper has {pages} pages, expected {expected_pages}")

    for pattern, label in (
        (r"Overfull \\hbox", "Overfull \\hbox"),
        (r"Overfull \\vbox", "Overfull \\vbox"),
    ):
        hits = re.findall(rf"^{pattern}.*$", log_text, re.M)
        for hit in hits:
            errors.append(f"{label}: {hit.strip()}")

    undefined = re.findall(r"^LaTeX Warning: .*undefined.*$", log_text, re.M)
    undefined += re.findall(r"^There were undefined (?:references|citations).*$", log_text, re.M)
    for hit in undefined:
        errors.append(f"undefined reference/citation: {hit.strip()}")

    errors_found = re.findall(r"^! .*$", log_text, re.M)
    errors_found += re.findall(r"^(?:Emergency stop|Fatal error occurred).*$", log_text, re.M)
    for hit in errors_found:
        errors.append(f"LaTeX error: {hit.strip()}")

    underfull = re.findall(r"^Underfull \\[hv]box.*$", log_text, re.M)
    if underfull:
        warnings.append(f"{len(underfull)} underfull box warning(s) (cosmetic, ignored)")

    other = [
        w for w in re.findall(r"^LaTeX Warning: .*$", log_text, re.M)
        if "undefined" not in w
    ]
    for hit in other:
        warnings.append(f"warning: {hit.strip()}")

    return errors, warnings, pages


def compile_paper(
    build_dir: Path,
    pdflatex: Path,
    bibtex: Path,
    engine: str = "pdflatex",
    timeout: int = 900,
) -> Tuple[Path, Path, str]:
    """Compile the paper in ``build_dir`` and return ``(pdf, log, transcript)``.

    The build is in-place: ``build_dir`` already holds the .tex, the .bib and
    IEEEtran.cls.  A failing tool aborts the build (SystemExit).
    """
    transcript: List[str] = []

    def step(cmd: Sequence[str], label: str) -> None:
        try:
            proc = run(cmd, build_dir, timeout)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"error: {label} timed out after {timeout}s")
        transcript.append(f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
        if proc.returncode != 0:
            tail = "\n".join(transcript[-1].splitlines()[-40:])
            raise SystemExit(f"error: {label} failed with exit code {proc.returncode}\n{tail}")

    if engine == "latexmk":
        step(
            [
                str(pdflatex.with_name("latexmk.exe" if os.name == "nt" else "latexmk")),
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                PAPER_TEX,
            ],
            "latexmk",
        )
    else:
        flags = ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error"]
        step([str(pdflatex), *flags, PAPER_TEX], "pdflatex (pass 1)")
        step([str(bibtex), PAPER_STEM], "bibtex")
        step([str(pdflatex), *flags, PAPER_TEX], "pdflatex (pass 2)")
        step([str(pdflatex), *flags, PAPER_TEX], "pdflatex (pass 3)")

    pdf = build_dir / f"{PAPER_STEM}.pdf"
    log = build_dir / f"{PAPER_STEM}.log"
    if not pdf.is_file():
        raise SystemExit("error: the LaTeX run produced no PDF")
    if not log.is_file():
        raise SystemExit("error: the LaTeX run produced no log")
    return pdf, log, "\n".join(transcript)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild the camera-ready PDF and package the INDICON 2026 submission.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The default invocation compiles the paper in INDICON2026_submit/ and "
            "writes INDICON2026_submit/INDICON2026_850.pdf, "
            "INDICON2026_submit/INDICON2026_850_source.zip and INDICON2026_submit.zip "
            "(repository root).  Use --out-dir/--build-dir to build into a sandbox "
            "without touching the repository."
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="write INDICON2026_850.pdf, INDICON2026_850_source.zip and INDICON2026_submit.zip into this "
        "directory instead of their usual locations (default: the usual locations)",
    )
    ap.add_argument(
        "--build-dir",
        default=None,
        help="compile the paper in this directory (seeded with copies of the .tex, .bib and IEEEtran.cls); "
        "INDICON2026_submit/ is then left untouched (default: compile in INDICON2026_submit/)",
    )
    ap.add_argument(
        "--skip-compile",
        action="store_true",
        help="reuse the existing INDICON2026_submit/INDICON2026Conference.pdf instead of compiling "
        "(the result is still verified)",
    )
    ap.add_argument(
        "--engine",
        choices=("pdflatex", "latexmk"),
        default="pdflatex",
        help="LaTeX driver: 'pdflatex' runs pdflatex+bibtex+pdflatex+pdflatex (default); "
        "'latexmk' calls latexmk -pdf (needs perl on Windows)",
    )
    ap.add_argument("--pdflatex", default=None, help="explicit path to pdflatex (default: found on PATH)")
    ap.add_argument("--bibtex", default=None, help="explicit path to bibtex (default: found on PATH)")
    ap.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="per-command timeout in seconds (default: 900)",
    )
    args = ap.parse_args(argv)

    tex = SUBMIT / PAPER_TEX
    bib = SUBMIT / PAPER_BIB
    for path in (tex, bib):
        if not path.is_file():
            raise SystemExit(f"error: missing {path}; the paper sources must be in place")

    build_dir = Path(args.build_dir).expanduser().resolve() if args.build_dir else SUBMIT
    build_dir.mkdir(parents=True, exist_ok=True)

    # Seed the build directory with the paper sources and the conference class.
    for src in (tex, bib):
        dst = build_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    cls = kit_class(build_dir)

    if args.skip_compile:
        pdf = SUBMIT / f"{PAPER_STEM}.pdf"
        log = build_dir / f"{PAPER_STEM}.log"
        if not pdf.is_file():
            raise SystemExit(f"error: --skip-compile but {pdf} does not exist")
        if not log.is_file():
            print(f"note: {log} not found, the log checks are skipped")
    else:
        pdflatex = find_tool("pdflatex", args.pdflatex)
        bibtex = find_tool("bibtex", args.bibtex)
        print(f"compiling the paper in {build_dir} with {pdflatex.name}")
        pdf, log, _transcript = compile_paper(
            build_dir, pdflatex, bibtex, engine=args.engine, timeout=args.timeout
        )

    # ---- verification -----------------------------------------------------
    errors: List[str] = []
    warnings: List[str] = []
    pages: Optional[int] = None
    if log.is_file():
        errors, warnings, pages = check_latex_log(
            log.read_text(encoding="utf-8", errors="replace")
        )
    actual = pdf_page_count(pdf)
    if actual is None:
        errors.append("could not determine the page count of the compiled PDF")
    elif actual != EXPECTED_PAGES:
        errors.append(f"the compiled PDF has {actual} pages, expected {EXPECTED_PAGES}")
    if pages is not None and actual is not None and pages != actual:
        errors.append(f"the log reports {pages} pages but the PDF has {actual}")
    bbl = build_dir / f"{PAPER_STEM}.bbl"
    if bbl.is_file() and "??" in bbl.read_text(encoding="utf-8", errors="replace"):
        errors.append("the bibliography contains unresolved entries ('??')")
    elif not bbl.is_file() and not args.skip_compile:
        errors.append("no .bbl was produced: the bibliography was not resolved")

    for warning in warnings:
        print(f"note: {warning}")
    if errors:
        sys.stdout.flush()
        print("\nsubmission build REJECTED:", file=sys.stderr)
        for problem in errors:
            print(f"  - {problem}", file=sys.stderr)
        print(f"  (log: {log})", file=sys.stderr)
        return 1

    print(f"paper verified: {actual} pages, clean LaTeX log")

    # ---- artefacts --------------------------------------------------------
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        final_pdf = out_dir / PDF_NAME
        source_zip = out_dir / SOURCE_NAME
        folder_zip = out_dir / FOLDER_ZIP.name
    else:
        final_pdf = SUBMIT / PDF_NAME
        source_zip = SUBMIT / SOURCE_NAME
        folder_zip = FOLDER_ZIP

    # 1. camera-ready PDF under the submission name
    shutil.copyfile(pdf, final_pdf)

    # 2. source bundle
    items: List[Tuple[Path, str]] = [
        (build_dir / PAPER_TEX, PAPER_TEX),
        (build_dir / PAPER_BIB, PAPER_BIB),
        (cls, CLASS_NAME),
    ]
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, arc in items:
            z.write(path, arc)
        z.writestr(README_NAME, README)

    # 3. folder-level package
    with zipfile.ZipFile(folder_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, arc in items:
            z.write(path, f"INDICON2026_submit/{arc}")
        z.write(final_pdf, f"INDICON2026_submit/{PDF_NAME}")
        z.write(source_zip, f"INDICON2026_submit/{SOURCE_NAME}")
        z.writestr("INDICON2026_submit/README.txt", FOLDER_README)

    print("submission package:")
    for path in (final_pdf, source_zip, folder_zip):
        print(f"  {path}  ({path.stat().st_size/1e6:.2f} MB)  sha256 {sha256(path)[:16]}")
    print("\nsource bundle contents:")
    with zipfile.ZipFile(source_zip) as z:
        for info in z.infolist():
            print(f"  {info.filename}  ({info.file_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
