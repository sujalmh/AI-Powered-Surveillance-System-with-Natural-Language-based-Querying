"""Build the INDICON 2026 submission package.

Produces the three artefacts the submission portal expects:

* ``INDICON2026_submit/INDICON2026_850.pdf``      -- the final camera-ready PDF
* ``INDICON2026_submit/INDICON2026_850_source.zip`` -- source bundle
  (paper .tex, References.bib, the conference IEEEtran.cls and a README with
  the exact compile sequence)
* ``INDICON2026_submit.zip``                       -- folder-level package with
  the PDF, the source bundle, the .tex and the .bib

The source bundle carries the conference's own ``IEEEtran.cls`` so that the
portal can compile it even where a different IEEEtran version is installed.

Run:
    python backend/scripts/package_submission.py
"""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
SUBMIT = ROOT / "INDICON2026_submit"
KIT = ROOT / "IEEE__INDICON_2026.zip"          # conference author kit
KIT_TEX = "INDICON2026Conference.tex"
PDF_NAME = "INDICON2026_850.pdf"
SOURCE_NAME = "INDICON2026_850_source.zip"
FOLDER_ZIP = ROOT / "INDICON2026_submit.zip"

README = """INDICON 2026 submission source
================================

Files
-----
INDICON2026Conference.tex  paper source (8 pages, IEEEtran conference class)
References.bib             bibliography (15 entries, IEEEtran style)
IEEEtran.cls              class file shipped with the INDICON 2026 author kit

Compile (pdfLaTeX + BibTeX, run from this directory)
----------------------------------------------------
    pdflatex -interaction=nonstopmode INDICON2026Conference.tex
    bibtex   INDICON2026Conference
    pdflatex -interaction=nonstopmode INDICON2026Conference.tex
    pdflatex -interaction=nonstopmode INDICON2026Conference.tex

The paper contains no external image files: the dataflow figure is a TikZ
picture inside the .tex source.

Paper ID: 850
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kit_class() -> Path:
    """Extract the conference IEEEtran.cls from the author kit."""
    import zipfile as zf

    with zf.ZipFile(KIT) as z:
        names = z.namelist()
        cls = next(n for n in names if n.endswith("IEEEtran.cls"))
        target = SUBMIT / "IEEEtran.cls"
        target.write_bytes(z.read(cls))
        return target


def main() -> int:
    pdf = SUBMIT / "INDICON2026Conference.pdf"
    tex = SUBMIT / "INDICON2026Conference.tex"
    bib = SUBMIT / "References.bib"
    for path in (pdf, tex, bib):
        if not path.exists():
            raise SystemExit(f"missing {path}; build the paper first")

    # 1. camera-ready PDF under the submission name
    final_pdf = SUBMIT / PDF_NAME
    shutil.copyfile(pdf, final_pdf)

    # 2. source bundle
    cls = kit_class()
    source_zip = SUBMIT / SOURCE_NAME
    items: List[Tuple[Path, str]] = [
        (tex, "INDICON2026Conference.tex"),
        (bib, "References.bib"),
        (cls, "IEEEtran.cls"),
    ]
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, arc in items:
            z.write(path, arc)
        z.writestr("README.txt", README)

    # 3. folder-level package
    with zipfile.ZipFile(FOLDER_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, arc in items:
            z.write(path, f"INDICON2026_submit/{arc}")
        z.write(final_pdf, f"INDICON2026_submit/{PDF_NAME}")
        z.write(source_zip, f"INDICON2026_submit/{SOURCE_NAME}")
        z.writestr("INDICON2026_submit/README.txt", README)

    print("submission package:")
    for path in (final_pdf, source_zip, FOLDER_ZIP):
        print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size/1e6:.2f} MB)  sha256 {sha256(path)[:16]}")
    print("\nsource bundle contents:")
    with zipfile.ZipFile(source_zip) as z:
        for info in z.infolist():
            print(f"  {info.filename}  ({info.file_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
