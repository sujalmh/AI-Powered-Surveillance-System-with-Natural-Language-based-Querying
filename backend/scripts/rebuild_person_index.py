"""Rebuild the person FAISS index without gender terms (revision E5).

Loads ``data/index/faiss_person/{index.bin,meta.json}``, recovers the two
unit branches from the stored 1152-d weighted concat
(``[0.6*v_unit || 0.4*t_unit] / norm``, so each block is re-normalized to
unit length), strips gender terms from ``attribute_text``, re-embeds the
stripped text with MiniLM (``all-MiniLM-L12-v2``, 384-d), re-fuses with the
same weights from ``backend.app.config.settings``
(``FUSION_VISUAL_WEIGHT``/``FUSION_TEXT_WEIGHT``, default 0.6/0.4),
L2-normalizes, and writes ``data/index/faiss_person_nogender/``.

Reconstruction is verified on a sample before writing: re-fusing the
recovered unit branches must reproduce the stored vector within 1e-4
(the same tolerance ``eval_fusion.py`` uses).

Run (repository root)::

    python backend/scripts/rebuild_person_index.py --verify-only
    python backend/scripts/rebuild_person_index.py --limit 64
    python backend/scripts/rebuild_person_index.py

The full rebuild re-embeds 1,778 short texts on CPU (no network: the model
must already be in the HF cache).  Outputs never overwrite the released
``results/fusion_tables.md``; the verification report goes to
``results/safety_run/person_rebuild_report.json`` by default.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.core.privacy import strip_gender_terms  # noqa: E402
from backend.app.eval.subset import (  # noqa: E402
    PERSON_INDEX,
    PERSON_META,
    PERSON_TEXT_DIM,
    PERSON_VISUAL_DIM,
)

_GENDER_CHECK_RE = re.compile(
    r"\b(male|female|man|woman|men|women|boy|girl|masculine|feminine|gender)\b",
    re.IGNORECASE,
)

RECON_TOL = 1e-4


def load_source(src_dir: Path):
    import faiss

    index = faiss.read_index(str(src_dir / "index.bin"))
    if index.d != PERSON_VISUAL_DIM + PERSON_TEXT_DIM:
        raise ValueError(f"unexpected dim {index.d}")
    meta = json.loads((src_dir / "meta.json").read_text(encoding="utf-8"))
    vectors = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    return vectors, meta


def split_unit_branches(vectors: np.ndarray):
    v = vectors[:, :PERSON_VISUAL_DIM]
    t = vectors[:, PERSON_VISUAL_DIM:]
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    return (v / np.maximum(vn, 1e-9)).astype(np.float32), (
        t / np.maximum(tn, 1e-9)
    ).astype(np.float32)


def fuse(visual_unit: np.ndarray, text_unit: np.ndarray, wv: float, wt: float):
    from backend.app.services.fusion import MultimodalFusion

    fuse_fn = MultimodalFusion(wv, wt)
    out = []
    degenerate = []
    for i, (v, t) in enumerate(zip(visual_unit, text_unit)):
        try:
            out.append(
                fuse_fn.fuse(
                    np.array(v, dtype=np.float32, copy=True),
                    np.array(t, dtype=np.float32, copy=True),
                )
            )
        except ValueError:
            # Degenerate stored row (both blocks ~zero, e.g. rows 1/143 of
            # the released index): keep a zero vector and report it.
            degenerate.append(int(i))
            out.append(np.zeros((PERSON_VISUAL_DIM + PERSON_TEXT_DIM,), dtype=np.float32))
    return np.vstack(out).astype(np.float32), degenerate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=str, default=str(PERSON_INDEX.parent))
    ap.add_argument("--dst-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N rows (0 = all)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check reconstruction + gender counts, write nothing")
    ap.add_argument("--report", type=str, default="results/safety_run/person_rebuild_report.json")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    default_dst = src_dir.parent / "faiss_person_nogender"
    dst_dir = Path(args.dst_dir) if args.dst_dir else default_dst
    wv = float(getattr(settings, "FUSION_VISUAL_WEIGHT", 0.6))
    wt = float(getattr(settings, "FUSION_TEXT_WEIGHT", 0.4))

    vectors, meta = load_source(src_dir)
    n_total = vectors.shape[0]
    n = n_total if not args.limit else min(int(args.limit), n_total)
    vectors = vectors[:n]
    meta = meta[:n]

    v_unit, t_unit = split_unit_branches(vectors)
    recon, recon_degenerate = fuse(v_unit, t_unit, wv, wt)
    ok_rows = np.array([i not in set(recon_degenerate) for i in range(n)])
    max_diff = float(np.max(np.abs(recon[ok_rows] - vectors[ok_rows]))) if ok_rows.any() else 0.0
    recon_ok = bool(max_diff < RECON_TOL)

    texts = [str(m.get("attribute_text") or "") for m in meta]
    gender_before = sum(1 for t in texts if _GENDER_CHECK_RE.search(t))
    stripped = [strip_gender_terms(t) for t in texts]
    gender_after = sum(1 for t in stripped if _GENDER_CHECK_RE.search(t))
    n_changed = sum(1 for a, b in zip(texts, stripped) if a != b)

    report: Dict[str, Any] = {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "n_total": n_total,
        "n_processed": n,
        "sample_only": bool(args.limit),
        "weights": {"visual": wv, "text": wt},
        "reconstruction_max_abs_diff": max_diff,
        "reconstruction_ok_tol_1e_4": bool(recon_ok),
        "reconstruction_degenerate_rows": [int(i) for i in recon_degenerate],
        "gender_rows_before": int(gender_before),
        "gender_rows_after_strip": int(gender_after),
        "texts_changed_by_strip": int(n_changed),
        "verify_only": bool(args.verify_only),
    }

    if not recon_ok:
        raise RuntimeError(f"reconstruction mismatch: {max_diff} >= {RECON_TOL}")

    new_vectors = None
    drift: Dict[str, Any] = {}
    if not args.verify_only:
        from backend.app.services.attribute_encoder import AttributeEncoder

        encoder = AttributeEncoder()
        new_text = np.asarray(encoder.encode_texts(stripped), dtype=np.float32)
        new_vectors, new_degenerate = fuse(v_unit, new_text, wv, wt)
        report["new_degenerate_rows"] = [int(i) for i in new_degenerate]

        cos = np.clip(
            np.sum(new_vectors.astype(np.float64) * vectors.astype(np.float64), axis=1),
            -1.0, 1.0,
        )
        same_mask = np.array([a == b for a, b in zip(texts, stripped)])
        drift = {
            "mean_cosine_old_vs_new": float(np.mean(cos)),
            "min_cosine_old_vs_new": float(np.min(cos)),
            "mean_cosine_unchanged_text_rows": float(np.mean(cos[same_mask])) if same_mask.any() else None,
            "mean_cosine_changed_text_rows": float(np.mean(cos[~same_mask])) if (~same_mask).any() else None,
        }
        report["drift"] = drift

        import faiss

        dst_dir.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(PERSON_VISUAL_DIM + PERSON_TEXT_DIM)
        index.add(np.ascontiguousarray(new_vectors.astype(np.float32)))
        faiss.write_index(index, str(dst_dir / "index.bin"))
        new_meta: List[Dict[str, Any]] = []
        for i, m in enumerate(meta):
            row = dict(m)
            row["faiss_id"] = int(i)
            row["attribute_text"] = stripped[i]
            new_meta.append(row)
        (dst_dir / "meta.json").write_text(
            json.dumps(new_meta, ensure_ascii=False), encoding="utf-8"
        )
        report["wrote"] = {
            "index": str(dst_dir / "index.bin"),
            "meta": str(dst_dir / "meta.json"),
            "ntotal": int(index.ntotal),
        }

    out_path = ROOT / args.report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"rows: {n}/{n_total}  recon_max_diff={max_diff:.2e} ok={recon_ok}")
    print(f"gender rows: {gender_before} -> {gender_after} (changed texts: {n_changed})")
    if drift:
        print(f"drift mean_cos={drift['mean_cosine_old_vs_new']:.4f} "
              f"min_cos={drift['min_cosine_old_vs_new']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
