"""Per-branch retrieval confidence from retrieved score lists (C1').

Pure functions (lists only, no FAISS/Mongo) so unit tests run anywhere.

For one query, each branch produces a score per candidate clip.  Three
cheap, label-free confidence signals are extracted from each branch's score
list:

* ``top1_margin``: gap between the best and second-best score.  A branch
  that clearly prefers one clip is more decisive than one with a flat top.
* ``topk_concentration``: share of the total score mass in the top-k.
  A peaked distribution is more confident than a uniform one.
* ``overlap_at_k``: fraction of the two branches' top-k id sets that agree.
  This is a *shared* diagnostic (it cannot favour either branch, so it
  cancels out of the alpha ratio); it is reported, not used in the weight.

``confidence_alpha`` turns the two per-branch confidences into a weight::

    alpha_conf = cs / (cs + ce)   (0.5 when both are ~0)

``combined_alpha`` blends the metadata prior with the score-driven weight::

    alpha = 0.5 * prior + 0.5 * alpha_conf

The metadata-only path of ``compute_adaptive_alpha`` is unchanged and
remains the fallback whenever score vectors are unavailable.
"""
from __future__ import annotations

from typing import Optional, Sequence

CONF_K_DEFAULT = 5
CONF_BLEND_DEFAULT = 0.5
_MARGIN_W = 0.5
_CONC_W = 0.5


def _sorted_desc(scores: Sequence[float]) -> list:
    return sorted((float(s) for s in scores), reverse=True)


def top1_margin(scores: Sequence[float]) -> float:
    """Gap between the best and second-best score (>= 0)."""
    vals = _sorted_desc(scores)
    if len(vals) < 2:
        return 0.0
    return max(0.0, vals[0] - vals[1])


def topk_concentration(scores: Sequence[float], k: int = CONF_K_DEFAULT) -> float:
    """Share of total score mass held by the top-k (in [0, 1])."""
    vals = [float(s) for s in scores]
    if not vals:
        return 0.0
    # Shift to non-negative so mass shares are meaningful.
    lo = min(vals)
    shifted = [v - lo for v in vals]
    total = sum(shifted)
    if total <= 1e-12:
        return 0.0
    top = sorted(shifted, reverse=True)[: max(1, k)]
    return float(sum(top) / total)


def overlap_at_k(
    ids_a: Sequence[str], ids_b: Sequence[str], k: int = CONF_K_DEFAULT
) -> float:
    """|top-k(A) ∩ top-k(B)| / k.  Shared diagnostic, not a branch weight."""
    sa = set(list(ids_a)[: max(1, k)])
    sb = set(list(ids_b)[: max(1, k)])
    return float(len(sa & sb) / max(1, k))


def branch_confidence(
    scores: Sequence[float],
    k: int = CONF_K_DEFAULT,
    margin_weight: float = _MARGIN_W,
    conc_weight: float = _CONC_W,
) -> float:
    """Per-branch confidence in [0, 1]: weighted mean of margin and conc.

    Callers should pass per-query min-max normalised scores so the margin
    is on a comparable [0, 1] scale across queries and branches.
    """
    margin = top1_margin(scores)
    conc = topk_concentration(scores, k=k)
    return float(margin_weight * margin + conc_weight * conc)


def confidence_alpha(
    struct_scores: Sequence[float],
    sem_scores: Sequence[float],
    k: int = CONF_K_DEFAULT,
) -> float:
    """Pure score-driven structured weight in [0, 1]."""
    cs = branch_confidence(struct_scores, k=k)
    ce = branch_confidence(sem_scores, k=k)
    if cs + ce <= 1e-12:
        return 0.5
    return float(cs / (cs + ce))


def combined_alpha(
    prior_alpha: float,
    struct_scores: Sequence[float],
    sem_scores: Sequence[float],
    k: int = CONF_K_DEFAULT,
    blend: float = CONF_BLEND_DEFAULT,
) -> float:
    """Half prior, half score-driven (blend tunable, default 0.5)."""
    conf = confidence_alpha(struct_scores, sem_scores, k=k)
    alpha = float(blend * float(prior_alpha) + (1.0 - blend) * conf)
    return max(0.0, min(1.0, alpha))


def confidence_diagnostics(
    struct_scores: Sequence[float],
    sem_scores: Sequence[float],
    struct_ids: Optional[Sequence[str]] = None,
    sem_ids: Optional[Sequence[str]] = None,
    k: int = CONF_K_DEFAULT,
) -> dict:
    """Label-free per-query diagnostics (margins, concentrations, overlap)."""
    out = {
        "struct_margin": top1_margin(struct_scores),
        "sem_margin": top1_margin(sem_scores),
        "struct_conc": topk_concentration(struct_scores, k=k),
        "sem_conc": topk_concentration(sem_scores, k=k),
        "struct_conf": branch_confidence(struct_scores, k=k),
        "sem_conf": branch_confidence(sem_scores, k=k),
    }
    if struct_ids is not None and sem_ids is not None:
        out["overlap_at_k"] = overlap_at_k(struct_ids, sem_ids, k=k)
    return out
