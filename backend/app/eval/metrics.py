"""Retrieval metrics for the INDICON paper.

Preserves paper definitions exactly:
- P@5: |retrieved@5 ∩ relevant| / 5
- R@10: |retrieved@10 ∩ relevant| / |relevant|
- F1@10 = H-mean(P@5, R@10)  (paper-specific, NOT standard P@10-based F1)
- macro-mean = mean of the 5 category means (n=30 each)
- Counting: EM (exact-match) + MAE complement supporting-clip P@5.

Pure numpy/python, no DB/FAISS required.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence
import math


def precision_at_k(retrieved: Sequence, relevant: Iterable, k: int = 5) -> float:
    rel = set(relevant)
    if k <= 0:
        return 0.0
    topk = list(retrieved)[:k]
    if not topk:
        return 0.0
    return sum(1 for r in topk if r in rel) / float(k)


def recall_at_k(retrieved: Sequence, relevant: Iterable, k: int = 10) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    topk = set(list(retrieved)[:k])
    return len(topk & rel) / float(len(rel))


def f1_at_10_paper(retrieved: Sequence, relevant: Iterable) -> float:
    """Paper F1@10 = H-mean(P@5, R@10). Returns 0 if both are 0."""
    p = precision_at_k(retrieved, relevant, 5)
    r = recall_at_k(retrieved, relevant, 10)
    if p + r <= 0:
        return 0.0
    return 2 * p * r / (p + r)


def macro_mean(category_means: Dict[str, float]) -> float:
    vals = list(category_means.values())
    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))


def exact_match(predicted: int, truth: int) -> float:
    return 1.0 if int(predicted) == int(truth) else 0.0


def mean_absolute_error(preds: Sequence[int], truths: Sequence[int]) -> float:
    preds, truths = list(preds), list(truths)
    if not preds:
        return 0.0
    return sum(abs(int(p) - int(t)) for p, t in zip(preds, truths)) / float(len(preds))


def per_query_scores(
    queries: List[Dict], retrieved_map: Dict[str, List[str]]
) -> List[Dict]:
    """Score a list of query dicts {qid, category, relevant:[...]}.

    retrieved_map: qid -> ranked clip-id list.
    Returns [{qid, category, p5, r10, f1}].
    """
    out = []
    for q in queries:
        qid = q["qid"]
        rel = q.get("relevant", [])
        ret = retrieved_map.get(qid, [])
        out.append(
            {
                "qid": qid,
                "category": q.get("category", "?"),
                "p5": precision_at_k(ret, rel, 5),
                "r10": recall_at_k(ret, rel, 10),
                "f1": f1_at_10_paper(ret, rel),
            }
        )
    return out


def category_means(scored: List[Dict], field: str = "p5") -> Dict[str, float]:
    from collections import defaultdict

    acc: Dict[str, List[float]] = defaultdict(list)
    for s in scored:
        acc[s["category"]].append(float(s[field]))
    return {c: (sum(v) / len(v) if v else 0.0) for c, v in acc.items()}
