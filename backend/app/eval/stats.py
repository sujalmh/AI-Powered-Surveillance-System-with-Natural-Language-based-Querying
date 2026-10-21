"""Statistical protocol: bootstrap CIs, paired tests, Holm, grouped resampling.

Implements paper §Eval protocol without scipy:
- query-level bootstrap (10k resamples) for mean P@5
- paired bootstrap on per-query deltas (adaptive - baseline)
- Wilcoxon signed-rank cross-check (normal approx, pure python)
- Holm correction for 5 categories
- grouped (cluster) bootstrap by video/identity

Deterministic via seed. No DB required.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 0
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI."""
    vals = list(values)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    alpha = 1.0 - ci
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return _mean(vals), lo, hi


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """Paired bootstrap on deltas d = a - b. Returns (mean_d, lo, hi, p_two_sided).

    p_two_sided = 2 * min(P(d_boot <= 0), P(d_boot >= 0)) approximation.
    """
    da = list(a)
    db = list(b)
    assert len(da) == len(db) and len(da) > 0
    n = len(da)
    deltas = [x - y for x, y in zip(da, db)]
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        m = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        boot_means.append(m)
    boot_means.sort()
    alpha = 1.0 - ci
    lo = boot_means[int((alpha / 2) * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot) - 1]
    m = _mean(deltas)
    # two-sided p: proportion of boot means on opposite side of 0 vs observed sign
    if m >= 0:
        p = sum(1 for x in boot_means if x <= 0) / n_boot
    else:
        p = sum(1 for x in boot_means if x >= 0) / n_boot
    p = min(1.0, 2.0 * p)
    return m, lo, hi, p


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value (normal approx, no scipy).

    Zero differences dropped; ties get average ranks. Returns 1.0 if n < 6
    (unreliable) or all differences are zero.
    """
    diffs = [a - b for a, b in zip(x, y) if (a - b) != 0]
    n = len(diffs)
    if n < 6:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return 1.0
    z = (w_plus - mu) / math.sqrt(var)
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


def holm_correction(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm step-down. Returns {name: reject_null}."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    out: Dict[str, bool] = {}
    for i, name in enumerate(names):
        thresh = alpha / (m - i)
        out[name] = pvals[name] <= thresh
    return out


def paired_grouped_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    groups: Sequence[str],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """Cluster-paired bootstrap on per-query deltas ``a - b``.

    Whole groups (video, identity, or a 5-minute bucket) are resampled with
    replacement, preserving the pairing between the two systems.  Returns
    ``(mean_delta, lo, hi, p_two_sided)``.
    """
    da = list(a)
    db = list(b)
    grps = list(groups)
    assert len(da) == len(db) == len(grps) and da
    by_group: Dict[str, List[float]] = {}
    for x, y, g in zip(da, db, grps):
        by_group.setdefault(g, []).append(x - y)
    keys = list(by_group.keys())
    g = len(keys)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pool: List[float] = []
        for _ in range(g):
            pool.extend(by_group[keys[rng.randrange(g)]])
        means.append(_mean(pool))
    means.sort()
    alpha = 1.0 - ci
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    m = _mean([x - y for x, y in zip(da, db)])
    if m >= 0:
        p = sum(1 for x in means if x <= 0) / n_boot
    else:
        p = sum(1 for x in means if x >= 0) / n_boot
    return m, lo, hi, min(1.0, 2.0 * p)


def grouped_bootstrap_ci(
    values: Sequence[float],
    groups: Sequence[str],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Cluster bootstrap: resample whole groups (videos/identities) with replacement.

    values[i] belongs to groups[i]. Each replicate draws len(unique_groups)
    groups with replacement and pools their values.
    """
    vals = list(values)
    grps = list(groups)
    assert len(vals) == len(grps)
    by_group: Dict[str, List[float]] = {}
    for v, g in zip(vals, grps):
        by_group.setdefault(g, []).append(v)
    keys = list(by_group.keys())
    g = len(keys)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pool: List[float] = []
        for _ in range(g):
            pool.extend(by_group[keys[rng.randrange(g)]])
        means.append(_mean(pool))
    means.sort()
    a = 1.0 - ci
    return _mean(vals), means[int((a / 2) * n_boot)], means[int((1 - a / 2) * n_boot) - 1]
