import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.eval.stats import (
    bootstrap_ci,
    grouped_bootstrap_ci,
    holm_correction,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci([0.5] * 20 + [0.9] * 10, n_boot=500, seed=0)
    assert lo <= m <= hi


def test_paired_and_holm():
    a = [0.8] * 30
    b = [0.6] * 30
    m, lo, hi, p = paired_bootstrap_ci(a, b, n_boot=500, seed=0)
    assert m > 0.15 and p < 0.05
    assert holm_correction({"find": 0.04, "track": 0.07})["find"] is False  # Holm: neither survives at n=5? check logic below
    # with only 2 tests Holm keeps find: 0.04 <= 0.05/2? No -> False. Documents marginality.


def test_grouped_matches_overall_mean():
    vals = [1.0, 0.0, 1.0, 0.0]
    m, lo, hi = grouped_bootstrap_ci(vals, ["v1", "v1", "v2", "v2"], n_boot=500, seed=0)
    assert m == 0.5 and lo <= hi


def test_wilcoxon_returns_p():
    p = wilcoxon_signed_rank([0.9] * 10, [0.5] * 10)
    assert 0.0 <= p <= 1.0
