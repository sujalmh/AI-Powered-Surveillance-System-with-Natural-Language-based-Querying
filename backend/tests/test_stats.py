import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.eval.stats import (
    bootstrap_ci,
    format_p,
    format_sd,
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


# ---------------------------------------------------------------------------
# Rendering of resolution-limited numbers
#
# A bootstrap p-value is estimated from a finite number of resamples, so
# it can only take the values 0, 2/n, 4/n, ...  A p-value of exactly zero
# therefore means "below the resolution floor", not "impossible", and a
# dispersion that rounds to zero is not zero.  Neither may be printed as a
# bare 0, because that reads like a broken measurement.
# ---------------------------------------------------------------------------

def test_a_zero_p_value_is_reported_as_below_the_resolution_floor():
    assert format_p(0.0, 10000) == "< 0.0002"
    assert format_p(0.0, 3000) == "< 0.0007"


def test_a_p_value_below_the_floor_is_also_reported_as_the_floor():
    assert format_p(1e-9, 10000) == "< 0.0002"


def test_a_p_value_above_the_floor_is_printed_with_four_decimals():
    assert format_p(0.04, 10000) == "0.0400"
    assert format_p(0.7973, 3000) == "0.7973"
    assert format_p(1.0, 3000) == "1.0000"


def test_a_p_value_is_never_printed_as_a_bare_zero():
    for n_boot in (1000, 3000, 10000):
        for p in (0.0, 1e-12, 2.0 / n_boot / 2):
            assert format_p(p, n_boot) != "0.0000"


def test_a_missing_p_value_is_reported_as_not_applicable():
    assert format_p(None, 10000) == "n/a"


def test_a_zero_standard_deviation_is_zero_but_a_tiny_one_is_not():
    assert format_sd(0.0) == "0.000"
    # 0.00044 is the random-projection baseline's spread: it is not zero,
    # but it rounds to zero at three decimals.
    assert format_sd(0.00044) == "< 0.001"
    # The threshold is the value that would round to 0.000.
    assert format_sd(0.00049) == "< 0.001"
    assert format_sd(0.00051) == "0.001"


def test_a_visible_standard_deviation_is_printed_with_three_decimals():
    assert format_sd(0.039) == "0.039"
    assert format_sd(0.0241) == "0.024"
