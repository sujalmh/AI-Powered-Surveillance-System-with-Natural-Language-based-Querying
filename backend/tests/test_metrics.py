import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.eval.metrics import (
    exact_match,
    f1_at_10_paper,
    macro_mean,
    mean_absolute_error,
    precision_at_k,
    recall_at_k,
)


def test_p5_r10_f1():
    ret = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    rel = ["a", "c", "z"]
    assert precision_at_k(ret, rel, 5) == 0.4  # a,c in top5
    assert abs(recall_at_k(ret, rel, 10) - 2 / 3) < 1e-9
    p, r = 0.4, 2 / 3
    assert abs(f1_at_10_paper(ret, rel) - 2 * p * r / (p + r)) < 1e-9


def test_macro_mean():
    assert macro_mean({"a": 1.0, "b": 0.0}) == 0.5


def test_counting_em_mae():
    assert exact_match(5, 5) == 1.0
    assert exact_match(4, 5) == 0.0
    assert mean_absolute_error([5, 5], [5, 3]) == 1.0
