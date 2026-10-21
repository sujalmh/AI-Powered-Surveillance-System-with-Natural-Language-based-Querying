"""Same-space fusion baselines as importable helpers (R1-7).

Novel code backend/app/services/fusion.py::MultimodalFusion is UNTOUCHED.
This module re-exports the fairer variants for eval scripts so the paper can
compare: concat (novel, 1 index) vs RRF (2 indices) vs zero-pad / norm-matched
/ random-proj addition (same-space).
"""
from backend.app.eval.baselines import norm_matched_add, random_proj_add, zero_pad_add

__all__ = ["zero_pad_add", "norm_matched_add", "random_proj_add"]
