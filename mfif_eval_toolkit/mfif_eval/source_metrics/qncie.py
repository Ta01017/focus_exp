"""qncie <- metricWang(A, B, F), TPAMI MFIF-Metrics.

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np

from .common import entropy, mutual_information


def qncie(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    vals = [entropy(a), entropy(b), entropy(f)]
    denom = max(vals) + 1e-12
    return float((mutual_information(a, f) + mutual_information(b, f)) / (2.0 * denom))
