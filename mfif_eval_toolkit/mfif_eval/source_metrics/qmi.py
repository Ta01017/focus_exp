"""qmi <- metricMI(A, B, F, 1), TPAMI MFIF-Metrics.

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np

from .common import entropy, mutual_information


def qmi(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    denom = entropy(a) + entropy(b)
    return float((mutual_information(a, f) + mutual_information(b, f)) / denom) if denom > 0 else 0.0
