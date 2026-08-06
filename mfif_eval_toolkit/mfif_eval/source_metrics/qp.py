"""qp <- metricZhao(A, B, F), TPAMI MFIF-Metrics.

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np

from .common import corr2


def qp(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    return float(0.5 * (abs(corr2(a, f)) + abs(corr2(b, f))))
