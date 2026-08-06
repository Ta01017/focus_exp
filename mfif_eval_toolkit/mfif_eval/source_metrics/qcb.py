"""qcb <- metricChenBlum(A, B, F), TPAMI MFIF-Metrics.

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np

from .common import EPS, fspecial_gaussian, filter2_same


def _contrast(image: np.ndarray) -> np.ndarray:
    low = filter2_same(fspecial_gaussian(11, 1.5), image)
    return np.abs(image - low)


def qcb(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    ca, cb, cf = _contrast(a), _contrast(b), _contrast(f)
    qaf = 2 * ca * cf / (ca * ca + cf * cf + EPS)
    qbf = 2 * cb * cf / (cb * cb + cf * cf + EPS)
    weights = ca + cb + EPS
    return float(np.nanmean((qaf * ca + qbf * cb) / weights))
