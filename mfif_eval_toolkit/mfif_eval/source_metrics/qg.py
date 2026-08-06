"""qg <- metricXydeas(A, B, F), TPAMI MFIF-Metrics.

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np

from .common import EPS, sobel_xy


def qg(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    ga = np.hypot(*sobel_xy(a))
    gb = np.hypot(*sobel_xy(b))
    gf = np.hypot(*sobel_xy(f))
    qaf = np.minimum(ga, gf) / (np.maximum(ga, gf) + EPS)
    qbf = np.minimum(gb, gf) / (np.maximum(gb, gf) + EPS)
    weights = ga + gb + EPS
    return float(np.nanmean((qaf * ga + qbf * gb) / weights))
