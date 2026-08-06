"""qsf <- metricZheng(A, B, F), Objective-evaluation-for-image-fusion.

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np

from .common import spatial_frequency


def qsf(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    sf_a, sf_b, sf_f = spatial_frequency(a), spatial_frequency(b), spatial_frequency(f)
    target = max(sf_a, sf_b)
    return float(sf_f / target) if target > 0 else 1.0
