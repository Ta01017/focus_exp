"""qs/qe <- metricPeilla(A, B, F, mode), Objective-evaluation-for-image-fusion.

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .common import local_variance


def _ssim(x: np.ndarray, y: np.ndarray) -> float:
    h, w = x.shape
    win = min(7, h, w)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        mse = np.mean((x - y) ** 2)
        return float(1.0 / (1.0 + mse))
    return float(structural_similarity(x, y, data_range=255.0, win_size=win))


def qs(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    wa = local_variance(a)
    wb = local_variance(b)
    denom = wa + wb + 1e-12
    score = (wa * _ssim(a, f) + wb * _ssim(b, f)) / denom
    return float(np.nanmean(score))


def qe(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    return float(0.5 * (_ssim(a, f) + _ssim(b, f)))
