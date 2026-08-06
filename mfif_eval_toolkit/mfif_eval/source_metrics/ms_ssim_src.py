"""ms_ssim_src <- analysis_MSSSIM(A, B, F), Objective-evaluation-for-image-fusion.

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .common import local_variance


def ms_ssim_src(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    scores = []
    aa, bb, ff = a.copy(), b.copy(), f.copy()
    for _ in range(4):
        h, w = ff.shape
        win = min(7, h, w)
        if win % 2 == 0:
            win -= 1
        if win < 3:
            break
        sa = structural_similarity(aa, ff, data_range=255.0, win_size=win)
        sb = structural_similarity(bb, ff, data_range=255.0, win_size=win)
        va, vb = np.mean(local_variance(aa)), np.mean(local_variance(bb))
        scores.append((va * sa + vb * sb) / (va + vb + 1e-12))
        if min(h, w) < 16:
            break
        aa, bb, ff = aa[::2, ::2], bb[::2, ::2], ff[::2, ::2]
    return float(np.prod(np.maximum(scores, 0)) ** (1.0 / len(scores))) if scores else 1.0
