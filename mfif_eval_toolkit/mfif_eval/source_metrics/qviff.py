"""qviff <- VIFF_Public(A, B, F), TPAMI MFIF-Metrics.

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .common import EPS, fspecial_gaussian


def _vif_pair(ref: np.ndarray, dist: np.ndarray) -> float:
    sigma_nsq = 2.0
    num = 0.0
    den = 0.0
    r = ref.astype(np.float64)
    d = dist.astype(np.float64)
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        sd = n / 5.0
        win = fspecial_gaussian(n, sd)
        if scale > 1:
            r = ndimage.convolve(r, win, mode="constant")[::2, ::2]
            d = ndimage.convolve(d, win, mode="constant")[::2, ::2]
        mu1 = ndimage.convolve(r, win, mode="constant")
        mu2 = ndimage.convolve(d, win, mode="constant")
        sigma1_sq = np.maximum(ndimage.convolve(r * r, win, mode="constant") - mu1 * mu1, 0)
        sigma2_sq = np.maximum(ndimage.convolve(d * d, win, mode="constant") - mu2 * mu2, 0)
        sigma12 = ndimage.convolve(r * d, win, mode="constant") - mu1 * mu2
        g = sigma12 / (sigma1_sq + EPS)
        sv_sq = sigma2_sq - g * sigma12
        g[sigma1_sq < EPS] = 0
        sv_sq[sigma1_sq < EPS] = sigma2_sq[sigma1_sq < EPS]
        sigma1_sq[sigma1_sq < EPS] = 0
        g[sv_sq <= EPS] = 0
        sv_sq[sv_sq <= EPS] = EPS
        num += float(np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq))))
        den += float(np.sum(np.log10(1 + sigma1_sq / sigma_nsq)))
    return num / den if den > 0 else 1.0


def qviff(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    return float(0.5 * (_vif_pair(a, f) + _vif_pair(b, f)))
