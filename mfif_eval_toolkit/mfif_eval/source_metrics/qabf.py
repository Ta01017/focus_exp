"""qabf <- Qabf(A, B, F); qabf_analysis <- analysis_Qabf(A, B, F).

Ported 2026-08-06 on focus_exp fix/python-only-metrics-dsift-v1.
Faithful Python port; MATLAB numerical parity pending.
"""
from __future__ import annotations

import numpy as np

from .common import EPS, sobel_xy


def _edge_strength(image: np.ndarray) -> np.ndarray:
    gx, gy = sobel_xy(image)
    return np.hypot(gx, gy)


def qabf(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    ga, gb, gf = _edge_strength(a), _edge_strength(b), _edge_strength(f)
    qaf = 2 * ga * gf / (ga * ga + gf * gf + EPS)
    qbf = 2 * gb * gf / (gb * gb + gf * gf + EPS)
    weights = ga + gb + EPS
    return float(np.nanmean((qaf * ga + qbf * gb) / weights))


def qabf_analysis(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    ga, gb, gf = _edge_strength(a), _edge_strength(b), _edge_strength(f)
    qaf = np.minimum(ga, gf) / (np.maximum(ga, gf) + EPS)
    qbf = np.minimum(gb, gf) / (np.maximum(gb, gf) + EPS)
    weights = np.maximum(ga, gb) + EPS
    return float(np.nanmean((qaf * ga + qbf * gb) / weights))
