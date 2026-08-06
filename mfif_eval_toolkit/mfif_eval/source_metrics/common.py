"""MATLAB-compatible helpers for source-image fusion metrics.

Ported on 2026-08-06 for focus_exp branch fix/python-only-metrics-dsift-v1.
The original reference functions are the MATLAB MFIF-Metrics and
Objective-evaluation-for-image-fusion implementations. MATLAB numerical
parity is pending unless recorded otherwise in run metadata.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


EPS = np.finfo(np.float64).eps


def rgb2gray_uint8_matlab_compatible(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            return arr.copy()
        return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxW or HxWx3 image, got shape {arr.shape}")
    rgb = arr[..., :3].astype(np.float64)
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def as_gray_double255(image: np.ndarray) -> np.ndarray:
    return rgb2gray_uint8_matlab_compatible(image).astype(np.float64)


def im2double(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    return arr.astype(np.float64)


def fspecial_gaussian(size: int | tuple[int, int], sigma: float) -> np.ndarray:
    if isinstance(size, int):
        rows = cols = size
    else:
        rows, cols = size
    y, x = np.mgrid[-(rows // 2): rows // 2 + 1, -(cols // 2): cols // 2 + 1]
    y = y[:rows, :cols]
    x = x[:rows, :cols]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    total = kernel.sum()
    return kernel / total if total else kernel


def conv2_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return signal.convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)


def filter2_same(kernel: np.ndarray, image: np.ndarray) -> np.ndarray:
    return signal.correlate2d(image, kernel, mode="same", boundary="fill", fillvalue=0)


def corr2(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64) - np.nanmean(a)
    y = np.asarray(b, dtype=np.float64) - np.nanmean(b)
    denom = np.sqrt(np.nansum(x * x) * np.nansum(y * y))
    return float(np.nansum(x * y) / denom) if denom > 0 else 0.0


def gradient2(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(np.asarray(image, dtype=np.float64))
    return gx, gy


def sobel_xy(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=np.float64)
    gx = ndimage.sobel(arr, axis=1, mode="constant")
    gy = ndimage.sobel(arr, axis=0, mode="constant")
    return gx, gy


def histogram256(image: np.ndarray) -> np.ndarray:
    values = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return np.bincount(values.ravel(), minlength=256).astype(np.float64)


def joint_histogram256(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.clip(np.rint(a), 0, 255).astype(np.uint8).ravel()
    y = np.clip(np.rint(b), 0, 255).astype(np.uint8).ravel()
    return np.bincount(x.astype(np.int64) * 256 + y.astype(np.int64), minlength=65536).reshape(256, 256).astype(np.float64)


def entropy_from_hist(hist: np.ndarray) -> float:
    total = float(np.sum(hist))
    if total <= 0:
        return 0.0
    p = hist.ravel() / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def entropy(image: np.ndarray) -> float:
    return entropy_from_hist(histogram256(image))


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    hxy = joint_histogram256(a, b)
    return entropy_from_hist(hxy.sum(axis=1)) + entropy_from_hist(hxy.sum(axis=0)) - entropy_from_hist(hxy)


def nanmean_scalar(value) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def ensure_same_shape(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> None:
    shapes = [np.asarray(x).shape[:2] for x in (a, b, f)]
    if len(set(shapes)) != 1:
        raise ValueError(f"source metric image shapes differ: {shapes}")


def spatial_frequency(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float64)
    m, n = arr.shape
    if m == 0 or n == 0:
        return 0.0
    df_c = arr[1:, :] - arr[:-1, :]
    df_r = arr[:, 1:] - arr[:, :-1]
    return float(np.sqrt(np.sum(df_c * df_c) / (m * n) + np.sum(df_r * df_r) / (m * n)))


def local_variance(image: np.ndarray, size: int = 7) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    mean = ndimage.uniform_filter(arr, size=size, mode="constant")
    mean2 = ndimage.uniform_filter(arr * arr, size=size, mode="constant")
    return np.maximum(mean2 - mean * mean, 0.0)
