"""Python DSIFT-MFIF runtime.

Ported from:
- DSIFT_Fusion.m
- DenseSIFT.m
- DSIFTNormalization.m
- SF.m
- generate_initmap.m
- img_extend.m
- refine_withmatching.m
- refine_withoutmatching.m

Original MATLAB files are retained as reference only, not used at runtime.
Port date: 2026-08-06. Branch: fix/python-only-metrics-dsift-v1.
MATLAB numerical parity: pending.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal
from skimage.morphology import remove_small_objects


def rgb2gray_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    rgb = arr[..., :3].astype(np.float64)
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def img_extend(image: np.ndarray, length: int) -> np.ndarray:
    return np.pad(image.astype(np.float64), ((length, length), (length, length)), mode="constant")


def spatial_frequency(image: np.ndarray) -> float:
    arr = image.astype(np.float64)
    m, n = arr.shape
    dc = arr[1:, :] - arr[:-1, :]
    dr = arr[:, 1:] - arr[:, :-1]
    return float(np.sqrt(np.sum(dc * dc) / (m * n) + np.sum(dr * dr) / (m * n)))


def _gaussian(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)[:size]
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return g / np.sum(g)


def _delta_gaussian(sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    g = _gaussian(4 * int(np.ceil(sigma)) + 1, sigma)
    gy, gx = np.gradient(g)
    gx = gx * 2.0 / (np.sum(np.abs(gx)) + 1e-12)
    gy = gy * 2.0 / (np.sum(np.abs(gy)) + 1e-12)
    return gx, gy


def dense_sift(image: np.ndarray, patch_size: int = 24, grid_spacing: int = 1) -> np.ndarray:
    image = image.astype(np.float64)
    max_value = float(np.max(image))
    if max_value > 0:
        image = image / max_value
    angle_nums = 8
    bin_nums = 4
    alpha = 9
    angles = np.arange(angle_nums, dtype=np.float64) * (2 * np.pi / angle_nums)
    gx_filter, gy_filter = _delta_gaussian(1.0)
    vertical = signal.correlate2d(image, gx_filter, mode="same", boundary="fill", fillvalue=0)
    horizontal = signal.correlate2d(image, gy_filter, mode="same", boundary="fill", fillvalue=0)
    magnitude = np.hypot(vertical, horizontal)
    theta = np.nan_to_num(np.arctan2(horizontal, vertical), nan=0.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    orient = np.empty((*image.shape, angle_nums), dtype=np.float32)
    for idx, angle in enumerate(angles):
        tmp = (cos_t * np.cos(angle) + sin_t * np.sin(angle)) ** alpha
        tmp = tmp * (tmp > 0)
        orient[..., idx] = (tmp * magnitude).astype(np.float32)
    half = patch_size // 2
    sample_resolution = patch_size / bin_nums
    weight = np.abs(np.arange(1, patch_size + 1) - (half - 0.5)) / sample_resolution
    weight = (1 - weight) * (weight <= 1)
    kernel = np.outer(weight, weight)
    for idx in range(angle_nums):
        orient[..., idx] = signal.convolve2d(orient[..., idx], kernel, mode="same", boundary="fill", fillvalue=0)
    h, w = image.shape
    loc_x = np.arange(half, w - half + 2, grid_spacing, dtype=int)
    loc_y = np.arange(half, h - half + 2, grid_spacing, dtype=int)
    sample_edges = np.linspace(1, patch_size + 1, bin_nums + 1)
    sx, sy = np.meshgrid(sample_edges[:-1], sample_edges[:-1])
    offsets_x = (sx.ravel(order="F") - half).astype(int)
    offsets_y = (sy.ravel(order="F") - half).astype(int)
    desc = np.zeros((len(loc_y), len(loc_x), angle_nums * bin_nums * bin_nums), dtype=np.float32)
    offset = 0
    for dx, dy in zip(offsets_x, offsets_y):
        yy = loc_y + dy
        xx = loc_x + dx
        desc[:, :, offset:offset + angle_nums] = orient[np.ix_(yy, xx, np.arange(angle_nums))]
        offset += angle_nums
    return desc


def dsift_normalization(desc: np.ndarray) -> np.ndarray:
    flat = desc.reshape((-1, desc.shape[-1]), order="C")
    norms = np.sqrt(np.sum(flat * flat, axis=1))
    idx = norms > 1
    out = flat.copy()
    if np.any(idx):
        values = out[idx] / norms[idx, None]
        values = np.minimum(values, 0.2)
        norms2 = np.sqrt(np.sum(values * values, axis=1))
        valid = norms2 > 0
        values[valid] = values[valid] / norms2[valid, None]
        out[idx] = values
    return out.reshape(desc.shape)


def generate_initmap(dsift1: np.ndarray, dsift2: np.ndarray, blocksize: int) -> tuple[np.ndarray, np.ndarray]:
    s1 = np.sum(dsift1, axis=2)
    s2 = np.sum(dsift2, axis=2)
    h, w = s1.shape
    score1 = np.zeros((h, w), dtype=np.float64)
    score2 = np.zeros((h, w), dtype=np.float64)
    counts = np.zeros((h, w), dtype=np.float64)
    for x in range(0, w - blocksize + 1):
        for y in range(0, h - blocksize + 1):
            a = np.sum(s1[y:y + blocksize, x:x + blocksize])
            b = np.sum(s2[y:y + blocksize, x:x + blocksize])
            if a > b:
                score1[y:y + blocksize, x:x + blocksize] += 1
            elif b > a:
                score2[y:y + blocksize, x:x + blocksize] += 1
            counts[y:y + blocksize, x:x + blocksize] += 1
    counts[counts < 1] = 1
    init1 = (score1 / counts > 0.99)
    init2 = (score2 / counts > 0.99)
    area = 3600
    temp1 = remove_small_objects(init1, min_size=area, connectivity=2)
    temp2 = remove_small_objects(init2, min_size=area, connectivity=2)
    init1 = ~remove_small_objects(~temp1, min_size=area, connectivity=2)
    init2 = ~remove_small_objects(~temp2, min_size=area, connectivity=2)
    return init1.astype(np.float64), init2.astype(np.float64)


def _argmin_matlab_order(values: np.ndarray) -> tuple[int, int]:
    idx = int(np.argmin(values.ravel(order="F")))
    return np.unravel_index(idx, values.shape, order="F")


def refine_withoutmatching(img1: np.ndarray, img2: np.ndarray, init1: np.ndarray, init2: np.ndarray) -> np.ndarray:
    h, w = img1.shape
    fmap = init1 + 0.5 * (1 - init1 - init2)
    r = 3
    for y in range(h):
        for x in range(w):
            if fmap[y, x] == 0.5 and y > r - 1 and y < h - r and x > r - 1 and x < w - r:
                fm1 = spatial_frequency(img1[y - r:y + r + 1, x - r:x + r + 1])
                fm2 = spatial_frequency(img2[y - r:y + r + 1, x - r:x + r + 1])
                fmap[y, x] = 1 if fm1 > fm2 else 0
    return fmap


def refine_withmatching(img1: np.ndarray, img2: np.ndarray, init1: np.ndarray, init2: np.ndarray, norm1: np.ndarray, norm2: np.ndarray) -> np.ndarray:
    h, w = img1.shape
    fmap = init1 + 0.5 * (1 - init1 - init2)
    r = 3
    for y in range(h):
        for x in range(w):
            if fmap[y, x] != 0.5 or not (y > r - 1 and y < h - r and x > r - 1 and x < w - r):
                continue
            v1 = norm1[y, x, :]
            v2 = norm2[y, x, :]
            b1 = norm1[y - r:y + r + 1, x - r:x + r + 1, :]
            b2 = norm2[y - r:y + r + 1, x - r:x + r + 1, :]
            yy1, xx1 = _argmin_matlab_order(np.sum((b2 - v1) ** 2, axis=2))
            yy2, xx2 = _argmin_matlab_order(np.sum((b1 - v2) ** 2, axis=2))
            p12y, p12x = y - r + yy1, x - r + xx1
            p21y, p21x = y - r + yy2, x - r + xx2
            if not (p12y > r - 1 and p12y < h - r and p12x > r - 1 and p12x < w - r):
                continue
            if not (p21y > r - 1 and p21y < h - r and p21x > r - 1 and p21x < w - r):
                continue
            fm1 = spatial_frequency(img1[y - r:y + r + 1, x - r:x + r + 1])
            fm12 = spatial_frequency(img2[p12y - r:p12y + r + 1, p12x - r:p12x + r + 1])
            fm2 = spatial_frequency(img2[y - r:y + r + 1, x - r:x + r + 1])
            fm21 = spatial_frequency(img1[p21y - r:p21y + r + 1, p21x - r:p21x + r + 1])
            if fm1 > fm12 and fm2 < fm21:
                fmap[y, x] = 1
            if fm1 < fm12 and fm2 > fm21:
                fmap[y, x] = 0
    return fmap


def dsift_fusion(
    a: np.ndarray,
    b: np.ndarray,
    scale: int = 48,
    block_size: int = 8,
    matching: bool = True,
    device: str = "auto",
    chunk_rows: int = 0,
) -> np.ndarray:
    del device, chunk_rows
    img1 = a.astype(np.float64)
    img2 = b.astype(np.float64)
    gray1 = rgb2gray_uint8(a).astype(np.float64)
    gray2 = rgb2gray_uint8(b).astype(np.float64)
    pad = int(scale / 4 - 1)
    patch = int(scale / 2)
    dsift1 = dense_sift(img_extend(gray1, pad), patch, 1)
    dsift2 = dense_sift(img_extend(gray2, pad), patch, 1)
    init1, init2 = generate_initmap(dsift1, dsift2, block_size)
    if matching:
        fmap = refine_withmatching(gray1, gray2, init1, init2, dsift_normalization(dsift1), dsift_normalization(dsift2))
    else:
        fmap = refine_withoutmatching(gray1, gray2, init1, init2)
    if img1.ndim == 3:
        fmap = np.repeat(fmap[:, :, None], img1.shape[2], axis=2)
    fused = img1 * fmap + img2 * (1 - fmap)
    return np.clip(np.rint(fused), 0, 255).astype(np.uint8)
