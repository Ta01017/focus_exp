#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity


ALIASES = {
    "dataset": ["dataset", "dataset_name", "set"],
    "method": ["method", "method_name", "model"],
    "sample_id": ["sample_id", "id", "name", "filename"],
    "a": ["a", "A", "image_a", "image_a_path", "a_path", "source_a", "input_a", "edit_image_0"],
    "b": ["b", "B", "image_b", "image_b_path", "b_path", "source_b", "input_b", "edit_image_1"],
    "gt": ["gt", "GT", "image", "target", "hq", "gt_path", "target_path", "image_path"],
    "pred": ["fused", "prediction", "pred", "output", "result", "fused_path", "prediction_path", "pred_path", "output_path"],
    "ma": ["m_a", "ma", "route_a", "route_a_path"],
    "mb": ["m_b", "mb", "route_b", "route_b_path"],
    "mg": ["m_g", "mg", "route_g", "route_g_path"],
}


def parse_args():
    p = argparse.ArgumentParser(description="Three-route MFIF evaluator: A preservation / B transfer / G generation")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--root", default="")
    p.add_argument("--erode-radius", type=int, default=-1, help="-1 => max(2, round(min(H,W)/256))")
    p.add_argument("--route-confidence", type=float, default=0.0,
                   help="0 => pure argmax and no uncertain pixels; >0 rejects max route probability below threshold")
    p.add_argument("--route-sum-tolerance", type=float, default=0.05,
                   help="maximum allowed mean absolute error of M_A+M_B+M_G from 1")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--patch-stride", type=int, default=32)
    p.add_argument("--g-patch-min-coverage", type=float, default=0.80)
    p.add_argument("--g-rsr-psnr-margin", type=float, default=0.20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--no-lpips", action="store_true")
    for x in ["dataset", "method", "id", "a", "b", "gt", "pred", "m-a", "m-b", "m-g"]:
        p.add_argument(f"--col-{x}", default="")
    return p.parse_args()


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def nanmean(xs):
    ys = [float(x) for x in xs if finite(x)]
    return float(np.mean(ys)) if ys else float("nan")


def find_col(fields, explicit, key, required):
    if explicit:
        if explicit not in fields:
            raise KeyError(f"column {explicit!r} not found; columns={fields}")
        return explicit
    for c in ALIASES[key]:
        if c in fields:
            return c
    if required:
        raise KeyError(f"cannot detect column for {key}; columns={fields}")
    return None


def resolve(root: Path, raw: Any):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None
    p = Path(s)
    return p if p.is_absolute() else root / p


def load_rgb(p: Path):
    if not p.is_file():
        raise FileNotFoundError(p)
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0


def load_gray(p: Path):
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".npy":
        x = np.squeeze(np.load(p)).astype(np.float32)
        if x.ndim != 2:
            raise ValueError(f"route map must be 2D: {p} {x.shape}")
        if float(np.nanmax(x)) > 1.5:
            x /= 255.0
        return np.clip(x, 0, 1)
    return np.asarray(Image.open(p).convert("L"), np.float32) / 255.0


def resize_rgb(x, hw, mode):
    h, w = hw
    if x.shape[:2] == (h, w):
        return x
    im = Image.fromarray(np.clip(np.round(x * 255), 0, 255).astype(np.uint8), "RGB")
    return np.asarray(im.resize((w, h), mode), np.float32) / 255.0


def resize_gray(x, hw):
    h, w = hw
    if x.shape[:2] == (h, w):
        return x
    im = Image.fromarray(np.clip(np.round(x * 255), 0, 255).astype(np.uint8), "L")
    return np.asarray(im.resize((w, h), Image.Resampling.BILINEAR), np.float32) / 255.0


def erode(mask, r):
    if r <= 0:
        return mask.astype(bool)
    x = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * r + 1
    y = 1 - F.max_pool2d(1 - x, k, stride=1, padding=r)
    return y[0, 0].numpy() > 0.5


def gray(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)


SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
SY = SX.T.copy()
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)


def conv(g, k):
    x = torch.from_numpy(g)[None, None]
    kk = torch.from_numpy(k)[None, None]
    return F.conv2d(x, kk, padding=(k.shape[0] // 2, k.shape[1] // 2))[0, 0].numpy()


def sobel(x):
    g = gray(x)
    gx, gy = conv(g, SX), conv(g, SY)
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def lapabs(x):
    return np.abs(conv(gray(x), LAP))


def blur_gray(g, sigma=1.5):
    r = max(1, int(round(3 * sigma)))
    xs = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(xs * xs) / (2 * sigma * sigma))
    k /= k.sum()
    x = torch.from_numpy(g)[None, None]
    x = F.conv2d(x, k[None, None, None, :], padding=(0, r))
    x = F.conv2d(x, k[None, None, :, None], padding=(r, 0))
    return x[0, 0].numpy()


def hfabs(x):
    g = gray(x)
    return np.abs(g - blur_gray(g))


def masked_mean(x, m):
    return float(np.mean(x[m])) if int(m.sum()) else float("nan")


def masked_mse(a, b, m):
    return float(np.mean(((a - b) ** 2)[m, :])) if int(m.sum()) else float("nan")


def masked_mae(a, b, m):
    return float(np.mean(np.abs(a - b)[m, :])) if int(m.sum()) else float("nan")


def psnr_from_mse(m):
    if not finite(m):
        return float("nan")
    if m <= 1e-12:
        return 120.0
    return float(-10 * math.log10(m))


def ssim_map(a, b):
    _, sm = structural_similarity(
        a, b, data_range=1.0, channel_axis=2, full=True,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    if sm.ndim == 3:
        sm = sm.mean(2)
    return sm.astype(np.float32)


class LP:
    def __init__(self, device="cuda", net="alex", enabled=True):
        self.enabled = enabled
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model = None
        if enabled:
            try:
                import lpips
            except Exception as e:
                raise RuntimeError("install lpips or use --no-lpips") from e
            self.model = lpips.LPIPS(net=net, spatial=True).to(self.device).eval()
            self.model.requires_grad_(False)

    def map(self, a, b):
        if not self.enabled:
            return None
        ta = torch.from_numpy(a.transpose(2, 0, 1))[None].to(self.device) * 2 - 1
        tb = torch.from_numpy(b.transpose(2, 0, 1))[None].to(self.device) * 2 - 1
        with torch.inference_mode():
            y = self.model(ta, tb)
            y = F.interpolate(y, size=a.shape[:2], mode="bilinear", align_corners=False)
        return y[0, 0].float().cpu().numpy()

    def masked(self, lm, m):
        if lm is None or not int(m.sum()):
            return float("nan")
        return float(np.mean(lm[m]))

    def crop(self, a, b):
        if not self.enabled:
            return float("nan")
        ta = torch.from_numpy(a.transpose(2, 0, 1))[None].to(self.device) * 2 - 1
        tb = torch.from_numpy(b.transpose(2, 0, 1))[None].to(self.device) * 2 - 1
        with torch.inference_mode():
            y = self.model(ta, tb)
        return float(y.mean().item())


def metrics(pred, ref, mask, sm, lm, lp):
    mse = masked_mse(pred, ref, mask)
    return {
        "psnr": psnr_from_mse(mse),
        "ssim": masked_mean(sm, mask),
        "lpips": lp.masked(lm, mask),
        "mae": masked_mae(pred, ref, mask),
        "mse": mse,
    }


def put(row, prefix, values):
    for k, v in values.items():
        row[f"{prefix}_{k}"] = v


def route_regions(ma, mb, mg, confidence=0.0):
    scores = np.stack([ma, mb, mg], axis=0).astype(np.float32)
    valid = np.all(np.isfinite(scores), axis=0)
    labels = np.argmax(scores, axis=0)
    if confidence > 0:
        valid &= np.max(scores, axis=0) >= confidence
    regions = {
        "a": valid & (labels == 0),
        "b": valid & (labels == 1),
        "g": valid & (labels == 2),
    }
    regions["uncertain"] = ~valid
    return regions


def g_nogt(pred, a, b, m):
    keys = [
        "sobel_pred", "sobel_a", "sobel_b", "sobel_gain_best_src",
        "laplacian_pred", "laplacian_a", "laplacian_b", "laplacian_gain_best_src",
        "highfreq_pred", "highfreq_a", "highfreq_b", "highfreq_gain_best_src",
    ]
    if not int(m.sum()):
        return {f"g_{k}": float("nan") for k in keys}
    sp, sa, sb = [masked_mean(z, m) for z in [sobel(pred), sobel(a), sobel(b)]]
    lp_, la, lb = [masked_mean(z, m) for z in [lapabs(pred), lapabs(a), lapabs(b)]]
    hp, ha, hb = [masked_mean(z, m) for z in [hfabs(pred), hfabs(a), hfabs(b)]]
    return {
        "g_sobel_pred": sp, "g_sobel_a": sa, "g_sobel_b": sb,
        "g_sobel_gain_best_src": sp - max(sa, sb),
        "g_laplacian_pred": lp_, "g_laplacian_a": la, "g_laplacian_b": lb,
        "g_laplacian_gain_best_src": lp_ - max(la, lb),
        "g_highfreq_pred": hp, "g_highfreq_a": ha, "g_highfreq_b": hb,
        "g_highfreq_gain_best_src": hp - max(ha, hb),
    }


def patches(h, w, p, s):
    if h < p or w < p:
        return
    ys, xs = list(range(0, h - p + 1, s)), list(range(0, w - p + 1, s))
    if ys[-1] != h - p:
        ys.append(h - p)
    if xs[-1] != w - p:
        xs.append(w - p)
    for y in ys:
        for x in xs:
            yield y, x


def g_rsr(pred, a, b, gt, m, lp, patch, stride, mincov, margin):
    valid = success = 0
    h, w = m.shape
    for y, x in patches(h, w, patch, stride):
        mm = m[y:y + patch, x:x + patch]
        if float(mm.mean()) < mincov:
            continue
        pp = pred[y:y + patch, x:x + patch]
        aa = a[y:y + patch, x:x + patch]
        bb = b[y:y + patch, x:x + patch]
        gg = gt[y:y + patch, x:x + patch]
        pps = psnr_from_mse(masked_mse(pp, gg, mm))
        aps = psnr_from_mse(masked_mse(aa, gg, mm))
        bps = psnr_from_mse(masked_mse(bb, gg, mm))
        if lp.enabled:
            pl, al, bl = lp.crop(pp, gg), lp.crop(aa, gg), lp.crop(bb, gg)
            lpok = pl < min(al, bl)
        else:
            lpok = True
        valid += 1
        if pps > max(aps, bps) + margin and lpok:
            success += 1
    return (success / valid if valid else float("nan")), valid, success


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def weighted_mean(rows, value_key, weight_key):
    vals = []
    weights = []
    for r in rows:
        v, w = r.get(value_key), r.get(weight_key, 0)
        if finite(v) and finite(w) and float(w) > 0:
            vals.append(float(v))
            weights.append(float(w))
    if not weights:
        return float("nan")
    return float(np.average(vals, weights=weights))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["method"])].append(r)
    outs = []
    skip = {
        "dataset", "method", "sample_id", "a_path", "b_path", "gt_path", "pred_path",
        "m_a_path", "m_b_path", "m_g_path", "g_valid_patch_count", "g_success_patch_count",
    }
    weighted_prefixes = ["a_gt", "a_src", "b_gt", "b_src", "g_gt", "g_src_a_gt", "g_src_b_gt"]
    for (ds, method), rs in groups.items():
        o = {"dataset": ds, "method": method, "n": len(rs)}
        nums = set()
        for r in rs:
            for k, v in r.items():
                if k not in skip and isinstance(v, (int, float, np.integer, np.floating)):
                    nums.add(k)
        for k in sorted(nums):
            values = [r.get(k, float("nan")) for r in rs]
            o[k + "_mean"] = nanmean(values)
            o[k + "_valid_images"] = sum(finite(v) for v in values)
        for prefix in weighted_prefixes:
            region = prefix[0]
            pixels = f"{region}_eval_pixels"
            for metric in ["mae", "mse", "ssim", "lpips"]:
                o[f"{prefix}_{metric}_pixel_weighted"] = weighted_mean(rs, f"{prefix}_{metric}", pixels)
            pooled_mse = o[f"{prefix}_mse_pixel_weighted"]
            o[f"{prefix}_psnr_pooled"] = psnr_from_mse(pooled_mse)
        nv = sum(int(r.get("g_valid_patch_count", 0)) for r in rs)
        ns = sum(int(r.get("g_success_patch_count", 0)) for r in rs)
        o["g_rsr_pooled"] = ns / nv if nv else float("nan")
        o["g_valid_patch_count_total"] = nv
        o["g_success_patch_count_total"] = ns
        outs.append(o)
    return outs


def main():
    args = parse_args()
    mf, out = Path(args.manifest), Path(args.output_dir)
    root = Path(args.root) if args.root else mf.parent
    with mf.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        recs = list(rd)
    if not recs:
        raise RuntimeError(f"empty manifest: {mf}")

    cd = find_col(fields, args.col_dataset, "dataset", False)
    cm = find_col(fields, args.col_method, "method", False)
    ci = find_col(fields, args.col_id, "sample_id", False)
    ca = find_col(fields, args.col_a, "a", True)
    cb = find_col(fields, args.col_b, "b", True)
    cp = find_col(fields, args.col_pred, "pred", True)
    cma = find_col(fields, args.col_m_a, "ma", True)
    cmb = find_col(fields, args.col_m_b, "mb", True)
    cmg = find_col(fields, args.col_m_g, "mg", True)
    cgt = find_col(fields, args.col_gt, "gt", False)
    print(f"[MANIFEST] {mf}\n[ROWS] {len(recs)}\n[COLUMNS] A={ca} B={cb} GT={cgt} PRED={cp} MA={cma} MB={cmb} MG={cmg}")

    lp = LP(args.device, args.lpips_net, not args.no_lpips)
    rows = []
    for i, it in enumerate(recs):
        ap, bp = resolve(root, it[ca]), resolve(root, it[cb])
        pp = resolve(root, it[cp])
        map_, mbp, mgp = resolve(root, it[cma]), resolve(root, it[cmb]), resolve(root, it[cmg])
        gtp = resolve(root, it[cgt]) if cgt else None
        mode = str(it.get("fused_mode", "")).strip().lower()
        virtual = None
        if mode in {"avgblend", "avg_blend", "avg blend"}:
            virtual = "avgblend"
        elif mode in {"no_generation", "nogeneration", "w/o generation", "wo generation"}:
            virtual = "nogeneration"
        if virtual is None and pp is None:
            raise RuntimeError(f"row {i}: missing prediction path; method={it.get('method')} mode={mode}")
        if any(x is None for x in [ap, bp, map_, mbp, mgp]):
            raise RuntimeError(f"row {i}: missing required path; method={it.get('method')} mode={mode}")

        a = load_rgb(ap)
        h, w = a.shape[:2]
        b = resize_rgb(load_rgb(bp), (h, w), Image.Resampling.BICUBIC)
        if virtual == "avgblend":
            pred = 0.5 * (a + b)
        elif virtual == "nogeneration":
            pred = a.copy()
        else:
            pred = resize_rgb(load_rgb(pp), (h, w), Image.Resampling.BICUBIC)
        ma = resize_gray(load_gray(map_), (h, w))
        mb = resize_gray(load_gray(mbp), (h, w))
        mg = resize_gray(load_gray(mgp), (h, w))
        gt = resize_rgb(load_rgb(gtp), (h, w), Image.Resampling.LANCZOS) if gtp is not None and gtp.is_file() else None

        route_sum = ma + mb + mg
        sum_error = float(np.nanmean(np.abs(route_sum - 1.0)))
        if not finite(sum_error) or sum_error > args.route_sum_tolerance:
            raise RuntimeError(
                f"row {i}: invalid route maps: mean|MA+MB+MG-1|={sum_error:.6f} > "
                f"{args.route_sum_tolerance}; sample={it.get(ci, i) if ci else i}"
            )
        raw = route_regions(ma, mb, mg, args.route_confidence)
        er = args.erode_radius if args.erode_radius >= 0 else max(2, int(round(min(h, w) / 256.0)))
        masks = {k: (erode(v, er) if k != "uncertain" else v) for k, v in raw.items()}
        raw_sum = raw["a"].astype(np.uint8) + raw["b"].astype(np.uint8) + raw["g"].astype(np.uint8) + raw["uncertain"].astype(np.uint8)
        if not np.all(raw_sum == 1):
            raise RuntimeError(f"row {i}: route partition is not exhaustive/exclusive")

        sid = str(it[ci]) if ci else (pp.stem if pp is not None else f"row_{i:06d}")
        row = {
            "dataset": it[cd] if cd else "dataset", "method": it[cm] if cm else "method", "sample_id": sid,
            "a_path": str(ap), "b_path": str(bp), "gt_path": str(gtp) if gtp else "",
            "pred_path": str(pp) if pp else f"<virtual:{virtual}>",
            "m_a_path": str(map_), "m_b_path": str(mbp), "m_g_path": str(mgp),
            "height": h, "width": w, "erode_radius": er,
            "route_sum_abs_error": sum_error, "route_sum_min": float(np.nanmin(route_sum)), "route_sum_max": float(np.nanmax(route_sum)),
            "a_raw_ratio": float(raw["a"].mean()), "b_raw_ratio": float(raw["b"].mean()),
            "g_raw_ratio": float(raw["g"].mean()), "uncertain_raw_ratio": float(raw["uncertain"].mean()),
            "a_eval_ratio": float(masks["a"].mean()), "b_eval_ratio": float(masks["b"].mean()), "g_eval_ratio": float(masks["g"].mean()),
            "a_eval_pixels": int(masks["a"].sum()), "b_eval_pixels": int(masks["b"].sum()), "g_eval_pixels": int(masks["g"].sum()),
        }

        sma, smb = ssim_map(pred, a), ssim_map(pred, b)
        lma, lmb = lp.map(pred, a), lp.map(pred, b)
        put(row, "a_src", metrics(pred, a, masks["a"], sma, lma, lp))
        put(row, "b_src", metrics(pred, b, masks["b"], smb, lmb, lp))
        row.update(g_nogt(pred, a, b, masks["g"]))

        if gt is not None:
            smg, lmg = ssim_map(pred, gt), lp.map(pred, gt)
            for reg in ["a", "b", "g"]:
                put(row, f"{reg}_gt", metrics(pred, gt, masks[reg], smg, lmg, lp))
            row["g_sobel_error"] = masked_mean(np.abs(sobel(pred) - sobel(gt)), masks["g"])
            row["g_laplacian_error"] = masked_mean(np.abs(lapabs(pred) - lapabs(gt)), masks["g"])
            smagt, smbgt = ssim_map(a, gt), ssim_map(b, gt)
            lmagt, lmbgt = lp.map(a, gt), lp.map(b, gt)
            ga = metrics(a, gt, masks["g"], smagt, lmagt, lp)
            gb = metrics(b, gt, masks["g"], smbgt, lmbgt, lp)
            put(row, "g_src_a_gt", ga)
            put(row, "g_src_b_gt", gb)
            row["g_psnr_gain_best_src"] = row["g_gt_psnr"] - max(ga["psnr"], gb["psnr"]) if all(finite(x) for x in [row["g_gt_psnr"], ga["psnr"], gb["psnr"]]) else float("nan")
            row["g_lpips_gain_best_src"] = min(ga["lpips"], gb["lpips"]) - row["g_gt_lpips"] if all(finite(x) for x in [row["g_gt_lpips"], ga["lpips"], gb["lpips"]]) else float("nan")
            rsr, nv, ns = g_rsr(pred, a, b, gt, masks["g"], lp, args.patch_size, args.patch_stride, args.g_patch_min_coverage, args.g_rsr_psnr_margin)
            row["g_rsr"], row["g_valid_patch_count"], row["g_success_patch_count"] = rsr, nv, ns
        else:
            for reg in ["a", "b", "g"]:
                for k in ["psnr", "ssim", "lpips", "mae", "mse"]:
                    row[f"{reg}_gt_{k}"] = float("nan")
            for k in ["g_sobel_error", "g_laplacian_error", "g_psnr_gain_best_src", "g_lpips_gain_best_src", "g_rsr"]:
                row[k] = float("nan")
            row["g_valid_patch_count"] = row["g_success_patch_count"] = 0
        rows.append(row)
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(recs):
            print(
                f"[{i + 1:04d}/{len(recs):04d}] {row['dataset']} | {row['method']} | {sid} | "
                f"A/B/G/U={row['a_raw_ratio']:.3f}/{row['b_raw_ratio']:.3f}/{row['g_raw_ratio']:.3f}/{row['uncertain_raw_ratio']:.3f} | "
                f"sum_err={sum_error:.6f}"
            )

    write_csv(out / "route_metrics_per_image.csv", rows)
    write_csv(out / "route_metrics_summary.csv", summarize(rows))
    print("[DONE]", out / "route_metrics_per_image.csv")
    print("[DONE]", out / "route_metrics_summary.csv")


if __name__ == "__main__":
    main()
