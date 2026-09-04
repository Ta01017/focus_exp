#!/usr/bin/env python3
"""Build ma/mb/mg for comparison methods using the suite's exact route code.

Input CSV columns:
  sample_id, focus_a, focus_b
Optional columns:
  source_a, valid, confidence

Relative paths are resolved against --root. Empty valid/confidence use an
all-one image, but for paper evaluation they should point to the same dataset
geometry-valid and route-confidence maps used by the main method.
"""

import argparse
import csv
import importlib
import inspect
import sys
from pathlib import Path

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, help="Directory containing infer_pixrestore.py")
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", default=".", help="Base directory for relative manifest paths")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--print-route-source", action="store_true")
    return p.parse_args()


def resolve(value, root: Path):
    value = (value or "").strip()
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def require_file(path, label):
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def main():
    args = parse_args()
    suite = Path(args.suite).resolve()
    root = Path(args.root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Import the authoritative implementation instead of duplicating it.
    sys.path.insert(0, str(suite))
    impl = importlib.import_module("infer_pixrestore")
    load_gray01 = impl.load_gray01
    mask_to_tensor = impl.mask_to_tensor
    focus_to_route = impl.focus_to_route
    save_mask = impl.save_mask

    if args.print_route_source:
        print(f"[ROUTE SOURCE] {inspect.getsourcefile(focus_to_route)}")
        print(inspect.getsource(focus_to_route))

    with Path(args.manifest).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("empty manifest")

    written = []
    for index, row in enumerate(rows):
        sid = (row.get("sample_id") or "").strip()
        if not sid:
            raise ValueError(f"row {index + 2}: empty sample_id")

        fa_path = resolve(row.get("focus_a"), root)
        fb_path = resolve(row.get("focus_b"), root)
        require_file(fa_path, f"{sid} focus_a")
        require_file(fb_path, f"{sid} focus_b")

        source_a = resolve(row.get("source_a"), root)
        if source_a is not None:
            require_file(source_a, f"{sid} source_a")
            with Image.open(source_a) as image:
                size = image.size
        else:
            with Image.open(fa_path) as image:
                size = image.size

        valid_path = resolve(row.get("valid"), root)
        confidence_path = resolve(row.get("confidence"), root)
        if valid_path is not None:
            require_file(valid_path, f"{sid} valid")
        if confidence_path is not None:
            require_file(confidence_path, f"{sid} confidence")

        # These calls and defaults match infer_pixrestore.py exactly.
        fa = load_gray01(fa_path, size, 0)
        fb = load_gray01(fb_path, size, 0)
        valid = load_gray01(valid_path, size, 1)
        confidence = load_gray01(confidence_path, size, 1)

        fa_t = mask_to_tensor(fa, "cpu")
        fb_t = mask_to_tensor(fb, "cpu")
        valid_t = mask_to_tensor(valid, "cpu")
        confidence_t = mask_to_tensor(confidence, "cpu")
        ma, mb, mg = focus_to_route(fa_t, fb_t, valid_t, confidence_t)

        max_sum_error = (ma + mb + mg - 1.0).abs().max().item()
        if max_sum_error > 1e-6:
            raise RuntimeError(f"{sid}: ma+mb+mg error={max_sum_error:.9g}")

        sample_out = out / sid
        sample_out.mkdir(parents=True, exist_ok=True)
        ma_path, mb_path, mg_path = (
            sample_out / "ma.png",
            sample_out / "mb.png",
            sample_out / "mg.png",
        )
        save_mask(ma, ma_path)
        save_mask(mb, mb_path)
        save_mask(mg, mg_path)
        written.append({
            "sample_id": sid,
            "m_a": str(ma_path),
            "m_b": str(mb_path),
            "m_g": str(mg_path),
            "route_sum_max_abs_error_float": f"{max_sum_error:.9g}",
        })

    output_manifest = out / "route_masks_manifest.csv"
    with output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(written[0]))
        writer.writeheader()
        writer.writerows(written)
    print(f"[DONE] rows={len(written)} manifest={output_manifest}")


if __name__ == "__main__":
    main()
