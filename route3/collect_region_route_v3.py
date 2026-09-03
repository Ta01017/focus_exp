#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_union(path, rows):
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--include-extra-real", action="store_true")
    args = p.parse_args()
    root = Path(args.output_root)
    methods = [
        "DSIFT", "FULX2.0_ORIGIN", "IFCNN", "FusionDiff", "ReDiffuse_ORIGIN", "SwinFusion", "ZMFF",
        "AvgBlend", "FullGen", "G_Diagnostic", "wo_Generation", "wo_Refiner", "Ours",
    ]
    datasets = ["CommonBlurGeometryVal200", "RealMFFAlignedVal110"]
    summary_rows, per_image_rows, missing = [], [], []
    for dataset in datasets:
        selected = list(methods)
        if dataset == "RealMFFAlignedVal110" and args.include_extra_real:
            selected += ["plus5k_Control", "plus5k_Severe"]
        for method in selected:
            base = root / "metrics" / dataset / method
            sp = base / "route_metrics_summary.csv"
            pp = base / "route_metrics_per_image.csv"
            if not sp.is_file() or not pp.is_file():
                missing.append(f"{dataset}/{method}")
                continue
            sr, pr = read_csv(sp), read_csv(pp)
            if len(sr) != 1:
                raise RuntimeError(f"expected one summary row: {sp}; got {len(sr)}")
            expected = 200 if dataset == "CommonBlurGeometryVal200" else 110
            if len(pr) != expected:
                raise RuntimeError(f"bad per-image row count: {pp}; expected={expected} got={len(pr)}")
            summary_rows.extend(sr)
            per_image_rows.extend(pr)
    if missing:
        raise RuntimeError("missing metric outputs:\n" + "\n".join(missing))
    write_union(root / "REGION_ROUTE_V3_ALL_SUMMARY.csv", summary_rows)
    write_union(root / "REGION_ROUTE_V3_ALL_PER_IMAGE.csv", per_image_rows)
    print("[DONE]", root / "REGION_ROUTE_V3_ALL_SUMMARY.csv")
    print("[DONE]", root / "REGION_ROUTE_V3_ALL_PER_IMAGE.csv")


if __name__ == "__main__":
    main()
