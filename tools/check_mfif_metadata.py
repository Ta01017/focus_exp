#!/usr/bin/env python3
"""Fast schema/path/geometry preflight for MFIF metadata files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
from metadata_dataset import inspect_item_paths, load_metadata


def image_size(path: Path) -> tuple[int, int]:
    if path is None or not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--require-gt", action="store_true")
    parser.add_argument("--max-check", type=int, default=-1)
    parser.add_argument("--print-eval-mode", action="store_true")
    args = parser.parse_args()
    metadata, indexed = load_metadata(args.metadata)
    if args.max_check >= 0:
        indexed = indexed[: args.max_check]
    counts: Counter[str] = Counter()
    errors: list[str] = []
    gt_count = 0
    widths, heights = [], []
    ids = []
    for index, item in indexed:
        try:
            sample = inspect_item_paths(item, index, metadata)
            sizes = [image_size(sample["a_path"]), image_size(sample["b_path"])]
            if sample["gt_path"] is not None:
                sizes.append(image_size(sample["gt_path"]))
                gt_count += 1
            elif args.require_gt:
                raise ValueError("GT path is missing")
            if len(set(sizes)) != 1:
                raise ValueError(f"A/B/GT dimensions differ: {sizes}")
            widths.append(sizes[0][0]); heights.append(sizes[0][1])
            ids.append(sample["sample_id"])
            counts[str(item.get("source_dataset") or "unspecified")] += 1
        except Exception as exc:
            errors.append(f"index={index}: {type(exc).__name__}: {exc}")
    if args.print_eval_mode:
        print("gt" if indexed and gt_count == len(indexed) else "no_gt")
        return 2 if errors else 0
    report = {
        "metadata": str(metadata), "samples_checked": len(indexed),
        "with_gt": gt_count, "source_datasets": dict(counts),
        "width_range": [min(widths), max(widths)] if widths else None,
        "height_range": [min(heights), max(heights)] if heights else None,
        "duplicate_sample_ids": len(ids) - len(set(ids)),
        "errors": errors[:20], "error_count": len(errors),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
