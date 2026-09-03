#!/usr/bin/env python3
"""Fast schema/path/geometry preflight for MFIF metadata files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--print-eval-mode", action="store_true")
    args = parser.parse_args()
    metadata, all_indexed = load_metadata(args.metadata)
    total_rows = len(all_indexed)
    indexed = all_indexed
    if 0 <= args.max_check < total_rows:
        if args.max_check == 0:
            indexed = []
        elif args.max_check == 1:
            indexed = [all_indexed[0]]
        else:
            # Cover the whole mixed metadata instead of checking only its first block.
            positions = [round(i * (total_rows - 1) / (args.max_check - 1))
                         for i in range(args.max_check)]
            indexed = [all_indexed[position] for position in positions]
    if args.print_eval_mode:
        # Mode detection only depends on schema; do not reopen every image after
        # the preceding geometry preflight has already read the validation set.
        print("gt" if all_indexed and all(
            isinstance(item, dict) and item.get("image") for _, item in all_indexed
        ) else "no_gt")
        return 0
    counts: Counter[str] = Counter()
    errors: list[str] = []
    gt_count = 0
    widths, heights = [], []
    ids = []
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    def inspect(pair):
        index, item = pair
        try:
            sample = inspect_item_paths(item, index, metadata)
            sizes = [image_size(sample["a_path"]), image_size(sample["b_path"])]
            has_gt = False
            if sample["gt_path"] is not None:
                sizes.append(image_size(sample["gt_path"]))
                has_gt = True
            elif args.require_gt:
                raise ValueError("GT path is missing")
            if len(set(sizes)) != 1:
                raise ValueError(f"A/B/GT dimensions differ: {sizes}")
            return {
                "width": sizes[0][0], "height": sizes[0][1],
                "sample_id": sample["sample_id"], "has_gt": has_gt,
                "source_dataset": str(item.get("source_dataset") or "unspecified"),
            }, None
        except Exception as exc:
            return None, f"index={index}: {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(indexed)))) as executor:
        for result, error in executor.map(inspect, indexed):
            if error:
                errors.append(error)
                continue
            widths.append(result["width"]); heights.append(result["height"])
            ids.append(result["sample_id"])
            gt_count += int(result["has_gt"])
            counts[result["source_dataset"]] += 1
    report = {
        "metadata": str(metadata), "total_rows": total_rows,
        "samples_checked": len(indexed),
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
