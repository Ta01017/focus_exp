#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadata_dataset import load_metadata, prepare_item, restore_a_size, save_inputs, write_run_files

from dsift_mfif import dsift_fusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--save-inputs", type=int, default=0)
    p.add_argument("--size-policy", default="error")
    p.add_argument("--device", default="auto")
    p.add_argument("--chunk-rows", type=int, default=0)
    p.add_argument("--scale", type=int, default=48)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--matching", type=int, default=1)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    metadata_path, items = load_metadata(args.metadata, args.start_index, args.max_samples)
    records = []
    for index, item in items:
        started = time.perf_counter()
        record = {
            "index": index,
            "sample_id": "",
            "source_a": "",
            "source_b": "",
            "gt": "",
            "prediction": "",
            "original_width": None,
            "original_height": None,
            "runtime_seconds": 0.0,
            "success": False,
            "error": "",
            "actual_iterations": None,
        }
        try:
            sample = prepare_item(item, index, metadata_path, size_policy=args.size_policy, mode="infer")
            record.update(
                {
                    "sample_id": sample["sample_id"],
                    "source_a": str(sample["a_path"]),
                    "source_b": str(sample["b_path"]),
                    "gt": str(sample["gt_path"] or ""),
                    "prediction": str((output_dir / f"{sample['sample_id']}_pred.png").resolve()),
                    "original_width": sample["original_size"][0],
                    "original_height": sample["original_size"][1],
                }
            )
            pred = Path(record["prediction"])
            if pred.exists() and not args.overwrite:
                record["success"] = True
                record["error"] = "skipped_existing"
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                a = sample["a"]
                b = sample["b"]
                fused = dsift_fusion(
                    __import__("numpy").asarray(a),
                    __import__("numpy").asarray(b),
                    scale=args.scale,
                    block_size=args.block_size,
                    matching=bool(args.matching),
                    device=args.device,
                    chunk_rows=args.chunk_rows,
                )
                image = restore_a_size(Image.fromarray(fused), sample)
                image.save(pred, format="PNG")
                if args.save_inputs:
                    save_inputs(sample, output_dir)
                record["success"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if args.strict:
                raise
        finally:
            record["runtime_seconds"] = time.perf_counter() - started
            records.append(record)
    write_run_files(
        output_dir,
        records,
        {
            "method": "DSIFT-MFIF-python",
            "metadata": str(metadata_path),
            "scale": args.scale,
            "block_size": args.block_size,
            "matching": bool(args.matching),
            "device": args.device,
            "chunk_rows": args.chunk_rows,
        },
    )
    print(f"[DONE] rows={len(records)} success={sum(1 for r in records if r['success'])}")
    return 0 if all(r["success"] for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
