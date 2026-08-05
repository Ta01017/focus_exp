#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["dataset", "sample_id", "mode", "method", "source_a", "source_b", "gt", "fused"]


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an evaluation manifest from a metadata inference manifest."
    )
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["gt", "no_gt"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include rows whose inference success column is false.",
    )
    args = parser.parse_args()

    with args.inference_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for source in reader:
            if not args.include_failed and "success" in source and not truthy(source["success"]):
                continue
            gt = source.get("gt", "").strip()
            if args.mode == "gt" and not gt:
                raise ValueError(
                    f"GT mode requires gt path, but sample {source.get('sample_id', '')} is empty"
                )
            rows.append(
                {
                    "dataset": args.dataset,
                    "sample_id": source.get("sample_id", ""),
                    "mode": args.mode,
                    "method": args.method,
                    "source_a": source.get("source_a", ""),
                    "source_b": source.get("source_b", ""),
                    "gt": gt if args.mode == "gt" else "",
                    "fused": source.get("prediction", ""),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] {args.output} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
