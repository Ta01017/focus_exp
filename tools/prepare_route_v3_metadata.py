#!/usr/bin/env python3
"""Derive authoritative route-v3 maps from focus_a/focus_b for inferred rows."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def pick(item, names):
    for name in names:
        if item.get(name) not in (None, ""):
            return item[name]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.metadata.resolve()
    items = json.loads(metadata.read_text(encoding="utf-8-sig"))
    with args.inference_manifest.open(newline="", encoding="utf-8-sig") as handle:
        inferred = [row for row in csv.DictReader(handle)
                    if str(row.get("success", "")).lower() in {"1", "true", "yes"}]
    rows = []
    for row_number, row in enumerate(inferred):
        index = int(row.get("index", row_number))
        item = items[index]
        edits = item.get("edit_image") or []
        focus_a = pick(item, ("focus_a", "focus_a_path", "mask_focus_a"))
        focus_b = pick(item, ("focus_b", "focus_b_path", "mask_focus_b"))
        if not focus_a and len(edits) >= 4:
            focus_a, focus_b = edits[2], edits[3]
        if not focus_a or not focus_b:
            raise ValueError(f"metadata index={index}: focus_a/focus_b unavailable")
        rows.append({
            "sample_id": row.get("sample_id") or str(index), "metadata_index": index,
            "source_a": row.get("source_a") or edits[0], "focus_a": focus_a, "focus_b": focus_b,
            "valid": pick(item, ("valid", "valid_mask", "geometry_valid")),
            "confidence": pick(item, ("confidence", "route_confidence", "confidence_mask")),
        })
    if not rows:
        raise ValueError("inference manifest has no successful rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converter_input = args.output_dir / "focus_ab_manifest.csv"
    with converter_input.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    subprocess.run([
        sys.executable, str(args.converter), "--suite", str(args.suite),
        "--manifest", str(converter_input), "--root", str(metadata.parent),
        "--output-dir", str(args.output_dir / "route_masks"),
    ], check=True)
    route_manifest = args.output_dir / "route_masks" / "route_masks_manifest.csv"
    with route_manifest.open(newline="", encoding="utf-8") as handle:
        routes = {row["sample_id"]: row for row in csv.DictReader(handle)}
    for row in rows:
        route = routes[row["sample_id"]]
        item = items[row["metadata_index"]]
        item["m_a"], item["m_b"], item["m_g"] = route["m_a"], route["m_b"], route["m_g"]
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] route-v3 derived rows={len(rows)} metadata={args.output_metadata}")


if __name__ == "__main__":
    raise SystemExit(main())
