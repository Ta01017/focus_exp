#!/usr/bin/env python3
"""Join MFIF metadata focus maps with a method inference manifest."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path, PureWindowsPath


FOCUS_A_KEYS = ("focus_a", "focusA", "fa", "focus_map_a", "focus_a_path",
                "focus_map_a_path", "focus_image_a", "edit_image_2", "m_a", "mask_a")
FOCUS_B_KEYS = ("focus_b", "focusB", "fb", "focus_map_b", "focus_b_path",
                "focus_map_b_path", "focus_image_b", "edit_image_3", "m_b", "mask_b")
FIELDS = ("dataset", "method", "sample_id", "source_a", "source_b", "gt",
          "prediction", "m_a", "m_b")


def resolve_path(raw: object, base: Path) -> str:
    if raw in (None, ""):
        return ""
    text = os.fspath(raw)
    if re.match(r"^[A-Za-z]:[\\/]", text):
        return str(Path(text) if os.name == "nt" else PureWindowsPath(text))
    path = Path(text.replace("\\", os.sep)).expanduser()
    return str(path.resolve() if path.is_absolute() else (base / path).resolve())


def first(item: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if item.get(key) not in (None, ""):
            return item[key]
    return None


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build(metadata: Path, inference_manifest: Path, output: Path,
          dataset: str, method: str) -> int:
    metadata = metadata.expanduser().resolve()
    inference_manifest = inference_manifest.expanduser().resolve()
    items = json.loads(metadata.read_text(encoding="utf-8-sig"))
    if not isinstance(items, list):
        raise ValueError("metadata top level must be a list")
    with inference_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        inference = list(csv.DictReader(handle))
    rows = []
    for row_number, source in enumerate(inference):
        if not truthy(source.get("success", "")):
            continue
        try:
            index = int(source.get("index", row_number))
        except ValueError as exc:
            raise ValueError(f"invalid inference index at row {row_number}: {source.get('index')}") from exc
        if index < 0 or index >= len(items):
            raise IndexError(f"inference index {index} outside metadata rows={len(items)}")
        item = items[index]
        edits = item.get("edit_image") or []
        if not isinstance(edits, list) or len(edits) < 2:
            raise ValueError(f"metadata index={index}: edit_image requires A and B")
        focus_a = first(item, FOCUS_A_KEYS)
        focus_b = first(item, FOCUS_B_KEYS)
        focus_pair = item.get("focus") or item.get("focus_maps") or item.get("masks")
        if isinstance(focus_pair, (list, tuple)):
            if focus_a is None and len(focus_pair) > 0:
                focus_a = focus_pair[0]
            if focus_b is None and len(focus_pair) > 1:
                focus_b = focus_pair[1]
        if focus_a is None and len(edits) > 2:
            focus_a = edits[2]
        if focus_b is None and len(edits) > 3:
            focus_b = edits[3]
        if focus_a is None or focus_b is None:
            raise ValueError(
                f"metadata index={index}: region evaluation requires focus_a/focus_b "
                "or edit_image[2]/edit_image[3]"
            )
        base = metadata.parent
        values = {
            "dataset": dataset,
            "method": method,
            "sample_id": source.get("sample_id") or str(index),
            "source_a": source.get("source_a") or resolve_path(edits[0], base),
            "source_b": source.get("source_b") or resolve_path(edits[1], base),
            "gt": source.get("gt") or resolve_path(item.get("image"), base),
            "prediction": source.get("prediction", ""),
            "m_a": resolve_path(focus_a, base),
            "m_b": resolve_path(focus_b, base),
        }
        for key in ("source_a", "source_b", "prediction", "m_a", "m_b"):
            path = Path(values[key])
            if not path.is_file():
                raise FileNotFoundError(f"index={index} {key} not found: {path}")
        if values["gt"] and not Path(values["gt"]).is_file():
            raise FileNotFoundError(f"index={index} gt not found: {values['gt']}")
        rows.append(values)
    if not rows:
        raise ValueError("inference manifest has no successful rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] region manifest={output} rows={len(rows)} method={method}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="RealSceneVal68")
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    build(args.metadata, args.inference_manifest, args.output, args.dataset, args.method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
