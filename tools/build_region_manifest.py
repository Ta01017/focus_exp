#!/usr/bin/env python3
"""Join normalized three-route maps with a method inference manifest."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image


ROUTE_A_KEYS = ("m_a", "ma", "route_a", "route_a_path", "mask_a")
ROUTE_B_KEYS = ("m_b", "mb", "route_b", "route_b_path", "mask_b")
ROUTE_G_KEYS = ("m_g", "mg", "route_g", "route_g_path", "mask_g")
FIELDS = ("dataset", "method", "sample_id", "source_a", "source_b", "gt",
          "prediction", "m_a", "m_b", "m_g", "fused_mode")


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


def validate_route_maps(paths: tuple[str, str, str], source_a: str,
                        tolerance: float = 0.05) -> None:
    with Image.open(source_a) as image:
        size = image.size
    maps = []
    for path in paths:
        with Image.open(path) as image:
            route = image.convert("L")
            if route.size != size:
                route = route.resize(size, Image.Resampling.BILINEAR)
            maps.append(np.asarray(route, dtype=np.float32) / 255.0)
    stack = np.stack(maps)
    error = float(np.mean(np.abs(stack.sum(axis=0) - 1.0)))
    if not np.all(np.isfinite(stack)) or error > tolerance:
        raise ValueError(
            f"invalid normalized route maps: mean|M_A+M_B+M_G-1|={error:.6f} > {tolerance}"
        )


def build(metadata: Path, inference_manifest: Path, output: Path,
          dataset: str, method: str, route_sum_tolerance: float = 0.05) -> int:
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
        route_a = first(item, ROUTE_A_KEYS)
        route_b = first(item, ROUTE_B_KEYS)
        route_g = first(item, ROUTE_G_KEYS)
        route_set = item.get("route_maps") or item.get("routes") or item.get("masks")
        if isinstance(route_set, (list, tuple)) and len(route_set) >= 3:
            route_a = route_a or route_set[0]
            route_b = route_b or route_set[1]
            route_g = route_g or route_set[2]
        # Only accept edit_image fallback when all three route maps are present.
        # Two old focus maps are not equivalent to the route-v3 protocol.
        if len(edits) >= 5:
            route_a = route_a or edits[2]
            route_b = route_b or edits[3]
            route_g = route_g or edits[4]
        if route_a is None or route_b is None or route_g is None:
            raise ValueError(
                f"metadata index={index}: route-v3 requires normalized m_a/m_b/m_g; "
                "two-map focus_a/focus_b data is not valid"
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
            "m_a": resolve_path(route_a, base),
            "m_b": resolve_path(route_b, base),
            "m_g": resolve_path(route_g, base),
            "fused_mode": "",
        }
        for key in ("source_a", "source_b", "prediction", "m_a", "m_b", "m_g"):
            path = Path(values[key])
            if not path.is_file():
                raise FileNotFoundError(f"index={index} {key} not found: {path}")
        if values["gt"] and not Path(values["gt"]).is_file():
            raise FileNotFoundError(f"index={index} gt not found: {values['gt']}")
        validate_route_maps(
            (values["m_a"], values["m_b"], values["m_g"]),
            values["source_a"], route_sum_tolerance,
        )
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
    parser.add_argument("--route-sum-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    build(args.metadata, args.inference_manifest, args.output, args.dataset, args.method,
          args.route_sum_tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
