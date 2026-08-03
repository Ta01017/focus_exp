"""Shared, dependency-light metadata.json handling for fusion baselines."""
from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageOps


SUFFIXES = ("_target", "_gt", "_a", "_src", "_source", "_input")
FIELDS = ("index", "sample_id", "source_a", "source_b", "gt", "prediction",
          "original_width", "original_height", "runtime_seconds", "success", "error")
FIELDS = FIELDS + ("actual_iterations",)


def bool01(value):
    value = str(value).strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError("expected 0/1 or false/true")
    return value in {"1", "true"}


def _portable_path(raw, base):
    """Resolve native paths and Windows paths, including backslashes on POSIX."""
    if raw in (None, ""):
        return None
    text = os.fspath(raw)
    if re.match(r"^[A-Za-z]:[\\/]", text):
        if os.name == "nt":
            return Path(text).resolve()
        # A Windows absolute path cannot be opened on POSIX, but preserving it
        # gives a useful per-sample error instead of silently rebasing it.
        return Path(str(PureWindowsPath(text)))
    text = text.replace("\\", os.sep)
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def sample_id(item, index):
    if item.get("source_index") is not None:
        raw = str(item["source_index"])
    elif item.get("image"):
        raw = PureWindowsPath(str(item["image"])).stem
    elif item.get("edit_image"):
        raw = PureWindowsPath(str(item["edit_image"][0])).stem
    else:
        raw = str(index)
    lowered = raw.lower()
    for suffix in SUFFIXES:
        if lowered.endswith(suffix):
            raw = raw[:-len(suffix)]
            break
    raw = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("._-") or str(index)
    return raw.zfill(6) if raw.isdigit() else raw


def load_metadata(metadata, start_index=0, max_samples=-1):
    metadata = Path(metadata).expanduser().resolve()
    with metadata.open("r", encoding="utf-8-sig") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError("metadata.json top level must be a list")
    stop = None if max_samples < 0 else start_index + max_samples
    return metadata, list(enumerate(items))[start_index:stop]


def prepare_item(item, index, metadata_path, size_policy="error"):
    if not isinstance(item, dict):
        raise ValueError("metadata item must be an object")
    edits = item.get("edit_image")
    if not isinstance(edits, list) or len(edits) < 2:
        raise ValueError("edit_image must contain at least two paths")
    base = metadata_path.parent
    a_path = _portable_path(edits[0], base)
    b_path = _portable_path(edits[1], base)  # deliberately ignore [2:]
    gt_path = _portable_path(item.get("image"), base)
    for label, path in (("A", a_path), ("B", b_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} image not found: {path}")
    try:
        with Image.open(a_path) as source:
            a = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(b_path) as source:
            b = ImageOps.exif_transpose(source).convert("RGB")
        a.load(); b.load()
    except Exception as exc:
        raise ValueError(f"damaged or unsupported image: {exc}") from exc
    original_size = a.size
    original_a = a.copy()
    if b.size != a.size:
        if size_policy == "error":
            raise ValueError(f"A/B size mismatch: A={a.size}, B={b.size}")
        if size_policy == "resize_b_to_a":
            b = b.resize(a.size, Image.Resampling.BICUBIC)
        elif size_policy == "center_crop_common":
            width, height = min(a.width, b.width), min(a.height, b.height)
            a = ImageOps.fit(a, (width, height), method=Image.Resampling.LANCZOS, centering=(.5, .5))
            b = ImageOps.fit(b, (width, height), method=Image.Resampling.LANCZOS, centering=(.5, .5))
        else:
            raise ValueError(f"unknown size policy: {size_policy}")
    return {"index": index, "sample_id": sample_id(item, index), "a_path": a_path,
            "b_path": b_path, "gt_path": gt_path, "a": a, "b": b,
            "original_size": original_size, "original_a": original_a, "working_size": a.size}


def base_record(index, item, metadata_path, output_dir):
    edits = item.get("edit_image", []) if isinstance(item, dict) else []
    base = metadata_path.parent
    sid = sample_id(item if isinstance(item, dict) else {}, index)
    return {"index": index, "sample_id": sid,
            "source_a": str(_portable_path(edits[0], base)) if len(edits) > 0 else "",
            "source_b": str(_portable_path(edits[1], base)) if len(edits) > 1 else "",
            "gt": str(_portable_path(item.get("image"), base) or "") if isinstance(item, dict) else "",
            "prediction": str((output_dir / f"{sid}_pred.png").resolve()),
            "original_width": None, "original_height": None, "runtime_seconds": 0.0,
            "success": False, "error": "", "actual_iterations": None}


def write_run_files(output_dir, records, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "inference_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with (output_dir / "inference_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            if not record["success"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)


def save_inputs(sample, output_dir):
    sample["a"].save(output_dir / f'{sample["sample_id"]}_input_a.png', format="PNG")
    sample["b"].save(output_dir / f'{sample["sample_id"]}_input_b.png', format="PNG")


def restore_a_size(result, sample):
    """Guarantee output size is the original oriented A size without stretching."""
    result = result.convert("RGB")
    if result.size == sample["original_size"]:
        return result
    canvas = sample.get("original_a", sample["a"]).copy()
    # center_crop_common changes sample['a']; use black only in the unreachable
    # case where a model returns a size unrelated to the working inputs.
    if canvas.size != sample["original_size"]:
        canvas = Image.new("RGB", sample["original_size"])
    left = (canvas.width - result.width) // 2
    top = (canvas.height - result.height) // 2
    canvas.paste(result, (left, top))
    return canvas


def timed():
    return time.perf_counter()
