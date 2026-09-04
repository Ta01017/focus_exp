"""Shared, dependency-light metadata.json handling for fusion baselines."""
from __future__ import annotations

import csv
import json
import os
import re
import time
import random
from pathlib import Path, PureWindowsPath

import numpy as np
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


def resolve_portable_path(raw, base):
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


# Backward-compatible private name used by the first inference adapters.
_portable_path = resolve_portable_path


def get_sample_id(item, index):
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
    raw = raw.zfill(6) if raw.isdigit() else raw
    # Names and source_index values are commonly reused by every constituent
    # dataset in a mixed metadata file.  Include both provenance (when
    # available) and the immutable metadata position so predictions can never
    # silently overwrite one another.
    dataset = re.sub(r"[^0-9A-Za-z._-]+", "_", str(item.get("source_dataset", ""))).strip("._-")
    prefix = f"{dataset}_" if dataset else ""
    return f"{prefix}{raw}__idx{int(index):06d}"


sample_id = get_sample_id


def load_metadata(metadata, start_index=0, max_samples=-1):
    metadata = Path(metadata).expanduser().resolve()
    with metadata.open("r", encoding="utf-8-sig") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError("metadata.json top level must be a list")
    stop = None if max_samples < 0 else start_index + max_samples
    return metadata, list(enumerate(items))[start_index:stop]


def validate_training_item(item, index=None):
    if not isinstance(item, dict):
        raise ValueError(f"metadata item {index} must be an object")
    if not item.get("image"):
        raise ValueError(f"metadata item {index} requires 'image' (GT) in train/val mode")


def inspect_item_paths(item, index, metadata_path):
    """Validate schema and resolve A/B/GT strings without opening any image."""
    if not isinstance(item, dict):
        raise ValueError(f"metadata index={index}: item must be an object")
    edits = item.get("edit_image")
    if not isinstance(edits, list) or len(edits) < 2:
        raise ValueError(f"metadata index={index}: edit_image must contain at least 2 paths")
    metadata_dir = Path(metadata_path).resolve().parent
    gt_value = item.get("image")
    return {"index": index, "sample_id": get_sample_id(item, index),
            "a_path": resolve_portable_path(edits[0], metadata_dir),
            "b_path": resolve_portable_path(edits[1], metadata_dir),
            "gt_path": resolve_portable_path(gt_value, metadata_dir) if gt_value else None,
            "source_index": item.get("source_index")}


def _open_rgb(path, label):
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} image not found: {path}")
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
        return image
    except Exception as exc:
        raise ValueError(f"damaged or unsupported {label} image {path}: {exc}") from exc


def prepare_item(item, index, metadata_path, size_policy="error", mode="infer"):
    if not isinstance(item, dict):
        raise ValueError("metadata item must be an object")
    edits = item.get("edit_image")
    if not isinstance(edits, list) or len(edits) < 2:
        raise ValueError("edit_image must contain at least two paths")
    if mode not in {"infer", "train", "val"}:
        raise ValueError("mode must be 'infer', 'train', or 'val'")
    if mode != "infer":
        validate_training_item(item, index)
    metadata_path = Path(metadata_path).resolve()
    base = metadata_path.parent
    a_path = resolve_portable_path(edits[0], base)
    b_path = resolve_portable_path(edits[1], base)  # deliberately ignore [2:]
    gt_path = resolve_portable_path(item.get("image"), base)
    a = _open_rgb(a_path, "A")
    b = _open_rgb(b_path, "B")
    target = _open_rgb(gt_path, "GT") if gt_path is not None else None
    original_size = a.size
    original_a = a.copy()
    images = [a, b] + ([target] if target is not None else [])
    sizes = [image.size for image in images]
    if len(set(sizes)) != 1:
        if size_policy == "error":
            raise ValueError(f"A/B/GT size mismatch: {sizes}")
        if size_policy in {"resize_b_to_a", "resize_all_to_a"}:
            b = b.resize(a.size, Image.Resampling.BICUBIC)
            if target is not None:
                target = target.resize(a.size, Image.Resampling.BICUBIC)
        elif size_policy == "center_crop_common":
            width = min(image.width for image in images)
            height = min(image.height for image in images)
            def center_crop(image):
                left, top = (image.width-width)//2, (image.height-height)//2
                return image.crop((left, top, left+width, top+height))
            a, b = center_crop(a), center_crop(b)
            if target is not None:
                target = center_crop(target)
        else:
            raise ValueError(f"unknown size policy: {size_policy}")
    return {"index": index, "sample_id": get_sample_id(item, index), "a_path": a_path,
            "b_path": b_path, "gt_path": gt_path, "a": a, "b": b,
            "image_a": a, "image_b": b, "target": target,
            "prompt": item.get("prompt", ""), "source_dataset": item.get("source_dataset", ""),
            "source_index": item.get("source_index"), "original_size": original_size,
            "original_a": original_a, "working_size": a.size}


def synchronized_preprocess(sample, size=None, crop_size=None, mode="val", seed=None,
                            hflip=False, vflip=False, rotate90=False,
                            operation_order="resize_then_crop", pad_multiple=None):
    """Apply one set of geometry parameters to A, B, and GT."""
    images = [sample["image_a"], sample["image_b"]]
    if sample.get("target") is not None:
        images.append(sample["target"])
    rng = random.Random(seed)
    def pad_all(values):
        width, height = values[0].size
        target_width, target_height = width, height
        if crop_size is not None:
            crop = (crop_size, crop_size) if isinstance(crop_size, int) else tuple(crop_size)
            target_width = max(target_width, crop[0])
            target_height = max(target_height, crop[1])
        if pad_multiple:
            target_width = ((target_width + pad_multiple - 1) // pad_multiple) * pad_multiple
            target_height = ((target_height + pad_multiple - 1) // pad_multiple) * pad_multiple
        if (target_width, target_height) == (width, height):
            return values
        # Inference/validation padding is placed on the bottom/right so the
        # original valid rectangle is exactly [:height, :width].  Training
        # undersize padding remains symmetric before the random crop.
        left = 0 if crop_size is None else (target_width - width) // 2
        right = target_width - width - left
        top = 0 if crop_size is None else (target_height - height) // 2
        bottom = target_height - height - top
        result = []
        for image in values:
            array = np.asarray(image)
            pads = ((top, bottom), (left, right))
            if array.ndim == 3:
                pads += ((0, 0),)
            result.append(Image.fromarray(np.pad(array, pads, mode="edge")))
        return result

    def resize_all(values):
        if size is None:
            return values
        output_size = (size, size) if isinstance(size, int) else tuple(size)
        return [image.resize(output_size, Image.Resampling.BICUBIC) for image in values]

    def crop_all(values):
        if crop_size is None:
            return values
        crop = (crop_size, crop_size) if isinstance(crop_size, int) else tuple(crop_size)
        width, height = values[0].size
        if width < crop[0] or height < crop[1]:
            raise ValueError(f"crop {crop} exceeds image size {(width, height)}")
        if mode == "train":
            left, top = rng.randint(0, width-crop[0]), rng.randint(0, height-crop[1])
        else:
            left, top = (width-crop[0])//2, (height-crop[1])//2
        return [image.crop((left, top, left+crop[0], top+crop[1])) for image in values]

    if operation_order == "resize_then_crop":
        images = crop_all(pad_all(resize_all(images)))
    elif operation_order == "crop_then_resize":
        images = resize_all(crop_all(pad_all(images)))
    else:
        raise ValueError("operation_order must be resize_then_crop or crop_then_resize")
    if mode == "train":
        if hflip and rng.random() < .5:
            images = [ImageOps.mirror(image) for image in images]
        if vflip and rng.random() < .5:
            images = [ImageOps.flip(image) for image in images]
        if rotate90:
            turns = rng.randrange(4)
            if turns:
                images = [image.rotate(90*turns, expand=True) for image in images]
    result = dict(sample)
    result["image_a"] = result["a"] = images[0]
    result["image_b"] = result["b"] = images[1]
    result["target"] = images[2] if len(images) == 3 else None
    result["working_size"] = images[0].size
    return result


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
