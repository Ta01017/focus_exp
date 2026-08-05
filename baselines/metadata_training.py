"""Torch dataset adapter shared by trainable fusion baselines."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metadata_dataset import (inspect_item_paths, load_metadata, prepare_item,
                              synchronized_preprocess)


def _tensor(image, channels, value_range):
    if channels == 1:
        image = image.convert("YCbCr").getchannel("Y")
        array = np.asarray(image, dtype=np.float32)[None, ...] / 255.0
    else:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array.copy())
    return tensor.mul(2).sub(1) if value_range == "minus_one_one" else tensor


def pil_rgb_to_tensor(image, value_range="minus_one_one"):
    """Canonical metadata RGB CHW conversion used by train/val/infer."""
    return _tensor(image.convert("RGB"), 3, value_range)


def normalized_tensor_to_rgb(tensor):
    """Convert CHW or 1xCHW [-1,1] tensor to a PIL RGB image."""
    from PIL import Image
    value = tensor.detach().cpu()
    if value.ndim == 4:
        value = value[0]
    array = ((value.clamp(-1, 1).numpy().transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)
    return Image.fromarray(array, "RGB")


class MetadataFusionDataset(Dataset):
    """A/B/GT metadata adapter with synchronized model-specific geometry."""

    def __init__(self, metadata, mode, *, size=None, crop_size=None, channels=3,
                 value_range="zero_one", size_policy="error", seed=0,
                 start_index=0, max_samples=-1, augment=False,
                 operation_order="resize_then_crop"):
        if mode not in {"train", "val"}:
            raise ValueError("MetadataFusionDataset mode must be train or val")
        self.metadata_path, self.items = load_metadata(metadata, start_index, max_samples)
        self.mode, self.size, self.crop_size = mode, size, crop_size
        self.channels, self.value_range = channels, value_range
        self.size_policy, self.seed, self.augment = size_policy, seed, augment
        self.operation_order = operation_order
        self.epoch = 0
        self._access_counter = 0

    def set_epoch(self, epoch):
        """Set distributed epoch; DataLoader worker RNG still changes per access."""
        self.epoch = int(epoch)
        self._access_counter = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, position):
        index, item = self.items[position]
        sample = prepare_item(item, index, self.metadata_path,
                              size_policy=self.size_policy, mode=self.mode)
        if self.mode == "train":
            worker_seed = torch.initial_seed()
            transform_seed = (self.seed + self.epoch * 1_000_003 + index * 97
                              + worker_seed + self._access_counter) % (2**32)
            self._access_counter += 1
        else:
            transform_seed = self.seed + index
        sample = synchronized_preprocess(
            sample, size=self.size, crop_size=self.crop_size, mode=self.mode,
            seed=transform_seed, hflip=self.augment, vflip=self.augment,
            rotate90=self.augment, operation_order=self.operation_order)
        a = _tensor(sample["image_a"], self.channels, self.value_range)
        b = _tensor(sample["image_b"], self.channels, self.value_range)
        target = _tensor(sample["target"], self.channels, self.value_range)
        return {
            "a": a, "b": b, "target": target,
            "A": a, "B": b, "GT": target,
            "index": index, "sample_id": sample["sample_id"],
            "a_path": str(sample["a_path"]), "b_path": str(sample["b_path"]),
            "gt_path": str(sample["gt_path"]), "A_path": str(sample["a_path"]),
            "B_path": str(sample["b_path"]), "GT_path": str(sample["gt_path"]),
            "prompt": sample["prompt"], "source_dataset": sample["source_dataset"],
            "source_index": "" if sample["source_index"] is None else str(sample["source_index"]),
        }


def describe_metadata_split(dataset, name):
    print(f"{name} metadata: {dataset.metadata_path}")
    print(f"{name} samples: {len(dataset)}")
    ids = []
    for position in range(min(3, len(dataset))):
        index, item = dataset.items[position]
        sample = inspect_item_paths(item, index, dataset.metadata_path)
        ids.append(sample["sample_id"])
        print(f"{name}[{position}] A={sample['a_path']} B={sample['b_path']} GT={sample['gt_path']}")
    inspected = [inspect_item_paths(item, index, dataset.metadata_path)
                 for index, item in dataset.items]
    all_ids = [sample["sample_id"] for sample in inspected]
    duplicates = len(all_ids) != len(set(all_ids))
    print(f"{name} duplicate sample_id: {duplicates}")
    return inspected


def warn_split_overlap(train_dataset, val_dataset, fail_on_overlap=False):
    train = describe_metadata_split(train_dataset, "train")
    val = describe_metadata_split(val_dataset, "val")
    fields = ("sample_id", "gt_path", "a_path", "source_index")
    found = {}
    for field in fields:
        left = {str(x[field]) for x in train if x.get(field) is not None}
        right = {str(x[field]) for x in val if x.get(field) is not None}
        overlap = left & right
        found[field] = overlap
        if overlap:
            print(f"WARNING: train/val {field} overlap ({len(overlap)}): {sorted(overlap)[:10]}")
    if fail_on_overlap and any(found.values()):
        raise ValueError("train/val overlap found and fail_on_overlap is enabled")
    return found
