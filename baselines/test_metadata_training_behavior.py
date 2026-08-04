import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ReDiffuse"))
sys.path.insert(0, str(ROOT / "SwinFusion"))

from metadata_dataset import load_metadata, prepare_item
from ReDiffuse.metadata_adapter import MetadataMFI_Dataset
from SwinFusion.data.dataset_metadata import DatasetMetadataMFF


def _fixture(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    yy, xx = np.mgrid[:320, :320]
    pattern = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=2).astype(np.uint8)
    for name in ("a", "b", "gt"):
        Image.fromarray(pattern).save(images / f"{name}.png")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps([
        {"image": "images/gt.png", "edit_image": ["images/a.png", "images/b.png"]},
        {"image": "images/gt.png", "edit_image": ["images/a.png", "images/b.png",
         "images/missing_focus_a.png", "images/missing_focus_b.png"]},
    ]), encoding="utf-8")
    return metadata


def _assert_synced(sample):
    assert sample["a"].shape == sample["b"].shape == sample["target"].shape
    assert (sample["a"] == sample["b"]).all()
    assert (sample["a"] == sample["target"]).all()


def test_rediffuse_train_crop_changes_and_is_synchronized(tmp_path):
    metadata = _fixture(tmp_path)
    dataset = MetadataMFI_Dataset(metadata, "train", True, 128, seed=19)
    assert dataset.crop_size == 256
    assert dataset.operation_order == "crop_then_resize"
    torch.manual_seed(101)
    outputs = [dataset[0] for _ in range(4)]
    for sample in outputs:
        _assert_synced(sample)
        assert tuple(sample["a"].shape) == (1, 128, 128)
    assert len({sample["a"].numpy().tobytes() for sample in outputs}) > 1


def test_rediffuse_validation_uses_valid_geometry_and_is_deterministic(tmp_path):
    metadata = _fixture(tmp_path)
    # Mirrors official valid config: resize=false, imgSize=-1.
    dataset = MetadataMFI_Dataset(metadata, "valid", False, -1, seed=19)
    first, second = dataset[0], dataset[0]
    _assert_synced(first)
    assert tuple(first["a"].shape) == (1, 320, 320)
    assert (first["a"] == second["a"]).all()
    source = (ROOT / "ReDiffuse" / "train.py").read_text(encoding="utf-8")
    assert 'config["dataset"].get("valid", config["dataset"]["train"])' in source
    assert 'val_cfg.get("resize"' in source and 'val_cfg.get("imgSize"' in source


def test_swinfusion_train_augmentation_changes_but_stays_synchronized(tmp_path):
    metadata = _fixture(tmp_path)
    dataset = DatasetMetadataMFF({"metadata": str(metadata), "phase": "train",
                                  "H_size": 128, "n_channels": 3, "seed": 23})
    torch.manual_seed(202)
    outputs = [dataset[0] for _ in range(4)]
    for sample in outputs:
        _assert_synced(sample)
    assert len({sample["a"].numpy().tobytes() for sample in outputs}) > 1
    torch.manual_seed(202)
    replay = DatasetMetadataMFF({"metadata": str(metadata), "phase": "train",
                                 "H_size": 128, "n_channels": 3, "seed": 23})
    assert (outputs[0]["a"] == replay[0]["a"]).all()


def test_two_and_four_edit_metadata_ignore_everything_after_b(tmp_path):
    metadata = _fixture(tmp_path)
    path, items = load_metadata(metadata)
    two = prepare_item(items[0][1], 0, path, mode="train")
    four = prepare_item(items[1][1], 1, path, mode="train")
    assert two["a_path"] == four["a_path"]
    assert two["b_path"] == four["b_path"]
