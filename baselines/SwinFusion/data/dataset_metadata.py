"""Metadata adapter for the official SwinFusion Multi-Focus route."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metadata_training import MetadataFusionDataset


class DatasetMetadataMFF(MetadataFusionDataset):
    def __init__(self, opt):
        phase = opt.get("phase", "train")
        mode = "train" if phase == "train" else "val"
        metadata = opt.get("metadata")
        if not metadata:
            raise ValueError("metadata_mff dataset requires a metadata path")
        super().__init__(
            metadata, mode, crop_size=opt.get("H_size") if mode == "train" else None,
            channels=opt.get("n_channels") or 1, value_range="zero_one",
            size_policy=opt.get("size_policy") or "error",
            seed=opt.get("seed") or 0, start_index=opt.get("start_index") or 0,
            max_samples=opt.get("max_samples") if opt.get("max_samples") is not None else -1,
            augment=mode == "train")
