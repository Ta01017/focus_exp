"""Dependency-light metadata Dataset for ReDiffuse."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadata_training import MetadataFusionDataset


class MetadataMFI_Dataset(MetadataFusionDataset):
    """Metadata RGB contract with synchronized crop-then-resize geometry."""
    def __init__(self, metadata, phase, resize, imgSzie, seed=0,
                 start_index=0, max_samples=-1, size_policy="error"):
        mode = "train" if phase == "train" else "val"
        super().__init__(metadata, mode, size=imgSzie if resize else None,
                         crop_size=256 if mode == "train" else None,
                         channels=3, value_range="minus_one_one", augment=False,
                         size_policy=size_policy, seed=seed,
                         start_index=start_index, max_samples=max_samples,
                         operation_order="crop_then_resize")
