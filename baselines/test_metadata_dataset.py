"""Command-line CPU smoke for metadata train/val tensors."""
import argparse
from pathlib import Path

from metadata_training import MetadataFusionDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--mode", choices=("train", "val"), default="train")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    dataset = MetadataFusionDataset(Path(args.metadata), args.mode, size=args.size,
                                    channels=args.channels, max_samples=args.max_samples,
                                    seed=args.seed)
    if not len(dataset):
        raise ValueError("metadata selection is empty")
    for position in range(len(dataset)):
        sample = dataset[position]
        if sample["a"].shape != sample["b"].shape or sample["a"].shape != sample["target"].shape:
            raise AssertionError(f"unsynchronized tensor shapes at {position}")
        print(position, sample["sample_id"], tuple(sample["a"].shape), sample["gt_path"])
    print("metadata dataset smoke: PASS")


if __name__ == "__main__":
    main()
