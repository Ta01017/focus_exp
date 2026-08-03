"""Batch metadata.json inference for the official IFCNN-MAX model."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT.parent))
from metadata_dataset import (base_record, bool01, load_metadata, prepare_item,
                              restore_a_size, save_inputs, write_run_files)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--start-index", type=int, default=0); p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--overwrite", type=bool01, default=False); p.add_argument("--save-inputs", type=bool01, default=False)
    p.add_argument("--size-policy", choices=("error", "resize_b_to_a", "center_crop_common"), default="error")
    return p


def checkpoint_path(value):
    if value:
        path = Path(value).expanduser().resolve()
    else:
        path = ROOT / "Code" / "snapshots" / "IFCNN-MAX.pth"
    if not path.is_file():
        raise FileNotFoundError("Official IFCNN-MAX checkpoint is required; use --checkpoint. " + str(path))
    return path


def main():
    args = parser().parse_args(); out = Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_path(args.checkpoint)
    meta, items = load_metadata(args.metadata, args.start_index, args.max_samples)
    from model import myIFCNN
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu explicitly")
    # Prevent torchvision from downloading an unrelated ResNet: the complete official state is loaded next.
    import model as model_module
    original = model_module.models.resnet101
    model_module.models.resnet101 = lambda **kw: original(weights=None)
    model = myIFCNN(fuse_scheme=0)
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    model.to(device).eval()
    records = []
    for index, item in items:
        rec = base_record(index, item, meta, out); started = time.perf_counter()
        try:
            pred_path = Path(rec["prediction"])
            sample = prepare_item(item, index, meta, args.size_policy)
            rec["original_width"], rec["original_height"] = sample["original_size"]
            if pred_path.exists() and not args.overwrite:
                rec.update(success=True, error="skipped_existing"); records.append(rec); continue
            mean = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
            std = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
            def tensor(im):
                x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
                return torch.from_numpy((x - mean) / std).unsqueeze(0).to(device)
            with torch.inference_mode(): result = model(tensor(sample["a"]), tensor(sample["b"]))[0]
            result = (result.cpu().numpy() * std + mean).clip(0, 1)
            image = Image.fromarray(np.uint8(np.rint(result.transpose(1, 2, 0) * 255)))
            restore_a_size(image, sample).save(pred_path, "PNG")
            if args.save_inputs: save_inputs(sample, out)
            rec["success"] = True
        except Exception as exc: rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["runtime_seconds"] = round(time.perf_counter() - started, 6); records.append(rec)
    config = vars(args) | {"metadata": str(meta), "checkpoint_loaded": str(ckpt.resolve()), "fusion_rule": "IFCNN-MAX"}
    write_run_files(out, records, config)
    if not any(r["success"] for r in records): raise SystemExit(2)


if __name__ == "__main__": main()
