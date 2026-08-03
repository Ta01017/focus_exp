"""Batch metadata.json inference using SwinFusion's official MFF network."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from metadata_dataset import base_record, bool01, load_metadata, prepare_item, restore_a_size, save_inputs, write_run_files


def main():
    p = argparse.ArgumentParser(); p.add_argument("--metadata", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint"); p.add_argument("--device", default="cuda:0"); p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=-1); p.add_argument("--overwrite", type=bool01, default=False)
    p.add_argument("--save-inputs", type=bool01, default=False); p.add_argument("--size-policy", choices=("error", "resize_b_to_a", "center_crop_common"), default="error")
    args = p.parse_args(); out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    default = ROOT / "Model/Multi_Focus_Fusion/Multi_Focus_Fusion/models/10000_E.pth"
    ckpt = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else default
    if not ckpt.is_file(): raise FileNotFoundError("Official SwinFusion Multi-Focus 10000_E checkpoint required: " + str(ckpt))
    from models.network_swinfusion1 import SwinFusion
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; pass --device cpu")
    # The released MFF checkpoint is a one-channel luminance model
    # (conv_first1_A.weight is [30, 1, 3, 3]). RGB is decoded first and A's
    # chroma is restored after the official luminance fusion path.
    model = SwinFusion(upscale=1, in_chans=1, img_size=128, window_size=8, img_range=1., depths=[6]*4,
                       embed_dim=60, num_heads=[6]*4, mlp_ratio=2, upsampler=None, resi_connection="1conv")
    state = torch.load(str(ckpt), map_location=device); model.load_state_dict(state.get("params", state), strict=True); model.to(device).eval()
    meta, items = load_metadata(args.metadata, args.start_index, args.max_samples); records = []
    for index, item in items:
        rec = base_record(index, item, meta, out); started = time.perf_counter()
        try:
            target = Path(rec["prediction"])
            sample = prepare_item(item, index, meta, args.size_policy); rec["original_width"], rec["original_height"] = sample["original_size"]
            if target.exists() and not args.overwrite: rec.update(success=True, error="skipped_existing"); records.append(rec); continue
            a_ycc = np.asarray(sample["a"].convert("YCbCr"))
            b_ycc = np.asarray(sample["b"].convert("YCbCr"))
            def tensor(y): return torch.from_numpy(y.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0).to(device)
            a, b = tensor(a_ycc[:, :, 0]), tensor(b_ycc[:, :, 0]); h, w = a.shape[-2:]; ph, pw = (-h) % 8, (-w) % 8
            if ph or pw:
                mode = "reflect" if h > ph and w > pw else "replicate"; a = F.pad(a, (0, pw, 0, ph), mode=mode); b = F.pad(b, (0, pw, 0, ph), mode=mode)
            with torch.inference_mode(): pred = model(a, b)[0, :, :h, :w].clamp(0, 1).cpu().numpy()
            fused_y = np.uint8(np.rint(pred[0] * 255))
            image = Image.fromarray(np.dstack((fused_y, a_ycc[:, :, 1:])), "YCbCr").convert("RGB")
            restore_a_size(image, sample).save(target, "PNG")
            if args.save_inputs: save_inputs(sample, out)
            rec["success"] = True
        except Exception as exc: rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["runtime_seconds"] = round(time.perf_counter() - started, 6); records.append(rec)
    write_run_files(out, records, vars(args) | {"metadata": str(meta), "checkpoint_loaded": str(ckpt), "model_config": "official_multi_focus"})
    if not any(r["success"] for r in records): raise SystemExit(2)


if __name__ == "__main__": main()
