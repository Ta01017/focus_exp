#!/usr/bin/env python3
"""Unified metadata inference adapter around the supplied Flux2 implementation."""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from metadata_dataset import base_record, bool01, load_metadata, prepare_item, restore_a_size, write_run_files


def load_external(path):
    spec = importlib.util.spec_from_file_location("focus_flux2_external", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Flux2 inference script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("build_pipeline", "run_one"):
        if not hasattr(module, name):
            raise AttributeError(f"Flux2 script must define {name}(): {path}")
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--external-script", required=True)
    p.add_argument("--lora-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--max-pixels", type=int, default=4194304)
    p.add_argument("--prompt", default="Multi-focus image fusion. Preserve sharp regions.")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--embedded-guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--overwrite", type=bool01, default=False)
    args = p.parse_args()
    script, lora = Path(args.external_script).resolve(), Path(args.lora_path).resolve()
    if not script.is_file(): raise FileNotFoundError(script)
    if not lora.is_file(): raise FileNotFoundError(lora)
    impl = load_external(script)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    pipe = impl.build_pipeline(device=args.device, torch_dtype=dtype, lora_path=str(lora))
    metadata, items = load_metadata(args.metadata, args.start_index, args.max_samples)
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in items:
        rec = base_record(index, item, metadata, out); started = time.perf_counter()
        try:
            sample = prepare_item(item, index, metadata, "error")
            rec["original_width"], rec["original_height"] = sample["original_size"]
            target = Path(rec["prediction"])
            if target.exists() and not args.overwrite:
                rec.update(success=True, error="skipped_existing"); records.append(rec); continue
            edits = item.get("edit_image") or []
            focus_a = None
            if len(edits) >= 3:
                from metadata_dataset import resolve_portable_path
                focus_a = str(resolve_portable_path(edits[2], metadata.parent))
            result = impl.run_one(
                pipe=pipe, a_path=str(sample["a_path"]), b_path=str(sample["b_path"]),
                focus_a_path=focus_a, prompt=item.get("prompt") or args.prompt,
                negative_prompt=args.negative_prompt, num_inference_steps=args.num_inference_steps,
                seed=args.seed + index, max_pixels=args.max_pixels, cfg_scale=args.cfg_scale,
                embedded_guidance=args.embedded_guidance, restore_to_original_size=True,
            )
            restore_a_size(result, sample).save(target, "PNG"); rec["success"] = True
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[FLUX2 ERROR] index={index} {rec['error']}", file=sys.stderr)
        rec["runtime_seconds"] = round(time.perf_counter() - started, 6); records.append(rec)
    config = vars(args).copy(); config.update(metadata=str(metadata), checkpoint_loaded=str(lora))
    write_run_files(out, records, config)
    if not any(row["success"] for row in records): raise SystemExit(2)


if __name__ == "__main__": main()
