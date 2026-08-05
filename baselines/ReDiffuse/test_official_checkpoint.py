"""Real CPython 3.8 smoke for the author's official-y checkpoint."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(ROOT.parent / "smoke_metadata" / "metadata.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-inference", action="store_true",
                        help="Run the standalone 2000-step metadata inference smoke (slow on CPU).")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "prepare_official_bytecode.py")], check=True)
    sys.path.insert(0, str(ROOT))
    from Condition_Noise_Predictor.Rot_E_UNet import NoisePred
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    u = config["Condition_Noise_Predictor"]["UNet"]
    model = NoisePred(3, 1, u["model_channels"], u["num_res_blocks"], u["dropout"],
                      u["time_embed_dim_mult"], u["down_sample_mult"])
    state = torch.load(ROOT / "weights" / "model.pt", map_location=args.device)
    result = model.load_state_dict(state, strict=True)
    print(f"missing_keys={result.missing_keys}")
    print(f"unexpected_keys={result.unexpected_keys}")
    model.to(args.device).eval(); torch.manual_seed(17)
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32, device=args.device),
                       torch.zeros(1, dtype=torch.long, device=args.device))
    print(f"official-y forward: PASS shape={tuple(output.shape)}")
    if not args.run_inference:
        print("metadata official-y inference: SKIPPED (pass --run-inference for full 2000 steps)")
        return
    out = ROOT / ".official_y_smoke"
    command = [sys.executable, str(ROOT / "infer_metadata.py"), "--metadata", args.metadata,
               "--output-dir", str(out), "--checkpoint", str(ROOT / "weights" / "model.pt"),
               "--checkpoint-mode", "official-y", "--device", args.device,
               "--sampling-steps", "2000", "--max-samples", "1", "--overwrite", "1"]
    subprocess.run(command, check=True)
    print("metadata official-y inference: PASS")


if __name__ == "__main__":
    main()
