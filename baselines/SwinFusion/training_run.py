"""Explicit SwinFusion scratch/official/resume run configuration."""
import json
import sys
from pathlib import Path


def _last(directory, suffix, required=False):
    files = []
    for path in Path(directory).glob(f"*_{suffix}.pth"):
        try:
            iteration = int(path.stem.rsplit("_", 1)[0])
        except ValueError:
            continue
        files.append((iteration, path.resolve()))
    if not files:
        if required:
            raise FileNotFoundError(f"no *_{suffix}.pth checkpoint in {directory}")
        return 0, None
    return max(files, key=lambda pair: pair[0])


def configure_training_run(opt, args):
    mode = args.init_mode
    if mode in {"scratch", "official"} and not args.output_dir:
        raise ValueError(f"--output-dir is required for --init-mode {mode}")
    if mode == "official" and not args.init_checkpoint_dir:
        raise ValueError("official mode requires --init-checkpoint-dir")
    if mode == "resume" and not args.resume_dir:
        raise ValueError("resume mode requires --resume-dir")

    resume_root = Path(args.resume_dir).resolve() if args.resume_dir else None
    output = Path(args.output_dir).resolve() if args.output_dir else resume_root
    if output.exists() and any(output.iterdir()):
        same_resume = mode == "resume" and output == resume_root
        if not same_resume and not args.overwrite_output:
            raise FileExistsError(f"output directory is non-empty: {output}; use --overwrite-output 1")
    output.mkdir(parents=True, exist_ok=True)
    paths = opt["path"]
    paths["task"] = str(output)
    paths["models"] = str(output / "checkpoints")
    paths["log"] = str(output / "logs")
    paths["options"] = str(output)
    paths["images"] = str(output / "validation")
    paths["tensorboard"] = str(output / "logs" / "tensorboard")

    current_step = 0
    paths["pretrained_netG"] = None
    paths["pretrained_netE"] = None
    paths["pretrained_optimizerG"] = None
    if mode == "official":
        source = Path(args.init_checkpoint_dir).resolve()
        _, g_path = _last(source, "G", required=True)
        _, e_path = _last(source, "E", required=True)
        paths["pretrained_netG"] = str(g_path)
        paths["pretrained_netE"] = str(e_path) if e_path else None
    elif mode == "resume":
        source = resume_root / "checkpoints" if (resume_root / "checkpoints").is_dir() else resume_root
        g_iter, g_path = _last(source, "G", required=True)
        e_iter, e_path = _last(source, "E", required=True)
        o_iter, optimizer_path = _last(source, "optimizerG", required=True)
        paths["pretrained_netG"] = str(g_path)
        paths["pretrained_netE"] = str(e_path) if e_path else None
        paths["pretrained_optimizerG"] = str(optimizer_path)
        if e_iter != g_iter or o_iter != g_iter:
            raise ValueError(f"resume checkpoint iterations disagree: G={g_iter}, E={e_iter}, optimizer={o_iter}")
        current_step = g_iter

    manifest = {"init_mode": mode, "output_dir": str(output),
                "init_checkpoint_dir": args.init_checkpoint_dir,
                "resume_dir": args.resume_dir, "current_step": current_step,
                "loaded_G": str(paths["pretrained_netG"] or ""),
                "loaded_E": str(paths["pretrained_netE"] or ""),
                "loaded_optimizerG": str(paths["pretrained_optimizerG"] or "")}
    manifest["arguments"] = vars(args)
    (output / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    rgb_mode = int(opt.get("n_channels", 1)) == 3
    contract = {"method": "SwinFusion", "task": "Multi-Focus Fusion",
                "dataset_format": "metadata", "color_space": "RGB" if rgb_mode else "OFFICIAL_CHECKPOINT_CONTRACT_UNKNOWN",
                "input_a": "edit_image[0]", "input_b": "edit_image[1]",
                "target": "image (validation pairing only; not official MFF loss)",
                "ignored_edit_images": "edit_image[2:]", "normalization": "[0, 1]",
                "formal_metadata_rgb_training": rgb_mode}
    (output / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return current_step, manifest
