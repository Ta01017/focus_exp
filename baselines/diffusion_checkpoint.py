from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def rgb_contract(method, steps=2000, **extra):
    return {"method": method, "dataset_format": "metadata", "color_space": "RGB",
            "input_a": "edit_image[0]", "input_b": "edit_image[1]",
            "target": "image", "ignored_edit_images": "edit_image[2:]",
            "normalization": "[-1, 1]", "diffusion_steps": int(steps), **extra}


def assert_rgb_contract(contract, method=None):
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no data_contract")
    if contract.get("dataset_format") != "metadata" or contract.get("color_space") != "RGB":
        raise ValueError(f"checkpoint data contract is not metadata RGB: {contract}")
    if method and contract.get("method") != method:
        raise ValueError(f"checkpoint method mismatch: expected {method}, got {contract.get('method')}")


def build_checkpoint(method, model, optimizer, scheduler, epoch, global_step, config, contract):
    return {"format_version": 2, "method": method, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch), "global_step": int(global_step), "config": config,
            "data_contract": contract,
            "random_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state(),
                             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}


def save_checkpoint(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint(**kwargs), path)


def load_model_init(path, model, method, allow_legacy=False, map_location="cpu"):
    # Full training checkpoints intentionally contain trusted Python/NumPy RNG
    # state, which is outside PyTorch's weights-only allowlist.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        assert_rgb_contract(checkpoint.get("data_contract"), method)
        state = checkpoint["model"]
    else:
        if not allow_legacy:
            raise ValueError("legacy state_dict has unknown data contract; pass explicit legacy opt-in")
        print("[LEGACY CHECKPOINT] Data contract is unknown.")
        state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint


def resume_training(path, model, optimizer, scheduler, method, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    required = {"model", "optimizer", "epoch", "global_step", "data_contract", "random_state"}
    missing = required - set(checkpoint) if isinstance(checkpoint, dict) else required
    if missing:
        raise ValueError(f"legacy/incomplete checkpoint cannot resume; missing={sorted(missing)}")
    assert_rgb_contract(checkpoint["data_contract"], method)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    state = checkpoint["random_state"]
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"]), checkpoint
