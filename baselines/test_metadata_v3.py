import hashlib
import json
import os
import random
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SwinFusion"))

from diffusion_checkpoint import (load_model_init, resume_training, rgb_contract,
                                  save_checkpoint)
from metadata_training import normalized_tensor_to_rgb, pil_rgb_to_tensor
from training_run import configure_training_run
from utils.utils_option import parse
from data.dataset_metadata import DatasetMetadataMFF


@pytest.mark.parametrize("rgb", [(255, 0, 0), (0, 0, 255)])
def test_rgb_channel_order_and_png_roundtrip(rgb, tmp_path):
    image = Image.new("RGB", (4, 3), rgb)
    tensor = pil_rgb_to_tensor(image)
    expected = torch.tensor(rgb, dtype=torch.float32).div(127.5).sub(1)
    assert torch.allclose(tensor[:, 0, 0], expected)
    output = tmp_path / "roundtrip.png"
    normalized_tensor_to_rgb(tensor).save(output)
    assert Image.open(output).convert("RGB").getpixel((0, 0)) == rgb


def test_full_checkpoint_resume_and_contract_rejection(tmp_path):
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
    loss = model(torch.ones(1, 2)).sum(); loss.backward(); optimizer.step(); scheduler.step()
    path = tmp_path / "full.pt"
    save_checkpoint(path, method="FusionDiff", model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=4, global_step=17, best_val_loss=0.125, config={"T": 2000},
                    contract=rgb_contract("FusionDiff", 2000))
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 2)
    epoch, step, checkpoint = resume_training(
        path, restored, restored_optimizer, restored_scheduler, "FusionDiff")
    assert (epoch, step) == (5, 17)
    assert checkpoint["format_version"] == 3 and checkpoint["best_val_loss"] == 0.125
    assert checkpoint["optimizer"]["state"]
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), restored.state_dict().values()))

    bad = torch.load(path, weights_only=False)
    bad["data_contract"]["color_space"] = "BGR"
    bad_path = tmp_path / "bad.pt"; torch.save(bad, bad_path)
    with pytest.raises(ValueError, match="metadata RGB"):
        load_model_init(bad_path, restored, "FusionDiff")
    legacy = tmp_path / "legacy.pt"; torch.save(model.state_dict(), legacy)
    with pytest.raises(ValueError, match="legacy"):
        load_model_init(legacy, restored, "FusionDiff")


@pytest.mark.parametrize("method", ["FusionDiff", "ReDiffuse"])
def test_resume_matches_continuous_optimizer_scheduler_rng(tmp_path, method):
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    model = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.5)
    def step(m, o):
        o.zero_grad(); loss=m(torch.rand(1, 2)).square().mean(); loss.backward(); o.step()
    step(model, optimizer); scheduler.step()
    checkpoint = tmp_path / "resume.pt"
    contract = (rgb_contract("ReDiffuse", model_mode="metadata-rgb", in_channels=9, out_channels=3)
                if method == "ReDiffuse" else rgb_contract("FusionDiff"))
    save_checkpoint(checkpoint, method=method, model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=0, global_step=1, best_val_loss=.4, config={},
                    contract=contract)
    step(model, optimizer)
    continuous = {k: v.clone() for k, v in model.state_dict().items()}
    continuous_random = (random.random(), float(np.random.rand()), torch.rand(()).item())

    resumed = torch.nn.Linear(2, 1); ro = torch.optim.AdamW(resumed.parameters(), lr=.9)
    rs = torch.optim.lr_scheduler.StepLR(ro, 1, gamma=.5)
    epoch, global_step, state = resume_training(checkpoint, resumed, ro, rs, method)
    assert (epoch, global_step, state["best_val_loss"]) == (1, 1, .4)
    assert rs.last_epoch == scheduler.last_epoch and ro.param_groups[0]["lr"] == scheduler.get_last_lr()[0]
    step(resumed, ro)
    assert all(torch.allclose(continuous[k], resumed.state_dict()[k]) for k in continuous)
    assert (random.random(), float(np.random.rand()), torch.rand(()).item()) == continuous_random


def _minimal_swin_option(tmp_path):
    option = {"task": "x", "gpu_ids": [7], "n_channels": 3, "scale": 1,
              "datasets": {}, "path": {"root": str(tmp_path)}, "netG": {}, "train": {}}
    path = tmp_path / "option.json"; path.write_text(json.dumps(option), encoding="utf-8")
    return path


@pytest.mark.parametrize(("visible", "logical"), [("2", [0]), ("1,3", [0, 1]), ("0,1,2", [0, 1, 2])])
def test_swin_external_gpu_mapping(monkeypatch, tmp_path, visible, logical):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    option = parse(str(_minimal_swin_option(tmp_path)))
    assert option["gpu_ids"] == logical
    assert option["num_gpu"] == len(logical)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible


def test_swin_output_and_tensorboard_are_isolated(tmp_path):
    output = tmp_path / "run"
    opt = {"n_channels": 3, "path": {"root": str(tmp_path)}}
    args = Namespace(init_mode="scratch", output_dir=str(output),
                     init_checkpoint_dir=None, resume_dir=None, overwrite_output=0)
    configure_training_run(opt, args)
    assert Path(opt["path"]["tensorboard"]).is_relative_to(output)
    source = (ROOT / "SwinFusion" / "models" / "model_plain.py").read_text(encoding="utf-8")
    assert "self.opt['path'].get(" in source and "'tensorboard'" in source


def test_swin_metadata_adapter_converts_rgb_to_single_channel_y(tmp_path):
    images = tmp_path / "images"; images.mkdir()
    for name, color in (("a", (255, 0, 0)), ("b", (0, 0, 255)), ("gt", (0, 255, 0))):
        Image.new("RGB", (16, 16), color).save(images / f"{name}.png")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps([{"image": "images/gt.png",
                                     "edit_image": ["images/a.png", "images/b.png"]}]), encoding="utf-8")
    sample = DatasetMetadataMFF({"metadata": str(metadata), "phase": "test",
                                 "n_channels": 3, "H_size": 8})[0]
    assert sample["A"].shape == sample["B"].shape == sample["GT"].shape == (1, 16, 16)
    expected_y = Image.open(images / "a.png").convert("YCbCr").getchannel("Y").getpixel((0, 0)) / 255
    assert sample["A"][0, 0, 0].item() == pytest.approx(expected_y)


def test_rediffuse_diffusion_cpu_device_follows_inputs(monkeypatch):
    # Diffusion imports optional image helpers that are irrelevant to this unit test.
    import types
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    fake_utils = types.ModuleType("utils"); fake_utils.tensor2img = lambda value: value
    monkeypatch.setitem(sys.modules, "utils", fake_utils)
    sys.path.insert(0, str(ROOT / "ReDiffuse"))
    sys.modules.pop("Diffusion", None)
    from Diffusion import GaussianDiffusion
    diffusion = GaussianDiffusion(8, "linear")
    source = torch.randn(2, 3, 8, 8)
    timestep = torch.tensor([0, 7], dtype=torch.long)
    noisy = diffusion.q_sample(source, timestep)
    assert noisy.device == source.device and noisy.dtype == source.dtype
    assert "cuda:3" not in (ROOT / "ReDiffuse" / "Diffusion.py").read_text(encoding="utf-8")


def test_v3_scripts_enforce_gpu_resume_and_validation_contracts():
    train = (ROOT.parent / "run_train_metadata_v3.sh").read_text(encoding="utf-8")
    infer = (ROOT.parent / "run_infer_metadata_v3.sh").read_text(encoding="utf-8")
    assert "INIT_CHECKPOINT and RESUME cannot be used together" in train
    assert 'if [[ -n "$RESUME" ]]' in train and "--sample-val" not in train
    assert 'export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_GPU"' in infer
    assert 'SWINFUSION_CHECKPOINT_MODE="${SWINFUSION_CHECKPOINT_MODE:-official-y}"' in infer


def test_official_b_conv_is_preserved_and_wrong_python_is_clear():
    pyc = ROOT / "ReDiffuse" / "Condition_Noise_Predictor" / "__pycache__" / "B_Conv.cpython-38.pyc"
    assert pyc.is_file()
    assert hashlib.sha256(pyc.read_bytes()).hexdigest() == "62fb37e52d4c4638daed9e6b5e4bf7d5cc3f337811159b17b9246ff8d67d5fa1"
    if sys.version_info[:2] != (3, 8):
        result = subprocess.run([sys.executable, str(ROOT / "ReDiffuse" / "prepare_official_bytecode.py")],
                                text=True, capture_output=True)
        assert result.returncode != 0
        assert "requires CPython 3.8" in result.stderr
