import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SwinFusion"))
sys.path.insert(0, str(ROOT / "ReDiffuse"))

from diffusion_sampling import validated_sampling_steps
from metadata_training import MetadataFusionDataset, warn_split_overlap
from ReDiffuse.runtime_check import require_official_b_conv
from SwinFusion.training_run import configure_training_run
from SwinFusion.utils import utils_option


def test_sampling_steps_reject_fake_respacing():
    config = {"diffusion_model": {"T": 2000}}
    assert validated_sampling_steps(config, None) == 2000
    assert validated_sampling_steps(config, 2000) == 2000
    with pytest.raises(ValueError, match="does not provide verified timestep respacing"):
        validated_sampling_steps(config, 10)


def test_rediffuse_missing_source_is_clear():
    with pytest.raises(RuntimeError, match="Do not reconstruct this file by guessing"):
        require_official_b_conv()


def test_external_cuda_visible_devices_is_preserved(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    config = ROOT / "SwinFusion" / "options" / "swinir" / "train_swinfusion_mff.json"
    utils_option.parse(str(config), is_train=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def _opt():
    return {"path": {"root": "unused"}}


def _args(mode, output, official=None, resume=None, overwrite=0):
    return Namespace(init_mode=mode, output_dir=str(output) if output else None,
                     init_checkpoint_dir=str(official) if official else None,
                     resume_dir=str(resume) if resume else None,
                     overwrite_output=overwrite)


def test_swinfusion_explicit_init_modes(tmp_path):
    scratch = tmp_path / "scratch"
    step, manifest = configure_training_run(_opt(), _args("scratch", scratch))
    assert step == 0 and not manifest["loaded_G"] and (scratch / "train_manifest.json").is_file()

    official = tmp_path / "official"
    official.mkdir()
    (official / "10000_G.pth").touch(); (official / "10000_E.pth").touch()
    step, manifest = configure_training_run(_opt(), _args("official", tmp_path / "finetune", official))
    assert step == 0 and manifest["loaded_G"].endswith("10000_G.pth")
    assert not manifest["loaded_optimizerG"]

    previous = tmp_path / "previous"
    checkpoints = previous / "checkpoints"
    checkpoints.mkdir(parents=True)
    for suffix in ("G", "E", "optimizerG"):
        (checkpoints / f"12000_{suffix}.pth").touch()
    step, manifest = configure_training_run(_opt(), _args("resume", None, resume=previous))
    assert step == 12000 and manifest["loaded_optimizerG"].endswith("12000_optimizerG.pth")


def test_overlap_inspection_does_not_decode_images(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps([{"image": "missing_gt.png",
                                     "edit_image": ["missing_a.png", "missing_b.png",
                                                    "also_missing_focus.png"],
                                     "source_index": 7}]), encoding="utf-8")
    train = MetadataFusionDataset(metadata, "train")
    val = MetadataFusionDataset(metadata, "val")
    import metadata_training
    monkeypatch.setattr(metadata_training, "prepare_item",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded image")))
    overlaps = warn_split_overlap(train, val)
    assert overlaps["sample_id"] and overlaps["a_path"] and overlaps["gt_path"]
    with pytest.raises(ValueError, match="overlap"):
        warn_split_overlap(train, val, fail_on_overlap=True)
