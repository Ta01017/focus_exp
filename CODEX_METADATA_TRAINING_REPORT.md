# Final metadata training and inference audit

## Environment

```text
commit: report accompanies the final commit from this audit
date: 2026-08-05
platform: Linux 6.6.87.2-microsoft-standard-WSL2 x86_64
primary Python: 3.12.3
primary PyTorch: 2.13.0+cpu
ReDiffuse Python: CPython 3.8.10
ReDiffuse PyTorch: 2.4.1+cpu
CUDA: unavailable; all GPU-only checks are 未验证
```

Older v2/v3/v4 conclusions are historical and superseded by this report.

## Status

| Method/Mode | Dataset | Forward | Backward | Checkpoint | Resume | Full inference | Ready |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DSIFT | PASS | N/A | N/A | N/A | N/A | 未验证（无 MATLAB） | No |
| IFCNN | PASS | 未验证 | N/A | 未验证 | N/A | 未验证 | No |
| SwinFusion official-y | PASS | PASS | PASS | strict PASS | PASS | PASS，1 RGB PNG | Yes on tested CPU path |
| SwinFusion metadata-y | PASS | PASS | PASS | contract + strict PASS | PASS | PASS，1 RGB PNG | Yes on tested CPU path |
| ZMFF | PASS | 未验证 | N/A | N/A | N/A | 未验证 | No |
| FusionDiff metadata-rgb | PASS | PASS | PASS | format v3 PASS | real model PASS | 未验证（2000 steps） | Train smoke only |
| ReDiffuse official-y | PASS | PASS | N/A | strict PASS | N/A | 未验证（2000 steps） | Forward only |
| ReDiffuse metadata-rgb | PASS | PASS | PASS | format v3 PASS | real model PASS | 未验证（2000 steps） | Train smoke only |

## Data contracts

- All metadata: `GT=image`, `A=edit_image[0]`, `B=edit_image[1]`; `edit_image[2:]` is ignored. Train/val require GT, inference does not. Relative paths resolve from the metadata file. UTF-8/BOM and synchronized A/B/GT geometry are covered by tests.
- SwinFusion: source RGB, model Y `[1,H,W]` in `[0,1]`; official source-based MFF loss uses A_Y/B_Y/pred_Y, while GT_Y is validation-only. Output is pred_Y plus A Cb/Cr, saved RGB. Checkpoint modes are `official-y` and `metadata-y` (`official` alias).
- FusionDiff: A/B/GT RGB, three channels, `[-1,1]`, T=2000. Only complete format-v3 checkpoints are saved.
- ReDiffuse official-y: RGB is converted with OpenCV RGB→YCrCb; model is 3→1. Cr and Cb are each fused from A/B with the author's absolute-distance-from-128 formula, then YCrCb→RGB.
- ReDiffuse metadata-rgb: A/B/GT RGB `[-1,1]`; model is 9→3 and only accepts a complete checkpoint with the matching data contract.

## Checkpoint boundary

FusionDiff and ReDiffuse checkpoints represent the state after training, validation, best-loss update and `scheduler.step()`. They contain model, optimizer, scheduler, epoch, global_step, best_val_loss, config, data contract and Python/NumPy/Torch/CUDA RNG. Validation uses `preserve_rng_state`; early-stop average loss divides by actual executed steps. Metadata training no longer writes legacy pure state_dict files.

## ReDiffuse CPython 3.8 verification

```text
requirements-py38 install: PASS
pip check: No broken requirements found
B_Conv SHA256: 62fb37e52d4c4638daed9e6b5e4bf7d5cc3f337811159b17b9246ff8d67d5fa1
pyc magic: 550d0d0a
import path: /mnt/d/focus_exp/baselines/ReDiffuse/Condition_Noise_Predictor/B_Conv.pyc
official checkpoint: missing_keys=[] unexpected_keys=[]
official-y forward: PASS (1,1,32,32)
metadata-rgb forward/backward: PASS (1,3,32,32)
```

## Commands executed

```bash
pytest -q
python3 -m py_compile <all modified Python files>
/tmp/mamba-root/envs/rediffuse38/bin/python -m py_compile <ReDiffuse modified Python files>
bash -n run_train_metadata_v3.sh run_infer_metadata_v3.sh
rg 'cuda:3|strict=False|edit_image\\[[23]\\]' <scoped paths>
rg 'metadata-rgb' baselines/SwinFusion
rg 'save_model\\(' baselines/ImageFusion/FusionDiff/train.py baselines/ReDiffuse/train.py
/tmp/mamba-root/envs/rediffuse38/bin/python baselines/ReDiffuse/prepare_official_bytecode.py
/tmp/mamba-root/envs/rediffuse38/bin/python baselines/ReDiffuse/test_official_checkpoint.py --device cpu
# Real SwinFusion network: strict official load, forward, original MFF backward, optimizer, save/resume, second step
# Real FusionDiff network: train step, format-v3 save, process-style reconstruction/resume, second step
# Real ReDiffuse 9→3 network under CPython 3.8: train step, save, reconstruction/resume, second step
# SwinFusion official-y and metadata-y inference, one sample each; output checked with file(1)
```

## Results and unresolved items

- Full pytest: **28 passed**.
- Four-image metadata with deliberately missing focus paths: PASS.
- Shell and Python 3.12/3.8 syntax: PASS.
- SwinFusion official-y and metadata-y inference each produced a 4×4 8-bit RGB PNG.
- FusionDiff complete 2000-step inference: 未验证; CPU runtime is prohibitive and no CUDA is available.
- ReDiffuse official-y complete 2000-step inference: 未验证 for the same environment constraint.
- ReDiffuse metadata-rgb complete 2000-step inference: 未验证; no completed self-trained full-sampling checkpoint and no CUDA.
- DSIFT full inference: 未验证 because MATLAB is unavailable.
- IFCNN and ZMFF full model inference: 未验证 in the primary environment.
- GPU mapping logic is unit-tested, but actual CUDA execution is 未验证 because CUDA is unavailable.
