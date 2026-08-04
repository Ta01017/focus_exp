# Audit of commit f329e3d6488984121256f48bd2d369e38df3c5f1

## Verdict

The metadata schema is wired correctly at code level:

- `edit_image[0]` is A.
- `edit_image[1]` is B.
- `edit_image[2:]` is ignored.
- `image` is GT in train/validation mode.
- Relative paths are resolved from the metadata file directory.
- UTF-8 BOM JSON is supported.

This is true for the shared Python reader and for the standalone MATLAB DSIFT adapter.

The commit is **not yet fully ready for an all-method one-click train/infer run**.

## Blocking and correctness issues

### 1. ReDiffuse source is incomplete

`Rot_E_UNet.py` imports:

```python
from . import B_Conv as fn
```

but `Condition_Noise_Predictor/B_Conv.py` is absent. Only
`__pycache__/B_Conv.cpython-38.pyc` is present. Do not reconstruct the source by
guessing and do not treat a version-specific `.pyc` as a portable source file.

Required action:

- obtain the real `B_Conv.py` from the authors or a verified original source;
- verify that `weights/model.pt` loads with `strict=True`;
- run one real forward and one sampling smoke test.

### 2. ReDiffuse metadata training preprocessing does not match the original code

Current adapter:

```python
size=imgSzie if resize else None
crop_size=256 if mode == "train" and not resize else None
```

The published config has `resize=true`, `imgSize=256`. Therefore the metadata
adapter resizes the whole image to 256 and performs no random crop.

The original directory dataset first selects a random 256×256 crop and then
applies the resize stage. With the default size 256, the resize is effectively
a no-op after cropping.

Required action:

- support model-specific operation order `crop_then_resize`;
- for ReDiffuse train metadata, apply a synchronized random 256 crop to A/B/GT;
- only after that, apply configured resize if enabled;
- validation must remain deterministic.

### 3. ReDiffuse validation incorrectly reuses training resize settings

Current `train.py` constructs the metadata validation dataset with
`train_resize` and `train_imgSize`. The official config has a separate valid
section with `resize=false` and `imgSize=-1`.

Required action:

```python
val_cfg = config["dataset"].get("valid", config["dataset"]["train"])
```

and pass the validation section's `resize` / `imgSize`.

### 4. Training augmentation is fixed per sample forever

`MetadataFusionDataset.__getitem__()` uses `seed + index` for training.
Consequently SwinFusion's crop/flip/rotation is exactly the same every epoch.

Required action:

- validation: keep deterministic `seed + index`;
- training: use a worker-seeded random value that changes on every access;
- maintain reproducibility by relying on the DataLoader worker seed;
- optionally add `set_epoch()` for distributed training.

### 5. FusionDiff has no released checkpoint in this repository

Training from scratch is available. Official-checkpoint inference is not
available unless the user supplies a verified trained checkpoint.

Required action:

- keep `--checkpoint` mandatory;
- document the exact checkpoint source and hash when one is obtained;
- never use random initialization for benchmark inference.

### 6. IFCNN has no complete supervised training entry in this snapshot

Do not invent an IFCNN training recipe and call it official. IFCNN should be
used for official-checkpoint inference unless a separately verified training
implementation is added and clearly labeled as a reimplementation.

### 7. SwinFusion training is not GT-supervised

The metadata adapter loads GT, but the preserved official MFF loss is based on
A/B fusion. GT is used for validation pairing/PSNR, not as the training target.

This is acceptable when the goal is to preserve the official baseline, but it
must be stated in the experiment table.

## Required Codex patch request

1. Fix issues 2–4 above.
2. Add tests proving:
   - ReDiffuse train preprocessing uses a changing synchronized 256 crop;
   - ReDiffuse validation uses the valid config and is deterministic;
   - SwinFusion training augmentation changes across repeated accesses while
     A/B/GT remain synchronized;
   - two-image and four-image metadata both use only the first two edits.
3. Do not mark ReDiffuse as PASS until `B_Conv.py` is restored and a strict
   checkpoint load plus real forward succeeds.
4. Remove committed `__pycache__` files and add them to `.gitignore`.
5. After patching, update `CODEX_METADATA_TRAINING_REPORT.md` with executed
   commands and separate:
   - static inspection,
   - dataset smoke,
   - model forward,
   - checkpoint load,
   - sampling smoke.

## Current safe scope

- Metadata inference adapter: code-level schema compatibility for all six.
- Reliable runnable inference depends on environment/checkpoints:
  - DSIFT: MATLAB required.
  - IFCNN: official checkpoint + compatible PyTorch/torchvision.
  - SwinFusion: official MFF checkpoint.
  - ZMFF: no checkpoint, per-sample optimization.
  - FusionDiff: user-supplied trained checkpoint.
  - ReDiffuse: blocked until `B_Conv.py` is restored.
- Reliable metadata training at this commit:
  - SwinFusion: code path exists; GPU smoke still required.
  - FusionDiff: code path exists; GPU smoke still required.
  - ReDiffuse: not approved because of source and preprocessing issues.
