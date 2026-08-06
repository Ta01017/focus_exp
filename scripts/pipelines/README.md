# Per-model train, infer, evaluate pipelines

每个模型一个入口：

- `swinfusion.sh`: supervised metadata-Y training, inference, evaluation
- `fusiondiff.sh`: supervised metadata-RGB training, inference, evaluation
- `rediffuse.sh`: supervised metadata-RGB training, inference, evaluation
- `ifcnn.sh`: official/pretrained inference and evaluation
- `zmff.sh`: zero-shot per-sample optimization inference and evaluation
- `dsift.sh`: Python DSIFT-MFIF non-learning inference and evaluation

所有脚本都支持三个阶段：

```bash
RUN_TRAIN=1 RUN_INFER=1 RUN_EVAL=1 bash scripts/pipelines/swinfusion.sh
```

`IFCNN`、`ZMFF`、`DSIFT` 没有可靠的监督 metadata 训练入口，`RUN_TRAIN=1` 时只会打印说明并跳过训练。

## Shared variables

- `ROOT`: 仓库根目录，默认自动识别
- `PYTHON`: Python 命令，默认 `python3`
- `OUTPUT_ROOT`: 输出根目录，默认 `outputs/pipeline`
- `CUDA_VISIBLE_GPU`: 物理 GPU 编号，默认 `0`
- `TRAIN_META`: 训练 metadata，仅监督训练模型需要
- `VAL_META`: 验证 metadata，仅监督训练模型需要
- `TEST_META`: 推理 metadata
- `MAX_SAMPLES`: 调试时限制样本数，默认 `-1`
- `OVERWRITE`: 推理是否覆盖已有结果，默认 `0`
- `SEED`: 随机种子，默认 `17`
- `RUN_TRAIN/RUN_INFER/RUN_EVAL`: 是否运行训练、推理、评估阶段

## Evaluation specs

推理后评估用 `EVAL_SPECS` 指定一个或多个验证集，每行格式：

```text
dataset|mode|metadata|inference_output_subdir|metrics
```

- `mode`: `gt` 或 `no_gt`
- `metadata`: 当前保留为记录位，评估 manifest 实际从 `inference_manifest.csv` 生成
- `inference_output_subdir`: 相对于 `$OUTPUT_ROOT/infer` 的输出子目录
- `metrics`: 可留空；`gt` 默认 `gt_main`，`no_gt` 默认 `no_gt_main`

示例，两个无 GT 验证集加一个有 GT 验证集：

```bash
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|SwinFusion-metadata-y|all_no_gt\nMFFW|no_gt|/data/meta/mffw.json|SwinFusion-metadata-y|rediffuse\nOwnPhone|gt|/data/meta/own_val.json|SwinFusion-metadata-y|gt_all'
```

评估会输出：

- `$OUTPUT_ROOT/eval/<MODEL>/<TAG>/results/<DATASET>/per_image.csv`
- `$OUTPUT_ROOT/eval/<MODEL>/<TAG>/results/<DATASET>/summary.csv`
- `$OUTPUT_ROOT/eval/<MODEL>/<TAG>/results/<DATASET>/skipped_metrics.csv`

无 GT 数据集遇到 PSNR/SSIM/LPIPS 等需要 GT 的指标时会跳过并记录 `requires_gt`。

## SwinFusion

```bash
ROOT=/mnt/d/focus_exp \
TRAIN_META=/data/meta/train.json \
VAL_META=/data/meta/val.json \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
TAG=swinfusion_metadata_y_v1 \
SWINFUSION_CKPT=/data/runs/mfif/train/SwinFusion/swinfusion_metadata_y_v1/models/latest_E.pth \
SWINFUSION_CHECKPOINT_MODE=metadata-y \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|SwinFusion-metadata-y|all_no_gt' \
bash scripts/pipelines/swinfusion.sh
```

官方 checkpoint 只推理评估：

```bash
RUN_TRAIN=0 \
TEST_META=/data/meta/lytro.json \
SWINFUSION_CKPT=/path/10000_E.pth \
SWINFUSION_CHECKPOINT_MODE=official-y \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|SwinFusion-official-y|all_no_gt' \
bash scripts/pipelines/swinfusion.sh
```

## FusionDiff

```bash
ROOT=/mnt/d/focus_exp \
TRAIN_META=/data/meta/train.json \
VAL_META=/data/meta/val.json \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
TAG=fusiondiff_metadata_rgb_v1 \
FUSIONDIFF_CKPT=/data/runs/mfif/train/FusionDiff/fusiondiff_metadata_rgb_v1/best_val_loss.pt \
SAMPLING_STEPS=2000 \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|__FUSIONDIFF_OUT__|all_no_gt' \
bash scripts/pipelines/fusiondiff.sh
```

`__FUSIONDIFF_OUT__` 会自动替换成 `FusionDiff-${checkpoint_stem}`。

## ReDiffuse

```bash
ROOT=/mnt/d/focus_exp \
REDIFFUSE_PYTHON=/tmp/mamba-root/envs/rediffuse38/bin/python \
TRAIN_META=/data/meta/train.json \
VAL_META=/data/meta/val.json \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
TAG=rediffuse_metadata_rgb_v1 \
REDIFFUSE_CKPT=/data/runs/mfif/train/ReDiffuse/rediffuse_metadata_rgb_v1/best_val_loss.pt \
REDIFFUSE_CHECKPOINT_MODE=metadata-rgb \
SAMPLING_STEPS=2000 \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|__REDIFFUSE_OUT__|all_no_gt' \
bash scripts/pipelines/rediffuse.sh
```

官方 checkpoint 只推理评估：

```bash
RUN_TRAIN=0 \
REDIFFUSE_PYTHON=/tmp/mamba-root/envs/rediffuse38/bin/python \
TEST_META=/data/meta/lytro.json \
REDIFFUSE_CKPT=/mnt/d/focus_exp/baselines/ReDiffuse/weights/model.pt \
REDIFFUSE_CHECKPOINT_MODE=official-y \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|__REDIFFUSE_OUT__|all_no_gt' \
bash scripts/pipelines/rediffuse.sh
```

## IFCNN

```bash
RUN_TRAIN=0 \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
IFCNN_CKPT=/path/IFCNN-MAX.pth \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|IFCNN|all_no_gt' \
bash scripts/pipelines/ifcnn.sh
```

## ZMFF

```bash
RUN_TRAIN=0 \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
ZMFF_ITERATIONS=1300 \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|ZMFF|all_no_gt' \
bash scripts/pipelines/zmff.sh
```

## DSIFT

```bash
RUN_TRAIN=0 \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|DSIFT|all_no_gt' \
bash scripts/pipelines/dsift.sh
```

DSIFT 正式推理入口为 Python-only。传统无 GT 指标评估也为 Python-only；QCNN 仍需官方 PyTorch 模型和权重。
