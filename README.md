# focus_exp

本仓库包含 6 个多焦点图像融合 baseline，并提供统一的 `metadata.json` 数据入口、训练脚本、推理脚本和评估工具。

核心规则固定为：

- `edit_image[0]`: 输入 A
- `edit_image[1]`: 输入 B
- `image`: GT，只在训练/有 GT 评估时需要
- `edit_image[2:]`: 永远忽略

## 1. 准备 metadata

最小格式如下：

```json
[
  {
    "image": "/data/gt/000001.png",
    "edit_image": ["/data/a/000001.png", "/data/b/000001.png"],
    "source_index": "000001"
  }
]
```

无 GT 验证集可以不写 `image`：

```json
[
  {
    "edit_image": ["/data/lytro/A/01.png", "/data/lytro/B/01.png"],
    "source_index": "01"
  }
]
```

训练和验证 metadata 必须有 `image`。推理 metadata 可以没有 `image`。

## 2. 安装依赖

公共 Python 依赖按你的环境安装。评估工具需要：

```bash
cd /mnt/d/focus_exp/mfif_eval_toolkit
python -m pip install -r requirements.txt
bash prepare_backends.sh
```

无 GT 传统指标依赖 MATLAB 和 Image Processing Toolbox。`ReDiffuse` 需要 Python 3.8 环境，例如：

```bash
REDIFFUSE_PYTHON=/tmp/mamba-root/envs/rediffuse38/bin/python
```

## 3. 一键流程

每个模型一个脚本，放在 [scripts/pipelines](</mnt/d/focus_exp/scripts/pipelines>)：

- `swinfusion.sh`
- `fusiondiff.sh`
- `rediffuse.sh`
- `ifcnn.sh`
- `zmff.sh`
- `dsift.sh`

每个脚本都支持三个阶段：

```bash
RUN_TRAIN=1 RUN_INFER=1 RUN_EVAL=1 bash scripts/pipelines/swinfusion.sh
```

`IFCNN`、`ZMFF`、`DSIFT` 没有可靠的监督训练入口。它们会跳过训练，只做推理和评估。

## 4. EVAL_SPECS 怎么写

推理后评估用 `EVAL_SPECS` 指定验证集。每行一个验证集：

```text
数据集名|gt或no_gt|metadata路径|推理输出子目录|指标组
```

`推理输出子目录` 是相对于 `$OUTPUT_ROOT/infer` 的目录名。留空时默认使用模型名，比如 `IFCNN`、`ZMFF`、`DSIFT`。

例子：

```bash
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|SwinFusion-metadata-y|all_no_gt\nOwnPhone|gt|/data/meta/own_val.json|SwinFusion-metadata-y|gt_all'
```

常用指标组：

- `gt_main`: PSNR, SSIM, LPIPS
- `gt_all`: 全部有 GT 指标
- `no_gt_main`: QMI, QAB/F, QCB, QVIFF, QCNN
- `all_no_gt`: 全部无 GT 指标
- `rediffuse`: ReDiffuse 论文常用指标组
- `all`: GT 和无 GT 指标都请求；不适用的会跳过并记录

无 GT 数据集遇到 PSNR、SSIM、LPIPS 这类需要 GT 的指标时不会报错，会在 `skipped_metrics.csv` 里标记 `requires_gt`。

## 5. SwinFusion

训练、推理、评估：

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

只用官方 checkpoint 推理评估：

```bash
RUN_TRAIN=0 \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
SWINFUSION_CKPT=/path/10000_E.pth \
SWINFUSION_CHECKPOINT_MODE=official-y \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|SwinFusion-official-y|all_no_gt' \
bash scripts/pipelines/swinfusion.sh
```

## 6. FusionDiff

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

`__FUSIONDIFF_OUT__` 会自动替换成实际输出目录，例如 `FusionDiff-best_val_loss`。

## 7. ReDiffuse

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

只用官方 checkpoint 推理评估：

```bash
RUN_TRAIN=0 \
REDIFFUSE_PYTHON=/tmp/mamba-root/envs/rediffuse38/bin/python \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
REDIFFUSE_CKPT=/mnt/d/focus_exp/baselines/ReDiffuse/weights/model.pt \
REDIFFUSE_CHECKPOINT_MODE=official-y \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|__REDIFFUSE_OUT__|all_no_gt' \
bash scripts/pipelines/rediffuse.sh
```

## 8. IFCNN

```bash
RUN_TRAIN=0 \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
IFCNN_CKPT=/path/IFCNN-MAX.pth \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|IFCNN|all_no_gt' \
bash scripts/pipelines/ifcnn.sh
```

## 9. ZMFF

```bash
RUN_TRAIN=0 \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
ZMFF_ITERATIONS=1300 \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|ZMFF|all_no_gt' \
bash scripts/pipelines/zmff.sh
```

## 10. DSIFT

```bash
RUN_TRAIN=0 \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|DSIFT|all_no_gt' \
bash scripts/pipelines/dsift.sh
```

DSIFT 需要 MATLAB。

## 11. 多验证集怎么跑

推理脚本一次只吃一个 `TEST_META`。如果要多个验证集，推荐每个验证集使用不同 `OUTPUT_ROOT` 跑一次推理，然后分别评估；如果确认不同验证集的 `sample_id` 不会重名，也可以共用同一个 `OUTPUT_ROOT` 后写多行 `EVAL_SPECS`。

例如 SwinFusion：

```bash
RUN_TRAIN=0 RUN_EVAL=0 \
TEST_META=/data/meta/lytro.json \
OUTPUT_ROOT=/data/runs/mfif_lytro \
SWINFUSION_CKPT=/path/model.pth \
SWINFUSION_CHECKPOINT_MODE=metadata-y \
bash scripts/pipelines/swinfusion.sh

RUN_TRAIN=0 RUN_EVAL=0 \
TEST_META=/data/meta/mffw.json \
OUTPUT_ROOT=/data/runs/mfif_mffw \
SWINFUSION_CKPT=/path/model.pth \
SWINFUSION_CHECKPOINT_MODE=metadata-y \
bash scripts/pipelines/swinfusion.sh

RUN_TRAIN=0 RUN_INFER=0 RUN_EVAL=1 \
OUTPUT_ROOT=/data/runs/mfif_lytro \
EVAL_SPECS=$'Lytro|no_gt|/data/meta/lytro.json|SwinFusion-metadata-y|all_no_gt\nMFFW|no_gt|/data/meta/mffw.json|SwinFusion-metadata-y|rediffuse' \
bash scripts/pipelines/swinfusion.sh
```

上面最后一步只评估 `/data/runs/mfif_lytro` 下的结果。若各验证集使用了不同 `OUTPUT_ROOT`，就分别运行一次评估，或者把对应 `inference_manifest.csv` 合并成你自己的总评估入口。

## 12. 输出在哪里

推理输出：

```text
$OUTPUT_ROOT/infer/<模型输出子目录>/
```

里面包括：

- `*_pred.png`: 融合结果
- `inference_manifest.csv`: 推理结果索引
- `errors.jsonl`: 失败样本
- `run_config.json`: 本次配置

评估输出：

```text
$OUTPUT_ROOT/eval/<MODEL>/<TAG>/results/<DATASET>/
```

里面包括：

- `per_image.csv`: 每张图的指标
- `summary.csv`: 均值、标准差、失败数
- `skipped_metrics.csv`: 跳过的指标和原因
- `run_metadata.json`: 指标来源和参数

更细的 baseline 说明见 [baselines/README.md](/mnt/d/focus_exp/baselines/README.md:1)。完整 pipeline 变量说明见 [scripts/pipelines/README.md](/mnt/d/focus_exp/scripts/pipelines/README.md:1)。

## 13. Smoke test

公共 CPU smoke：

```bash
python -m pytest -q
```

测试 metadata 位于：

```text
baselines/smoke_metadata/metadata.json
```
