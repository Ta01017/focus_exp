# focus_exp

本仓库包含 6 个多焦点图像融合 baseline，并提供统一的 `metadata.json` 数据入口、训练脚本、推理脚本和评估工具。

## 当前混合训练集一键运行

服务器上的 `mfif_train_mix_v1` 与 `real_mfif_zedd_selfshot_v4_0901` 已有专用入口：

```bash
cd /path/to/focus_exp
GPUS=0,1,2 OUTPUT_ROOT=/data/runs/mfif_mix_v1 bash run_mix_v1_all.sh
```

`GPUS` 可以传 1、2 或 3 张当前空闲卡，例如 `GPUS=2`、`GPUS=2,4`、`GPUS=1,3,6`；不传时会使用 `nvidia-smi` 列出的所有卡。SwinFusion 训练阶段使用这些卡做 DataParallel，之后 SwinFusion、IFCNN、ZMFF、DSIFT 按卡分组并行完成推理和评估。FusionDiff 与 ReDiffuse 不在该入口中。

脚本默认数据路径正是：

```text
/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/mfif_train_mix_v1/mfif_train_mix_v1/metadata_train_mix_v1_balanced.json
/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/real_mfif_zedd_selfshot_v4_0901/metadata_val_final.json
```

训练数据允许不同样本具有不同宽高。每个样本的 A/B/GT 会先校验同尺寸，再使用完全同步的补边、随机裁剪和增强。这里不是先把整张图拉伸放大：当宽或高小于裁剪窗口时，只在四周复制边缘像素补足；达到裁剪尺寸后，从 A/B/GT 的同一坐标裁出 patch。SwinFusion 默认裁剪 `128x128`，FusionDiff 和 ReDiffuse 默认裁剪 `256x256`；后两者按原配置在裁剪后统一为 `256x256`。验证推理按每张原图处理，只临时补齐到网络所需的 8 倍数，保存前裁回原始宽高。

FusionDiff、ReDiffuse 的 metadata 训练适配同样支持这个混合尺寸训练集，但按要求没有加入 `run_mix_v1_all.sh`。需要单独运行时仍使用 `scripts/pipelines/fusiondiff.sh` 和 `scripts/pipelines/rediffuse.sh`。IFCNN、ZMFF、DSIFT 不做监督 batch 训练，因此不需要训练 patch 对齐：推理时逐张读取原尺寸，样本内 A/B 尺寸一致即可。

常用配置：

```bash
# 快速试跑
GPUS=0 TRAIN_MAX_SAMPLES=20 INFER_MAX_SAMPLES=10 MAX_TRAIN_STEPS=20 \
  OUTPUT_ROOT=/data/runs/mfif_mix_smoke bash run_mix_v1_all.sh

# 完整训练（默认也是 20000 步）；显存不足时减小 batch 或裁剪尺寸
GPUS=0,1 TRAIN_BATCH_SIZE=4 TRAIN_CROP_SIZE=128 MAX_TRAIN_STEPS=20000 \
  OUTPUT_ROOT=/data/runs/mfif_mix_v1 bash run_mix_v1_all.sh

# 已训练完，只重跑推理与评估
RUN_TRAIN=0 GPUS=0,1,2 OUTPUT_ROOT=/data/runs/mfif_mix_v1 \
  bash run_mix_v1_all.sh
```

可覆盖变量包括 `TRAIN_META`、`VAL_META`、`PYTHON`、`TAG`、`NUM_WORKERS`、`IFCNN_CKPT`、`SWINFUSION_CKPT`、`ZMFF_ITERATIONS`、`EVAL_METRICS` 和 `OVERWRITE`。验证 metadata 全部含 GT 时会自动使用 `all` 指标并同时启用源图指标；没有 GT 时自动使用 `all_no_gt`。

推理和评估全部成功后，脚本默认把 DSIFT、IFCNN、SwinFusion、ZMFF 复制整理到 `RealSceneVal68/<方法>/{manifest,metrics,predictions}`。归档会先在目标旁建立临时目录并校验数量，成功后才覆盖这四种方法的旧子目录；原始运行输出不会删除，FULX2.0_ORIGIN、FusionDiff、ReDiffuse_ORIGIN 不会改动。使用 `RUN_ARCHIVE=0` 可关闭，或用 `ARCHIVE_ROOT=/path` 指定其他位置。

一键流程还会在普通评估后运行四区域指标。它从验证 metadata 的 `focus_a/focus_b`（或 `edit_image[2]/edit_image[3]`）生成 region manifest，默认调用服务器上的 `pixrestore_mfif_paper_suite_v7_20260831/tools/region_eval_v2.py`，参数与 `run_region_real_v2.sh` 一致。结果写入 `$OUTPUT_ROOT/region_eval`，并随归档复制为 `manifest/region_manifest_v2.csv`、`metrics/region_metrics_per_image.csv`、`metrics/region_metrics_summary.csv` 和 `metrics/region_eval.log`。可用 `RUN_REGION_EVAL=0` 关闭，或通过 `REGION_EVAL`、`REGION_PYTHON` 指定评估器和环境。

注意：四种方法中只有 SwinFusion 有监督训练流程；IFCNN 加载官方 checkpoint，ZMFF 是逐样本零样本优化，DSIFT 是非学习算法。这三种显示“跳过训练”属于预期行为。

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

正式推理和评估路径为 Python-only。无 GT 传统指标是对公开 MATLAB 实现的 Python 移植，当前状态为 faithful Python port; MATLAB numerical parity pending/verified per metric。`ReDiffuse` 需要 Python 3.8 环境，例如：

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

DSIFT 正式推理入口为 Python-only。原 `.m` 文件仅保留为 reference only, not used at runtime。

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
