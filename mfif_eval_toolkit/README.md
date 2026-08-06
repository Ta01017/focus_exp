# MFIF Evaluation Toolkit

统一评测双图多焦点融合、带视角偏差融合和单图/双图去模糊结果。

## 1. 支持的指标

### 有 GT

- `psnr`
- `ssim`
- `lpips`
- `mae`
- `mse`
- `ms_ssim_gt`

### 无 GT / 双源融合

| CLI 名称 | 显示名称 | 论文代码映射 |
|---|---|---|
| `qmi` | QMI | `metricMI(A,B,F,1)` |
| `qsf` | QSF | `metricZheng(A,B,F)` |
| `qs` | QS | `metricPeilla(A,B,F,1)` |
| `qcb` | QCB | `metricChenBlum(A,B,F)` |
| `qabf` | QAB/F | `Qabf(A,B,F)` |
| `qabf_analysis` | Qabf alternate | `analysis_Qabf(A,B,F)` |
| `qncie` | QNCIE | `metricWang(A,B,F)` |
| `qg` | QG | `metricXydeas(A,B,F)` |
| `qp` | QP | `metricZhao(A,B,F)` |
| `qe` | QE | `metricPeilla(A,B,F,3)` |
| `qviff` | QVIFF | `VIFF_Public(A,B,F)` |
| `ms_ssim_src` | source MS-SSIM | `analysis_MSSSIM(A,B,F)` |
| `qcnn` | QCNN | TPAMI 2024 官方 `model.py + resnet34.pth` |

> **重要：** Qabf、QAB/F、QG 在不同论文和仓库中命名容易混淆。本工具不会擅自合并：
> `qabf`、`qabf_analysis`、`qg` 分别调用三个公开入口，并在结果中保留独立列。正式论文中必须写清楚函数和仓库来源。

## 2. 内置指标组

- `gt_main`: PSNR, SSIM, LPIPS
- `gt_all`: 所有有 GT 指标
- `no_gt_main`: QMI, QAB/F, QCB, QVIFF, QCNN
- `ips`: QMI, QSF, QS, QCB, QAB/F, QNCIE
- `rediffuse`: QAB/F, QMI, QG, QP, QE, source MS-SSIM
- `all_no_gt`: 上述所有无 GT 指标的并集
- `all`: 有 GT和无 GT全部指标；每一行只计算与其 mode 相符的指标

## 3. 安装

```bash
cd mfif_eval_toolkit
python -m pip install -r requirements.txt
bash prepare_backends.sh
```

传统无 GT 指标使用仓库内 Python port，原 MATLAB 实现仅作为 reference only, not used at runtime。当前状态为 faithful Python port; MATLAB numerical parity pending/verified per metric。QCNN 使用 TPAMI 论文官方 PyTorch 模型和权重。

## 4. Manifest 格式

CSV 每行对应一个“样本 × 方法”：

```csv
dataset,sample_id,mode,method,source_a,source_b,gt,fused
Lytro,01,no_gt,Ours,/path/01_A.jpg,/path/01_B.jpg,,/path/ours/01.png
MFI-WHU,01,gt,Ours,/path/01_A.png,/path/01_B.png,/path/01_GT.png,/path/ours/01.png
```

- `mode`: `gt` 或 `no_gt`
- 无 GT 行的 `gt` 留空
- 同一个样本的不同方法各占一行
- 所有图必须尺寸一致；本工具不会偷偷 resize、crop 或二次配准

可以用目录自动生成：

```bash
python scripts/build_manifest_from_dirs.py \
  --dataset Lytro \
  --mode no_gt \
  --source-a-dir /data/Lytro/A \
  --source-b-dir /data/Lytro/B \
  --fused ReDiffuse=/results/ReDiffuse/Lytro \
  --fused Ours=/results/Ours/Lytro \
  --output manifests/lytro.csv
```

有 GT：

```bash
python scripts/build_manifest_from_dirs.py \
  --dataset Own-Phone \
  --mode gt \
  --source-a-dir /data/own/A \
  --source-b-dir /data/own/B \
  --gt-dir /data/own/GT \
  --fused P2IKT=/results/P2IKT \
  --fused Ours=/results/Ours \
  --output manifests/own_phone.csv
```

如果默认文件名归一化规则不适用，用 `--strip-a/--strip-b/--strip-gt/--strip-fused` 传正则。例如去掉 `_target`：

```bash
--strip-gt '(?i)_target$'
```

合并多个数据集 manifest：

```bash
python scripts/merge_manifests.py manifests/*.csv --output manifests/all.csv
```

## 5. 运行示例

### 单个无 GT 数据集，当前暂定五项

```bash
python evaluate.py \
  --manifest manifests/all.csv \
  --datasets Lytro \
  --mode no_gt \
  --metrics no_gt_main \
  --output-dir outputs/lytro_main
```

### Lytro，测试所有候选无 GT 指标

```bash
python evaluate.py \
  --manifest manifests/all.csv \
  --datasets Lytro \
  --metrics all_no_gt \
  --output-dir outputs/lytro_all_metrics
```

### MFFW，测试 ReDiffuse 指标组

```bash
python evaluate.py \
  --manifest manifests/all.csv \
  --datasets MFFW \
  --metrics rediffuse \
  --output-dir outputs/mffw_rediffuse
```

### 同时测试所有无 GT 数据集，但仍按数据集分别汇总

```bash
python evaluate.py \
  --manifest manifests/all.csv \
  --mode no_gt \
  --metrics all_no_gt \
  --output-dir outputs/all_no_gt
```

`summary.csv` 的分组键是 `dataset, mode, method`，因此 Lytro 和 MFFW 不会被混合平均。

### 有 GT 数据集

```bash
python evaluate.py \
  --manifest manifests/all.csv \
  --mode gt \
  --metrics gt_main \
  --output-dir outputs/all_gt
```

### GT 与无 GT 一次运行

```bash
python evaluate.py \
  --manifest manifests/all.csv \
  --metrics all \
  --output-dir outputs/all_datasets_all_metrics
```

默认情况下，有 GT 行计算 GT 指标，无 GT 行计算双源指标。无 GT 行遇到 PSNR、SSIM、LPIPS、MAE、MSE、MS-SSIM(GT) 等需要 GT 的指标时会跳过，不会报错中断；跳过原因会写入 `per_image.csv` 的 `skipped_metrics` 列，并单独保存到 `skipped_metrics.csv`。

若希望在有 GT 数据上同时计算 QMI、QAB/F、QCNN 等源图指标，用：

```bash
python evaluate.py \
  --manifest manifests/all.csv \
  --mode gt \
  --metrics all \
  --source-metrics-on-gt \
  --output-dir outputs/gt_with_all_source_metrics
```

### 只跑单个指标或多个指定指标

```bash
python evaluate.py --manifest manifests/all.csv --metrics qcb --output-dir outputs/qcb_only
python evaluate.py --manifest manifests/all.csv --metrics qmi,qabf,qcnn --output-dir outputs/selected
```

### 只测试某几个方法

```bash
python evaluate.py \
  --manifest manifests/all.csv \
  --methods ReDiffuse,Ours \
  --metrics all \
  --output-dir outputs/rediffuse_vs_ours
```

## 6. 输出

- `per_image.csv`: 每个样本的每项指标与错误信息
- `summary.csv`: 按 dataset/mode/method 分别统计均值、标准差、样本数、失败数
- `skipped_metrics.csv`: 记录每个样本被跳过的指标和原因，例如 `requires_gt`
- `run_metadata.json`: 记录指标来源、方向、后端和本次参数

## 7. 关于 QCNN

QCNN 官方脚本按每层 13×13 特征 patch 计算，分辨率很大时会消耗大量显存和时间。本工具保持原始尺寸和官方逻辑，不默认缩放。若 4K 图运行困难，应在论文中统一规定 resize 协议后再修改，不能只对某些方法缩放。

## 8. 公平性约束

- 无 GT 公开集：输入 A、B、输出 F 必须严格对应且尺寸相同。
- 有 GT 自有视角偏差集：输出直接与 A 坐标系 GT 比较。
- 不在评测阶段对每个模型单独做 ECC、Homography、光流或裁剪。
- 如果 GT 本身需要固定校正，应在生成 manifest 前统一处理，所有方法使用完全相同的 GT。

## 9. A、B、GT 位于同一个文件夹

支持。使用 `--include-a`、`--include-b`、`--include-gt` 按**文件名 stem**筛选各自角色，再用 `--strip-*` 删除角色后缀，得到共同的 `sample_id`。

例如目录：

```text
/data/mixed/
├── 000001_a.png
├── 000001_b.png
├── 000001_gt.png
├── 000002_a.png
├── 000002_b.png
└── 000002_gt.png
```

结果位于单独目录时：

```bash
python scripts/build_manifest_from_dirs.py \
  --dataset Own-Phone \
  --mode gt \
  --source-a-dir /data/mixed \
  --source-b-dir /data/mixed \
  --gt-dir /data/mixed \
  --include-a '(?i)_a$' \
  --include-b '(?i)_b$' \
  --include-gt '(?i)_gt$' \
  --strip-a '(?i)_a$' \
  --strip-b '(?i)_b$' \
  --strip-gt '(?i)_gt$' \
  --fused Ours=/results/ours \
  --output manifests/own_phone.csv
```

如果命名是 `005899_a.png`、`005899_b.png`、`005899_target.png`：

```bash
--include-a '(?i)_a$'       --strip-a '(?i)_a$' \
--include-b '(?i)_b$'       --strip-b '(?i)_b$' \
--include-gt '(?i)_target$' --strip-gt '(?i)_target$'
```

A、B、GT 和不同方法输出全都在同一目录也支持。`--fused` 可使用：

```text
METHOD=/same/directory::INCLUDE_REGEX
```

例如：

```text
/data/all/
├── 000001_a.png
├── 000001_b.png
├── 000001_gt.png
├── 000001_ours.png
└── 000001_rediffuse.png
```

生成 manifest：

```bash
python scripts/build_manifest_from_dirs.py \
  --dataset Own-Mixed \
  --mode gt \
  --source-a-dir /data/all \
  --source-b-dir /data/all \
  --gt-dir /data/all \
  --include-a '(?i)_a$' \
  --include-b '(?i)_b$' \
  --include-gt '(?i)_gt$' \
  --strip-a '(?i)_a$' \
  --strip-b '(?i)_b$' \
  --strip-gt '(?i)_gt$' \
  --strip-fused '(?i)_(?:ours|rediffuse)$' \
  --fused 'Ours=/data/all::(?i)_ours$' \
  --fused 'ReDiffuse=/data/all::(?i)_rediffuse$' \
  --output manifests/own_mixed.csv
```

注意：

- `--include-*` 决定哪些文件属于 A、B、GT 或某个输出方法。
- `--strip-*` 决定匹配时从文件名中删除什么后缀。
- 三者归一化后的 stem 必须一致，例如都变成 `000001`。
- 正则参数建议始终加单引号，避免 shell 解释特殊字符。
