# Metadata training audit report

审计日期：2026-08-03。结论中的 “PASS” 仅表示本机实际执行通过；“代码检查”不等同于真实模型运行。

| Method | Metadata inference | Metadata train | Metadata val | Official checkpoint | Resume | Smoke status | Remaining issue |
| ------ | ------------------ | -------------- | ------------ | ------------------- | ------ | ------------ | --------------- |
| DSIFT | 已接入，批量逐项容错 | 不适用 | 不适用 | 不适用 | 不适用 | 代码检查 | 本机无 MATLAB，未运行算法 |
| IFCNN | 已接入，严格 checkpoint | 官方仓库无完整入口 | adapter 可复用，无官方训练验证循环 | 仓库含 IFCNN-MAX/MEAN/SUM | 不适用 | Dataset PASS；模型未运行 | 环境缺 `torchvision` |
| SwinFusion MFIF | 已接入 | 原入口已切换 metadata Dataset | test loader 已切换 metadata，GT 配对 | 仓库含 MFIF G/E/optimizerG | 原 G/E/optimizer 自动恢复 | Dataset/静态检查 PASS | 未运行 GPU step；MFF loss 源码硬编码 CUDA |
| FusionDiff | 已接入，缺权重明确失败 | 原 `train.py` 已切换 | 确定 seed，执行一批 diffusion validation loss | 仓库未提供 | `--resume` 模型及可选 optimizer | Dataset/静态检查 PASS | 无官方 checkpoint；未运行模型 step |
| ReDiffuse | 已接入，严格 `model.pt` | 原 `train.py` 已切换 | 确定 seed，执行一批 diffusion validation loss | `weights/model.pt` 存在但加载失败 | `--resume` 模型及可选 optimizer | Dataset PASS；checkpoint FAIL | 源码缺少 `Condition_Noise_Predictor/B_Conv.py`，仅有 Python 3.8 pyc |
| ZMFF | 已接入逐样本优化 | 不适用，未新增监督训练 | GT 只供后续评测 | 不适用 | 不适用 | Dataset/代码检查 PASS | CPU 完整 zero-shot 优化耗时，未运行模型 |

## 数据流

- DSIFT：metadata → A/B（忽略其余 edit_image）→ MATLAB 尺寸策略 → 原 `DSIFT_Fusion`。
- IFCNN：metadata → A/B → ImageNet normalization → 原 IFCNN-MAX 融合；GT 不参与推理。
- SwinFusion MFIF：metadata → RGB A/B/GT → 同步 128 patch/flip/rotation → Y `[0,1]` tensor → 原 SwinFusion + 原 `fusion_loss_mff(A,B,fused)`；GT 与验证结果配对，但不擅自改变官方无监督 MFF loss。
- FusionDiff：metadata → RGB A/B/GT → 同步固定 resize → `[-1,1]` → 原 `GaussianDiffusion.train_losses(model,A,B,GT,t,concat_type,loss_scale)`。
- ReDiffuse：metadata → RGB A/B/GT → 同步 256 crop/resize → Y `[-1,1]` → 原 rotation-equivariant noise predictor 和 diffusion loss。
- ZMFF：metadata → A/B → 每个样本重设 seed 并重新构造网络、noise/mask inputs、optimizer → 原 zero-shot loss；GT 不进入优化。

## 主要修改文件及目的

- `baselines/metadata_dataset.py`：公共路径解析、BOM JSON、GT 强制加载、train/val/infer 模式、完整样本信息和同步几何预处理。
- `baselines/metadata_training.py`：统一 torch Dataset 字段、各模型可选通道/normalization、split 摘要和重叠警告。
- `baselines/metadata_smoke_common.py`、`baselines/smoke_metadata/*`、`pytest.ini`：两图/四图、缺失 focus、GT 来源、同步/确定性和坏样本 smoke。
- `baselines/ImageFusion/FusionDiff/dataset.py`、`train.py`：保留目录 Dataset，同时让原训练/验证循环真正迭代 metadata，增加 CLI 与 resume。
- `baselines/ReDiffuse/my_dataset.py`、`train.py`：同上，保留原 Y 通道和 256 patch 协议。
- `baselines/SwinFusion/data/dataset_metadata.py`、`data/select_dataset.py`、`main_train_swinfusion.py`：只为 MFIF 路线增加 metadata train/test Dataset，保留官方训练组件与多权重恢复逻辑。
- 根目录及各 baseline README：Windows 可复制命令、方法限制和 checkpoint 约束。

DSIFT、IFCNN、SwinFusion、FusionDiff、ReDiffuse、ZMFF 的已有 `infer_metadata` 适配经过字段审计：均只使用 `edit_image[0:2]`。DSIFT 与 ZMFF 未增加监督训练。

## 实际测试

```text
python3 -m py_compile baselines/metadata_dataset.py baselines/metadata_training.py ...
python3 baselines/metadata_smoke_common.py
python3 -m pytest -q baselines/IFCNN/test_metadata_smoke.py baselines/SwinFusion/test_metadata_smoke.py baselines/ImageFusion/FusionDiff/test_metadata_smoke.py baselines/ReDiffuse/test_metadata_smoke.py baselines/ZMFF/test_metadata_smoke.py
MetadataFusionDataset train/val tensor smoke
IFCNN official checkpoint inference attempt
ReDiffuse strict model.pt load attempt
```

结果：Python 静态编译通过；公共 smoke 通过；pytest 5 passed；torch train/val Dataset smoke 通过。IFCNN 模型尝试因缺少 `torchvision` 失败。ReDiffuse checkpoint 尝试在加载模型定义阶段失败，因为仓库缺 `B_Conv.py`。MATLAB、CUDA、FusionDiff checkpoint 均不可用，因此 DSIFT 真实推理、四个深度模型的一步训练/采样没有声称通过。

## 未解决问题

1. ReDiffuse 官方快照在当前仓库不完整：不知道缺失的 `B_Conv.py` 是否与现有 Python 3.8 pyc 完全一致，不能猜写替代实现。
2. FusionDiff 没有官方 checkpoint，正式推理必须由用户提供；随机初始化只允许训练 smoke，不能作为有效结果。
3. 当前环境缺 `torchvision`、MATLAB 和 CUDA，无法完成相应真实模型 smoke。
4. SwinFusion 官方 `fusion_loss_mff` 内部使用 `.cuda()`，CPU model step 不可运行；未修改该论文 loss。
5. 正式 train/val 必须使用不同 metadata；公共 smoke 为测试同步逻辑而故意复用同一文件，启动时会打印重叠警告。
