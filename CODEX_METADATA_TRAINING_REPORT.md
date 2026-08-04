# Metadata training audit report

审计日期：2026-08-03。结论中的 “PASS” 仅表示本机实际执行通过；“代码检查”不等同于真实模型运行。

| Method | Metadata inference | Metadata train | Metadata val | Official checkpoint | Resume | Smoke status | Remaining issue |
| ------ | ------------------ | -------------- | ------------ | ------------------- | ------ | ------------ | --------------- |
| DSIFT | 已接入，批量逐项容错 | 不适用 | 不适用 | 不适用 | 不适用 | 代码检查 | 本机无 MATLAB，未运行算法 |
| IFCNN | 已接入，严格 checkpoint | 官方仓库无完整入口 | adapter 可复用，无官方训练验证循环 | 仓库含 IFCNN-MAX/MEAN/SUM | 不适用 | Dataset PASS；模型未运行 | 环境缺 `torchvision` |
| SwinFusion MFIF | 已接入 | 原入口已切换 metadata Dataset | test loader 已切换 metadata，GT 配对 | 仓库含 MFIF G/E/optimizerG | 原 G/E/optimizer 自动恢复 | Dataset/静态检查 PASS | 未运行 GPU step；MFF loss 源码硬编码 CUDA |
| FusionDiff | 已接入，缺权重明确失败 | 原 `train.py` 已切换 | 确定 seed，执行一批 diffusion validation loss | 仓库未提供 | `--resume` 模型及可选 optimizer | Dataset/静态检查 PASS | 无官方 checkpoint；未运行模型 step |
| ReDiffuse | 已接入，严格 `model.pt` | 原 `train.py` 已切换；同步 crop-then-resize | 使用 valid 配置且确定 | `weights/model.pt` 存在但加载失败 | `--resume` 模型及可选 optimizer | Dataset PASS；checkpoint BLOCKED | 源码缺少 `Condition_Noise_Predictor/B_Conv.py` |
| ZMFF | 已接入逐样本优化 | 不适用，未新增监督训练 | GT 只供后续评测 | 不适用 | 不适用 | Dataset/代码检查 PASS | CPU 完整 zero-shot 优化耗时，未运行模型 |

## 数据流

- DSIFT：metadata → A/B（忽略其余 edit_image）→ MATLAB 尺寸策略 → 原 `DSIFT_Fusion`。
- IFCNN：metadata → A/B → ImageNet normalization → 原 IFCNN-MAX 融合；GT 不参与推理。
- SwinFusion MFIF：metadata → RGB A/B/GT → 每次访问生成新的、三图同步的 128 patch/flip/rotation → Y `[0,1]` tensor → 原 SwinFusion + 原 `fusion_loss_mff(A,B,fused)`；GT 与验证结果配对，但不擅自改变官方无监督 MFF loss。
- FusionDiff：metadata → RGB A/B/GT → 同步固定 resize → `[-1,1]` → 原 `GaussianDiffusion.train_losses(model,A,B,GT,t,concat_type,loss_scale)`。
- ReDiffuse：metadata → RGB A/B/GT → 训练时先同步随机 256 crop、再按 train 配置 resize → Y `[-1,1]` → 原 rotation-equivariant noise predictor 和 diffusion loss。验证独立读取 valid 配置，不随机裁剪。
- ZMFF：metadata → A/B → 每个样本重设 seed 并重新构造网络、noise/mask inputs、optimizer → 原 zero-shot loss；GT 不进入优化。

## 主要修改文件及目的

- `baselines/metadata_dataset.py`：公共路径解析、BOM JSON、GT 强制加载、train/val/infer 模式、完整样本信息和同步几何预处理。
- `baselines/metadata_training.py`：统一 torch Dataset 字段、各模型可选通道/normalization、split 摘要和重叠警告。
- `baselines/metadata_smoke_common.py`、`baselines/smoke_metadata/*`、`pytest.ini`：两图/四图、缺失 focus、GT 来源、同步/确定性和坏样本 smoke。
- `baselines/test_metadata_training_behavior.py`：ReDiffuse 操作顺序/动态同步裁剪、valid 几何确定性、SwinFusion 动态同步增强及两图/四图规则回归测试。
- `baselines/ImageFusion/FusionDiff/dataset.py`、`train.py`：保留目录 Dataset，同时让原训练/验证循环真正迭代 metadata，增加 CLI 与 resume。
- `baselines/ReDiffuse/my_dataset.py`、`train.py`：同上，保留原 Y 通道和 256 patch 协议。
- `baselines/ReDiffuse/metadata_adapter.py`：不依赖缺失 ReDiffuse 网络源码即可测试的 crop-then-resize Dataset adapter。
- `baselines/SwinFusion/data/dataset_metadata.py`、`data/select_dataset.py`、`main_train_swinfusion.py`：只为 MFIF 路线增加 metadata train/test Dataset，保留官方训练组件与多权重恢复逻辑。
- 根目录及各 baseline README：Windows 可复制命令、方法限制和 checkpoint 约束。

DSIFT、IFCNN、SwinFusion、FusionDiff、ReDiffuse、ZMFF 的已有 `infer_metadata` 适配经过字段审计：均只使用 `edit_image[0:2]`。DSIFT 与 ZMFF 未增加监督训练。

## 实际测试（2026-08-04 修复后）

```text
python3 -m py_compile baselines/metadata_dataset.py baselines/metadata_training.py ...
python3 baselines/metadata_smoke_common.py
python3 -m pytest -q baselines/IFCNN/test_metadata_smoke.py baselines/SwinFusion/test_metadata_smoke.py baselines/ImageFusion/FusionDiff/test_metadata_smoke.py baselines/ReDiffuse/test_metadata_smoke.py baselines/ZMFF/test_metadata_smoke.py
python3 -m pytest -q baselines/test_metadata_training_behavior.py baselines/IFCNN/test_metadata_smoke.py baselines/SwinFusion/test_metadata_smoke.py baselines/ImageFusion/FusionDiff/test_metadata_smoke.py baselines/ReDiffuse/test_metadata_smoke.py baselines/ZMFF/test_metadata_smoke.py
MetadataFusionDataset train/val tensor smoke
IFCNN official checkpoint inference attempt
ReDiffuse strict model.pt load attempt
```

### Static inspection

Python 静态编译通过。确认 ReDiffuse train Dataset 为 `crop_then_resize`，validation 从 `config["dataset"]["valid"]` 读取 resize/imgSize；确认 SwinFusion 每 epoch 调用 Dataset/Sampler `set_epoch`。所有已提交 `__pycache__` 内容已删除，根 `.gitignore` 已忽略 Python/pytest cache。

### Dataset smoke

公共 smoke 通过；针对性 pytest 为 **9 passed**。实际证明 ReDiffuse 重复训练访问产生变化的同步 256 crop、验证确定且遵守 valid 几何，SwinFusion 重复训练访问增强变化且 A/B/GT 始终同步；随机序列在重新设置 worker RNG 后可复现；两图和四图只读取 A/B。

### Model forward

未通过/未执行。IFCNN 环境缺 `torchvision`；ReDiffuse 缺 `B_Conv.py`；SwinFusion/FusionDiff 缺可用 CPU 路径或所需环境。不得从 Dataset PASS 推断 model forward PASS。

### Checkpoint load

IFCNN 未进入加载（缺 `torchvision`）。ReDiffuse 未进入严格 state load（模型定义因缺 `B_Conv.py` 失败），状态为 **BLOCKED，不是 PASS**。FusionDiff 仓库没有官方 checkpoint。SwinFusion MFIF 权重文件存在，但本轮未执行真实严格加载。

### Sampling smoke

未执行：缺 MATLAB/CUDA/完整 ReDiffuse 源码或 FusionDiff checkpoint。没有声称任何真实 sampling PASS。

## 未解决问题

1. ReDiffuse 官方快照在当前仓库不完整：缺失 `B_Conv.py` 已明确阻塞 checkpoint/forward/sampling；旧 Python 3.8 pyc 已按要求删除，不能猜写替代实现。必须从作者或经核验的原始来源取得源码后再验证。
2. FusionDiff 没有官方 checkpoint，正式推理必须由用户提供；随机初始化只允许训练 smoke，不能作为有效结果。
3. 当前环境缺 `torchvision`、MATLAB 和 CUDA，无法完成相应真实模型 smoke。
4. SwinFusion 官方 `fusion_loss_mff` 内部使用 `.cuda()`，CPU model step 不可运行；未修改该论文 loss。
5. 正式 train/val 必须使用不同 metadata；公共 smoke 为测试同步逻辑而故意复用同一文件，启动时会打印重叠警告。
