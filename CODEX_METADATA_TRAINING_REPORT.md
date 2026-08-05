# Metadata training audit report

> Current status: v4。下方 v2/v3 内容仅为历史记录；如有冲突，以本节为准。

## v4 当前结论（2026-08-05）

| Method | Internal color | Metadata train | Metadata infer | Forward | Optimizer | Resume | Sampling | Ready | Blocker |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DSIFT | Original MATLAB | N/A | 已接入 | N/A | N/A | N/A | 未验证 | 否 | MATLAB 不可用 |
| IFCNN | Official | 无官方训练循环 | 已接入 | 未验证 | N/A | N/A | 未验证 | 部分 | 当前依赖环境 |
| SwinFusion | Y | 已接入 | 已接入 | official strict load/forward 已验证 | 已验证 | 已实现 | N/A | 是 | GPU 正式长训练未执行 |
| ZMFF | RGB | N/A | 已接入 | 逐样本 | N/A | N/A | 未验证 | 部分 | 完整优化耗时 |
| FusionDiff | RGB | 已接入 | 已接入 | 已验证 | 已验证 | 公共完整恢复测试 PASS | 2000 步未验证 | 是 | 正式权重和完整采样耗时 |
| ReDiffuse | RGB | 已接入 | 已接入 | 未验证 | 未验证 | 公共完整恢复测试 PASS | 未验证 | 否 | 本机无 CPython 3.8，B_Conv 真实验证阻塞 |

SwinFusion 最终数据流：metadata 公共层读取 RGB → A/B/GT 同步几何变换 → 专用 adapter 提取单通道 Y `[1,H,W]`、范围 `[0,1]` → 官方单通道 SwinFusion 和未经修改的 source-based MFF loss。GT Y 只用于验证和配对。推理输出融合 Y，并与输入 A 的 Cb/Cr 合并后保存 RGB PNG。`official-y` 加载作者单通道权重；`metadata-y` 额外严格检查相邻 run 的 metadata RGB→Y data contract。

ReDiffuse `Diffusion.py` 已删除全局设备和 `cuda:3`：系数跟随 `t.device`，初始噪声跟随 source tensor，timestep 跟随输入设备。当前 CPython 3.12 下 CPU 扩散设备单测 PASS；因无 Python 3.8.10，官方 B_Conv 导入、官方 `model.pt` strict load、真实模型 forward/optimizer/inference 仍为未验证。B_Conv SHA256 为 `62fb37e52d4c4638daed9e6b5e4bf7d5cc3f337811159b17b9246ff8d67d5fa1`，准备后的导入路径应为 `Condition_Noise_Predictor/B_Conv.pyc`，但本机未声称导入成功。

FusionDiff/ReDiffuse 训练验证仅保留 `smoke` 和 `loss`。周期性完整 diffusion sampling validation 尚未实现；图像质量验证应运行独立 metadata 推理入口。训练脚本允许指定同一非空 output 目录完整 resume，不删除旧日志或 checkpoint，并拒绝同时设置 init 和 resume。一键推理会实际导出 `CUDA_VISIBLE_DEVICES`，Python 入口统一使用映射后的逻辑 `cuda:0`。

本轮实际执行：`pytest -q baselines/test_metadata_v3.py baselines/test_metadata_training_behavior.py baselines/test_metadata_v2.py` 为 **20 passed**；Python 静态编译及两个 v3 shell 的 `bash -n` 均 PASS；`rg "cuda:3" baselines/ReDiffuse`（排除文档）为空。作者 SwinFusion `10000_E.pth` 在 CPU 上 `strict=True` 加载并完成 `(1,1,128,128)` 前向；原 `fusion_loss_mff` 在不改源码的测试 harness 中用单通道 A/B/fused 完成 backward 和 Adam step。`CUDA_VISIBLE_GPU=2 METHOD=all ... run_infer_metadata_v3.sh` 确认日志及子进程环境为物理卡 2、逻辑 `cuda:0`；随后 ZMFF 因当前环境缺 `cv2` 失败，其余缺 checkpoint/MATLAB 的方法按脚本策略跳过。使用 `/bin/true` 的脚本级 resume smoke 证明非空目录允许继续且 init+resume 返回 3。完整 2000 步采样和长训练未执行。

审计日期：2026-08-03。结论中的 “PASS” 仅表示本机实际执行通过；“代码检查”不等同于真实模型运行。

## v2 verification matrix（2026-08-05）

| Method | Dataset smoke | Model forward | Optimizer step | Checkpoint strict load | Sampling smoke | Train ready | Infer ready | Blocker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DSIFT | PASS（公共 schema） | N/A | N/A | N/A | 未验证 | N/A | 未验证 | MATLAB 未安装 |
| IFCNN | PASS | 未验证 | N/A | 未验证 | 未验证 | No official train loop | 未验证 | 缺 torchvision |
| SwinFusion | PASS | PASS（真实 MFIF 网络，CPU） | PASS（原 MFF loss） | PASS（10000_G，strict=True） | N/A | 代码路径完成；完整入口未验证 | checkpoint forward PASS | 完整 CLI 缺 cv2/timm；测试以仅补齐 import 的 harness 执行 |
| ZMFF | PASS | Per-image 未验证 | N/A | N/A | 未验证 | N/A | 未验证 | 完整逐样本优化未运行 |
| FusionDiff | PASS | PASS（真实 NoisePred，CPU） | PASS（原 diffusion loss） | PASS（smoke save/load，strict=True） | 未验证 | 代码路径及最小 step PASS | 正式推理未验证 | 仓库无正式 checkpoint；cv2 缺失 |
| ReDiffuse | PASS | 未验证 | 未验证 | 未验证 | 未验证 | No | No | Missing B_Conv.py |

SwinFusion MFIF 是官方 A/B 无监督融合 loss；metadata GT 仅用于验证配对/PSNR，不是训练 loss target。FusionDiff 正式 checkpoint 来源和 SHA-256 当前均“不知道”，因此没有把 smoke checkpoint 当作正式权重。

### v2 已执行命令与结果

```text
python3 -m pytest ... test_metadata_training_behavior.py test_metadata_v2.py ...
# 14 passed

python3 -m py_compile baselines/SwinFusion/main_train_swinfusion.py ...
bash -n run_train_metadata_v2.sh run_infer_metadata_v2.sh
# PASS

CUDA_VISIBLE_DEVICES=2 python3 -c "... utils_option.parse(...); assert env == '2'"
# PASS；外部 GPU 可见性未被覆盖

# SwinFusion：用 torch 实现缺失 timm 小工具、空的未使用 torchvision import，
# 并只在 CPU harness 中将 loss 内硬编码 Tensor.cuda() 映射为 CPU identity；
# 网络、官方 checkpoint、forward、原 fusion_loss_mff 和 optimizer 均未替换。
# 结果：strict checkpoint load + 128x128 forward PASS；原 MFF loss optimizer step PASS。

# FusionDiff：为缺失且本次路径未使用的 cv2/utils import 提供空 harness，
# 运行真实 NoisePred + GaussianDiffusion.train_losses + AdamW step，随后 save/load strict=True。
# 结果：forward + optimizer + checkpoint strict load PASS。

METHOD=rediffuse bash run_train_metadata_v2.sh
METHOD=rediffuse bash run_infer_metadata_v2.sh
# 均按预期 exit 3，并明确报告 Missing B_Conv.py
```

没有执行 2000 步正式 sampling，因此 Sampling smoke 保持“未验证”。非 2000 步校验单元测试 PASS：直接修改 `T` 的伪少步采样会抛出 `ValueError`。

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

## v3 metadata RGB 修复（2026-08-05）

统一契约为 `GT=image`、`A=edit_image[0]`、`B=edit_image[1]`，忽略 `edit_image[2:]`。公共读取层返回 PIL RGB；SwinFusion formal metadata、FusionDiff、ReDiffuse 的训练/验证/推理均保持 RGB。FusionDiff/ReDiffuse 使用 `[-1,1]`，保存时按 CHW RGB 转 HWC RGB PNG，不进行 RGB/BGR 交换。checkpoint 内含模型、optimizer、scheduler、epoch、global_step、配置、data_contract 及 Python/NumPy/Torch/CUDA RNG；旧权重默认拒绝，初始化和完整恢复为两个参数。

| Method | Metadata RGB train | Metadata RGB infer | Model forward | Optimizer step | Checkpoint load | Full resume | Sampling | Ready | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SwinFusion | 是（scratch/resume） | 是 | PASS（既有真实 harness） | PASS（既有真实 harness） | official strict PASS（兼容模式） | 已实现；本轮静态/单测 | 未执行 | 部分 | 当前环境无 CUDA；official 颜色契约未知 |
| FusionDiff | 是 | 是 | PASS（既有真实 harness） | PASS（既有真实 harness） | RGB full checkpoint PASS | PASS | 2000 步未执行 | 是（需训练权重） | 正式采样耗时高 |
| ReDiffuse | 是 | 是 | 未验证 | 未验证 | 自训 full checkpoint公共逻辑 PASS；official 未验证 | 公共逻辑 PASS | 未执行 | 否 | 当前没有 CPython 3.8，不能导入官方 pyc |

SwinFusion 将外部 `CUDA_VISIBLE_DEVICES=3,5` 映射为程序内逻辑 `[0,1]`，不会覆盖外部设置；输出统一隔离到 run 目录的 `checkpoints/`、`logs/`、`logs/tensorboard/`、`validation/`。正式 RGB scratch/resume 与颜色契约未知的 official 兼容模式分开。

ReDiffuse 的 `B_Conv` 来自作者发布的 CPython 3.8 字节码，未猜写源码，也未替换为普通卷积。源文件：`baselines/ReDiffuse/Condition_Noise_Predictor/__pycache__/B_Conv.cpython-38.pyc`；SHA256：`62fb37e52d4c4638daed9e6b5e4bf7d5cc3f337811159b17b9246ff8d67d5fa1`；magic：`550d0d0a`。准备脚本验证解释器、magic 和导入后复制到 `Condition_Noise_Predictor/B_Conv.pyc`。由于本机只有 CPython 3.12，实际导入路径和 official checkpoint strict load均未验证。

本轮实际测试：

```text
pytest -q baselines/test_metadata_v3.py baselines/test_metadata_training_behavior.py baselines/test_metadata_v2.py
16 passed
bash -n run_train_metadata_v3.sh run_infer_metadata_v3.sh
PASS
python3 -m py_compile <v3 修改的 Python 文件>
PASS
python3.8 baselines/ReDiffuse/prepare_official_bytecode.py
未执行：系统无 python3.8；用当前解释器运行会按预期明确拒绝
```

新增一键用法示例：

```bash
METHOD=all TRAIN_META=/data/train.json VAL_META=/data/val.json OUTPUT_ROOT=/runs/train TAG=rgb_v3 bash run_train_metadata_v3.sh
METHOD=all METADATA=/data/test.json OUTPUT_ROOT=/runs/infer SWINFUSION_CKPT=/runs/swin/checkpoints/1000_G.pth FUSIONDIFF_CKPT=/runs/fd/checkpoints/latest.pt REDIFFUSE_CKPT=/runs/rd/checkpoints/latest.pt bash run_infer_metadata_v3.sh
```
