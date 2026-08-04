# Metadata adapters

公共读取器 `metadata_dataset.py` 支持 UTF-8/BOM、Windows/Linux/相对路径、EXIF transpose、RGB 解码、两图/四图 metadata，以及 `infer`、`train`、`val` 三种模式。训练和验证强制要求 `image` GT。`metadata_training.py` 负责 A/B/GT 同步 resize、crop、flip 和 90 度旋转；默认尺寸不一致策略是 `error`。

## CPU Dataset smoke

```cmd
python -m pytest -q
python baselines\test_metadata_dataset.py --metadata baselines\smoke_metadata\metadata.json --max-samples 2
```

## SwinFusion MFIF metadata training

```cmd
cd baselines\SwinFusion
python main_train_swinfusion.py --opt options\swinir\train_swinfusion_mff.json --train-metadata ..\smoke_metadata\metadata.json --val-metadata ..\smoke_metadata\metadata.json --max-samples 2 --max-train-steps 1 --num-workers 0 --seed 17
```

正式运行必须使用不同的 train/val metadata。原 MFIF 网络、`mff` loss、optimizer、scheduler、128 patch 和灰度 `[0,1]` normalization 保持不变。模型保存关系仍为 `*_G.pth`、EMA `*_E.pth` 和 `*_optimizerG.pth`；入口会按同一 iteration 自动恢复它们。

## FusionDiff metadata training

```cmd
cd baselines\ImageFusion\FusionDiff
python train.py --dataset-format metadata --train-metadata ..\..\smoke_metadata\metadata.json --val-metadata ..\..\smoke_metadata\metadata.json --max-samples 2 --max-train-steps 1 --num-workers 0 --seed 17
```

## ReDiffuse metadata training

```cmd
cd baselines\ReDiffuse
python train.py --dataset-format metadata --train-metadata ..\smoke_metadata\metadata.json --val-metadata ..\smoke_metadata\metadata.json --max-samples 2 --max-train-steps 1 --num-workers 0 --seed 17
```

FusionDiff 与 ReDiffuse 的 metadata 路径只替换数据来源，训练仍调用各自 `GaussianDiffusion.train_losses`，保持 timestep、noise schedule、条件排列和 loss。`--resume PATH` 可载入模型状态；若 checkpoint 还含 `optimizer`，也恢复 optimizer。

## Metadata inference

```cmd
python baselines\IFCNN\infer_metadata.py --metadata baselines\smoke_metadata\metadata.json --output-dir outputs\IFCNN --checkpoint path\IFCNN-MAX.pth --device cpu --max-samples 2
python baselines\SwinFusion\infer_metadata.py --metadata baselines\smoke_metadata\metadata.json --output-dir outputs\SwinFusion --checkpoint path\MFIF_G.pth --device cpu --max-samples 2
python baselines\ImageFusion\FusionDiff\infer_metadata.py --metadata baselines\smoke_metadata\metadata.json --output-dir outputs\FusionDiff --checkpoint path\model.pt --device cpu --max-samples 2
python baselines\ReDiffuse\infer_metadata.py --metadata baselines\smoke_metadata\metadata.json --output-dir outputs\ReDiffuse --checkpoint baselines\ReDiffuse\weights\model.pt --device cpu --max-samples 2
python baselines\ZMFF\infer_metadata.py --metadata baselines\smoke_metadata\metadata.json --output-dir outputs\ZMFF --device cpu --iterations 1 --seed 17 --max-samples 2
baselines\DSIFT-MFIF\run_metadata_dsift.bat baselines\smoke_metadata\metadata.json outputs\DSIFT 0 2
```

IFCNN 源码中没有完整官方训练入口，因此没有编造监督训练流程；可复用公共 `MetadataFusionDataset`，但可靠范围仅为官方 checkpoint 推理。DSIFT 是非学习 MATLAB 方法。ZMFF 是逐样本 zero-shot 优化，两者都没有监督训练入口。

Linux 一键入口为根目录 `run_train_metadata_v2.sh` 与 `run_infer_metadata_v2.sh`。SwinFusion 训练必须设置独立 `OUTPUT_ROOT/TAG` 和显式 `INIT_MODE=scratch|official|resume`。FusionDiff/ReDiffuse 正式推理固定使用训练配置的 `T=2000`；未实现验证过的 DDIM/timestep respacing，不能用修改 `T` 模拟少步采样。
