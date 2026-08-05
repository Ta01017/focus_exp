<h1 align="center">
ReDiffuse: Rotation Equivariant Diffusion Model for Multi-focus Image Fusion
</h1>
<p align="center">
  <!-- <a href="https://github.com/yayayacc/MUR/"><b>[🌐 PyPi Package]</b></a> • -->
  <a href="https://arxiv.org/abs/2603.21129"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/MorvanLi/ReDiffuse/"><b>[🐱 GitHub]</b></a>
</p>

<p align="center"> Repo for "ReDiffuse: Rotation Equivariant Diffusion Model for Multi-focus Image Fusion</a>"</p>
<a href="https://arxiv.org/abs/2603.21129" target="_blank">

## 🔥 News

- [2026/03/28] 🔥🔥🔥 Our github repo is released !!!
- [2026/03/22] 🔥🔥🔥 Our paper is released!!!

## 🌐 Usage

### ⚙ Network Architecture

Our ReDiffuse is implemented in ``Condition_Noise_Predictor/Rot_E_UNet.py``.

### 🏊 Training
**1. Virtual Environment**

```
# create virtual environment
conda create -n rediffuse python=3.8.10
conda activate rediffuse
# Install the PyTorch wheel matching the server CUDA runtime. Example for CUDA 12.1:
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-py38.txt
```
**2. Data Preparation**

Download the Real-MFF dataset and place it under the directory ``'./Dataset/Multi-Focus-Images/train'`` following this structure:

 ```
train_data/
└── Real-MFF/
    ├── imageA/
    │   ├── 001_A.png
    │   ├── 002_A.png
    │   └── ...
    ├── imageB/
    │   ├── 001_B.png
    │   ├── 002_B.png
    │   └── ...
    └── Fusion/
        ├── 001_F.png
        ├── 002_F.png
        └── ...
 ```
Note: Please ensure the directory structure and file naming conventions remain consistent to avoid errors during training.

**3. ReDiffuse Training**

Run 
```
python train.py
```

### Checkpoint modes

`official-y` and `metadata-rgb` checkpoints are structurally incompatible.

- `official-y`: CPython 3.8.10 plus the author's published `B_Conv.cpython-38.pyc`; the official `weights/model.pt` is a 3-input/1-output noise predictor. Metadata RGB A/B are converted to Y, and the predicted Y is combined with input A's Cb/Cr to save RGB.
- `metadata-rgb`: the supervised metadata path trains a 9-input/3-output model from scratch. It stays RGB and only accepts a complete self-trained checkpoint whose data contract says `model_mode=metadata-rgb`.

Prepare the official bytecode before either mode:

```bash
python prepare_official_bytecode.py
```

Metadata smoke（正式训练必须使用不同 train/val 文件）：

```cmd
python train.py --model-mode metadata-rgb --dataset-format metadata --train-metadata ..\smoke_metadata\metadata.json --val-metadata ..\smoke_metadata\metadata.json --max-samples 2 --max-train-steps 1 --num-workers 0 --seed 17
python infer_metadata.py --checkpoint-mode official-y --metadata ..\smoke_metadata\metadata.json --output-dir ..\..\outputs\ReDiffuse-official --checkpoint weights\model.pt --device cpu --max-samples 1
python infer_metadata.py --checkpoint-mode metadata-rgb --metadata ..\smoke_metadata\metadata.json --output-dir ..\..\outputs\ReDiffuse-rgb --checkpoint path\best_val_loss.pt --device cpu --max-samples 1
```

Metadata supervised training preserves the rotation-equivariant network, synchronized crop, diffusion loss, and condition order, but uses RGB 9→3. The official model remains Y 3→1 and is inference-only in this pipeline. Both loaders use `strict=True`; there is no automatic mode inference or structural fallback.

正式推理必须使用训练配置的 2000 步。少步采样尚未实现，不得通过直接修改 `T` 重建 noise schedule。仓库保留作者发布的 CPython 3.8 字节码；`prepare_official_bytecode.py` 校验版本、magic 和 SHA256，然后创建不提交的运行时 `Condition_Noise_Predictor/B_Conv.pyc` 并实际导入。不得猜写源码、换成普通卷积或使用 `strict=False`。
We also provide the pre-trained weights in the ``./weights/model.pt``.
### 🏄 Testing

**1. Test datasets**

Test datasets are provided in the following folders:

- `./Dataset/Multi-Focus-Images/valid/Lytro/`
- `./Dataset/Multi-Focus-Images/valid/MFFW/`
- `./Dataset/Multi-Focus-Images/valid/MFI-WHU/`
- `./Dataset/Multi-Focus-Images/valid/Road-MF/`

**2. ReDiffuse Testing**

Run 
```
python main.py
```

Testing results will be saved to the `./generate_imgs/` directory by default.
