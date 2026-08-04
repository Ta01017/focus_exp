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
# select pytorch version yourself
# install reufuse requirements
pip install -r requirements.txt
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

Metadata smoke（正式训练必须使用不同 train/val 文件）：

```cmd
python ..\test_metadata_dataset.py --metadata ..\smoke_metadata\metadata.json --channels 1 --size 256 --max-samples 2
python train.py --dataset-format metadata --train-metadata ..\smoke_metadata\metadata.json --val-metadata ..\smoke_metadata\metadata.json --max-samples 2 --max-train-steps 1 --num-workers 0 --seed 17
python infer_metadata.py --metadata ..\smoke_metadata\metadata.json --output-dir ..\..\outputs\ReDiffuse --checkpoint weights\model.pt --device cpu --max-samples 2
```

Metadata 模式保留旋转等变网络、Y 通道、同步 256 crop、扩散 loss 和条件排列。checkpoint 不存在或状态不匹配时推理会失败，不会使用随机权重生成结果。
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

