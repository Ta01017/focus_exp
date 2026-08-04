# ZMFF

This is an implementation of the following paper.

```
@article{Hu2023ZMFF,
title = {ZMFF: Zero-shot multi-focus image fusion},
author={Hu, Xingyu and Jiang, Junjun and Liu, Xianming and Ma, Jiayi},
journal = {Information Fusion},
volume = {92},
pages = {127-138},
year = {2023},
}
```

## Preparation

### Dependencies

- python >= 3.6
- pytorch >= 1.5.0
- NVIDIA GPU
- CUDA

### Data Preparation

- [Lytro](http://mansournejati.ece.iut.ac.ir/content/lytro-multi-focus-dataset)
- [Real-MFF](https://github.com/Zancelot/Real-MFF)
- [MFI-WHU](https://github.com/HaoZhang1018/MFI-WHU)

## Testing

For testing, please run:

```shell
python main.py
```

Metadata zero-shot smoke：

```cmd
python infer_metadata.py --metadata ..\smoke_metadata\metadata.json --output-dir ..\..\outputs\ZMFF --device cpu --iterations 1 --seed 17 --start-index 0 --max-samples 2
```

每个样本都会重新设置 seed，并重新构造网络、mask/noise inputs 和 optimizer；GT 不参与优化，`edit_image[2:]` 被忽略。ZMFF 不是普通监督训练方法，因此没有新增离线训练入口。

## results
The output results will be stored in `./result`.

Our results in [Lytro](https://drive.google.com/file/d/112b3HypIQoO0-mQH6GfHDTY2dk1TRhK2/view?usp=share_link), [Real-MFF](https://drive.google.com/file/d/1fV4fLpjK8v-AgFn53Ikrlq6y_Dt5-790/view?usp=share_link), [MFI-WHU](https://drive.google.com/file/d/1Q8h23h3DD_odVg0PfZ6FwNgkBsv_6_CL/view?usp=share_link) datasets can be downloaded.


