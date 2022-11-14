# SwinIR-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257v1).

## Table of contents

- [SwinIR-PyTorch](#swinir-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test SwinIR_default_sr_x4](#test-swinir_default_sr_x4)
        - [Train SwinIR_default_sr_x4](#train-swinir_default_sr_x4)
        - [Resume SwinIR_default_sr_x4](#resume-train-swinir_default_sr_x4)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [SwinIR: Image Restoration Using Swin Transformer](#swinir-image-restoration-using-swin-transformer)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `swinirnet_config.py` file and `swinirgan_config.py` file.

### Test SwinIR_default_sr_x4

Modify the `swinirnet_config.py` file.

- line 31: `g_arch_name` change to `swinir_default_sr_x4`.
- line 36: `upscale_factor` change to `4`.
- line 38: `mode` change to `test`.
- line 40: `exp_name` change to `SwinIRNet_default_sr_x4-Set5`.
- line 84: `lr_dir` change to `f"./data/Set5/LRbicx{upscale_factor}"`.
- line 86: `gt_dir` change to `f"./data/Set5/GTmod12"`.
- line 88: `g_model_weights_path` change to `./results/pretrained_models/`.

```bash
python3 test.py
```

### Train SwinIR_default_sr_x4

Modify the `swinirnet_config.py` file.

- line 31: `g_arch_name` change to `swinir_default_sr_x4`.
- line 36: `upscale_factor` change to `4`.
- line 38: `mode` change to `train`.
- line 40: `exp_name` change to `SwinIRNet_default_sr_x4-DIV2K`.
- line 84: `lr_dir` change to `f"./data/Set5/LRbicx{upscale_factor}"`.
- line 86: `gt_dir` change to `f"./data/Set5/GTmod12"`.

```bash
python3 train_swinirnet.py
```

### Resume train SwinIR_default_sr_x4

Modify the `swinirnet_config.py` file.

- line 31: `g_arch_name` change to `swinir_default_sr_x4`.
- line 36: `upscale_factor` change to `4`.
- line 38: `mode` change to `train`.
- line 40: `exp_name` change to `SwinIRNet_default_sr_x4-DIV2K`.
- line 57: `resume_g_model_weights_path` change to `./samples/SwinIRNet_default_sr_x4-DIV2K/g_epoch_xxx.pth.tar`.
- line 84: `lr_dir` change to `f"./data/Set5/LRbicx{upscale_factor}"`.
- line 86: `gt_dir` change to `f"./data/Set5/GTmod12"`.

```bash
python3 train_swinirnet.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/2108.10257v1.pdf](https://arxiv.org/pdf/2108.10257v1.pdf)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

|         Method          | Scale |   Set5 (PSNR)    |    Set5 (SSIM)     |   Set14 (PSNR)   |    Set14 (SSIM)    |
|:-----------------------:|:-----:|:----------------:|:------------------:|:----------------:|:------------------:|
| SwinIRNet_default_sr_x2 |   2   | 38.35(**38.10**) | 0.9620(**0.9617**) | 34.14(**33.72**) | 0.9227(**0.9196**) |
| SwinIRNet_default_sr_x3 |   3   |   34.89(**-**)   |   0.9312(**-**)    |   30.77(**-**)   |   0.8503(**-**)    |
| SwinIRNet_default_sr_x4 |   4   |   32.72(**-**)   |   0.9021(**-**)    |   28.94(**-**)   |   0.7914(**-**)    |

|           Method            | Scale |   Set5 (PSNR)    |    Set5 (SSIM)     |   Set14 (PSNR)   |    Set14 (SSIM)    |
|:---------------------------:|:-----:|:----------------:|:------------------:|:----------------:|:------------------:|
| SwinIRNet_lightweight_sr_x2 |   2   | 38.14(**37.85**) | 0.9611(**0.9606**) | 33.86(**33.39**) | 0.9206(**0.9168**) |
| SwinIRNet_lightweight_sr_x3 |   3   |   34.62(**-**)   |   0.9289(**-**)    |   30.54(**-**)   |   0.8463(**-**)    |
| SwinIRNet_lightweight_sr_x4 |   4   |   32.44(**-**)   |   0.8976(**-**)    |   28.77(**-**)   |   0.7858(**-**)    |

```bash
# Download `SwinIRNet_default_sr_x4-DIV2K-8c4a7569.pth.tar` weights to `./results/pretrained_models/SwinIRNet_default_sr_x4-DIV2K-8c4a7569.pth.tar`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input:

<span align="center"><img width="1020" height="768" src="figure/img_012.png"/></span>

Output:

<span align="center"><img width="1020" height="768" src="figure/sr_img_012.png"/></span>

```text
Build `swinir_default_sr_x4` model successfully.
Load `swinir_default_sr_x4` model weights `./results/pretrained_models/SwinIRNet_default_sr_x4-DIV2K-8c4a7569.pth.tar` successfully.
SR image save to `./figure/sr_img_012.png`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### SwinIR: Image Restoration Using Swin Transformer

_Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte_ <br>

**Abstract** <br>
Image restoration is a long-standing low-level vision problem that aims to restore high-quality images from low-quality
images (e.g., downscaled, noisy and compressed images). While state-of-the-art image restoration methods are based on
convolutional neural networks, few attempts have been made with Transformers which show impressive performance on
high-level vision tasks. In this paper, we propose a strong baseline model SwinIR for image restoration based on the
Swin Transformer. SwinIR consists of three parts: shallow feature extraction, deep feature extraction and high-quality
image reconstruction. In particular, the deep feature extraction module is composed of several residual Swin Transformer
blocks (RSTB), each of which has several Swin Transformer layers together with a residual connection. We conduct
experiments on three representative tasks: image super-resolution (including classical, lightweight and real-world image
super-resolution), image denoising (including grayscale and color image denoising) and JPEG compression artifact
reduction. Experimental results demonstrate that SwinIR outperforms state-of-the-art methods on different tasks by up to
0.14∼0.45dB, while the total number of parameters can be reduced by up to 67%.

[[Code]](https://github.com/JingyunLiang/SwinIR) [[Paper]](https://arxiv.org/pdf/2108.10257v1.pdf)

```bibtex
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
```
