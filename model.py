# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT widthARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import collections.abc
import math
import os
import warnings
from itertools import repeat
from typing import Any, cast, Dict, List, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torch.nn.init import trunc_normal_
from torch.nn.utils import spectral_norm
from torch.utils import checkpoint
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "SwinIR", "DiscriminatorUNet", "FeatureLoss",
    "swinir_default_sr_x2", "swinir_default_sr_x3", "swinir_default_sr_x4", "swinir_default_sr_x8",
    "swinir_lightweight_sr_x2", "swinir_lightweight_sr_x3", "swinir_lightweight_sr_x4", "swinir_lightweight_sr_x8",
    "swinir_real_sr_x2", "swinir_real_sr_x3", "swinir_real_sr_x4", "swinir_real_sr_x8",
    "discriminator_unet",
]

feature_extractor_net_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(net_cfg_name: str, batch_norm: bool = False) -> nn.Sequential:
    net_cfg = feature_extractor_net_cfgs[net_cfg_name]
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in net_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class _FeatureExtractor(nn.Module):
    def __init__(
            self,
            net_cfg_name: str = "vgg19",
            batch_norm: bool = False,
            num_classes: int = 1000) -> None:
        super(_FeatureExtractor, self).__init__()
        self.features = _make_layers(net_cfg_name, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class SwinIR(nn.Module):
    """PyTorch implements of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        image_size (tuple(int)): Input image size. Default: (64, 64)
        patch_size (tuple(int)): Patch size. Default: (1, 1)
        in_channels (int): Number of input image channels. Default: 3
        out_channels (int): Number of output image channels. Default: 3
        channels (int): Number of model feature channels. Default: 64
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attention_drop (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale_factor: Upscale factor. 2/3/4/8 for image SR, 1 for de-noising and compress artifact reduction
        upsample_method: The upsample module. "default_sr"/"lightweight_sr"/"real_sr"/None
        resi_connection: The convolutional block before residual connection. "1conv"/"3conv"

    """

    def __init__(
            self,
            image_size: tuple[int, int] = (64, 64),
            patch_size: tuple[int, int] = (1, 1),
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            embed_dim: int = 96,
            depths: list = None,
            num_heads: list = None,
            window_size: int = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_scale: float = None,
            drop_rate: float = 0.,
            attention_drop: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: nn.Module = nn.LayerNorm,
            ape: bool = False,
            patch_norm: bool = True,
            use_checkpoint: bool = False,
            upsample_method: str = "default_sr",
            resi_connection: str = "1conv",
            mean_value: list = None,
            upscale_factor: int = 4,
    ) -> None:
        super(SwinIR, self).__init__()
        if depths is None:
            depths = [6, 6, 6, 6]
        if num_heads is None:
            num_heads = [6, 6, 6, 6]
        if mean_value is None and in_channels == 3:
            self.register_buffer("mean", torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, in_channels, 1, 1))
        else:
            self.register_buffer("mean", torch.Tensor(mean_value).view(1, in_channels, 1, 1))

        self.window_size = window_size
        self.ape = ape
        self.patch_norm = patch_norm
        self.upsample_method = upsample_method
        self.upscale_factor = upscale_factor

        # Low frequency information extraction layer
        self.conv1 = nn.Conv2d(in_channels, embed_dim, (3, 3), (1, 1), (1, 1))

        # High frequency information extraction block
        self.patch_embed = _PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # merge non-overlapping patches into image
        self.patch_unembed = _PatchUnEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=embed_dim,
            embed_dim=embed_dim)

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        stochastic_depth_decay_rule = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build Residual Swin Transformer blocks (RSTB)
        trunk = []
        for i in range(len(depths)):
            trunk.append(
                _RSTB(dim=embed_dim,
                      input_resolution=(self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]),
                      depth=depths[i],
                      num_heads=num_heads[i],
                      window_size=window_size,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      drop=drop_rate,
                      attention_drop=attention_drop,
                      drop_path=stochastic_depth_decay_rule[sum(depths[:i]):sum(depths[:i + 1])],
                      norm_layer=norm_layer,
                      downsample=None,
                      use_checkpoint=use_checkpoint,
                      image_size=image_size,
                      patch_size=patch_size,
                      resi_connection=resi_connection))
        self.trunk = nn.Sequential(*trunk)
        self.norm = norm_layer(embed_dim)

        # High-frequency information linear fusion layer
        if resi_connection == "1conv":
            self.conv2 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, (3, 3), (1, 1), (1, 1)),
            )
        elif resi_connection == "3conv":
            self.conv2 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, (1, 1), (1, 1), (0, 0)),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(embed_dim // 4, embed_dim, (3, 3), (1, 1), (1, 1)),
            )
        else:
            warnings.warn("`resi_connection` only supports `1conv` and `3conv`. Default use `1conv`.")
            self.conv2 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, (3, 3), (1, 1), (1, 1)),
            )

        # Zoom block
        if self.upsample_method == "default_sr":
            # for classical SR
            self.conv3 = nn.Sequential(
                nn.Conv2d(embed_dim, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(True),
            )

            upsampling = []
            if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
                for _ in range(int(math.log(upscale_factor, 2))):
                    upsampling.append(_UpsampleBlock(channels, 2))
            elif upscale_factor == 3:
                upsampling.append(_UpsampleBlock(channels, 3))
            self.upsampling = nn.Sequential(*upsampling)

            self.conv_out = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))
        elif self.upsample_method == "lightweight_sr":
            self.upsampling = _LightweightUpsampleBlock(embed_dim, out_channels, upscale_factor)
        elif self.upsample_method == "real_sr":
            self.conv3 = nn.Sequential(
                nn.Conv2d(embed_dim, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(True),
            )

            if self.upscale_factor == 2:
                self.upsampling1 = nn.Sequential(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                    nn.LeakyReLU(0.2, True),
                )
            elif self.upscale_factor == 3:
                self.upsampling1 = nn.Sequential(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                    nn.LeakyReLU(0.2, True),
                )
            elif self.upscale_factor == 4:
                self.upsampling1 = nn.Sequential(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                    nn.LeakyReLU(0.2, True),
                )
                self.upsampling2 = nn.Sequential(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                    nn.LeakyReLU(0.2, True),
                )
            elif self.upscale_factor == 8:
                self.upsampling1 = nn.Sequential(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                    nn.LeakyReLU(0.2, True),
                )
                self.upsampling2 = nn.Sequential(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                    nn.LeakyReLU(0.2, True),
                )
                self.upsampling3 = nn.Sequential(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                    nn.LeakyReLU(0.2, True),
                )

            self.conv4 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.conv_out = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))
        else:
            # for image de-noising and JPEG compression artifact reduction
            self.conv_out = nn.Conv2d(embed_dim, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all layer
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        height, width = x.shape[2:]
        x = self._check_image_size(x)

        # (x - self.mean) * image_range
        x = torch.sub(x, self.mean)
        x = torch.mul(x, 1.)

        if self.upsample_method == "default_sr":
            x = self.conv1(x)
            x = torch.add(self.conv2(self._forward_features(x)), x)
            x = self.conv3(x)
            x = self.upsampling(x)
            x = self.conv_out(x)
        elif self.upsample_method == "lightweight_sr":
            x = self.conv1(x)
            x = torch.add(self.conv2(self._forward_features(x)), x)
            x = self.upsampling(x)
        elif self.upsample_method == "real_sr":
            x = self.conv1(x)
            x = torch.add(self.conv2(self._forward_features(x)), x)
            x = self.conv3(x)

            if self.upscale_factor == 2:
                x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            elif self.upscale_factor == 3:
                x = self.upsampling1(F_torch.interpolate(x, scale_factor=3, mode="nearest"))
            elif self.upscale_factor == 4:
                x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
                x = self.upsampling2(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            elif self.upscale_factor == 8:
                x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
                x = self.upsampling2(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
                x = self.upsampling3(F_torch.interpolate(x, scale_factor=2, mode="nearest"))

            x = self.conv4(x)
            x = self.conv_out(x)
        else:
            # for image de-noising and JPEG compression artifact reduction
            conv1 = self.conv1(x)
            features = torch.add(self.conv2(self._forward_features(conv1)), conv1)
            features = self.conv_out(features)
            x = torch.add(x, features)

        # (x / image_range) + self.mean
        x = torch.div(x, 1.)
        x = torch.add(x, self.mean)

        x = x[:, :, :height * self.upscale_factor, :width * self.upscale_factor]
        x = torch.clamp_(x, 0.0, 1.0)

        return x

    def _forward_features(self, x: Tensor) -> Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.trunk:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def _check_image_size(self, x: Tensor) -> Tensor:
        _, _, height, width = x.size()

        mod_pad_height = (self.window_size - height % self.window_size) % self.window_size
        mod_pad_width = (self.window_size - width % self.window_size) % self.window_size
        x = F_torch.pad(x, (0, mod_pad_width, 0, mod_pad_height), "reflect")

        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)


class DiscriminatorUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            upsample_method: str = "bilinear",
    ) -> None:
        super(DiscriminatorUNet, self).__init__()
        self.upsample_method = upsample_method

        self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1), (1, 1))
        self.down_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, int(channels * 2), (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 2), int(channels * 4), (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 4), int(channels * 8), (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 8), int(channels * 4), (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 4), int(channels * 2), (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 2), channels, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)

        # Down-sampling
        down1 = self.down_block1(out1)
        down2 = self.down_block2(down1)
        down3 = self.down_block3(down2)

        # Up-sampling
        down3 = F_torch.interpolate(down3, scale_factor=2, mode=self.upsample_method, align_corners=False)
        up1 = self.up_block1(down3)

        up1 = torch.add(up1, down2)
        up1 = F_torch.interpolate(up1, scale_factor=2, mode=self.upsample_method, align_corners=False)
        up2 = self.up_block2(up1)

        up2 = torch.add(up2, down1)
        up2 = F_torch.interpolate(up2, scale_factor=2, mode=self.upsample_method, align_corners=False)
        up3 = self.up_block3(up2)

        up3 = torch.add(up3, out1)

        out = self.conv2(up3)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class FeatureLoss(nn.Module):
    """Implement feature loss based on VGG19 model
    Using advanced feature map layers from later layers will focus more on the texture content of the image

    Paper reference list:
        - `Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        - `ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        - `Perceptual Extreme Super Resolution Network with Receptive Field Block <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            net_cfg_name: str,
            batch_norm: bool,
            num_classes: int,
            model_weights_path: str,
            feature_nodes: list,
            feature_normalize_mean: list,
            feature_normalize_std: list,
    ) -> None:
        super(FeatureLoss, self).__init__()
        # Define the feature extraction model
        model = _FeatureExtractor(net_cfg_name, batch_norm, num_classes)
        # Load the pre-trained model
        if model_weights_path is None:
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif model_weights_path is not None and os.path.exists(model_weights_path):
            checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
            if "state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError("Model weight file not found")
        # Extract the output of the feature extraction layer
        self.feature_extractor = create_feature_extractor(model, feature_nodes)
        # Select the specified layers as the feature extraction layer
        self.feature_extractor_nodes = feature_nodes
        # input normalization
        self.normalize = transforms.Normalize(feature_normalize_mean, feature_normalize_std)
        # Freeze model parameters without derivatives
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> list[Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
        device = sr_tensor.device

        losses = []
        # input normalization
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        # Get the output of the feature extraction layer
        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        # Compute feature loss
        for i in range(len(self.feature_extractor_nodes)):
            losses.append(F_torch.l1_loss(sr_feature[self.feature_extractor_nodes[i]],
                                          gt_feature[self.feature_extractor_nodes[i]]))

        losses = torch.Tensor([losses]).to(device)

        return losses


class _UpsampleBlock(nn.Sequential):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsampling_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsampling_block(x)

        return x


class _LightweightUpsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int):
        super(_LightweightUpsampleBlock, self).__init__()
        self.lightweight_upsampling_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.lightweight_upsampling_block(x)

        return x


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
class _PatchEmbed(nn.Module):
    """Image to Patch Embedding

    Args:
        image_size (tuple[int,int]): Image size.  Default: (224, 224).
        patch_size (tuple[int,int]): Patch token size. Default: (4, 4).
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None

    """

    def __init__(
            self,
            image_size: tuple[int, int] = (224, 224),
            patch_size: tuple[int, int] = 4,
            embed_dim: int = 96,
            norm_layer: nn.Module = None
    ) -> None:
        super(_PatchEmbed, self).__init__()
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C

        if self.norm is not None:
            x = self.norm(x)

        return x


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
class _PatchUnEmbed(nn.Module):
    """Image to Patch Unembedding

    Args:
        image_size (tuple[int, int]): Image size.  Default: 224.
        patch_size (tuple[int, int]): Patch token size. Default: 4.
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.

    """

    def __init__(
            self,
            image_size: tuple[int, int] = (224, 224),
            patch_size: tuple[int, int] = (4, 4),
            in_channels: int = 3,
            embed_dim: int = 96
    ) -> None:
        super(_PatchUnEmbed, self).__init__()
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

    def forward(self, x: Tensor, x_size: tuple) -> Tensor:
        batch_size, height_width, channels = x.shape
        out = x.transpose(1, 2).view(batch_size, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C

        return out


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
class _RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attention_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        image_size: Input image size. Default: (224, 224).
        patch_size: Patch size. Default: (4, 4).
        resi_connection: The convolutional block before residual connection.

    """

    def __init__(
            self,
            dim: int,
            input_resolution: tuple,
            depth: int,
            num_heads: int,
            window_size: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_scale: float = None,
            drop: float = 0.,
            attention_drop: float = 0.,
            drop_path: float | list[float] = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            downsample: nn.Module = None,
            use_checkpoint: bool = False,
            image_size: tuple[int, int] = (224, 224),
            patch_size: tuple[int, int] = (4, 4),
            resi_connection: str = "1conv"
    ) -> None:
        super(_RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = _BasicLayer(dim=dim,
                                          input_resolution=input_resolution,
                                          depth=depth,
                                          num_heads=num_heads,
                                          window_size=window_size,
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop,
                                          attention_drop=attention_drop,
                                          drop_path=drop_path,
                                          norm_layer=norm_layer,
                                          downsample=downsample,
                                          use_checkpoint=use_checkpoint)

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, (3, 3), (1, 1), (1, 1))
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, (3, 3), (1, 1), (1, 1))
            )

        self.patch_embed = _PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = _PatchUnEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=0,
            embed_dim=dim)

    def forward(self, x: Tensor, x_size: tuple) -> Tensor:
        out = self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size)))
        out = torch.add(out, x)

        return out


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
class _BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attention_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim: int,
            input_resolution: tuple,
            depth: int,
            num_heads: int,
            window_size: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_scale: float = None,
            drop: float = 0.,
            attention_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            downsample: nn.Module = None,
            use_checkpoint: bool = False
    ) -> None:
        super(_BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            _SwinTransformerBlock(dim=dim,
                                  input_resolution=input_resolution,
                                  num_heads=num_heads,
                                  window_size=window_size,
                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  qk_scale=qk_scale,
                                  drop=drop,
                                  attention_drop=attention_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: Tensor, x_size: tuple) -> Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)

        return x


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
class _PatchMerging(nn.Module):
    """Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self,
            input_resolution: tuple,
            dim: int,
            norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        super(_PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: B, H*W, C
        """
        height, width = self.input_resolution
        batch_size, height_width, channels = x.shape
        assert height_width == height * width, "input feature has wrong size"
        assert height % 2 == 0 and width % 2 == 0, f"x size ({height}*{width}) are not even."

        out = x.view(batch_size, height, width, channels)

        x0 = out[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = out[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = out[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = out[:, 1::2, 1::2, :]  # B H/2 W/2 C
        out = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        out = out.view(batch_size, -1, 4 * channels)  # B H/2*W/2 4*C

        out = self.norm(out)
        out = self.reduction(out)

        return out


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
class _SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attention_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self,
            dim: int,
            input_resolution: tuple,
            num_heads: int,
            window_size: int = 7,
            shift_size: int = 0,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_scale: float = None,
            drop: float = 0.,
            attention_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm):
        super(_SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don"t partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attention = _WindowAttention(
            dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attention_drop=attention_drop, projection_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, mlp_hidden_dim, None, drop)

        if self.shift_size > 0:
            attention_mask = self._calculate_mask(self.input_resolution)
        else:
            attention_mask = None

        self.register_buffer("attention_mask", attention_mask)

    def forward(self, x: Tensor, x_size: tuple) -> Tensor:
        identity = x
        height, width = x_size
        batch_size, height_width, channels = x.shape

        out = self.norm1(x)
        out = out.view(batch_size, height, width, channels)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(out, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = out

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)  # nW*B,window_size*window_size, C

        # width-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attention_windows = self.attention(x_windows, mask=self.attention_mask)  # nW*B, window_size*window_size, C
        else:
            attention_windows = self.attention(x_windows, mask=self._calculate_mask(x_size).to(out.device))

        # merge windows
        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attention_windows, self.window_size, height, width)  # B height" width" C

        # reverse cyclic shift
        if self.shift_size > 0:
            out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            out = shifted_x
        out = out.view(batch_size, height * width, channels)

        # FFN
        out = self.drop_path(out)
        out = torch.add(out, identity)
        out = torch.add(out, self.drop_path(self.mlp(self.norm2(out))))

        return out

    def _calculate_mask(self, x_size: tuple) -> Tensor:
        # calculate attention mask for Sw-MSA
        height, width = x_size
        image_mask = torch.zeros((1, height, width, 1))  # 1 height width 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                image_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(image_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float(0.0))

        return attention_mask


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
class _WindowAttention(nn.Module):
    """WindowAttention based multi-head self attention (width-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attention_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        projection_drop (float, optional): Dropout ratio of projection output. Default: 0.0
    """

    def __init__(
            self,
            dim: int,
            window_size: tuple,
            num_heads: int,
            qkv_bias: bool = True,
            qk_scale: float = None,
            attention_drop: float = 0.,
            projection_drop: float = 0.
    ) -> None:
        super(_WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_drop = nn.Dropout(attention_drop)
        self.projection = nn.Linear(dim, dim)

        self.projection_drop = nn.Dropout(projection_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Any = None) -> Tensor:
        """

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, width_height*width_height, width_height*width_height)
        """
        num_windows_batch_size, num_windows, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(num_windows_batch_size, num_windows, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch.script happy (cannot use tensor as tuple)

        q = q * self.scale
        attention = (q @ k.transpose(-2, -1))

        # Wh*Ww,Wh*Ww,nH
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size[0] * self.window_size[1],
                                                             self.window_size[0] * self.window_size[1],
                                                             -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_width = mask.shape[0]
            attention = attention.view(num_windows_batch_size // num_width,
                                       num_width,
                                       self.num_heads,
                                       num_windows,
                                       num_windows)
            attention = torch.add(attention, mask.unsqueeze(1).unsqueeze(0))
            attention = attention.view(-1, self.num_heads, num_windows, num_windows)
            attention = self.softmax(attention)
        else:
            attention = self.softmax(attention)

        attention = self.attention_drop(attention)

        x = (attention @ v).transpose(1, 2).reshape(num_windows_batch_size, num_windows, channels)
        x = self.projection(x)
        x = self.projection_drop(x)

        return x


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
def window_partition(x: Tensor, window_size: int) -> Tensor:
    """

    Args:
        x (Tensor): (B, height, width, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)

    """
    batch_size, height, width, channels = x.shape
    x = x.view(batch_size, height // window_size, window_size, width // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)

    return windows


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
def window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    """

    Args:
        windows (Tensor): (num_windows*B, window_size, window_size, C)
        window_size (int): widthindow size
        height (int): heighteight of image
        width (int): widthidth of image

    Returns:
        out: (B, height, width, C)

    """
    batch_size = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)

    return x


# Modify from `https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py`.
class _MLP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int = None,
            out_channels: int = None,
            drop: float = 0.
    ) -> None:
        super(_MLP, self).__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)

        return x


# Modify from `https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/drop.py`.
def _drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True) -> Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as "Drop Connect" is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I"ve opted for
    changing the layer and argument names to "drop path" rather than mix DropConnect as a layer name and use
    "survival rate" as the argument.
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    x = torch.mul(x, random_tensor)

    return x


# Modify from `https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/drop.py`.
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor):
        return _drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# Copy from `https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/helpers.py`.
def _ntuple(n: Any):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


_to_1tuple = _ntuple(1)
_to_2tuple = _ntuple(2)
_to_3tuple = _ntuple(3)
_to_4tuple = _ntuple(4)
_to_ntuple = _ntuple


def swinir_default_sr_x2(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=180,
                   depths=[6, 6, 6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="default_sr",
                   resi_connection="1conv",
                   upscale_factor=2,
                   **kwargs)

    return model


def swinir_default_sr_x3(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=180,
                   depths=[6, 6, 6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="default_sr",
                   resi_connection="1conv",
                   upscale_factor=3,
                   **kwargs)

    return model


def swinir_default_sr_x4(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=180,
                   depths=[6, 6, 6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="default_sr",
                   resi_connection="1conv",
                   upscale_factor=4,
                   **kwargs)

    return model


def swinir_default_sr_x8(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=180,
                   depths=[6, 6, 6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="default_sr",
                   resi_connection="1conv",
                   upscale_factor=8,
                   **kwargs)

    return model


def swinir_lightweight_sr_x2(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=60,
                   depths=[6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="lightweight_sr",
                   resi_connection="1conv",
                   upscale_factor=2,
                   **kwargs)

    return model


def swinir_lightweight_sr_x3(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=60,
                   depths=[6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="lightweight_sr",
                   resi_connection="1conv",
                   upscale_factor=3,
                   **kwargs)

    return model


def swinir_lightweight_sr_x4(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=60,
                   depths=[6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="lightweight_sr",
                   resi_connection="1conv",
                   upscale_factor=4,
                   **kwargs)

    return model


def swinir_lightweight_sr_x8(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=60,
                   depths=[6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="lightweight_sr",
                   resi_connection="1conv",
                   upscale_factor=8,
                   **kwargs)

    return model


def swinir_real_sr_x2(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=180,
                   depths=[6, 6, 6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="real_sr",
                   resi_connection="1conv",
                   upscale_factor=2,
                   **kwargs)

    return model


def swinir_real_sr_x3(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=180,
                   depths=[6, 6, 6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="real_sr",
                   resi_connection="1conv",
                   upscale_factor=3,
                   **kwargs)

    return model


def swinir_real_sr_x4(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=180,
                   depths=[6, 6, 6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="real_sr",
                   resi_connection="1conv",
                   upscale_factor=4,
                   **kwargs)

    return model


def swinir_real_sr_x8(**kwargs) -> SwinIR:
    model = SwinIR(window_size=8,
                   embed_dim=180,
                   depths=[6, 6, 6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsample_method="real_sr",
                   resi_connection="1conv",
                   upscale_factor=8,
                   **kwargs)

    return model


def discriminator_unet(**kwargs) -> DiscriminatorUNet:
    model = DiscriminatorUNet(**kwargs)

    return model
