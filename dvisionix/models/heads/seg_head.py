# -*- coding: utf-8 -*-
"""分割头。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ...registry import HEADS


@HEADS.register()
@HEADS.register(name="seg_head")
class SegHead(BaseModel):
    """简单分割头：1x1 卷积将特征图映射到类别 logits。

    输入: 特征图 (B, in_channels, H, W)；输出: logits (B, num_classes, H, W)。
    若给定 output_size，则插值到目标尺寸。
    """

    def __init__(self, in_channels, num_classes, output_size=None, dropout=0.0):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.output_size = output_size
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.output_size is not None and out.shape[-2:] != tuple(self.output_size):
            out = F.interpolate(out, size=tuple(self.output_size), mode="bilinear", align_corners=False)
        return out


@HEADS.register()
@HEADS.register(name="fcn_head")
class FCNHead(BaseModel):
    """FCN 风格分割头：conv3x3 + conv1x1 -> logits。"""

    def __init__(self, in_channels, num_classes, mid_channels=256, output_size=None):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.output_size = output_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.output_size is not None and out.shape[-2:] != tuple(self.output_size):
            out = F.interpolate(out, size=tuple(self.output_size), mode="bilinear", align_corners=False)
        return out


@HEADS.register()
@HEADS.register(name="deeplabv3_head")
class DeepLabV3Head(BaseModel):
    """DeepLabV3 分割头（ASPP 空洞空间金字塔池化）。"""

    def __init__(self, in_channels, num_classes, atrous_rates=(6, 12, 18), output_size=None):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.output_size = output_size

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, 256, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=atrous_rates[0], dilation=atrous_rates[0]), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=atrous_rates[1], dilation=atrous_rates[1]), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=atrous_rates[2], dilation=atrous_rates[2]), nn.ReLU(inplace=True))
        self.pool_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 256, 1), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(256 * 5, 256, 1), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        pool = F.interpolate(self.pool_branch(x), size=size, mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x), pool], dim=1))
        out = self.out_conv(fused)
        if self.output_size is not None and out.shape[-2:] != tuple(self.output_size):
            out = F.interpolate(out, size=tuple(self.output_size), mode="bilinear", align_corners=False)
        return out


@HEADS.register()
@HEADS.register(name="unet_decoder")
class UNetDecoder(BaseModel):
    """U-Net 风格解码器：多尺度特征（高->低）上采样 + 跳跃连接 -> logits。

    输入：backbone features_only 输出列表 [f1(高), ..., fn(低)]。
    输出：(B, num_classes, H, W)（与最高层特征分辨率一致）。
    """

    def __init__(self, in_channels_list, num_classes, base_channels=64, output_size=None):
        super().__init__(task_type="segmentation")
        self.in_channels_list = list(in_channels_list)
        self.num_classes = num_classes
        self.output_size = output_size

        rev = list(reversed(self.in_channels_list))
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        cin = rev[0]
        for i in range(1, len(rev)):
            up = nn.ConvTranspose2d(cin, rev[i], kernel_size=2, stride=2)
            conv = self._double_conv(rev[i] + rev[i], rev[i])
            self.ups.append(up)
            self.convs.append(conv)
            cin = rev[i]
        self.final = nn.Conv2d(cin, num_classes, 1)

    @staticmethod
    def _double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        x = feats[-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = feats[len(feats) - 2 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.convs[i](x)
        out = self.final(x)
        if self.output_size is not None and out.shape[-2:] != tuple(self.output_size):
            out = F.interpolate(out, size=tuple(self.output_size), mode="bilinear", align_corners=False)
        return out


__all__ = ["SegHead", "FCNHead", "DeepLabV3Head", "UNetDecoder"]
