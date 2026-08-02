# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: U-Net 风格解码器（UNetDecoder）。
"""U-Net 风格解码器（UNetDecoder）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="unet_decoder")
class UNetDecoder(BaseModel):
    """U-Net 风格解码器：多尺度特征（高->低）上采样 + 跳跃连接 -> logits。"""

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

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
        """UNetDecoder 前向：多尺度编码特征 -> logits (B, C, H, W)。"""
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
            out = F.interpolate(
                out, size=tuple(self.output_size), mode="bilinear", align_corners=False
            )
        return out


__all__ = ["UNetDecoder"]
