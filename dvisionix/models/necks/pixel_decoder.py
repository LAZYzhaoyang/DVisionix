# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: PixelDecoder 颈部（Mask2Former 像素解码器，与 FPN/PANet 同级）。
"""PixelDecoder 颈部（Mask2Former 像素解码器，与 FPN/PANet 同级）。"""

import torch.nn as nn
import torch.nn.functional as F

from ...registry import NECKS
from ..base import BaseModel


@NECKS.register()
@NECKS.register(name="pixel_decoder")
class PixelDecoder(BaseModel):
    """轻量 FPN 像素解码器：多尺度特征自顶向下融合到统一通道。

    供 Mask2Former 等需要"多尺度 -> 统一通道像素特征"的模型使用；
    输出 List[Tensor]，最精细在前。
    """

    def __init__(self, in_channels, d_model: int = 256):
        super().__init__()
        self.in_channels = list(in_channels)
        self.d_model = d_model
        self.out_channels = d_model
        self.lateral = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in self.in_channels])
        self.fpn = nn.ModuleList(
            [nn.Conv2d(d_model, d_model, 3, padding=1) for _ in range(len(self.in_channels))]
        )

    def forward(self, feats):
        """PixelDecoder 前向：多尺度特征 -> 多尺度掩码特征（最精细在前）。"""
        laterals = [conv(f) for conv, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False
            )
        return [conv(f) for conv, f in zip(self.fpn, laterals)]


__all__ = ["PixelDecoder"]
