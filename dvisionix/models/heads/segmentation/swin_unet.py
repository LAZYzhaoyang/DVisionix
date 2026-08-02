# -*- coding: utf-8 -*-
"""Swin-UNet 风格分割解码器（PatchExpand 逐级上采样 + 跳连融合，compact）。"""

import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel
from ...layers import PatchExpand


@HEADS.register()
@HEADS.register(name="swin_unet_decoder")
class SwinUNetDecoder(BaseModel):
    """Swin-UNet 风格解码器：多尺度特征统一通道后，从最深层 PatchExpand 上采样并与跳连特征相加。

    多尺度输入（input_style="multi_scale"）；PatchExpand = LN + Linear(2x) + PixelShuffle(2)。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(self, in_channels_list, num_classes, d_model: int = 64):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.d_model = d_model
        self.in_channels_list = list(in_channels_list)
        levels = len(in_channels_list)

        self.align = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in in_channels_list])
        self.expands = nn.ModuleList([PatchExpand(d_model) for _ in range(levels - 1)])
        self.merge = nn.ModuleList([nn.Conv2d(d_model // 2, d_model, 1) for _ in range(levels - 1)])
        self.out_conv = nn.Conv2d(d_model, num_classes, 1)

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        feats = [conv(f) for conv, f in zip(self.align, feats)]
        x = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            x = self.expands[i](x)  # (B, d//2, 2H, 2W)
            x = self.merge[i](x)  # (B, d, 2H, 2W)
            skip = feats[i]
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip
        return self.out_conv(x)


__all__ = ["SwinUNetDecoder"]
