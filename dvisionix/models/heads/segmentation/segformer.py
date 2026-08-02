# -*- coding: utf-8 -*-
"""SegFormer 分割头（分层 transformer 特征 + MLP 解码）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="segformer_head")
class SegFormerHead(BaseModel):
    """SegFormer MLP 解码头：多级特征上采样到最高分辨率后 concat -> MLP -> logits。

    输入：多尺度特征列表（高分辨率在前）；输出 (B, num_classes, H, W)（与最高层同分辨率）。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(self, in_channels_list, num_classes, channels=256):
        super().__init__(task_type="segmentation")
        self.in_channels_list = list(in_channels_list)
        self.num_classes = num_classes
        self.channels = channels
        self.linear_fuse = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in in_channels_list])
        self.fuse_conv = nn.Conv2d(channels * len(in_channels_list), channels, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        target = feats[0].shape[-2:]
        fused = []
        for i, f in enumerate(feats):
            f = self.linear_fuse[i](f)
            if f.shape[-2:] != target:
                f = F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            fused.append(f)
        x = self.fuse_conv(torch.cat(fused, dim=1))
        return self.linear_pred(self.dropout(F.relu(x)))


__all__ = ["SegFormerHead"]
