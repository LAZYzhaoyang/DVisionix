# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: SegFormerV3 分割头（overlap embed + MixFFN + SE 通道注意力融合解码）。
"""SegFormerV3 分割头（overlap embed + MixFFN + SE 通道注意力融合解码）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel
from ...layers import LayerNorm2d, MixFFN, SEBlock


@HEADS.register()
@HEADS.register(name="segformer_v3_head")
class SegFormerV3Head(BaseModel):
    """SegFormerV3-lite 解码头：每层 overlap patch embed（4x4 stride2）+ MixFFN 增强，
    多尺度上采样融合后经 SE 通道注意力与深度可分离解码输出。

    相比 V2 增加 SE 融合与更深的解码头，多尺度输入（input_style="multi_scale"）即插即用。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(
        self,
        in_channels_list,
        num_classes,
        d_model: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.d_model = d_model
        self.in_channels_list = list(in_channels_list)
        levels = len(in_channels_list)

        self.overlap = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for c in in_channels_list:
            self.overlap.append(
                nn.Sequential(
                    nn.Conv2d(c, d_model, kernel_size=4, stride=2, padding=1),
                    LayerNorm2d(d_model),
                    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                )
            )
            self.blocks.append(nn.ModuleList([MixFFN(d_model) for _ in range(num_blocks)]))
        self.se = SEBlock(d_model * levels, reduction=8)
        self.decode = nn.Sequential(
            nn.Conv2d(d_model * levels, d_model, kernel_size=3, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(d_model, num_classes, 1),
        )

    def forward(self, feats):
        """SegFormerV3Head 前向：多层特征 -> logits (B, C, H, W)。"""
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        outs = []
        for i, f in enumerate(feats):
            x = self.overlap[i](f)
            for blk in self.blocks[i]:
                x = blk(x)
            outs.append(x)
        target = outs[0].shape[-2:]
        fused = torch.cat(
            [F.interpolate(o, size=target, mode="bilinear", align_corners=False) for o in outs],
            dim=1,
        )
        fused = fused * self.se(fused)
        return self.decode(fused)


__all__ = ["SegFormerV3Head"]
